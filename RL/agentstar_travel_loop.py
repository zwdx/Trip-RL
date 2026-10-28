"""veRL v0.9.1 loop adapter for the author's TravelAgentFormat and environment.

Generated token IDs are retained; veRL's Continuous Token builder supplies
context boundaries and masks. Task/tool/reward logic stays in the author modules.
"""
import asyncio
import copy
import logging
import json
import math
import os
import time
from uuid import uuid4

import numpy as np
import ray
from omegaconf import open_dict

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase, AgentLoopManager, AgentLoopWorker, AgentLoopMetrics,
    AgentLoopOutput, register,
)
from verl.utils.ray_utils import auto_await
from rllm.agents.travel_agent import TravelAgentFormat
from rllm.environments.tools.travel_tool_env import TravelToolEnvironment
from rllm.rewards.travel_reward import (
    sum_travel_reward_fn, success_travel_reward_fn, planner_r1_reward_fn,
)

TRAVEL_TOOLS = ['SearchFlight', 'SearchCity', 'SearchAttraction',
                'SearchRestaurant', 'SearchAccommodation', 'GoogleDistanceMatrix']



def render_action_history(tokenizer, messages):
    """Drop previous reasoning from model input; keep the original audit messages."""
    history = copy.deepcopy(messages)
    for message in history:
        if message.get('role') == 'assistant':
            content = message.get('content', '')
            if '</think>' in content:
                message['content'] = content.split('</think>', 1)[1].lstrip()
            message.pop('reasoning_content', None)
    # This checkpoint's template otherwise inserts empty thinking into tool history.
    branch = 'if loop.index0 > ns.last_query_index'
    template = tokenizer.chat_template
    if template.count(branch) != 1:
        raise ValueError('Unsupported history-thinking template; refusing to alter model input')
    template = template.replace(branch, branch + ' and preserve_thinking')
    ids = tokenizer.apply_chat_template(
        history, tokenize=True, add_generation_prompt=True, chat_template=template,
        enable_thinking=True, preserve_thinking=False)
    return list(ids['input_ids'] if hasattr(ids, 'keys') else ids)


def split_action_history_outputs(output):
    """Native V1 broadcasts the final session advantage to each turn output."""
    samples = output.extra_fields.pop('turn_samples', [])
    if not samples or not any(output.response_mask):
        return output
    outputs = []
    for index, sample in enumerate(samples):
        extra = {key: value for key, value in output.extra_fields.items()
                 if key not in ('calls', 'steps', 'messages')}
        extra['turn_index'] = index
        outputs.append(AgentLoopOutput(
            prompt_ids=sample['prompt_ids'], response_ids=sample['response_ids'],
            response_mask=[1] * len(sample['response_ids']),
            response_logprobs=sample['response_logprobs'],
            reward_score=output.reward_score, num_turns=output.num_turns,
            metrics=output.metrics if index == len(samples)-1 else AgentLoopMetrics(),
            extra_fields=extra))
    return outputs



def invalid_tool_call(actions, schemas):
    """Only deterministic schema errors are attributed to the policy."""
    definitions = {schema['function']['name']: schema['function']['parameters'] for schema in schemas}
    for action in actions:
        function = action.get('function', {})
        name = function.get('name')
        if name == 'finish':
            continue
        if name not in definitions:
            return f'Unknown tool: {name}'
        arguments = function.get('arguments', {})
        try:
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (ValueError, TypeError):
            return 'Tool arguments are not valid JSON'
        if not isinstance(arguments, dict):
            return 'Tool arguments must be an object'
        schema = definitions[name]
        if set(schema.get('required', [])) - arguments.keys():
            return f'Missing required arguments: {name}'
        # All six travel tools accept exactly their declared named parameters.
        if arguments.keys() - schema.get('properties', {}).keys():
            return f'Unknown arguments: {name}'
        for key, value in arguments.items():
            rule = schema.get('properties', {}).get(key, {})
            if rule.get('type') == 'string' and not isinstance(value, str):
                return f'Argument must be a string: {name}.{key}'
            if 'enum' in rule and value not in rule['enum']:
                return f'Invalid argument value: {name}.{key}'
    return None


def shape_failure_reward(task_reward, termination, token_fraction, turn_fraction, policy, validate=False):
    """Budget costs are deterministic; wall-clock service time is never penalized."""
    if validate or not policy.get('enabled', False):
        return float(task_reward), 0., 0.
    buffer = float(policy.get('soft_buffer_fraction', .2))
    cap = float(policy.get('max_soft_penalty', 1.))
    invalid = float(policy.get('invalid_action_penalty', 1.))
    if not 0 < buffer <= 1 or any(not math.isfinite(x) or x < 0 for x in (cap, invalid)):
        raise ValueError('Invalid failure reward parameters')
    usage = min(1., max(0., float(token_fraction), float(turn_fraction)))
    soft = cap if usage >= 1. else cap * max(0., (usage-(1.-buffer))/buffer)
    # A single hard failure score avoids counting malformed truncated text twice.
    if termination in ('FORMAT_ERROR', 'INVALID_TOOL_CALL'):
        return -invalid, 0., invalid
    return float(task_reward)-soft, soft, 0.


def curriculum_stage(step, total):
    if total <= 0 or not 1 <= step <= total:
        raise ValueError(f'Invalid curriculum progress {step}/{total}')
    return 'SUM' if step / total <= 0.33 else 'MACRO' if step / total <= 0.67 else 'SUCCESS'


def attach_progress(batch, total):
    """Read restored trainer metadata before dispatching samples to Ray workers."""
    validate = bool(batch.meta_info.get('validate', False))
    step = int(batch.meta_info.get('global_steps', -1))
    if not validate:
        curriculum_stage(step, total)
    batch.non_tensor_batch['agentstar_progress'] = np.array(
        [{'step': step, 'total': total, 'validate': validate} for _ in range(len(batch))], dtype=object)


class AgentStarLoopWorker(AgentLoopWorker):
    def __init__(self, config, *args, **kwargs):
        os.environ['AGENT_STAR_FORMATTER_URL'] = config.agentstar.formatter_url
        os.environ['AGENT_STAR_FORMATTER_MODEL'] = config.agentstar.formatter_model
        super().__init__(config, *args, **kwargs)


class AgentStarLoopManager(AgentLoopManager):
    async def _init_agent_loop_workers(self):
        # This version-pinned adapter reuses the native teacher's OpenAI endpoint.
        # ponytail: one teacher endpoint; add request routing only for multiple replicas.
        self.config = copy.deepcopy(self.config)
        if not self.config.agentstar.formatter_url:
            if not self.teacher_client or len(self.teacher_client) != 1:
                raise ValueError('Provide agentstar.formatter_url when no single OPD teacher is present')
            client = next(iter(self.teacher_client.values()))
            addresses = await client._load_balancer.get_all_servers.remote()
            if len(addresses) != 1:
                raise ValueError('Shared formatter requires one teacher replica; otherwise supply formatter_url')
            with open_dict(self.config.agentstar):
                self.config.agentstar.formatter_url = f'http://{addresses[0]}/v1'
                self.config.agentstar.formatter_model = self.config.distillation.teacher_models.teacher_model.model_path
        self.agent_loop_workers_class = ray.remote(AgentStarLoopWorker)
        await super()._init_agent_loop_workers()

    @auto_await
    async def generate_sequences(self, prompts):
        attach_progress(prompts, int(self.config.trainer.total_training_steps))
        output = await super().generate_sequences(prompts)
        if not prompts.meta_info.get('validate', False) and not output.batch['response_mask'].any():
            raise RuntimeError('All trajectories were filtered; stop before an empty optimizer update')
        return output


logger = logging.getLogger(__name__)

RETRYABLE_ROLLOUT_ERRORS = (ConnectionError, OSError)


@register('agentstar_travel')
class AgentStarTravelLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        timeout = float(kwargs['extra_info']['agentstar_settings'].get('timeout_seconds', 900))
        start = time.monotonic()
        self._trace = {'phase':'setup', 'phase_started':start, 'timing':{}, 'calls':[], 'steps':[], 'messages':[]}
        try:
            output = await asyncio.wait_for(self._run(sampling_params, **kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning('AgentStar trajectory timed out after %.0fs; returning a masked sample', timeout)
            output = self._masked_output(kwargs, reason='TIMEOUT')
        except RETRYABLE_ROLLOUT_ERRORS as exc:
            logger.warning('AgentStar rollout hit a retryable error (%s); retrying once', exc)
            try:
                output = await asyncio.wait_for(self._run(sampling_params, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning('AgentStar retry timed out after %.0fs; returning a masked sample', timeout)
                output = self._masked_output(kwargs, reason='TIMEOUT')
            except Exception as retry_exc:
                logger.exception('AgentStar retry failed (%s); returning a masked sample', retry_exc)
                output = self._masked_output(kwargs, reason='ERROR')
        except Exception as exc:
            logger.exception('AgentStar rollout failed (%s); returning a masked sample', exc)
            output = self._masked_output(kwargs, reason='ERROR')
        elapsed = time.monotonic() - start
        self._record_phase(None)
        output.extra_fields['phase_seconds'] = dict(self._trace['timing'])
        output.extra_fields['elapsed_seconds'] = elapsed
        output.extra_fields['timeout_seconds'] = timeout
        if elapsed >= timeout * 0.75:
            logger.warning('AgentStar trajectory slow: elapsed=%.1fs timeout=%.0fs termination=%s turns=%s',
                           elapsed, timeout, output.extra_fields.get('termination'), output.num_turns)
        return output

    def _record_phase(self, phase):
        trace = self._trace
        now = time.monotonic()
        previous = trace['phase']
        if previous is not None:
            trace['timing'][previous] = trace['timing'].get(previous, 0.) + now-trace['phase_started']
        trace['phase'], trace['phase_started'] = phase, now

    def _masked_output(self, kwargs, reason):
        settings = kwargs['extra_info']['agentstar_settings']
        task = kwargs['extra_info'].get('task_info') or {}
        progress = kwargs.get('agentstar_progress')
        prompt_ids = getattr(self, '_initial_ids', None)
        if not prompt_ids:
            prompt_ids = self.tokenizer.encode('timeout', add_special_tokens=False)
        if not prompt_ids:
            fallback = self.tokenizer.eos_token_id
            if fallback is None:
                fallback = self.tokenizer.pad_token_id
            prompt_ids = [0 if fallback is None else fallback]
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.pad_token_id
        if eos is None:
            eos = 0
        stage = 'SUM'
        if settings.get('reward_type') == 'PLANNER_R1' and progress is not None:
            try:
                stage = ('VALIDATION_SUCCESS' if progress['validate']
                         else curriculum_stage(progress['step'], progress['total']))
            except (KeyError, TypeError, ValueError):
                stage = 'SUM'
        return AgentLoopOutput(
            prompt_ids=list(prompt_ids),
            response_ids=[eos],
            response_mask=[0],
            response_logprobs=[0.0],
            reward_score=0.0,
            num_turns=len(self._trace.get('calls', [])),
            metrics=self._trace.get('metrics', AgentLoopMetrics()),
            extra_fields={
                'task_id': task.get('id', ''),
                'termination': reason,
                'judgement': {},
                'reward_extra_info': {
                    'acc': 0.0,
                    'task_reward': 0.0,
                    'training_reward':0.0, 'soft_budget_penalty':0.0,
                    'invalid_action_penalty':0.0, 'model_failure':0.0,
                    'infrastructure_failure':1.0,
                    'curriculum_stage': stage,
                    'termination': reason,
                },
                'curriculum_progress': progress,
                'calls': copy.deepcopy(self._trace.get('calls', [])),
                'steps': copy.deepcopy(self._trace.get('steps', [])),
                'messages': copy.deepcopy(self._trace.get('messages', [])),
                'failure_phase': self._trace['phase'],
                'pending_action': copy.deepcopy(self._trace.get('pending_action')),
                'partial_response_ids': list(self._trace.get('runtime_ids', []))[len(prompt_ids):],
                'partial_response_logprobs': list(self._trace.get('logprobs', [])),
                'policy': 'timeout_zero_mask',
                'skip_teacher': True,
            },
        )

    async def _run(self, sampling_params, **kwargs):
        self._initial_ids = None
        task = copy.deepcopy(kwargs['extra_info']['task_info'])
        settings = kwargs['extra_info']['agentstar_settings']
        tools = TRAVEL_TOOLS
        progress = kwargs.get('agentstar_progress')
        stage = 'SUM'
        reward_fn = sum_travel_reward_fn
        if settings.get('reward_type') == 'PLANNER_R1':
            if progress is None:
                raise ValueError('Curriculum requires trainer progress, never agent turn counts')
            if progress['validate']:
                stage = 'VALIDATION_SUCCESS'
            else:
                stage = curriculum_stage(progress['step'], progress['total'])

            def reward_fn(task_info, action, **_):
                if progress['validate']:
                    return success_travel_reward_fn(task_info, action)
                return planner_r1_reward_fn(task_info, action,
                    current_step=progress['step'], total_steps=progress['total'])
        agent = TravelAgentFormat(tools=tools, system_prompt=settings['system_prompt'])
        env = TravelToolEnvironment(task=task, tools=tools,
            reward_fn=reward_fn, max_steps=settings.get('max_turns', 60))
        observation, info = env.reset()
        agent.update_from_env(observation, 0., False, info)
        runtime_ids = await self.ct_build_initial_tokens(agent.chat_completions)
        initial_ids = runtime_ids.copy()
        self._initial_ids = initial_ids
        self._trace['runtime_ids'] = runtime_ids
        self._trace['messages'] = agent.chat_completions
        if len(initial_ids) > self.rollout_config.prompt_length:
            raise ValueError('Initial prompt must not be truncated')
        mask, logprobs, calls, steps = [], [], [], []
        metrics = AgentLoopMetrics()
        self._trace.update(calls=calls, steps=steps, metrics=metrics, turn_samples=[])
        reward, metadata, termination = 0., {}, 'MAX_STEPS'
        failure_policy = settings.get('failure_reward', {'enabled': False})
        learn_failures = bool(failure_policy.get('enabled', False))
        validate = bool(progress and progress['validate'])
        invalid_reason = None
        length_fraction = 0.
        response_limit = self.rollout_config.response_length
        request_id = kwargs.get('request_id') or uuid4().hex
        for turn in range(settings.get('max_turns', 60)):
            remaining = min(response_limit-len(mask), settings['max_context']-len(runtime_ids))
            if remaining <= 0:
                termination = 'TRUNCATION'
                break
            generation_ids = runtime_ids
            if not settings.get('preserve_thinking', True):
                generation_ids = await asyncio.to_thread(render_action_history, self.tokenizer, agent.chat_completions)
                remaining = min(remaining, settings['max_context']-len(generation_ids))
                if remaining <= 0:
                    termination = 'TRUNCATION'
                    break
            params = {**sampling_params, 'max_tokens':min(settings['max_tokens'], remaining),
                      'stop_token_ids':[248046,248044]}
            if progress is not None and progress['validate']:
                params['repetition_penalty'] = settings.get('validation_repetition_penalty', 1.1)
            call = {'turn':turn, 'prompt_tokens':len(generation_ids), 'completion_tokens':0,
                    'max_tokens':params['max_tokens'], 'status':'running'}
            calls.append(call)
            self._record_phase('generation')
            start = time.monotonic()
            generated = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=generation_ids, sampling_params=params)
            metrics.generate_sequences += time.monotonic()-start
            self._record_phase('processing')
            backend_stop = getattr(generated, 'stop_reason', None)
            finish_reason = getattr(generated, 'extra_fields', {}).get('finish_reason')
            call.update(stop_reason=backend_stop, finish_reason=finish_reason)
            if backend_stop == 'aborted' or finish_reason == 'abort':
                raise RuntimeError('Backend aborted generation; exclude infrastructure failure')
            tokens = list(generated.token_ids)
            if not tokens or len(tokens) > remaining:
                raise RuntimeError('Empty or over-budget backend token output')
            raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
            # The native generation prompt owns the opening thinking token.
            prefix = self.tokenizer.decode(generation_ids[-12:], skip_special_tokens=False)
            response = '<think>\n'+raw if prefix.endswith('<think>\n') and not raw.startswith('<think>') else raw
            for eos in ('<|im_end|>', '<|endoftext|>'):
                if response.endswith(eos):
                    response = response[:-len(eos)]
            generated_logprobs = getattr(generated, 'log_probs', None)
            if settings.get('require_rollout_logprobs', False) and not generated_logprobs:
                raise RuntimeError('Training backend must return sampled-token logprobs')
            if not settings.get('preserve_thinking', True):
                self._trace['turn_samples'].append({
                    'prompt_ids':list(generation_ids), 'response_ids':tokens,
                    'response_logprobs':list(generated_logprobs) if generated_logprobs else None})
            merged, mask, aligned_logprobs = await self.ct_merge_assistant_token(
                runtime_ids, tokens, mask, logprobs if (logprobs or generated_logprobs) else None,
                assistant_logprobs=generated_logprobs)
            logprobs = aligned_logprobs or []
            runtime_ids = merged.token_ids
            self._trace.update(runtime_ids=runtime_ids, logprobs=logprobs)
            call.update(completion_tokens=len(tokens), status='completed')
            action = agent.update_from_model(response).action
            before_observation = copy.deepcopy(agent.chat_completions)
            self._trace.update(messages=before_observation, pending_action=action)
            actions = [action] if isinstance(action, dict) else action if isinstance(action, list) else []
            finished = any(item.get('function', {}).get('name') == 'finish' for item in actions if isinstance(item, dict))
            format_failure = any('Format error occurs' in str(item.get('function', {}).get('arguments', {})) for item in actions)
            if learn_failures:
                invalid_reason = invalid_tool_call(actions, env.tools.json)
                if invalid_reason:
                    termination = 'INVALID_TOOL_CALL'
                    break
                if format_failure:
                    termination = 'TRUNCATION' if call['finish_reason'] == 'length' else 'FORMAT_ERROR'
                    if termination == 'TRUNCATION':
                        length_fraction = 1.
                    break
                if turn + 1 >= settings.get('max_turns', 60) and not finished:
                    termination = 'MAX_STEPS'
                    break
            terminal = isinstance(action, str) or env.step_count+1 >= env.max_steps or finished
            self._record_phase('reward' if terminal else 'tools')
            start = time.monotonic()
            observation, reward, done, info = await asyncio.to_thread(env.step, action)
            metrics.tool_calls += time.monotonic()-start
            self._record_phase('processing')
            metadata = info.get('metadata', {})
            if learn_failures and any(str(value).startswith('Error:') for value in observation.get('tool_outputs', {}).values()):
                raise RuntimeError('Tool infrastructure failure after valid schema')
            error = str(metadata.get('error', ''))
            if settings.get('fail_on_reward_error', False) and error and 'Format error occurs' not in error:
                raise RuntimeError(f'Reward infrastructure failure: {error}')
            steps.append({'response':response,'action':action,'reward':reward,'metadata':metadata})
            agent.update_from_env(observation, reward, done, info)
            self._trace['messages'] = agent.chat_completions
            if done:
                termination = ('FORMAT_ERROR' if 'Format error occurs in the previous turn!' in
                               str(metadata.get('error','')) else 'ENV_DONE')
                # A valid final answer at the budget boundary is still a completed answer.
                if not learn_failures and turn + 1 >= settings.get('max_turns', 60):
                    termination = 'MAX_STEPS'
                break
            if len(mask) >= response_limit or len(runtime_ids) >= settings['max_context']:
                termination = 'TRUNCATION'
                break
            merged, mask, aligned_logprobs = await self.ct_merge_context_msg(
                before_observation, agent.chat_completions, runtime_ids, mask,
                logprobs if logprobs else None)
            logprobs = aligned_logprobs or []
            runtime_ids = merged.token_ids
            self._trace.update(runtime_ids=runtime_ids, logprobs=logprobs)
            if len(mask) >= response_limit or len(runtime_ids) > settings['max_context']:
                termination = 'TRUNCATION'
                break
        if termination in ('TRUNCATION','MAX_STEPS', 'FORMAT_ERROR', 'INVALID_TOOL_CALL'):
            reward = 0.
            if not learn_failures and termination in ('TRUNCATION','MAX_STEPS'):
                mask = [0]*len(mask)
        task_reward = float(reward)
        length_fraction = max(length_fraction, len(mask)/response_limit,
                              len(runtime_ids)/settings['max_context'])
        reward, soft_penalty, invalid_penalty = shape_failure_reward(
            task_reward, termination, length_fraction,
            len(calls)/settings.get('max_turns', 60), failure_policy, validate)
        response_ids = runtime_ids[len(initial_ids):]
        assert len(response_ids) == len(mask)
        if len(response_ids) > response_limit:
            response_ids, mask = response_ids[:response_limit], mask[:response_limit]
        judgement = metadata.get('judgement', {})
        return AgentLoopOutput(prompt_ids=initial_ids,response_ids=response_ids,response_mask=mask,
            response_logprobs=logprobs[:response_limit] if logprobs else None,
            reward_score=float(reward),num_turns=len(calls),metrics=metrics,
            extra_fields={'task_id':task['id'],'termination':termination,'judgement':metadata,
                          'turn_samples':self._trace['turn_samples'],
                           'reward_extra_info': {'acc':float(judgement.get('success_pass', 0.))
                              if termination not in ('TRUNCATION', 'MAX_STEPS') else 0.,
                              'task_reward':task_reward, 'training_reward':float(reward),
                              'soft_budget_penalty':soft_penalty, 'invalid_action_penalty':invalid_penalty,
                              'model_failure':float(termination in ('TRUNCATION','MAX_STEPS','FORMAT_ERROR','INVALID_TOOL_CALL')),
                              'infrastructure_failure':0., 'curriculum_stage':stage,
                              'termination':termination},
                          'curriculum_progress':progress,
                          'calls':calls,'steps':steps,'messages':agent.chat_completions,
                          'invalid_action_reason':invalid_reason,
                          'failure_reward_policy':failure_policy,
                          'policy':'model_failure_rewards_infra_masked' if learn_failures else 'author_overlong_filter'})
