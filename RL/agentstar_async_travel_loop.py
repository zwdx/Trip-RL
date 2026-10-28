"""Agent-STAR adapter to the pinned native V1 TransferQueue implementation."""
import asyncio
import copy
import gzip
import json
from pathlib import Path
from uuid import uuid4
import os
import time

import ray
import torch
import transfer_queue as tq
from omegaconf import open_dict
from tensordict import NonTensorData, NonTensorStack

from agentstar_travel_loop import AgentStarTravelLoop, curriculum_stage, split_action_history_outputs
from verl.experimental.agent_loop.agent_loop import register
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync


class VersionTrackingClient:
    """Keep versions across tool turns without mutating the shared worker client."""
    def __init__(self, client):
        self.client = client
        self.minimum = self.maximum = None

    async def generate(self, *args, **kwargs):
        output = await self.client.generate(*args, **kwargs)
        versions = output.extra_fields
        low, high = versions.get('min_global_steps'), versions.get('max_global_steps')
        if low is None or high is None or low > high:
            raise ValueError('Async generation requires valid policy-version bounds')
        self.minimum = low if self.minimum is None else min(low, self.minimum)
        self.maximum = high if self.maximum is None else max(high, self.maximum)
        return output


@register('agentstar_travel_async')
class AgentStarAsyncTravelLoop(AgentStarTravelLoop):
    async def run(self, sampling_params, **kwargs):
        original = self.server_manager
        tracker = VersionTrackingClient(original)
        self.server_manager = tracker
        try:
            output = await super().run(sampling_params, **kwargs)
        finally:
            self.server_manager = original
        if tracker.minimum is None and any(output.response_mask):
            raise ValueError('Unversioned trainable trajectory')
        # No generated tokens in a timeout: use its submission version, with a zero mask.
        fallback = int(kwargs['global_steps'])
        output.extra_fields['admission_wait_seconds'] = kwargs.get('admission_wait_seconds', 0.)
        output.extra_fields.update(min_global_steps=tracker.minimum if tracker.minimum is not None else fallback,
                                   max_global_steps=tracker.maximum if tracker.maximum is not None else fallback)
        await asyncio.to_thread(self._save_trajectory, output, kwargs)
        if not kwargs['extra_info']['agentstar_settings'].get('preserve_thinking', True):
            return split_action_history_outputs(output)
        return output

    def _save_trajectory(self, output, sample):
        folder = Path(self.config.trainer.default_local_dir)/'trajectory_audit'/str(sample['global_steps'])
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{sample.get('uid', 'unknown')}_{sample.get('session_id', 'unknown')}_{uuid4().hex}.json.gz"
        # Raw student tokens and timeout progress; teacher tensors are scored later.
        with gzip.open(folder/name, 'wt', encoding='utf-8') as stream:
            json.dump({'uid':sample.get('uid'), 'session_id':sample.get('session_id'),
                       'submission_step':sample['global_steps'], 'output':output.model_dump()}, stream, ensure_ascii=False, default=lambda value:value.tolist())


def attach_async_progress(prompts, total):
    validate = bool(prompts['validate']) if 'validate' in prompts else False
    step = int(prompts['global_steps'])
    if not validate:
        curriculum_stage(step, total)
    prompts['agentstar_progress'] = NonTensorData({'step':step, 'total':total, 'validate':validate})


def fill_masked_teacher(output, pad_token):
    # Keep the same fields as successful OPD samples when teacher scoring is skipped.
    if output.extra_fields.get('skip_teacher', False):
        width = len(output.prompt_ids) + len(output.response_ids)
        output.extra_fields['teacher_ids'] = torch.full((width, 1), pad_token or 0, dtype=torch.int32)
        output.extra_fields['teacher_logprobs'] = torch.zeros((width, 1), dtype=torch.float32)


class AgentStarAsyncLoopWorker(AgentLoopWorkerTQ.__ray_metadata__.modified_class):
    def __init__(self, config, *args, **kwargs):
        os.environ['AGENT_STAR_FORMATTER_URL'] = config.agentstar.formatter_url
        os.environ['AGENT_STAR_FORMATTER_MODEL'] = config.agentstar.formatter_model
        super().__init__(config, *args, **kwargs)
        rollout = config.actor_rollout_ref.rollout
        replicas = config.trainer.nnodes * config.trainer.n_gpus_per_node // rollout.tensor_model_parallel_size
        slots = replicas * rollout.max_num_seqs // rollout.agent.num_workers
        if slots < 1:
            raise ValueError('Agent workers exceed student sequence capacity')
        self._admission = asyncio.Semaphore(slots)

    async def _run_agent_loop(self, *args, **kwargs):
        started = time.monotonic()
        async with self._admission:
            kwargs['admission_wait_seconds'] = time.monotonic()-started
            return await super()._run_agent_loop(*args, **kwargs)

    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        if not validate and self.distillation_enabled:
            items = output if isinstance(output, list) else [output]
            for item in items:
                fill_masked_teacher(item, self.tokenizer.pad_token_id)
            # Native TQ scores only the final output; score earlier turns using
            # each turn's actual input after removing historical reasoning.
            for item in items[:-1]:
                await self._compute_teacher_logprobs(
                    item, prompt_ids=item.prompt_ids, response_ids=item.response_ids,
                    validate=validate, sample_kwargs=kwargs)
        return await super()._agent_loop_postprocess(output, validate, **kwargs)


class AgentStarAsyncLoopManager(AgentLoopManagerTQ):
    async def _init_agent_loop_workers(self):
        self.config = copy.deepcopy(self.config)
        if not self.config.agentstar.formatter_url:
            if not self.teacher_client or len(self.teacher_client) != 1:
                raise ValueError('Shared formatter requires exactly one teacher model')
            client = next(iter(self.teacher_client.values()))
            addresses = await client._load_balancer.get_all_servers.remote()
            if len(addresses) != 1:
                raise ValueError('Shared formatter requires one replica, or an explicit formatter_url')
            with open_dict(self.config.agentstar):
                self.config.agentstar.formatter_url = f'http://{addresses[0]}/v1'
                self.config.agentstar.formatter_model = self.config.distillation.teacher_models.teacher_model.model_path
        self.agent_loop_workers_class = ray.remote(AgentStarAsyncLoopWorker)
        await super()._init_agent_loop_workers()

    def generate_sequences(self, prompts):
        attach_async_progress(prompts, int(self.config.trainer.total_training_steps))
        return super().generate_sequences(prompts)


class AgentStarAsyncReplayBuffer(ReplayBufferAsync):
    def sample(self, global_steps, partition_id, batch_size):
        batch, metrics = super().sample(global_steps, partition_id, batch_size)
        if partition_id == 'train':
            data = tq.kv_batch_get(keys=batch.keys, partition_id=partition_id, select_fields=['response_mask', 'extra_fields'])
            mask = data['response_mask']
            # Count each trajectory once, even when no-history mode emits many turns.
            sessions = {}
            for key, extra in zip(batch.keys, data['extra_fields'], strict=True):
                session, index = key.rsplit('_', 1)
                if session not in sessions or int(index) > sessions[session][0]:
                    sessions[session] = (int(index), extra.get('reward_extra_info', {}))
            summaries = [value[1] for value in sessions.values()]
            for field in ('task_reward', 'training_reward', 'soft_budget_penalty',
                          'invalid_action_penalty', 'model_failure', 'infrastructure_failure', 'acc'):
                metrics[f'training/trajectory/{field}_mean'] = sum(float(row.get(field, 0.)) for row in summaries)/len(summaries)
            metrics['training/trajectory/count'] = len(summaries)
            keep = [key for key, row in zip(batch.keys, mask.unbind(), strict=True) if row.any()]
            if not keep:
                raise RuntimeError('All trajectories were filtered; stop before an empty optimizer update')
            # Exclude infrastructure failures from group reward statistics too,
            # rather than letting their placeholder zero rewards shift GRPO means.
            metrics['training/filtered_infrastructure_rows'] = len(batch.keys)-len(keep)
            batch = batch.select_keys(keep)
        return batch, metrics
