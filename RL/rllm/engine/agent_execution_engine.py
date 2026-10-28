import asyncio
import copy
import logging
import math
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

from rllm.agents.agent import Action, BaseAgent, Trajectory
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.env_utils import (
    compute_mc_return,
    compute_trajectory_reward,
)
from rllm.parser import ChatTemplateParser
from rllm.utils import colorful_print


logger = logging.getLogger(__name__)


########################################################
# Helper functions for ARPO
########################################################
def _entropy_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Compute mean token-level entropy (nats) from logits or logprobs.
    """
    if logits is None or logits.numel() == 0:
        return 0.0
    
    if logits.ndim == 1:
        # logprobs (log probabilities) from vLLM: H = -Σ exp(logp) * logp
        logprobs = logits.float()
        probs = torch.exp(logprobs).clamp(min=1e-12, max=1.0)
        entropy = -(probs * logprobs).sum().item()
        return float(entropy)
    else:
        # raw logits [seq_len, vocab_size]: H = -Σ p * log(p)
        probs = torch.softmax(logits.float() / max(temperature, 1e-8), dim=-1)
        probs = probs.clamp(min=1e-12)
        entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
        return float(entropy)


def _entropy_from_token_ids(token_ids: list, tokenizer) -> float:
    """Empirical entropy (nats) of token sequence from counts. Fallback when logits unavailable."""
    if not token_ids:
        return 0.0
    try:
        t = torch.as_tensor(token_ids, dtype=torch.long)
    except Exception:
        return 0.0
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        t = t[t != pad_id]
    if t.numel() == 0:
        return 0.0
    unique_ids, counts = torch.unique(t, return_counts=True)
    probs = counts.float() / counts.sum()
    probs = probs.clamp_min(1e-12)
    return float(-(probs * probs.log()).sum().item())


class _PathState:
    """
    One path in action-level ARPO. Pure data: snapshot of agent+env at *start* of step_idx.
    """

    __slots__ = (
        "idx", "step_idx", "response_tokens", "response_masks",
        "H_initial", "response_token_len", "prompt_tokens", "origin_task",
        "chat_completions", "trajectory_steps", "env_step_count",
    )

    def __init__(self, idx: int, step_idx: int = 0):
        self.idx = idx
        self.step_idx = step_idx
        self.response_tokens = []
        self.response_masks = []
        self.H_initial = None
        self.response_token_len = 0
        self.prompt_tokens = None
        self.origin_task = None
        self.chat_completions = None  # agent messages at start of this step
        self.trajectory_steps = None  # agent._trajectory.steps at start of this step
        self.env_step_count = 0  # env.step_count at start of this step

    def copy_before_step(self):
        other = _PathState(self.idx, self.step_idx)
        other.response_tokens = list(self.response_tokens)
        other.response_masks = list(self.response_masks)
        other.H_initial = self.H_initial
        other.response_token_len = self.response_token_len
        other.prompt_tokens = list(self.prompt_tokens) if self.prompt_tokens else None
        other.origin_task = copy.deepcopy(self.origin_task) if self.origin_task is not None else None
        other.chat_completions = copy.deepcopy(self.chat_completions) if self.chat_completions else None
        other.trajectory_steps = copy.deepcopy(self.trajectory_steps) if self.trajectory_steps else None
        other.env_step_count = self.env_step_count
        return other


class AgentExecutionEngine:
    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=128,  # The number of active agents
        trajectory_timeout=None,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=64,  # The number of concurrent env operations
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}

        self.config = config
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.max_env_workers = max_workers
        self.overlong_filter = overlong_filter

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length
        self.disable_thinking = self.config.get("rllm", {}).get("disable_thinking", False) if self.config is not None else False

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]

        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e9)

        if env_class is not None:
            assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=self.disable_thinking)
        else:
            self.chat_parser = chat_parser

        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", {})  # for openai api requests

        assert self.engine_name in ["openai", "verl"], "Currently only openai and verl are supported as rollout engine"
        if self.engine_name == "openai":
            from rllm.engine.rollout.openai_engine import OpenAIEngine

            self.rollout_engine = OpenAIEngine(
                **rollout_engine_args,
                api_retries=api_retries,
                tokenizer=self.tokenizer,
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                disable_thinking=self.disable_thinking,
            )
        elif self.engine_name == "verl":
            from rllm.engine.rollout.verl_engine import VerlEngine

            self.rollout_engine = VerlEngine(
                config=self.config,
                rollout_manager=rollout_engine,
                tokenizer=self.tokenizer,
                disable_thinking=self.disable_thinking,
            )

    async def get_model_response(self, prompt, application_id, **kwargs) -> str:
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response text

        Raises:
            NotImplementedError: If the engine type is not supported
        """

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        if self.engine_name == "openai":
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, enforce_max_prompt_length=False, **sampling_params)
            return output.text
        elif self.engine_name == "verl":
            meta_data = sampling_params.pop("meta_info", {})
            validate = meta_data.get("validate", False)
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, validate=validate, enforce_max_prompt_length=False, **sampling_params)
            return output.text
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def update_envs_and_agents(self, envs, agents):
        """
        Update the environments and agents.

        Args:
            envs: List of environments to use
            agents: List of agents to use
        """
        assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        self.envs = envs
        # For keeping track of the environment index in the batch.
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents

    async def run_agent_trajectory_async_arpo(
        self,
        idx: int,
        application_id: str,
        *,
        initial_rollouts: int = 4,
        max_rollouts: int = 8,
        arpo_alpha: float = 0.5,
        arpo_beta: float = 0.3,
        arpo_tau: float = 0.6,
        **kwargs,
    ):
        """
        Action-level ARPO: branch at high-entropy steps (fork path), collect up to max_rollouts
        trajectories in token format. Advantage remains GRPO. `mode` is only for collection API
        compatibility (default Token); ARPO always performs action-level rollout.
        """
        assert max_rollouts >= 1, "max_rollouts must be at least 1"
        initial_rollouts = max(1, min(initial_rollouts, max_rollouts))

        # Action-level ARPO with frontier and branching (always token-form trajectories)
        if not hasattr(self, "executor") or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=self.max_env_workers)
        loop = asyncio.get_event_loop()
        agent = self.agents[idx]
        env = self.envs[idx]

        # meta_info 只在 validate 场景用于区分采样/验证；ARPO 当前只用于训练阶段，
        # 这里先从 kwargs 中移除，避免透传到 vLLM SamplingParams 造成 Unexpected keyword 'meta_info'。
        _ = kwargs.pop("meta_info", None)
        use_logits = hasattr(self.rollout_engine, "get_model_response_with_logits")
        branch_budget = max(0, max_rollouts - initial_rollouts)
        collected = []
        frontier = deque([_PathState(idx, 0) for _ in range(initial_rollouts)])
        total_time = 0.0
        llm_time = 0.0
        env_time = 0.0
        reward_time = None
        termination_reason = None
        reward = 0.0

        # --- 唯一写 path → agent/env 的地方：把 path 对应的“当前步开始前”状态应用到 agent/env ---
        async def _apply_path_to_agent_env(path: _PathState):
            """Make agent and env reflect path. Path is the single source of truth for 'state at start of path.step_idx'."""
            if path.step_idx == 0:
                observation, info = await loop.run_in_executor(self.executor, env.reset)
                info["max_steps"] = self.max_steps
                agent.reset()
                agent.update_from_env(observation=observation, reward=0.0, done=False, info=info)
                # 立刻把当前 agent/env 快照进 path，之后 step_idx>0 都从 path 恢复
                path.origin_task = copy.deepcopy(observation)
                path.chat_completions = copy.deepcopy(agent.chat_completions)
                path.trajectory_steps = copy.deepcopy(agent._trajectory.steps)
                path.env_step_count = getattr(env, "step_count", None)
                prompt_tokens, _ = convert_messages_to_tokens_and_masks(
                    agent.chat_completions, tokenizer=self.tokenizer, parser=self.chat_parser,
                    contains_first_msg=True, contains_generation_msg=True,
                )
                if len(prompt_tokens) > self.max_prompt_length:
                    raise Exception(f"Trajectory {path.idx}: initial prompt length exceeded max_prompt_length")
                path.prompt_tokens = prompt_tokens
                return path.origin_task

            if path.chat_completions is None or path.trajectory_steps is None:
                raise ValueError(
                    f"ARPO path step_idx={path.step_idx} missing chat_completions or trajectory_steps; "
                    "must call _snapshot_agent_env_to_path before appending to frontier."
                )
            if hasattr(agent, "messages"):
                agent.messages[:] = copy.deepcopy(path.chat_completions)
            if hasattr(agent, "_trajectory"):
                agent._trajectory.steps[:] = copy.deepcopy(path.trajectory_steps)
            if hasattr(agent, "current_observation"):
                agent.current_observation = path.trajectory_steps[-1].observation
            if hasattr(env, "step_count") and path.env_step_count is not None:
                env.step_count = path.env_step_count
            return path.origin_task

        # --- 唯一写 agent/env → path 的地方：当前步执行完后，把“下一步开始前”的快照写入 path ---
        def _snapshot_agent_env_to_path(path: _PathState):
            """Write current agent/env state into path for the *next* step; call only before pushing path to frontier."""
            path.step_idx += 1
            path.chat_completions = copy.deepcopy(agent.chat_completions)
            path.trajectory_steps = copy.deepcopy(agent._trajectory.steps)
            path.env_step_count = getattr(env, "step_count", None)

        def _build_token_result(path, trajectory, origin_task, judgement_dict, formatted_plan, termination_reason, reward_time, env_time, llm_time, total_time):
            compute_trajectory_reward(trajectory)
            compute_mc_return(trajectory, gamma=self.gamma)
            prompt_tokens = path.prompt_tokens
            if prompt_tokens is None:
                messages = agent.chat_completions
                prompt_tokens, _ = convert_messages_to_tokens_and_masks(
                    messages, tokenizer=self.tokenizer, parser=self.chat_parser,
                    contains_first_msg=True, contains_generation_msg=True,
                )
            return {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(path.response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(path.response_masks, dtype=torch.long),
                "trajectory_reward": trajectory.reward,
                "judgement": judgement_dict,
                "formatted_plan": formatted_plan,
                "idx": env.idx,
                "chat_completions": agent.chat_completions, # TODO: 是不是也可以 path.chat_completions？
                "termination": termination_reason,
                "item": origin_task,
                "metrics": {"steps": len(trajectory.steps), "reward_time": reward_time, "env_time": env_time, "llm_time": llm_time, "total_time": total_time},
            }

        while frontier and len(collected) < max_rollouts:
            path = frontier.popleft()
            try:
                origin_task = await _apply_path_to_agent_env(path)
            except Exception as e:
                logger.warning("ARPO apply_path_to_agent_env failed for path %s: %s", path.idx, e)
                continue
            # agent/env 现已等于 path 对应的“当前步开始前”状态
            prompt_messages = agent.chat_completions.copy()
            response_token_len = path.response_token_len
            # enforce_max_prompt_length 对齐标准 rollout
            if self.enforce_max_prompt_length:
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    trajectory = agent.trajectory
                    if hasattr(trajectory, "task") and origin_task is not None:
                        trajectory.task = origin_task
                    token_result = _build_token_result(path, trajectory, origin_task, {}, [], termination_reason, None, env_time, llm_time, total_time)
                    if self.overlong_filter and termination_reason in ("TRUNCATION", "MAX_STEPS", "TIMEOUT"):
                        token_result["response_masks"] = torch.zeros_like(token_result["response_masks"])
                    collected.append(token_result)
                    continue
            max_tokens = self.max_response_length - response_token_len if not self.enforce_max_prompt_length else self.max_response_length
            step_kwargs = {**kwargs, "max_tokens": max_tokens}

            if use_logits:
                start_time = time.time()
                output, logits = await self.rollout_engine.get_model_response_with_logits(
                    prompt_messages, application_id=str(uuid.uuid4()), validate=False, enforce_max_prompt_length=False,
                    **step_kwargs
                )
                delta_time = time.time() - start_time
                llm_time += delta_time
                total_time += delta_time
                response = output.text
                completion_ids = getattr(output, "completion_ids", None)
                if completion_ids is None:
                    completion_ids = self.tokenizer.encode(response, add_special_tokens=False)
                H_t = _entropy_from_logits(logits, temperature=1.0) if logits is not None else _entropy_from_token_ids(completion_ids, self.tokenizer)
            else:
                start_time = time.time()
                response = await self.get_model_response(prompt_messages, str(uuid.uuid4()), **step_kwargs)
                delta_time = time.time() - start_time
                llm_time += delta_time
                total_time += delta_time
                completion_ids = self.tokenizer.encode(response, add_special_tokens=False)
                H_t = _entropy_from_token_ids(completion_ids, self.tokenizer)

            if path.H_initial is None:
                path.H_initial = H_t
            delta_H = H_t - path.H_initial
            P_t = arpo_alpha + arpo_beta * delta_H
            if P_t > arpo_tau and branch_budget > 0:
                print(f"[NEW BRANCH] path.idx: {path.idx}, step_idx: {path.step_idx}, H_t: {H_t}, path.H_initial: {path.H_initial}, delta_H: {delta_H}, P_t: {P_t}, branch_budget: {branch_budget}")
                fork = path.copy_before_step()
                frontier.append(fork)
                branch_budget -= 1

            action = agent.update_from_model(response).action
            env_start_time = time.time()
            try:
                remaining = None
                if self.trajectory_timeout is not None:
                    remaining = max(1, self.trajectory_timeout - total_time)
                next_observation, reward, done, info = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, env.step, action),
                    timeout=remaining if remaining is not None else self.trajectory_timeout,
                )
            except asyncio.TimeoutError:
                next_observation, reward, done, info = origin_task, 0.0, True, {}
                termination_reason = "ENV_TIMEOUT"
            env_delta = time.time() - env_start_time
            env_time += env_delta
            total_time += env_delta
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len
            agent.update_from_env(observation=next_observation, reward=reward, done=done, info=info)
            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            assistant_message, env_messages = get_recent_assistant_user_messages(agent.chat_completions)
            assert assistant_message is not None and env_messages is not None
            assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks(
                [assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False
            )
            env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(
                env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True
            )
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            path.response_token_len = response_token_len

            truncation = False
            if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
                truncation_length = self.max_response_length - response_token_len
                if truncation_length < 0:
                    assistant_msg_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                    assistant_msg_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                    env_msg_tokens, env_msg_masks = [], []
                truncation = True
            path.response_tokens.extend(assistant_msg_tokens)
            path.response_masks.extend(assistant_msg_masks)
            if not truncation:
                path.response_tokens.extend(env_msg_tokens)
                path.response_masks.extend(env_msg_masks)

            termination_reason = "TRUNCATION" if truncation else ("ENV_DONE" if done else "MAX_STEPS" if path.step_idx + 1 >= self.max_steps else None)
            if truncation:
                cur_step.reward = 0.0 if (response_token_len - len(env_msg_tokens) > self.max_response_length) else cur_step.reward
                done = True
            if path.step_idx + 1 >= self.max_steps and not done:
                termination_reason = "MAX_STEPS"
                done = True
            if self.trajectory_timeout is not None and total_time >= self.trajectory_timeout and not done:
                termination_reason = "TIMEOUT"
                done = True

            if done or (path.step_idx + 1 >= self.max_steps):
                if hasattr(env, "compute_final_reward"):
                    start_time_reward = time.time()
                    reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
                    reward_time = time.time() - start_time_reward
                    cur_step = agent.get_current_state()
                    cur_step.reward = reward
                await loop.run_in_executor(self.executor, env.close)
                trajectory = agent.trajectory
                task_for_item = path.origin_task or origin_task
                if task_for_item is not None and hasattr(trajectory, "task"):
                    trajectory.task = task_for_item
                jd = info.get("metadata", {}).get("judgement", {})
                fp = info.get("metadata", {}).get("formatted_plan", [])
                token_result = _build_token_result(path, trajectory, task_for_item, jd, fp, termination_reason, reward_time, env_time, llm_time, total_time)
                if self.overlong_filter and termination_reason in ("TRUNCATION", "MAX_STEPS", "TIMEOUT"):
                    token_result["response_masks"] = torch.zeros_like(token_result["response_masks"])
                collected.append(token_result)
            else:
                _snapshot_agent_env_to_path(path)
                frontier.append(path)

        # 若 action-level ARPO 分支提前结束，按论文做法，补齐到 max_rollouts 条
        while len(collected) < max_rollouts:
            extra = await self.run_agent_trajectory_with_retry(
                idx=idx,
                application_id=str(uuid.uuid4()),
                seed=0,
                mode="Token",
                **kwargs,
            )
            collected.append(extra)

        return collected

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agents[idx]
        env = self.envs[idx]
        # env_id = env.env_id
        # print(f"Running agent trajectory asynchronously for agent {agent} and environment {env} with index {idx}")

        termination_reason = None
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        # for step return
        episode_steps = []

        # Reset environment with the task using the executor
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        origin_task, judgement_dict, formatted_plan = observation.copy(), {}, []
        info["max_steps"] = self.max_steps

        # Reset agent
        agent.reset()
        # Update agent internal state from environment.
        agent.update_from_env(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )
        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=True, contains_generation_msg=True)
        prompt_token_len = len(prompt_tokens)
        # Note, this should never happen!
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}, retrying")

        for step_idx in range(self.max_steps):
            # Get action from agent
            prompt_messages = agent.chat_completions.copy()
            # Max remaining tokens left for the response
            # For enforced max prompt at each step, no need to deduct here
            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - response_token_len
            else:
                max_tokens = self.max_response_length

                # since max prompt is enforced, we filter out too long prompts.
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            kwargs["max_tokens"] = max_tokens

            start_time = time.time()
            response = await self.get_model_response(prompt_messages, application_id, **kwargs)
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time
            # Update steps
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
            }
            episode_steps.append(prompt_response_pair)

            # Update agent with model response
            action: Action = agent.update_from_model(response)
            action = action.action

            # Take step in environment using the executor
            start_time = time.time()

            try:
                next_observation, reward, done, info = await asyncio.wait_for(loop.run_in_executor(self.executor, env.step, action), timeout=(self.trajectory_timeout - total_time))
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n", "red")
                reward = 0

                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len

            # Update agent internal state.
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            # Check and convert to tokens if necessary
            assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assert env_messages is not None or mode != "Token", "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assistant_msg_tokens, assistant_msg_masks = [], []
            env_msg_tokens, env_msg_masks = [], []
            if assistant_message:
                assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks([assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False)
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True)

            # Update repsonse token length
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            # Reached maximum number of tokens for the trajectory
            if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
                # Truncation length
                truncation_length = self.max_response_length - response_token_len
                # Truncate the response and masks
                if truncation_length < 0:
                    truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                    truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                else:
                    # Edge case where the response is exactly the max response length.
                    truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                    truncated_response_masks = assistant_msg_masks + env_msg_masks
                # Update token collections
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)

                cur_step = agent.get_current_state()
                if response_token_len - len(env_msg_tokens) > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                # handle returning
                break

            # Update the token version of trajectory
            response_tokens.extend(assistant_msg_tokens)
            response_masks.extend(assistant_msg_masks)
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Check if episode is done
            if done:
                termination_reason = "ENV_DONE"
                judgement_dict = info.get("metadata", {}).get("judgement", {})
                formatted_plan = info.get("metadata", {}).get("formatted_plan", [])
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        masked_out = False
        if self.overlong_filter:
            if termination_reason == "TRUNCATION" or termination_reason == "MAX_STEPS" or termination_reason == "TIMEOUT":
                # Mask out the entire response for overlong trajectories if the reward is 0.
                response_masks = [0] * len(response_masks)
                masked_out = True

        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward
        # Closing environment using the executor.
        await loop.run_in_executor(self.executor, env.close)
        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

        trajectory: Trajectory = agent.trajectory
        # Aggregate final trajectory statistics
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        if mode == "Text":
            trajectory.task = origin_task

            import os 
            if os.environ.get("AGENT_TYPE", "") == "Travel":
                trajectory.info = {"judgement": judgement_dict, "formatted_plan": formatted_plan}
            elif os.environ.get("AGENT_TYPE", "") == "Retail": 
                trajectory.info = {"judgement": judgement_dict} 
                
            return trajectory
        elif mode == "Token":
            token_result = {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "trajectory_reward": trajectory.reward,
                "judgement": judgement_dict,
                "formatted_plan": formatted_plan,
                "idx": env.idx,
                "chat_completions": agent.chat_completions,
                "termination": termination_reason,
                "item": origin_task,
                "metrics": {
                    # Total number of steps taken in the trajectory
                    "steps": len(trajectory.steps),
                    # Time to calculate reward
                    "reward_time": reward_time,
                    # Total time spent in environment execution (env.step)
                    "env_time": env_time,
                    # Time to calculate response tokens
                    "llm_time": llm_time,
                    # Total time spent in the trajectory
                    "total_time": total_time,
                },
            }
            return token_result
        elif mode == "Conversation":
            return agent.chat_completions
        elif mode == "Step":
            steps_result = {
                "steps": episode_steps,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
            }
            return steps_result

    async def run_agent_trajectory_with_retry(self, idx, application_id, seed=0, mode="Text", **kwargs):
        for _ in range(self.retry_limit):
            try:
                return await asyncio.wait_for(self.run_agent_trajectory_async(idx, application_id=application_id, seed=seed, mode=mode, **kwargs), timeout=7200)
            except Exception:
                traceback.print_exc()
                continue
        traceback.print_exc()
        raise Exception(f"Trajectory {idx} cannot complete. Please check the log message")

    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
        if timing_raw is None:
            timing_raw = {}
        import os

        # 仅在训练阶段且 ARPO 环境变量打开时启用 ARPO；
        # 验证阶段 meta_info.validate=True，此时强制走默认 rollout。
        meta_info = kwargs.get("meta_info") or {}
        is_validate = isinstance(meta_info, dict) and meta_info.get("validate", False)
        arpo_enabled = (not is_validate) and os.getenv("ARPO", "").strip().lower() == "true"
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
        assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"  # type: ignore
        if not hasattr(self, "executor") or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=self.max_env_workers)
        semaphore = asyncio.Semaphore(self.n_parallel_agents)

        if self.engine_name == "verl":
            self.rollout_engine.wake_up()

        async def launch_one_trajectory_task(env_idx: int):
            async with semaphore:
                try:
                    # Default rollout (GRPO path): one trajectory per env_idx
                    # ARPO rollout: may return multiple trajectories per env_idx (action-level branching)
                    if arpo_enabled and mode == "Token":
                        result = await self.run_agent_trajectory_async_arpo(
                            idx=env_idx,
                            application_id=str(uuid.uuid4()),
                            **kwargs,
                        )
                    else:
                        result = await self.run_agent_trajectory_with_retry(
                            idx=env_idx,
                            application_id=str(uuid.uuid4()),
                            seed=reset_seed,
                            mode=mode,
                            **kwargs,
                        )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    raise e
                return result

        # Create all N conceptual tasks. Their execution will be throttled by the semaphore
        # and the availability of agent/env indices.
        tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.envs))]

        tasks_completed = 0
        trajectories_yielded = 0
        for coro in asyncio.as_completed(tasks_to_run):
            try:
                result = await coro
                tasks_completed += 1
                # GRPO: result is one trajectory; ARPO: result is a list of trajectories
                if arpo_enabled and mode == "Token":
                    for one in result:
                        trajectories_yielded += 1
                        yield one
                    colorful_print(
                        f"Number of Env-Indices {tasks_completed}/{len(self.envs)} completed; Trajectories yielded={trajectories_yielded}",
                        "cyan",
                    )
                else:
                    trajectories_yielded += 1
                    colorful_print(f"Number of Trajectories {tasks_completed}/{len(self.envs)} completed", "cyan")
                    yield result
            except Exception as e:
                raise e

        if self.engine_name == "verl":
            self.rollout_engine.sleep()

        self.executor.shutdown(wait=False, cancel_futures=True)

    async def execute_tasks(self, tasks: list[dict]):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.

        Args:
            tasks: List of tasks to process
            max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)

        Returns:
            A list of trajectories, one for each task.
        """
        if not hasattr(self, "executor") or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=self.max_env_workers)

        max_concurrent = self.n_parallel_agents

        # Initialize results list to store trajectories for all tasks
        all_trajectories = {}

        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        # Track completed trajectories
        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with semaphore:
                # Get an available index
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), "Agent is not initalized or not inheriting from BaseAgent"
                    self.agents[index].trajectory.task = task  # type: ignore
                    res = await self.run_agent_trajectory_async(index, application_id=task_id)
                    res.task = task
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                finally:
                    # Put the index back in the queue when done
                    await index_queue.put(index)

        # Run all tasks concurrently
        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]

        self.executor.shutdown(wait=False, cancel_futures=True)

        return ordered_trajectories

    def shutdown(self):
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown()
            self.executor = None


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
