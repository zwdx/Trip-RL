"""CPU check: history removal, native session GRPO, and per-turn teacher routing."""
import asyncio
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), '/data/xiongzhou/agent_star_runs/qwen35_async_migration_20261024/deps']
from transformers import AutoTokenizer
from agentstar_travel_loop import render_action_history, split_action_history_outputs
from agentstar_async_travel_loop import AgentStarAsyncLoopWorker
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, AgentLoopMetrics
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
from verl.trainer.ppo import core_algos
from verl import DataProto
import numpy as np
import torch
from tensordict import TensorDict

model = '/data/xiongzhou/agent_star_runs/qwen35_2b_sft_ep6_mb2_gpu13_20261021/train/checkpoint-90'
tokenizer = AutoTokenizer.from_pretrained(model)
messages = [
    {'role': 'system', 'content': 'Use tools.'},
    {'role': 'user', 'content': 'Plan a trip.'},
    {'role': 'assistant', 'content': '<think>HIDDEN_OLD_REASONING</think><tool_call>{"name":"SearchCity","arguments":{"state":"Ohio"}}</tool_call>'},
    {'role': 'user', 'content': '<tool_response>TOOL_RESULT_COLUMBUS</tool_response>'},
    {'role': 'assistant', 'content': '<think>SECOND_OLD_REASONING</think><tool_call>{"name":"SearchCity","arguments":{"state":"Texas"}}</tool_call>'},
    {'role': 'user', 'content': '<tool_response>TOOL_RESULT_AUSTIN</tool_response>'},
]
original = copy.deepcopy(messages)
ids = render_action_history(tokenizer, messages)
text = tokenizer.decode(ids, skip_special_tokens=False)
assert 'HIDDEN_OLD_REASONING' not in text and 'SECOND_OLD_REASONING' not in text
assert text.count('<think>') == 1 and '</think>' not in text
assert '<think>\n\n</think>' not in text
assert 'SearchCity' in text and 'TOOL_RESULT_COLUMBUS' in text
assert text.endswith('<think>\n')
assert messages == original
first = tokenizer.encode('<think>first</think><tool_call>{}</tool_call>', add_special_tokens=False)
second = tokenizer.encode('second</think><answer>done</answer>', add_special_tokens=False)
output = AgentLoopOutput(prompt_ids=[1], response_ids=first+second,
    response_mask=[1]*len(first+second), response_logprobs=[-.5]*len(first+second),
    reward_score=2., num_turns=2, metrics=AgentLoopMetrics(),
    extra_fields={'reward_extra_info': {'acc':0.}, 'min_global_steps':1, 'max_global_steps':1,
                  'turn_samples':[
        {'prompt_ids':[1], 'response_ids':first, 'response_logprobs':[-.5]*len(first)},
        {'prompt_ids':ids, 'response_ids':second, 'response_logprobs':[-.5]*len(second)}]})
outputs = split_action_history_outputs(output)
assert len(outputs) == 2
assert outputs[1].prompt_ids == ids and outputs[0].response_ids == first
assert all(item.reward_score == 2. for item in outputs)
assert all(len(item.response_ids) == len(item.response_logprobs) for item in outputs)
# Native GRPO uses one reward per session despite unequal numbers of turns.
data = DataProto(batch=TensorDict({
    'response_mask': torch.ones(3,2),
    'token_level_rewards': torch.tensor([[0.,1.],[0.,1.],[0.,0.]])}, batch_size=[3]),
    non_tensor_batch={'uid':np.array(['prompt']*3,dtype=object)})
result = compute_advantage_for_multi_trajectories(data,
    ['prompt_0_0','prompt_0_1','prompt_1_0'],core_algos.AdvantageEstimator.GRPO,num_repeat=2)
a = result.batch['advantages']
assert torch.allclose(a[0], a[1]) and a[0,0] > 0 and a[2,0] < 0
async def teacher_check():
    worker = SimpleNamespace(distillation_enabled=True,
        tokenizer=SimpleNamespace(pad_token_id=0), _compute_teacher_logprobs=AsyncMock())
    native = AgentLoopWorkerTQ.__ray_metadata__.modified_class
    with patch.object(native, '_agent_loop_postprocess', new=AsyncMock()) as post:
        # Bind the real method through a lightweight subclass to retain super().
        class CheckWorker(AgentStarAsyncLoopWorker):
            def __init__(self):
                self.distillation_enabled=True
                self.tokenizer=worker.tokenizer
                self._compute_teacher_logprobs=worker._compute_teacher_logprobs
        await CheckWorker()._agent_loop_postprocess(outputs, False, uid='prompt', session_id=0)
        worker._compute_teacher_logprobs.assert_awaited_once()
        assert worker._compute_teacher_logprobs.call_args.kwargs['prompt_ids'] == [1]
        post.assert_awaited_once()
asyncio.run(teacher_check())
print('ACTION_ONLY_HISTORY_CPU_CHECK_PASSED')
