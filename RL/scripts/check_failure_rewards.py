"""CPU-only regression using the real travel loop, parser, and token builder."""
import asyncio
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path[:0] = [str(root), '/data/xiongzhou/agent_star_runs/qwen35_async_migration_20261024/deps']
from transformers import AutoTokenizer
import agentstar_travel_loop as travel
from agentstar_async_travel_loop import AgentStarAsyncReplayBuffer
from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync
from transfer_queue import KVBatchMeta
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.llm_server import LLMServerClient, FullyAsyncLLMServerClient
import torch
from tensordict import TensorDict, NonTensorStack

policy = {'enabled':True, 'soft_buffer_fraction':.2, 'max_soft_penalty':1., 'invalid_action_penalty':1.}
shape = travel.shape_failure_reward
assert shape(3.,'ENV_DONE',.8,.8,policy) == (3.,0.,0.)
assert abs(shape(3.,'ENV_DONE',.9,.1,policy)[0]-2.5) < 1e-6
assert shape(0.,'MAX_STEPS',.5,1.,policy)[0] == -1.
assert shape(0.,'TRUNCATION',1.,.5,policy)[0] == -1.
assert shape(0.,'FORMAT_ERROR',1.,1.,policy)[0] == -1.
assert shape(1.,'ENV_DONE',1.,1.,policy,validate=True) == (1.,0.,0.)
assert shape(0.,'TRUNCATION',1.,1.,{'enabled':False}) == (0.,0.,0.)
for bad in [0.,1.1]:
    try: shape(0.,'ENV_DONE',0.,0.,{**policy,'soft_buffer_fraction':bad})
    except ValueError: pass
    else: raise AssertionError('invalid policy accepted')
schemas = [{'type':'function','function':{'name':'SearchCity','parameters':{
    'type':'object','properties':{'state':{'type':'string'}},'required':['state']}}}]
assert travel.invalid_tool_call([{'function':{'name':'SearchCity','arguments':'{"state":"Ohio"}'}}],schemas) is None
for function in [{'name':'Unknown','arguments':'{}'}, {'name':'SearchCity','arguments':'{}'},
                 {'name':'SearchCity','arguments':'{"state":1}'}, {'name':'SearchCity','arguments':'['},
                 {'name':'SearchCity','arguments':'{"state":"Ohio","unknown":1}'}]:
    assert travel.invalid_tool_call([{'function':function}],schemas)

model = '/data/xiongzhou/agent_star_runs/qwen35_2b_sft_ep6_mb2_gpu13_20261021/train/checkpoint-90'
tokenizer = AutoTokenizer.from_pretrained(model)
class Environment:
    mode = 'ok'
    calls = 0
    def __init__(self, task, tools, reward_fn, max_steps):
        self.task=task; self.step_count=0; self.max_steps=max_steps
        self.tools=SimpleNamespace(json=schemas)
    def reset(self): return self.task,{}
    def step(self, actions):
        type(self).calls += 1
        self.step_count += 1
        finish = any(x['function']['name']=='finish' for x in actions)
        if finish:
            if self.mode == 'infra':
                return {},0.,True,{'metadata':{'error':'Error formatting planning: HTTP 429'}}
            return {},2.,True,{'metadata':{'judgement':{'success_pass':1}}}
        if self.mode == 'tool_infra':
            return {'tool_outputs':{'id':'Error: backend unavailable'}},0.,False,{'metadata':{}}
        return {'tool_outputs':{actions[0]['id']:'Columbus, Ohio'}},0.,False,{'metadata':{}}

async def run_case(responses, max_turns=60, mode='ok', timeout=30.):
    loop=object.__new__(travel.AgentStarTravelLoop)
    loop.tokenizer=tokenizer
    loop.loop=asyncio.get_running_loop()
    loop.rollout_config=SimpleNamespace(prompt_length=2048,response_length=63488)
    loop.continuous_token_builder=QwenContinuousTokenBuilder(tokenizer,
        chat_template_kwargs={'enable_thinking':True,'preserve_thinking':False})
    inputs=[]
    class Client:
        async def generate(self, request_id, prompt_ids, sampling_params):
            inputs.append(list(prompt_ids))
            content,finish=responses[len(inputs)-1]
            if content is None:
                await asyncio.sleep(1.)
            ids=tokenizer.encode(content,add_special_tokens=False)
            return TokenOutput(token_ids=ids,log_probs=[-.5]*len(ids),
                stop_reason='aborted' if finish=='abort' else 'completed',
                extra_fields={'finish_reason':finish})
    loop.server_manager=Client()
    settings={'system_prompt':'Use one travel tool or answer.', 'max_turns':max_turns,
        'max_context':65536,'max_tokens':32768,'timeout_seconds':timeout,
        'require_rollout_logprobs':True,'fail_on_reward_error':True,
        'preserve_thinking':False,'failure_reward':policy}
    Environment.mode=mode;Environment.calls=0
    with patch.object(travel,'TravelToolEnvironment',Environment):
        output=await loop.run({'temperature':1.},extra_info={
            'task_info':{'id':'demo','query':'Plan a trip from Boston to Columbus.'},
            'agentstar_settings':settings},agentstar_progress={'step':1,'total':155,'validate':False})
    return output,inputs

answer='done</think><answer>valid plan</answer><|im_end|>'
tool='OLD_THOUGHT</think><tool_call>{"name":"SearchCity","arguments":{"state":"Ohio"}}</tool_call><|im_end|>'
async def checks():
    # Exercise the real async client's rebuilt TokenOutput, not a fake finish_reason field.
    client=object.__new__(FullyAsyncLLMServerClient)
    client.config=SimpleNamespace()
    for reason in ['stop','length']:
        native=TokenOutput(token_ids=[248046],log_probs=[-.5],stop_reason='completed',
            extra_fields={'finish_reason':reason,'global_steps':0})
        with patch.object(LLMServerClient,'generate',new=AsyncMock(return_value=native)):
            result=await client.generate('check',prompt_ids=[1],sampling_params={'max_tokens':1})
        assert result.extra_fields['finish_reason']==reason
    out,_=await run_case([(answer,'stop')])
    assert out.extra_fields['termination']=='ENV_DONE' and out.reward_score==2.
    assert any(out.response_mask)
    out,_=await run_case([(answer,'stop')],max_turns=1)
    assert out.extra_fields['termination']=='ENV_DONE' and out.reward_score==1.
    out,_=await run_case([(tool,'stop')],max_turns=1)
    assert out.extra_fields['termination']=='MAX_STEPS' and out.reward_score==-1.
    assert any(out.response_mask) and Environment.calls==0
    out,_=await run_case([('unfinished reasoning','length')])
    assert out.extra_fields['termination']=='TRUNCATION' and out.reward_score==-1.
    assert any(out.response_mask) and out.response_logprobs
    out,_=await run_case([(answer,'length')])
    assert out.extra_fields['termination']=='ENV_DONE' and out.reward_score==2.
    out,_=await run_case([('partial generation','abort')])
    assert out.extra_fields['termination']=='ERROR' and not any(out.response_mask)
    assert out.extra_fields['skip_teacher']
    out,_=await run_case([('missing closing tag','stop')])
    assert out.extra_fields['termination']=='FORMAT_ERROR' and out.reward_score==-1.
    out,_=await run_case([(tool.replace('SearchCity','Unknown'),'stop')])
    assert out.extra_fields['termination']=='INVALID_TOOL_CALL' and out.reward_score==-1.
    assert any(out.response_mask)
    for responses,mode in [([(answer,'stop')],'infra'), ([(tool,'stop')],'tool_infra')]:
        out,_=await run_case(responses,mode=mode)
        assert out.extra_fields['termination']=='ERROR' and not any(out.response_mask)
        assert out.extra_fields['skip_teacher'] and out.reward_score==0.
    out,_=await run_case([(None,'stop')],timeout=.01)
    assert out.extra_fields['termination']=='TIMEOUT' and not any(out.response_mask)
    out,inputs=await run_case([(tool,'stop'),(answer,'stop')])
    rendered = tokenizer.decode(inputs[1],skip_special_tokens=False)
    assert 'OLD_THOUGHT' not in rendered
    assert rendered.count('<think>')==1 and '</think>' not in rendered
    assert 'SearchCity' in tokenizer.decode(inputs[1]) and 'Columbus' in tokenizer.decode(inputs[1])
    turns=travel.split_action_history_outputs(out)
    assert len(turns)==2 and turns[1].prompt_ids==inputs[1]
    assert all(x.reward_score==2. and any(x.response_mask) for x in turns)
asyncio.run(checks())
# A masked service failure must not enter GRPO group mean/std, even at reward 0.
meta=KVBatchMeta(keys=['p_0_0','p_1_0','p_2_0'],tags=[{}]*3,partition_id='train')
data=TensorDict({'response_mask':torch.tensor([[1,1],[0,0],[1,1]]),
    'extra_fields':NonTensorStack.from_list([{'reward_extra_info':{'task_reward':2.}},
        {'reward_extra_info':{'infrastructure_failure':1.}},
        {'reward_extra_info':{'training_reward':-1.,'model_failure':1.}}])},batch_size=[3])
with patch.object(ReplayBufferAsync,'sample',return_value=(meta,{})), \
     patch('agentstar_async_travel_loop.tq.kv_batch_get',return_value=data):
    batch,metrics=AgentStarAsyncReplayBuffer.sample(object.__new__(AgentStarAsyncReplayBuffer),1,'train',32)
    assert batch.keys==['p_0_0','p_2_0'] and len(batch)==2
    assert metrics['training/filtered_infrastructure_rows']==1
    assert metrics['training/trajectory/count']==3
    assert abs(metrics['training/trajectory/infrastructure_failure_mean']-1/3)<1e-6
print('FAILURE_REWARD_LOOP_CPU_CHECK_PASSED')
