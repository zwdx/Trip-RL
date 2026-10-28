"""CPU regression against native V1 interfaces. No model training is claimed here."""
import asyncio
import json
import gzip
import time
import os
from pathlib import Path
import sys
import tempfile
from omegaconf import OmegaConf, open_dict
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
import torch
from tensordict import TensorDict, NonTensorData
from agentstar_async_travel_loop import (attach_async_progress, VersionTrackingClient,
    AgentStarAsyncTravelLoop, AgentStarAsyncLoopWorker, AgentStarAsyncReplayBuffer, fill_masked_teacher)
from examples.travel.train_qwen35_curriculum_opd_async import load_config, require_runtime
sys.path.insert(0,str(ROOT/'scripts'))
from check_qwen35_curriculum_opd import check_losses
from verl.utils.tensordict_utils import assign_non_tensor_data
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker, _agent_loop_registry


def check_progress():
    for step in (1,51,52,103,104,155):
        batch=TensorDict({'index':torch.arange(2)},batch_size=[2])
        assign_non_tensor_data(batch,'global_steps',step)
        attach_async_progress(batch,155)
        assert batch['agentstar_progress'] == {'step':step,'total':155,'validate':False}
    batch['validate']=NonTensorData(True)
    assign_non_tensor_data(batch,'global_steps',0)
    attach_async_progress(batch,155)
    assert batch['agentstar_progress']['validate']
    print('ASYNC_SUBMISSION_CURRICULUM_BOUNDARIES_PASS')


def prepared_route_fixture(cfg):
    from examples.travel.train_qwen35_curriculum_opd import prepare
    checked=OmegaConf.create(OmegaConf.to_container(cfg,resolve=False))
    with tempfile.TemporaryDirectory(prefix='agentstar_async_route_') as tmp:
        with open_dict(checked):
            checked.trainer.use_v1=False
            checked.trainer.default_local_dir=tmp
        prepare(checked)
        rows=[json.loads(line) for line in (Path(tmp)/'train.jsonl').read_text().splitlines()]
    targets={item.name:item._target_ for item in OmegaConf.load(cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path)}
    assert all(targets.get(row['agent_name'])=='agentstar_async_travel_loop.AgentStarAsyncTravelLoop' for row in rows)
    print('REAL_PREPARED_DATA_AGENT_NAMES_RESOLVE_TO_ASYNC_LOOP_PASS')
    return rows[0]


async def check_transport(cfg):
    from transformers import AutoTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    tokenizer=AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path)
    system=json.loads(Path(cfg.agentstar.system_prompt_source).read_text())['system_prompt']
    prepared=prepared_route_fixture(cfg)
    task=prepared['extra_info']['task_info']
    texts=['Search. </think>\n<tool_call>{"name":"SearchCity","arguments":{"state":"California"}}</tool_call>',
           'Done. </think>\n<answer>synthetic check only</answer>']
    class Server:
        def __init__(self): self.ids=[]; self.stall=False
        async def generate(self, request_id, prompt_ids, sampling_params):
            if self.stall and len(self.ids)==1:
                await asyncio.sleep(10)
            text=texts[len(self.ids)]; version=4+len(self.ids); self.ids.append(request_id)
            tokens=tokenizer.encode(text,add_special_tokens=False)+[tokenizer.eos_token_id]
            return SimpleNamespace(token_ids=tokens,log_probs=[-1.]*len(tokens),finish_reason='stop',
                                   extra_fields={'min_global_steps':version,'max_global_steps':version})
    server=Server()
    from verl.utils.config import omega_conf_to_dataclass
    teachers=omega_conf_to_dataclass(cfg.distillation).teacher_models
    selector=AgentLoopWorker(config=cfg,llm_client=server,teacher_client={key:SimpleNamespace() for key in teachers})
    assert _agent_loop_registry[prepared['agent_name']]['_target_']=='agentstar_async_travel_loop.AgentStarAsyncTravelLoop'
    async def raw_output(output,validate,**kwargs): return output
    selector._agent_loop_postprocess=raw_output
    settings={'system_prompt':system,'max_turns':3,'max_context':65536,'max_tokens':32768,
              'timeout_seconds':20,'reward_type':'PLANNER_R1','require_rollout_logprobs':True,'fail_on_reward_error':True}
    scores={k:1 for k in ('commonsense_macro_score','commonsense_micro_score','hard_macro_score','hard_micro_score','success_pass')}
    with patch.object(AgentStarAsyncTravelLoop, '_save_trajectory'), patch('rllm.rewards.travel_reward.fine_grained_travel_reward_fn',return_value=(True,scores,[])):
        output=await selector._run_agent_loop({'temperature':1.,'top_p':1.,'top_k':-1},
            trajectory={'step':4,'sample_index':0,'rollout_n':0,'validate':False},
            agent_name=prepared['agent_name'],trace=False,global_steps=4,
            extra_info={'task_info':task,'agentstar_settings':settings},
            agentstar_progress={'step':4,'total':155,'validate':False})
    assert output.num_turns==2 and output.reward_score==5.
    assert len(output.response_ids)==len(output.response_mask)==len(output.response_logprobs)
    assert 0 in output.response_mask and 1 in output.response_mask
    assert len(set(server.ids))==1 and selector.llm_client is server
    assert output.extra_fields['min_global_steps']==4 and output.extra_fields['max_global_steps']==5
    print('ASYNC_REAL_TOOL_MASK_LOGPROBS_AND_MULTI_TURN_VERSIONS_PASS (mocked final reward)')

    assert all(output.extra_fields['phase_seconds'].get(phase,0)>0 for phase in ('generation','tools','reward'))
    async def timeout_case(phase):
        server.ids.clear();server.stall=phase=='generation'
        settings['timeout_seconds']=3
        def score(*args,**kwargs):
            if phase=='reward':time.sleep(4)
            return True,scores,[]
        with patch.object(AgentStarAsyncTravelLoop,'_save_trajectory'), patch('rllm.rewards.travel_reward.fine_grained_travel_reward_fn',side_effect=score):
            partial=await selector._run_agent_loop({'temperature':1.,'top_p':1.,'top_k':-1},
                trajectory={'step':4,'sample_index':0,'rollout_n':0,'validate':False},
                agent_name=prepared['agent_name'],trace=False,global_steps=4,
                extra_info={'task_info':task,'agentstar_settings':settings},
                agentstar_progress={'step':4,'total':155,'validate':False})
        e=partial.extra_fields
        assert e['termination']=='TIMEOUT' and e['failure_phase']==phase,(phase,e)
        assert e['phase_seconds'][phase]>=2.5
        assert len(e['calls'])==2 and e['steps'] and e['messages'] and e['partial_response_ids']
        assert partial.response_mask==[0] and e['skip_teacher']
        if phase=='generation':assert e['calls'][-1]['status']=='running'
        if phase=='reward':assert 'answer' in str(e['pending_action'])
        with tempfile.TemporaryDirectory() as tmp:
            archive=AgentStarAsyncTravelLoop.__new__(AgentStarAsyncTravelLoop)
            archive.config=OmegaConf.create({'trainer':{'default_local_dir':tmp}})
            archive._save_trajectory(partial,{'global_steps':4,'uid':'cpu','session_id':0})
            file=next(Path(tmp).rglob('*.json.gz'))
            with gzip.open(file,'rt') as stream:saved=json.load(stream)
            assert saved['output']['extra_fields']['failure_phase']==phase
        return partial
    await timeout_case('generation');await timeout_case('reward')
    print('GENERATION_AND_REWARD_TIMEOUT_PHASE_TIMING_PARTIAL_HISTORY_GZIP_ARCHIVE_PASS')

    # Exercise native TQ postprocessing, not a rewritten serialization imitation.
    worker=AgentStarAsyncLoopWorker.__new__(AgentStarAsyncLoopWorker)
    worker.tokenizer=tokenizer;worker.processor=None;worker.mm_processor_kwargs={}
    worker.config=cfg;worker.distillation_enabled=True;worker.reward_loop_worker_handles=None
    worker.rollout_config=cfg.actor_rollout_ref.rollout
    emitted=[]
    async def teacher(out,**kwargs):
        width=len(out.prompt_ids)+len(out.response_ids)
        out.extra_fields['teacher_ids']=torch.ones(width,1,dtype=torch.int32)
        out.extra_fields['teacher_logprobs']=torch.full((width,1),-0.5)
    worker._compute_teacher_logprobs=teacher
    async def put(**kwargs): emitted.append(kwargs)
    masked=AgentLoopOutput(prompt_ids=[1],response_ids=[2],response_mask=[0],response_logprobs=[0.],
        reward_score=0.,num_turns=0,metrics=AgentLoopMetrics(),
        extra_fields={'skip_teacher':True,'min_global_steps':4,'max_global_steps':4,'reward_extra_info':{}})
    # Teacher function above is mocked; separately prove zero-mask schema survives the native serializer.
    fill_masked_teacher(masked,tokenizer.pad_token_id)
    row=masked.as_dict()
    assert row['teacher_ids'].shape==(2,1) and row['teacher_logprobs'].eq(0).all()
    with patch('verl.trainer.ppo.v1.agent_loop_tq.tq.async_kv_batch_put',side_effect=put):
        await worker._agent_loop_postprocess(output,False,uid='prompt_a',session_id=0,global_steps=4)
    assert len(emitted)==1
    record=emitted[0];assert record['keys']==['prompt_a_0_0']
    assert record['tags'][0]['min_global_steps']==4 and record['tags'][0]['max_global_steps']==5
    fields=record['fields'];assert 'teacher_ids' in fields and 'rollout_log_probs' in fields
    assert fields['loss_mask'].sum()==sum(output.response_mask)
    print('NATIVE_TQ_TEACHER_SERIALIZATION_AND_MASKED_SCHEMA_PASS')


async def check_group_completion(cfg):
    worker=AgentStarAsyncLoopWorker.__new__(AgentStarAsyncLoopWorker)
    worker.config=cfg
    gates=[asyncio.Event() for _ in range(8)]
    statuses=[]
    async def session(*args,session_id,**kwargs): await gates[session_id].wait()
    async def put(**kwargs): statuses.append(kwargs['tag']['status'])
    worker._run_agent_loop=session
    with patch('verl.trainer.ppo.v1.agent_loop_tq.tq.async_kv_put',side_effect=put):
        task=asyncio.create_task(worker._run_prompt({'uid':'group'}, {}, {'validate':False}))
        for event in gates[:7]: event.set()
        await asyncio.sleep(.02)
        assert statuses==['running'] and not task.done()
        gates[-1].set();await task
    assert statuses==['running','finished']
    print('NATIVE_GRPO_EIGHT_SESSION_COMPLETION_BARRIER_PASS')


def check_empty_batch_guard():
    from agentstar_async_travel_loop import ReplayBufferAsync
    sampler=AgentStarAsyncReplayBuffer.__new__(AgentStarAsyncReplayBuffer)
    batch=SimpleNamespace(keys=['a'],partition_id='train')
    with patch.object(ReplayBufferAsync,'sample',return_value=(batch,{})):
        with patch('agentstar_async_travel_loop.tq.kv_batch_get',return_value={'response_mask':torch.zeros(1,2)}):
            try: sampler.sample(1,'train',1)
            except RuntimeError as error: assert 'empty optimizer update' in str(error)
            else: raise AssertionError('Empty update accepted')
        with patch('agentstar_async_travel_loop.tq.kv_batch_get',return_value={'response_mask':torch.tensor([[1,0]])}):
            assert sampler.sample(1,'train',1)[0] is batch
    print('ASYNC_EMPTY_BATCH_FAIL_BEFORE_OPTIMIZER_PASS')


def check_behavior_logprobs(cfg):
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    assert cfg.algorithm.rollout_correction.bypass_mode
    holder=SimpleNamespace(config=cfg)
    batch=SimpleNamespace(keys=['a'],partition_id='train')
    values=torch.tensor([[-.3,-.7]])
    stored=[]
    with patch('verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get',return_value=TensorDict({'rollout_log_probs':values.clone()},batch_size=[1])):
        with patch('verl.trainer.ppo.v1.trainer_base.tq.kv_batch_put',side_effect=lambda **kwargs:stored.append(kwargs)):
            assert PPOTrainer._compute_old_log_prob(holder,batch,{}) is batch
    assert torch.equal(stored[0]['fields']['old_log_probs'],values)
    print('NATIVE_ASYNC_BEHAVIOR_LOGPROBS_RETAINED_AS_OLD_POLICY_PASS')



async def check_admission():
    worker=AgentStarAsyncLoopWorker.__new__(AgentStarAsyncLoopWorker)
    worker._admission=asyncio.Semaphore(2)
    entered=[];release=asyncio.Event()
    async def native(self,*args,**kwargs):
        entered.append(kwargs)
        await release.wait()
    with patch.object(AgentStarAsyncLoopWorker.__mro__[1],'_run_agent_loop',native):
        tasks=[asyncio.create_task(worker._run_agent_loop(sample=i)) for i in range(3)]
        await asyncio.sleep(0.02)
        assert len(entered)==2,entered
        release.set();await asyncio.gather(*tasks)
    assert len(entered)==3 and entered[-1]['admission_wait_seconds']>=0.015
    print('OUTER_ADMISSION_LIMIT_QUEUE_OUTSIDE_TRAJECTORY_TIMEOUT_PASS')


def main():
    require_runtime()
    cfg=load_config()
    assert cfg.trainer.n_gpus_per_node==2 and cfg.distillation.n_gpus_per_node==1
    assert cfg.data.train_batch_size==32 and cfg.actor_rollout_ref.rollout.n==8
    assert cfg.agentstar.timeout_seconds==1200
    assert cfg.distillation.distillation_loss.use_task_rewards
    check_progress();check_losses();asyncio.run(check_transport(cfg));asyncio.run(check_admission())
    asyncio.run(check_group_completion(cfg));check_empty_batch_guard();check_behavior_logprobs(cfg)
    print('QWEN35_CURRICULUM_OPD_ASYNC_CPU_CHECK_PASS; no live model training or teacher scoring')


if __name__=='__main__': main()
