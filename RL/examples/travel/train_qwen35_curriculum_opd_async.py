"""Independent V1 colocate-async entry; original synchronous files stay immutable."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

from omegaconf import OmegaConf, open_dict
from examples.travel.train_qwen35_curriculum_opd import BACKEND, RL_ROOT, digest, load_config as load_sync, prepare as prepare_sync

HERE = Path(__file__).parent
DEPS = Path('/data/xiongzhou/agent_star_runs/qwen35_async_migration_20261024/deps')


def load_config(overrides=()):
    cfg = load_sync()
    with open_dict(cfg):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(HERE/'qwen35_curriculum_opd_async.yaml'))
        cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(HERE/'agentstar_async_travel_loop.yaml')
        cfg.trainer.default_local_dir = str(cfg.trainer.default_local_dir).replace('qwen35_curriculum_opd_', 'qwen35_curriculum_opd_async_', 1)
        cfg.ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH = f'{DEPS}:{RL_ROOT}'
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))


def prepare(cfg):
    with open_dict(cfg):
        cfg.ray_kwargs.ray_init.runtime_env.env_vars.AGENT_STAR_FORMATTER_DIAGNOSTICS_DIR = str(
            Path(cfg.trainer.default_local_dir).resolve()/'formatter_failures')
    if not cfg.trainer.use_v1 or cfg.trainer.v1.trainer_mode != 'colocate_async':
        raise ValueError('This independent entry requires V1 colocate_async')
    if cfg.trainer.n_gpus_per_node != 2 or cfg.distillation.n_gpus_per_node != 1:
        raise ValueError('Expected two student GPUs and one teacher GPU')
    if cfg.trainer.v1.sampler.max_off_policy_strategy != 'wait':
        raise ValueError('Keep dropless wait semantics in this initial migration')
    if cfg.actor_rollout_ref.rollout.agent.agent_loop_manager_class != 'agentstar_async_travel_loop.AgentStarAsyncLoopManager':
        raise ValueError('Use the checked TransferQueue adapter')
    aliases = {item.name:item._target_ for item in OmegaConf.load(cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path)}
    for name in ('agentstar_travel', cfg.actor_rollout_ref.rollout.agent.default_agent_loop):
        if aliases.get(name) != 'agentstar_async_travel_loop.AgentStarAsyncTravelLoop':
            raise ValueError(f'Prepared data agent name must resolve to the async loop: {name}')
    # Reuse every original tokenizer, dataset, budget and loss check without editing that entry.
    # Staging is temporary: only the final async protocol can own the output directory.
    with tempfile.TemporaryDirectory(prefix='agentstar_async_prepare_') as tmp:
        checked = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        with open_dict(checked):
            checked.trainer.use_v1 = False
            checked.trainer.default_local_dir = tmp
            checked.trainer.resume_mode = 'auto'
        prepare_sync(checked)
        with open_dict(cfg):
            cfg.trainer.total_training_steps = checked.trainer.total_training_steps
        protocol = json.loads((Path(tmp)/'protocol.json').read_text())
        protocol['async_execution'] = {
            'trainer_v1': OmegaConf.to_container(cfg.trainer.v1, resolve=True),
            'transfer_queue': OmegaConf.to_container(cfg.transfer_queue, resolve=True),
            'rollout': OmegaConf.to_container(cfg.actor_rollout_ref.rollout, resolve=True),
            'rollout_correction': OmegaConf.to_container(cfg.algorithm.rollout_correction, resolve=True),
            'student_gpus': cfg.trainer.n_gpus_per_node,
            'teacher_gpus': cfg.distillation.n_gpus_per_node,
            'teacher_config': OmegaConf.to_container(cfg.distillation, resolve=True),
            'gen_batch_size': cfg.data.gen_batch_size,
            'curriculum': 'submission_step_frozen_in_trajectory',
            'dependency': json.loads((DEPS.parent/'env-spec.json').read_text()),
        }
        tracked = [Path(__file__), RL_ROOT/'agentstar_async_travel_loop.py',
                   HERE/'qwen35_curriculum_opd_async.yaml', HERE/'agentstar_async_travel_loop.yaml']
        tracked += list((BACKEND/'verl/trainer/ppo/v1').glob('*.py'))
        tracked.append(BACKEND/'verl/workers/rollout/vllm_rollout/vllm_async_server.py')
        tracked.append(BACKEND/'verl/workers/rollout/llm_server.py')
        protocol['implementation_hashes'].update({str(p): digest(p) for p in tracked})
        out = Path(cfg.trainer.default_local_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        manifest = out/'protocol.json'
        checkpoints = any(out.glob('global_step_*')) or (out/'latest_checkpointed_iteration.txt').exists()
        if checkpoints and (not manifest.exists() or cfg.trainer.resume_mode == 'disable'):
            raise ValueError('Existing checkpoints require their matching async protocol and resume mode')
        if manifest.exists() and json.loads(manifest.read_text()) != protocol:
            raise ValueError(f'Async protocol changed; choose a new output directory: {out}')
        if cfg.trainer.resume_mode == 'resume_path' and Path(cfg.trainer.resume_from_path).resolve().parent != out:
            raise ValueError('Checkpoint must belong to this async output directory')
        for split in ('train','val'):
            target, content = out/f'{split}.jsonl', (Path(tmp)/f'{split}.jsonl').read_text()
            if target.exists() and target.read_text() != content:
                raise ValueError(f'Prepared input changed: {target}')
            target.write_text(content)
            cfg.data[f'{split}_files'] = [str(target)]
        manifest.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
        OmegaConf.save(cfg, out/'resolved_config.yaml', resolve=True)
        print(f'ASYNC_PREPARED_NOT_TRAINED: {out}', flush=True)
        return out


def require_runtime():
    from verl.trainer.ppo.v1.trainer_base import _tq_supports_checkpoint
    if not _tq_supports_checkpoint():
        raise RuntimeError('TransferQueue >=0.1.9 with save/load APIs required; no unsafe resume fallback')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args, overrides=parser.parse_known_args()
    if any('=' not in x for x in overrides):
        parser.error('Overrides must be key=value')
    os.chdir(RL_ROOT)
    cfg=load_config(overrides)
    prepare(cfg)
    if not args.prepare_only:
        require_runtime()
        from verl.trainer.main_ppo import run_ppo, TaskRunnerV1
        run_ppo(cfg, TaskRunnerV1)


if __name__ == '__main__':
    main()
