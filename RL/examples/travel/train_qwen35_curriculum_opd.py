"""Agent-STAR's frozen Qwen3.5 curriculum + native veRL PG-OPD recipe.

Run from RL with /data/xiongzhou/venvs/agentstar_rl/bin/python -m
examples.travel.train_qwen35_curriculum_opd --prepare-only [key=value ...].
Omit --prepare-only to run. Resume with trainer.default_local_dir=<existing run>
and trainer.resume_mode=auto; the saved semantic protocol must still match.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

RL_ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path('/data/xiongzhou/agent_star_runs/verl-v0.9.1-source')
BACKEND_COMMIT = '1876b06d0a3e4e71e06230be10af14492ca8a75b'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(overrides=()):
    import verl
    if not Path(verl.__file__).resolve().is_relative_to(BACKEND):
        raise RuntimeError(f'Wrong veRL import: {verl.__file__}')
    revision = subprocess.check_output(['git', '-C', str(BACKEND), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != BACKEND_COMMIT:
        raise RuntimeError(f'Unexpected veRL revision: {revision}')
    with initialize_config_dir(config_dir=str(BACKEND / 'verl/trainer/config'), version_base=None):
        base = compose(config_name='ppo_trainer')
    recipe = OmegaConf.load(Path(__file__).with_name('qwen35_curriculum_opd.yaml'))
    OmegaConf.set_struct(base, False)
    cfg = OmegaConf.merge(base, recipe)
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    with open_dict(cfg):
        if not cfg.trainer.default_local_dir:
            stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
            cfg.trainer.default_local_dir = f'/data/xiongzhou/agent_star_runs/qwen35_curriculum_opd_{stamp}'
        cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(
            Path(__file__).with_name('agentstar_travel_loop.yaml'))
    return cfg


def read_tasks(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    for row in rows:
        if not isinstance(row.get('query'), str) or not row['query'].strip():
            raise ValueError(f'Missing query in {path}')
        for key in ('org', 'dest', 'days', 'date', 'people_number', 'budget', 'local_constraint'):
            if key not in row:
                raise ValueError(f'Missing {key} in {path}')
    keys = [' '.join(row['query'].split()).casefold() for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f'Duplicate normalized queries in {path}')
    return rows


def prepare(cfg):
    """Build native JSONL inputs; do not modify source tasks or reset resume state."""
    from transformers import AutoTokenizer
    from agentstar_travel_loop import TRAVEL_TOOLS
    from rllm.agents.travel_agent import TravelAgentFormat
    from verl.utils.config import validate_config
    from verl.trainer.ppo.utils import need_critic, need_reference_policy

    if Path.cwd().resolve() != RL_ROOT:
        raise ValueError(f'Run from {RL_ROOT}: the original tool databases use relative paths')
    if cfg.trainer.use_v1 or cfg.algorithm.adv_estimator != 'grpo':
        raise ValueError('This adapter targets the pinned synchronous RayPPOTrainer with GRPO')
    if cfg.algorithm.filter_groups.enable:
        raise ValueError('Do not discard equal-task-reward groups that still have OPD signal')
    if cfg.data.gen_batch_size not in (None, cfg.data.train_batch_size):
        raise ValueError('Keep generation and training prompt batch sizes equal')
    if cfg.actor_rollout_ref.actor.use_kl_loss or cfg.algorithm.use_kl_in_reward:
        raise ValueError('Reference KL is disabled in this recipe')
    loss = cfg.distillation.distillation_loss
    if (loss.loss_mode != 'k1' or not loss.use_policy_gradient or not loss.use_task_rewards
            or loss.loss_max_clamp != 10.0 or loss.distillation_loss_coef < 0):
        raise ValueError('Expected PG k1 + task rewards, Rmax=10, nonnegative lambda')
    if cfg.actor_rollout_ref.rollout.n != 8 or cfg.actor_rollout_ref.actor.ppo_epochs != 1:
        raise ValueError('Keep group size eight and one PPO epoch in the frozen recipe')
    context = int(cfg.agentstar.max_context)
    if context != cfg.data.max_prompt_length + cfg.data.max_response_length:
        raise ValueError('Prompt + trajectory suffix budgets must equal the context budget')
    if cfg.agentstar.single_turn_max_tokens > cfg.data.max_response_length:
        raise ValueError('Single-turn generation exceeds trajectory budget')
    if min(cfg.agentstar.single_turn_max_tokens, cfg.agentstar.max_turns, cfg.agentstar.timeout_seconds) <= 0:
        raise ValueError('Generation, turn and timeout budgets must be positive')
    if cfg.distillation.enabled:
        teacher = cfg.distillation.teacher_models.teacher_model
        if teacher.inference.max_model_len < context + 1:
            raise ValueError('Teacher needs the entire student prefix plus its dummy output token')
        if cfg.agentstar.formatter_url is None and teacher.inference.enable_sleep_mode:
            raise ValueError('The shared teacher/formatter must stay awake')
    elif not cfg.agentstar.formatter_url:
        raise ValueError('A pure-GRPO run still needs an explicit strict formatter endpoint')

    paths = {'train': cfg.agentstar.train_source, 'val': cfg.agentstar.val_source}
    tasks = {split: read_tasks(path) for split, path in paths.items()}
    query_keys = lambda rows: {' '.join(row['query'].split()).casefold() for row in rows}
    if query_keys(tasks['train']) & query_keys(tasks['val']):
        raise ValueError('Train/validation query overlap')
    tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path)
    teacher_tokenizer = AutoTokenizer.from_pretrained(cfg.distillation.teacher_models.teacher_model.model_path)
    if (tokenizer.get_vocab() != teacher_tokenizer.get_vocab()
            or tokenizer.eos_token_id != teacher_tokenizer.eos_token_id
            or tokenizer.pad_token_id != teacher_tokenizer.pad_token_id):
        raise ValueError('Teacher/student vocabulary IDs do not match')
    system_config = json.loads(Path(cfg.agentstar.system_prompt_source).read_text())
    system = system_config['system_prompt']
    agent = TravelAgentFormat(tools=TRAVEL_TOOLS, system_prompt=system)
    settings = {'system_prompt': system, 'reward_type': 'PLANNER_R1',
                'max_turns': int(cfg.agentstar.max_turns), 'max_context': context,
                'max_tokens': int(cfg.agentstar.single_turn_max_tokens),
                'validation_repetition_penalty': float(system_config['generation']['repetition_penalty']),
                'timeout_seconds': int(cfg.agentstar.timeout_seconds),
                'require_rollout_logprobs': True, 'fail_on_reward_error': True,
                'preserve_thinking': bool(cfg.data.apply_chat_template_kwargs.get('preserve_thinking', True)),
                'failure_reward': OmegaConf.to_container(cfg.agentstar.failure_reward, resolve=True)
                    if cfg.agentstar.get('failure_reward') is not None else {'enabled': False}}
    prepared, lengths = {}, {}
    template_kwargs = OmegaConf.to_container(cfg.data.apply_chat_template_kwargs)
    for split, rows in tasks.items():
        prepared[split], lengths[split] = [], []
        for index, source in enumerate(rows):
            task = dict(source)
            task.setdefault('id', f'{split}_{index:04d}')
            agent.reset()
            agent.update_from_env(task, 0., False, {})
            prompt = agent.chat_completions
            ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, **template_kwargs)
            ids = list(ids['input_ids'] if hasattr(ids, 'keys') else ids)
            if len(ids) > cfg.data.max_prompt_length:
                raise ValueError(f'Initial prompt would be truncated: {task["id"]}: {len(ids)}')
            lengths[split].append(len(ids))
            prepared[split].append({'prompt': prompt, 'data_source': 'agentstar_travel',
                'agent_name': 'agentstar_travel', 'reward_model': {'ground_truth': ''},
                'extra_info': {'index': index, 'task_info': task, 'agentstar_settings': settings}})
    available = len(tasks['train'])
    if cfg.data.train_max_samples > 0:
        available = min(available, cfg.data.train_max_samples)
    steps_per_epoch = available // int(cfg.data.train_batch_size)
    if not steps_per_epoch:
        raise ValueError('Training data is smaller than a full prompt batch')
    requested_steps = cfg.trainer.total_training_steps
    natural_steps = steps_per_epoch * int(cfg.trainer.total_epochs)
    total = int(requested_steps) if requested_steps is not None else natural_steps
    if not 1 <= total <= natural_steps:
        raise ValueError('Training-step cap must be attainable within total_epochs')
    with open_dict(cfg):
        cfg.trainer.total_training_steps = total
    protocol = {'backend_commit': BACKEND_COMMIT,
        'implementation_hashes': {str(path): digest(path) for path in (
            Path(__file__), RL_ROOT / 'agentstar_travel_loop.py',
            RL_ROOT / 'rllm/rewards/travel_reward.py',
            RL_ROOT / 'rllm/environments/tools/travel_tool_env.py',
            RL_ROOT / 'rllm/rewards/travel_utils/strict_transcription.py',
            BACKEND / 'verl/experimental/agent_loop/agent_loop.py',
            BACKEND / 'verl/trainer/distillation/losses.py')},
        'source_hashes': {split: digest(path) for split, path in paths.items()},
        'system_prompt_sha256': hashlib.sha256(system.encode()).hexdigest(),
        'student_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'student': cfg.actor_rollout_ref.model.path,
        'student_config': OmegaConf.to_container(cfg.actor_rollout_ref.model, resolve=True),
        'teacher': cfg.distillation.teacher_models.teacher_model.model_path,
        'settings': settings, 'data': {'train_count': len(tasks['train']), 'val_count': len(tasks['val']),
            'batch_size': cfg.data.train_batch_size, 'train_max_samples': cfg.data.train_max_samples,
            'seed': cfg.data.seed, 'epochs': cfg.trainer.total_epochs, 'total_steps': total},
        'distillation': {'enabled': cfg.distillation.enabled, 'lambda': loss.distillation_loss_coef,
                         'mode': 'PG_k1', 'Rmax': loss.loss_max_clamp},
        'actor': OmegaConf.to_container(cfg.actor_rollout_ref.actor, resolve=True),
        'sampling': {k: cfg.actor_rollout_ref.rollout[k] for k in
                     ('n', 'temperature', 'top_p', 'top_k', 'seed')},
        'validation': OmegaConf.to_container(cfg.actor_rollout_ref.rollout.val_kwargs, resolve=True)}
    run_dir = Path(cfg.trainer.default_local_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / 'protocol.json'
    has_checkpoint = any(run_dir.glob('global_step_*')) or (run_dir/'latest_checkpointed_iteration.txt').exists()
    if has_checkpoint and (not manifest_path.exists() or cfg.trainer.resume_mode == 'disable'):
        raise ValueError('Existing checkpoints require their saved protocol and an explicit resume mode')
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != protocol:
        raise ValueError(f'Run protocol changed: use a new output directory, not {run_dir}')
    if cfg.trainer.resume_mode == 'resume_path':
        checkpoint_run = Path(cfg.trainer.resume_from_path).resolve().parent
        if checkpoint_run != run_dir:
            raise ValueError('Resume checkpoint must belong to default_local_dir')
    for split, rows in prepared.items():
        target = run_dir / f'{split}.jsonl'
        text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        if target.exists() and target.read_text() != text:
            raise ValueError(f'Prepared data changed: {target}')
        target.write_text(text)
        with open_dict(cfg):
            cfg.data[f'{split}_files'] = [str(target)]
    manifest_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    validate_config(cfg, use_reference_policy=need_reference_policy(cfg), use_critic=need_critic(cfg))
    OmegaConf.save(cfg, run_dir / 'resolved_config.yaml', resolve=True)
    print(json.dumps({'status': 'PREPARED_NOT_TRAINED', 'run_dir': str(run_dir),
        'train_count': len(tasks['train']), 'val_count': len(tasks['val']),
        'steps_per_epoch': steps_per_epoch, 'total_training_steps': total,
        'max_prompt_tokens': {k: max(v) for k, v in lengths.items()}}, ensure_ascii=False), flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args, overrides = parser.parse_known_args()
    if any('=' not in item for item in overrides):
        parser.error('Overrides must be key=value')
    os.chdir(RL_ROOT)
    cfg = load_config(overrides)
    prepare(cfg)
    if not args.prepare_only:
        from verl.trainer.main_ppo import run_ppo
        from verl.trainer.main_ppo_v0 import TaskRunner
        run_ppo(cfg, TaskRunner)


if __name__ == '__main__':
    main()
