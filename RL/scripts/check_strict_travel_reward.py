"""CPU-only check: python RL/scripts/check_strict_travel_reward.py [RUN_ROOT].

HTTP and scoring dependencies are isolated; this is not a teacher/GPU smoke test.
An optional RUN_ROOT replays saved strict_final outputs without rescoring them.
"""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('strict_transcription',
    ROOT / 'rllm/rewards/travel_utils/strict_transcription.py')
strict = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(strict)
source = ROOT / 'rllm/rewards/travel_reward.py'
tree = ast.parse(source.read_text())
names = {'extract_answer', 'call_format_model', 'invoke_format_model_post_process',
         'fine_grained_travel_reward_fn', 'sum_travel_reward_fn', 'success_travel_reward_fn',
         'macro_travel_reward_fn', 'planner_r1_reward_fn'}
nodes = [node for node in tree.body if
         isinstance(node, ast.FunctionDef) and node.name in names or
         isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
         t.id == 'FORMAT_PLANNING' for t in node.targets)]
ns = {name: getattr(strict, name) for name in
      ('STRICT_SYSTEM_PROMPT', 'bind_source', 'finalize_plan')}
ns.update(os=os, json=json, time=SimpleNamespace(sleep=lambda _: None),
          RewardOutput=lambda **kwargs: SimpleNamespace(**kwargs))
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)


def rejected(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('An invalid transcription was accepted')


head = '| Day | Date / City Route | Transportation | Breakfast | Attractions | Lunch | Dinner | Accommodation |\n'
raw = head + '| March 8, 2022 | Seattle | Flight F123, Seattle to Washington | - | Example Diner, Seattle | Henry\'s, Seattle | The Culinary Pitaara, Seattle | Rooptop, Seattle |\n'
target = [{'day': 1, 'current_city': 'Seattle',
           'transportation': 'Flight Number: F123, from Seattle to Washington',
           'breakfast': '-', 'attraction': 'Example Diner, Seattle;',
           'lunch': "Henry's, Seattle", 'dinner': 'The Culinary Pitaara, Seattle',
           'accommodation': 'Rooptop, Seattle'}]
assert strict.source_table(raw)[1]['lunch'] == "Henry's, Seattle"
assert strict.bound_entity_value('−','breakfast') == '-'
assert 'AUTHORITATIVE SOURCE COLUMN BINDINGS' in strict.bind_source(raw)
converted = copy.deepcopy(target)
converted[0]['current_city'] = 'from Seattle to Washington'
converted[0]['attraction'] = 'Example Diner, Seattle'
assert strict.finalize_plan(raw, converted) == target
for field, value in [('accommodation', 'Rooftop, Seattle'), ('lunch', 'Example Diner, Seattle'),
                     ('breakfast', "Henry's, Seattle"), ('transportation', 'Flight Number: 123')]:
    bad = copy.deepcopy(target)
    bad[0][field] = value
    if field == 'transportation':
        rejected(lambda: strict.finalize_plan(raw, bad))
    else:
        assert strict.finalize_plan(raw, bad) == target
for invalid in ({}, [{'day': 1}], [], [{**target[0], 'day': 2}]):
    rejected(lambda: strict.finalize_plan(raw, copy.deepcopy(invalid)))
extra = raw.replace('Rooptop, Seattle |', 'Wrong Park, Seattle | Correct Hotel, Seattle |')
assert strict.source_table(extra)[1]['accommodation'] == 'Wrong Park, Seattle'
assert strict.finalize_plan(extra, copy.deepcopy(target))[0]['accommodation']=='Wrong Park, Seattle'
missing = head + '| 1 | Alpha -> Beta -> Gamma | - | - | - | - | - |\n'
assert strict.source_table(missing)[1]['accommodation'] == '-'
assert strict.source_cities(strict.source_table(missing)) == {1: 'from Alpha to Beta to Gamma'}
calendar = raw + raw.splitlines()[1].replace('March 8', 'March 9') + '\n'
assert list(strict.source_table(calendar)) == [1, 2]
assert len(strict.source_table(raw + '\n| Item | Cost |\n| Hotel | 100 |')) == 1
assert len(strict.source_table(raw + '\n| Day | Breakfast | Lunch | Dinner |\n| Mar 8 | - | Wrong | Wrong |')) == 1
paren = copy.deepcopy(target)
paren[0]['accommodation'] = 'Hotel!, Area, Seattle'
assert strict.finalize_plan(raw.replace('Rooptop, Seattle', 'Hotel! (Area, Seattle)'), paren) == paren
extract = ns['extract_answer']
assert extract('<think>example <answer>wrong</answer></think><answer>right</answer>') == 'right'
assert extract('<answer>draft <answer>final</answer></answer>') == 'final'
rejected(lambda: extract('<think>not closed'))
rejected(lambda: extract('<answer>not closed'))

calls = []
reply = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(target)}}]}
def post(url, **kwargs):
    calls.append((url, kwargs))
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: reply)
ns['requests'] = SimpleNamespace(post=post)
old_url = os.environ.get('AGENT_STAR_FORMATTER_URL')
old_model = os.environ.get('AGENT_STAR_FORMATTER_MODEL')
os.environ['AGENT_STAR_FORMATTER_URL'] = 'http://formatter.invalid/v1'
os.environ['AGENT_STAR_FORMATTER_MODEL'] = 'Qwen3.8-27B-FP8'
try:
    invoke = ns['invoke_format_model_post_process']
    assert invoke('Synthetic one-day trip', '<think>...</think><answer>' + raw + '</answer>') == target
    payload = calls[-1][1]['json']
    assert payload['temperature'] == 0 and payload['top_p'] == 1 and payload['seed'] == 42
    assert payload['model'] == 'Qwen3.8-27B-FP8'
    assert payload['chat_template_kwargs'] == {'enable_thinking': False}
    assert 'AUTHORITATIVE SOURCE COLUMN BINDINGS' in payload['messages'][1]['content']
    assert payload['messages'][0]['content'] == strict.STRICT_SYSTEM_PROMPT
    count = len(calls)
    assert invoke('query', 'answer not found') == [] and len(calls) == count
    for reason, content in [('length', json.dumps(target)), ('stop', ''), ('stop', 'invalid JSON')]:
        reply['choices'][0] = {'finish_reason': reason, 'message': {'content': content}}
        rejected(lambda: ns['call_format_model']([], max_retries=2))
        assert len(calls) == count + 2
        count += 2
    os.environ.pop('AGENT_STAR_FORMATTER_URL')
    rejected(lambda: ns['call_format_model']([], max_retries=2))
    assert len(calls) == count
    os.environ['AGENT_STAR_FORMATTER_URL'] = 'http://formatter.invalid/v1'
    reply['choices'][0] = {'finish_reason': 'stop', 'message': {'content': json.dumps(target)}}
    scored = []
    def score(info, plan):
        scored.append(plan)
        return dict(commonsense_micro_score=.5, commonsense_macro_score=0,
                    hard_micro_score=.25, hard_macro_score=0, success_pass=0)
    ns['eval_one_sample'] = score
    info = {'query': 'Synthetic query'}
    for name, args, expected in [('sum_travel_reward_fn', (), .75),
                                ('macro_travel_reward_fn', (), 0),
                                ('success_travel_reward_fn', (), 0),
                                ('planner_r1_reward_fn', (1, 10), .75)]:
        output = ns[name](info, raw, *args)
        assert output.reward == expected and output.metadata['formatted_plan'] == target
    assert len(scored) == 4
    assert ns['sum_travel_reward_fn'](info, 'Format error occurs in the previous turn!').reward == 0
finally:
    for key, value in [('AGENT_STAR_FORMATTER_URL', old_url), ('AGENT_STAR_FORMATTER_MODEL', old_model)]:
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
print('STRICT_TRAVEL_REWARD_CHECK_PASS', flush=True)

if sys.argv[1:]:
    run = Path(sys.argv[1])
    total, recognized, failures = 0, 0, []
    for condition in ('epoch4', 'epoch6'):
        path = run / 'strict_final' / condition / 'transcription_details.jsonl'
        for line in path.read_text().splitlines():
            record = json.loads(line)
            total += 1
            recognized += bool(strict.source_table(record['raw_plan']))
            try:
                assert strict.finalize_plan(record['raw_plan'], copy.deepcopy(record['plan'])) == record['plan']
            except (ValueError, AssertionError) as error:
                failures.append({'condition': condition, 'idx': record['item']['idx'], 'error': str(error)})
    print(json.dumps({'records': total, 'recognized_tables': recognized, 'failures': failures}, ensure_ascii=False), flush=True)
    assert not failures, 'Saved strict results require fidelity review; do not weaken checks silently'
    old_cases = json.loads((run / 'formatter_review_cases.json').read_text())
    assert len(old_cases) == 8
    for record in old_cases:
        try:
            bound = strict.finalize_plan(record['raw_plan'], copy.deepcopy(record['formatted_plan']))
            assert bound != record['formatted_plan'], 'Old conversion error reached scoring unchanged'
        except ValueError:
            pass
    print('PREVIOUS_EIGHT_CONVERSION_ERRORS_BOUND_OR_REJECTED', flush=True)
    print('STRICT_TRAVEL_REWARD_REPLAY_PASS', flush=True)
