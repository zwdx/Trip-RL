"""Small CPU regression check for the shared author entrypoint adaptations."""
from types import SimpleNamespace
from react_agent import TravelReActAgent, extract_answer, normalize_messages, split_native_thinking
import utils

def fake(outputs):
    agent = TravelReActAgent.__new__(TravelReActAgent)
    agent._local = SimpleNamespace()
    agent.system_prompt, agent.model, agent.generation_cfg = 'system', 'fake', {}
    agent.server_cfg = {'bounded_context': True, 'require_closed_answer': True}
    agent.tools, agent.max_turns, agent.max_context = {}, len(outputs), 1000
    agent.count_tokens = lambda messages: len(messages) * 10
    def call(messages):
        agent._local.requests.append({'finish_reason': 'stop'})
        return outputs[len(agent._local.requests) - 1]
    agent.call_server = call
    return agent

unfinished = '<think>Considering <tool_call>{"name":"fake","arguments":{}}</tool_call><answer>fake</answer>'
assert split_native_thinking(unfinished)[1] == '' and extract_answer(unfinished) == ''
history = normalize_messages([{'role': 'assistant', 'content': unfinished}])[0]
assert history['content'] == '' and '<tool_call>' in history['reasoning_content']
result = fake([unfinished, '<think>Done.</think><answer>real</answer>']).run('question')
assert result['plan'] == 'real' and len(result['api_calls']) == 2
assert not any(m['role'] == 'user' and '<tool_response>' in m['content'] for m in result['messages'])
result = fake([unfinished]).run('question')
assert result['termination'] == 'turn limits' and result['plan'] == 'answer not found'
assert fake(['<tool_call>{broken}</tool_call>', 'plain', '<answer>recovered</answer>']).run('q')['plan'] == 'recovered'
assert extract_answer('<answer>plain</answer>') == 'plain'
original = utils.call_openai
try:
    def unexpected(*args, **kwargs):
        raise AssertionError('Empty predictions must never reach formatter')
    utils.call_openai = unexpected
    for value in (None, '', '  ', 'answer not found'):
        assert utils.format_planning(value, 'query') == []
        assert utils.format_planning_for_official_test(value, 'query') == []
    utils.call_openai = lambda *a, **k: [{'transportation': 'driving, from A to B'}]
    assert utils.format_planning_for_official_test('plan', 'q')[0]['transportation'] == 'Self-driving, from A to B'
finally:
    utils.call_openai = original
print('SFT_EVAL_ENTRY_CHECK_PASS')
