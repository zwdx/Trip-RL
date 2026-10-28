"""CPU regression: display equivalence, fidelity rejection, retry and diagnostics."""
import ast
import copy
import json
import os
from pathlib import Path
import runpy
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import urlsplit
from uuid import uuid4
import requests

root = Path(__file__).resolve().parents[1]
strict = runpy.run_path(str(root/'rllm/rewards/travel_utils/strict_transcription.py'))
source = root/'rllm/rewards/travel_reward.py'
tree = ast.parse(source.read_text())
functions = {'_record_formatter_failure', 'call_format_model', 'invoke_format_model_post_process', 'extract_answer'}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions or
         isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='FORMAT_PLANNING' for t in n.targets)]
ns = dict(strict, os=os, Path=Path, requests=requests, json=json, time=time,
          uuid4=uuid4, urlsplit=urlsplit)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)

literal = strict['literal']
assert literal('**[Bansara’s]**') == literal("[Bansara's]")
assert literal('*Light & Love* vibrant, historic, sleeps 4') == 'Light & Love vibrant, historic, sleeps 4'
assert literal('_Light & Love_') == 'Light & Love'
assert literal('Star*House') == 'Star*House'
assert literal('Café') != literal('Cafe') and literal('NAME') != literal('name')
plan = [{'day':1, 'current_city':'Seattle', 'transportation':'-', 'breakfast':'-',
         'attraction':'-', 'lunch':'-', 'dinner':"Bansara's, Seattle",
         'accommodation':'Light & Love vibrant, historic, sleeps 4, Seattle'}]
raw = "Dinner: **[Bansara’s]**, Seattle. Hotel: *Light & Love* vibrant, historic, sleeps 4, Seattle."
assert strict['finalize_plan'](raw, copy.deepcopy(plan)) == plan
for field, wrong in [('dinner','Different Bistro, Seattle'), ('accommodation','Light & Dove, Seattle')]:
    bad=copy.deepcopy(plan);bad[0][field]=wrong
    try:strict['finalize_plan'](raw,bad)
    except ValueError:pass
    else:raise AssertionError('Real entity change accepted')
head='| Day | City | Transportation | Breakfast | Attractions | Lunch | Dinner | Accommodation |\n'
table=head+'| 1 | Seattle | Flight F123 | - | - | - | Bansara’s, Seattle | *Light & Love* vibrant, historic, sleeps 4, Seattle |\n'
bad=copy.deepcopy(plan);bad[0]['day']=2
try:strict['finalize_plan'](table,bad)
except ValueError:pass
else:raise AssertionError('Day change accepted')
bad=copy.deepcopy(plan);bad[0]['transportation']='Flight Number: F999'
try:strict['finalize_plan'](table,bad)
except ValueError:pass
else:raise AssertionError('Flight change accepted')

def reply(status=200):
    response=requests.Response();response.status_code=status
    response.url='https://integrate.api.nvidia.com/v1/chat/completions'
    response.headers={'x-request-id':'test-request','Content-Type':'application/json'}
    response._content=json.dumps({'error':'test server fault'} if status!=200 else
        {'choices':[{'finish_reason':'stop','message':{'content':'[]'}}]}).encode()
    return response

with tempfile.TemporaryDirectory() as directory:
    env={'AGENT_STAR_FORMATTER_URL':'https://integrate.api.nvidia.com/v1',
         'NVIDIA_API_KEY':'TEST_SECRET_DO_NOT_LOG', 'AGENT_STAR_FORMATTER_API_KEY_FILE':'',
         'AGENT_STAR_FORMATTER_DIAGNOSTICS_DIR':directory}
    session=Mock()
    ns.update(_formatter_retries=lambda:2,_formatter_timeout=lambda:(10,300),
              _formatter_session=lambda:session,_formatter_semaphore=lambda:__import__('threading').BoundedSemaphore(1),
              _wait_formatter_rate_limit=lambda:None)
    with patch.dict(os.environ,env), patch.object(time,'sleep') as sleep:
        session.post.side_effect=[reply(500),reply()]
        assert ns['call_format_model']([])==[] and session.post.call_count==2
        sleep.assert_called_once_with(2.)
        files=list(Path(directory).glob('*.json'));assert len(files)==1
        record=json.loads(files[0].read_text());assert record['response_status']==500
        assert 'test server fault' in record['response_body'] and record['response_headers']['x-request-id']=='test-request'
        assert 'TEST_SECRET_DO_NOT_LOG' not in files[0].read_text()
        assert record['request']['chat_template_kwargs']=={'enable_thinking':False}
        assert 'stop_token_ids' not in record['request']
        session.post.reset_mock();sleep.reset_mock()
        session.post.side_effect=requests.exceptions.ReadTimeout('test timeout')
        try:ns['call_format_model']([])
        except requests.exceptions.ReadTimeout:pass
        else:raise AssertionError('Timeout was swallowed')
        assert session.post.call_count==1 and not sleep.called
        session.post.reset_mock();session.post.side_effect=[reply(500),reply(500)]
        try:ns['call_format_model']([])
        except requests.exceptions.HTTPError:pass
        else:raise AssertionError('Persistent 500 swallowed')
        assert session.post.call_count==2
        bad=copy.deepcopy(plan);bad[0]['dinner']='Different Bistro, Seattle'
        ns['call_format_model']=lambda *a,**k:bad
        try:ns['invoke_format_model_post_process']('query','<answer>'+raw+'</answer>')
        except ValueError:pass
        else:raise AssertionError('Fidelity failure was swallowed')
        records=[json.loads(x.read_text()) for x in Path(directory).glob('*.json')]
        assert any(x.get('phase')=='fidelity' and x.get('source')==raw and x.get('converted_plan')==bad for x in records)
print('FORMATTER_RELIABILITY_CPU_CHECK_PASSED')
