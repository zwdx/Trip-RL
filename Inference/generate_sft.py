"""Collect checked teacher trajectories using Agent-STAR's tools and evaluator."""
import argparse
import ast
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from openai import OpenAI
from eval import eval_one_sample
from prompt import FORMAT_PLANNING, TRAVEL_SYSTEM_PROMPT
from react_agent import TOOL_MAP, TravelReActAgent, normalize_messages, extract_answer


# Reuse the published pure checker without importing the GPU/RL dependency tree.
_author_path = Path(__file__).resolve().parents[1] / "RL/rllm/agents/travel_agent.py"
_author_class = next(node for node in ast.parse(_author_path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "TravelAgentFormat")
_author_method = next(node for node in _author_class.body if isinstance(node, ast.FunctionDef) and node.name == "format_check")
_author_namespace = {"json": json, "re": re}
exec(compile(ast.Module(body=[_author_method], type_ignores=[]), str(_author_path), "exec"), _author_namespace)
_author_format_check = _author_namespace["format_check"]


def qwen_system_prompt():
    # Keep the released tools/guidelines; the native chat template owns reasoning.
    intro, remainder = TRAVEL_SYSTEM_PROMPT.split("Critical Format Requirements:", 1)
    tools_and_guidelines = remainder.split("Available tools:", 1)[1].split("Example response format:", 1)[0]
    return intro + """Response protocol:
Use the model's native reasoning channel. Do not discuss or reproduce protocol tags in reasoning.
After reasoning, emit exactly ONE of the following, without any surrounding text:
1. <tool_call>{"name":"SearchCity","arguments":{"state":"Texas"}}</tool_call>
   Substitute a listed tool name and its JSON arguments. The payload must be a JSON object with exactly name and arguments. Never use XML function/parameter syntax or multiple calls.
2. <answer>the complete day-by-day itinerary</answer>
Close every action tag. Tool responses arrive in the next user message; never invent them.

Available tools:""" + tools_and_guidelines + """
Planning and final-answer requirements:
- Use the exact entity name and database city returned by each tool, keeping separate attractions separate. A street-address municipality does not replace the tool's city.
- Never repeat a restaurant across any meals/days or an attraction across days.
- Self-driving cannot be mixed with flights or taxis during this trip. Respect the user's transportation requirements.
- Give a compact daily table with day, current city/route, transportation, breakfast, attractions, lunch, dinner, accommodation; include exact names and cities. For flights include number, route and times. Use '-' only where no activity is planned. Include a short budget calculation for the whole group and accommodation occupancy/minimum-night checks. Avoid repeated summaries, addresses, phone numbers, decorative headings and optional alternatives.
Now begin the conversation with the user.
"""


QWEN_FORMAT_PLANNING = FORMAT_PLANNING + """
Extraction fidelity requirements:
- Transcribe the itinerary as written; do not optimize it, fix its choices, invent missing meals, move activities to another day/city, or select alternatives to satisfy constraints.
- Preserve exact entity names. If the text lists two distinct attractions joined with '&' or 'and', emit separate 'Name, City;' entries. Do not split an ampersand that belongs to one entity name.
- The city label is the itinerary/database city, not a street-address suburb. Example: Grounds For Sculpture visited during a Trenton day, with a Hamilton Twp street address, is 'Grounds For Sculpture, Trenton;'. Explicit travel to another itinerary city must be preserved.
- Meals before an intercity journey remain in the departure city, meals after it in the arrival city. A restaurant chosen in the wrong city must remain wrong; never relocate it to make it valid.
- Retain all planned attractions separately and output all daily keys. Output only the JSON array.
"""


def protocol_error(text):
    valid, reason = _author_format_check(None, text)
    if not valid:
        return reason
    # The released inference runner executes one action per response.
    tail = text.split("</think>", 1)[1]
    if tail.count("<tool_call>") > 1:
        return "multiple_tool_calls"
    return None

def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def parse_plan(text):
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Some formatters omit the array brackets; add only the container, never values.
        result = json.loads("[" + text + "]")
    return [result] if isinstance(result, dict) else result


def evaluate_plan(item, plan):
    keys = {"day", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"}
    if not isinstance(plan, list) or len(plan) != int(item["days"]) or any(not isinstance(day, dict) or not keys <= day.keys() for day in plan):
        raise ValueError("Invalid daily itinerary schema")
    return eval_one_sample(copy.deepcopy(item), plan)


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def prepare_candidates(run, config):
    target = run / "candidates.jsonl"
    if target.exists():
        return rows(target)
    repo = Path(config["repo"])
    excluded = set()
    for path in (repo / "RL/rllm/data").glob("travel*.jsonl"):
        excluded.update(item["query"].strip() for item in rows(path))
    buckets = {level: [] for level in ("easy", "medium", "hard")}
    seen = set(excluded)
    for item in rows(run / "data/TravelTotal_17K.jsonl"):
        query = item["query"].strip()
        if query in seen:
            continue
        seen.add(query)
        if isinstance(item.get("local_constraint"), str):
            item["local_constraint"] = ast.literal_eval(item["local_constraint"])
        buckets[item["level"]].append(item)
    rng = random.Random(config["seed"])
    for bucket in buckets.values():
        rng.shuffle(bucket)
    candidates = []
    while any(buckets.values()) and len(candidates) < config["candidate_max"]:
        for bucket in buckets.values():
            if bucket and len(candidates) < config["candidate_max"]:
                candidates.append(bucket.pop())
    target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in candidates))
    write_json(run / "candidate_manifest.json", {"seed": config["seed"], "count": len(candidates), "levels": dict(collections.Counter(x["level"] for x in candidates)), "excluded_queries": len(excluded), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    return candidates


def collect(run, limit=None, workers=None):
    config = json.loads((run / "config.json").read_text())
    version = config.get("protocol_version", 2)
    system_prompt = qwen_system_prompt() if version >= 3 else TRAVEL_SYSTEM_PROMPT
    format_prompt = QWEN_FORMAT_PLANNING if version >= 3 else FORMAT_PLANNING
    prompt_hashes = {"planner": hashlib.sha256(system_prompt.encode()).hexdigest(), "formatter": hashlib.sha256(format_prompt.encode()).hexdigest()}
    write_json(run / f"prompts_protocol_v{version}.json", {"planner": system_prompt, "formatter": format_prompt, "sha256": prompt_hashes})
    endpoints = config.get("endpoints") or [{"url": config["url"], "concurrency": config["concurrency"], "pid_file": "server.pid"}]
    if len({x["url"] for x in endpoints}) != len(endpoints) or any(int(x["concurrency"]) < 1 for x in endpoints):
        raise ValueError("Teacher endpoints must be distinct and have positive concurrency limits")
    for endpoint in endpoints:
        for _ in range(90):
            try:
                with urllib.request.urlopen(endpoint["url"] + "/models", timeout=2):
                    break
            except OSError:
                time.sleep(2)
        else:
            raise RuntimeError(f'Teacher server {endpoint["url"]} did not become ready within 180 seconds')
    candidates = prepare_candidates(run, config)
    history = list({record["query"]: record for record in rows(run / "attempts.jsonl")}.values())
    reopen = set(json.loads((run / "protocol_v2_reopen.json").read_text())) if (run / "protocol_v2_reopen.json").exists() else set()
    history = [x for x in history if x["query"] not in reopen or x.get("protocol_version", 1) >= 2]
    completed = {x["query"] for x in history}
    accepted = [x for x in history if x.get("accepted")]
    goal = limit or config["target_successes"]
    concurrency = workers or (config["concurrency"] if history else 4)
    agents = [TravelReActAgent(system_prompt=system_prompt, model=config["served_model"], server_cfg={"url": endpoint["url"], "api_key": "EMPTY", "tokenizer_path": config["teacher_model"], "raw_response_tokens": True}, generation_cfg={**config["generation"], "max_tokens": config["max_tokens"]}, max_context=config["max_context"] - config["max_tokens"], max_turns=config["max_turns"]) for endpoint in endpoints]
    formatters = [OpenAI(base_url=endpoint["url"], api_key="EMPTY", timeout=1800, max_retries=2) for endpoint in endpoints]
    endpoint_active = collections.Counter()
    stopped = threading.Event()
    partials = run / (f"partials_protocol_v{version}" if version >= 3 else "partials")
    partials.mkdir(exist_ok=True)
    for index, original in enumerate(history):
        if not original.get("error", "").startswith("JSONDecodeError") or not original.get("formatter"):
            continue
        record = copy.deepcopy(original)
        try:
            record["plan"] = extract_answer(record["messages"][-1]["content"])
            record["formatted_plan"] = parse_plan(record["formatter"]["content"])
            record["scores"] = evaluate_plan(record["item"], record["formatted_plan"])
            record["accepted"] = record["scores"]["success_pass"] == 1
            record["rejection"] = None if record["accepted"] else "task_constraints"
            record["trajectory_tokens"] = agents[0].count_tokens(record["messages"])
            record["repaired_error"] = record.pop("error")
            record["formatter_container_normalized"] = True
            append(run / "attempts.jsonl", record)
            history[index] = record
            if record["accepted"]:
                accepted.append(record)
                append(run / "successful_trajectories.jsonl", record)
        except (ValueError, KeyError, TypeError) as exc:
            print("PILOT_RECHECK_FAILED", record["candidate_index"], str(exc), flush=True)

    def task(index, item, endpoint_index):
        agent, formatter = agents[endpoint_index], formatters[endpoint_index]
        started = time.time()
        agent._local.seed = config["seed"] + index * 1000
        agent._local.checkpoint_path = partials / f"{index}.json"
        record = {"candidate_index": index, "query": item["query"], "item": item, "accepted": False, "seed": agent._local.seed, "protocol_version": version, "prompt_sha256": prompt_hashes, "endpoint_url": endpoints[endpoint_index]["url"]}
        try:
            result = agent.run(item["query"], response_validator=protocol_error)
            record.update(result)
            errors = [protocol_error(message["content"]) for message in result["messages"] if message["role"] == "assistant"]
            # The legacy agent can append a missing closing tag; check the raw API text too.
            for call in result["api_calls"]:
                raw = call["content"]
                if call["reasoning_content"]:
                    raw = f'<think>\n{call["reasoning_content"]}\n</think>\n\n{raw}'
                errors.append(protocol_error(call.get("raw_response") or raw))
            record["protocol_errors"] = [error for error in errors if error]
            record["truncated_calls"] = sum(call["finish_reason"] != "stop" for call in result["api_calls"])
            if record["protocol_errors"] or record["truncated_calls"] or result["termination"] != "answer":
                record["rejection"] = "protocol_or_truncation"
                return record
            if stopped.is_set():
                record["rejection"] = "target_reached"
                return record
            prompt = format_prompt.replace("{{{user_query}}}", item["query"]).replace("{{{planning_text}}}", result["plan"])
            fmt = config["formatter"]
            response = formatter.chat.completions.create(model=config["served_model"], messages=[{"role": "system", "content": "You are a helpful assistant that formats text into a strict JSON format."}, {"role": "user", "content": prompt}], temperature=fmt["temperature"], top_p=fmt["top_p"], max_tokens=fmt["max_tokens"], presence_penalty=fmt["presence_penalty"], frequency_penalty=0.0, seed=config["seed"], extra_body={"top_k": fmt["top_k"], "min_p": 0.0, "repetition_penalty": fmt["repetition_penalty"], "chat_template_kwargs": {"enable_thinking": fmt["enable_thinking"]}})
            choice = response.choices[0]
            record["formatter"] = {"content": choice.message.content, "finish_reason": choice.finish_reason, "usage": response.usage.model_dump() if response.usage else None}
            if choice.finish_reason != "stop":
                raise ValueError("Formatter output truncated")
            plan = parse_plan(choice.message.content)
            record["formatted_plan"] = plan
            record["scores"] = evaluate_plan(item, plan)
            record["accepted"] = record["scores"]["success_pass"] == 1
            record["rejection"] = None if record["accepted"] else "task_constraints"
            record["trajectory_tokens"] = agent.count_tokens(result["messages"])
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record.setdefault("messages", [])
            record.setdefault("api_calls", getattr(agent._local, "requests", []))
            record["rejection"] = "exception"
        finally:
            record["elapsed_seconds"] = round(time.time() - started, 2)
        return record

    def status(active, state="running"):
        snapshot = {"protocol_version": version, "phase_completed": sum(x.get("protocol_version") == version for x in history), "phase_accepted": sum(x.get("protocol_version") == version for x in accepted), "state": state, "target": goal, "accepted": len(accepted), "completed": len(history), "active": active, "concurrency": concurrency, "rejections": dict(collections.Counter(x.get("rejection") for x in history if not x.get("accepted"))), "accepted_levels": dict(collections.Counter(x["item"]["level"] for x in accepted)), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        snapshot["endpoints"] = [{"url": endpoint["url"], "gpu": endpoint.get("gpu"), "limit": endpoint["concurrency"], "active": endpoint_active[i] if state == "running" else 0} for i, endpoint in enumerate(endpoints)]
        write_json(run / "status.json", snapshot)
        print("PROGRESS", json.dumps(snapshot), flush=True)

    exclusion_file = run / "excluded_semantic_duplicates.json"
    excluded = {item["query"] for item in json.loads(exclusion_file.read_text())} if exclusion_file.exists() else set()
    waiting = iter((i, item) for i, item in enumerate(candidates) if item["query"] not in completed and item["query"] not in excluded)
    with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
        active = {}
        while len(accepted) < goal:
            control = run / "control.json"
            if control.exists():
                concurrency = int(json.loads(control.read_text())["concurrency"])
                if not 1 <= concurrency <= config["concurrency"]:
                    raise ValueError("Concurrency must be between 1 and the configured engine limit")
            capacity = min(concurrency, max(1, goal - len(accepted)))
            while len(active) < capacity:
                available = [i for i, endpoint in enumerate(endpoints) if endpoint_active[i] < int(endpoint["concurrency"])]
                if not available:
                    break
                pair = next(waiting, None)
                if pair is None:
                    break
                endpoint_index = min(available, key=endpoint_active.__getitem__)
                future = pool.submit(task, *pair, endpoint_index)
                active[future] = (pair[0], endpoint_index)
                endpoint_active[endpoint_index] += 1
            if not active:
                break
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                _, endpoint_index = active.pop(future)
                endpoint_active[endpoint_index] -= 1
                record = future.result()
                append(run / "attempts.jsonl", record)
                history.append(record)
                if record.get("accepted"):
                    accepted.append(record)
                    append(run / "successful_trajectories.jsonl", record)
                if record.get("error", "").startswith(("APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError")):
                    concurrency = max(4, concurrency // 2)
                    write_json(control, {"concurrency": concurrency, "reason": record["error"]})
            if not workers and len(history) >= 4 and any(x.get("scores") for x in history[-8:]) and not any(x.get("error") for x in history[-8:]):
                concurrency = min(config["concurrency"], 16 if len(history) < 8 else 64)
            status(len(active))
        stopped.set()
        for future in active:
            future.cancel()
        for future in active:
            endpoint_active[active[future][1]] -= 1
            if not future.cancelled():
                record = future.result()
                append(run / "attempts.jsonl", record)
                history.append(record)
                if record.get("accepted"):
                    accepted.append(record)
                    append(run / "successful_trajectories.jsonl", record)
    selected = accepted[:goal]
    data = [{"messages": record["messages"], "source_id": record["item"].get("id"), "level": record["item"]["level"], "query": record["query"], "protocol_version": record.get("protocol_version", 1), "seed": record["seed"]} for record in selected]
    (run / "sft_messages.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in data))
    write_json(run / "dataset_info.json", {f"agent_star_qwen38_{goal}": {"file_name": "sft_messages.jsonl", "formatting": "sharegpt", "columns": {"messages": "messages"}, "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}}})
    status(0, "complete" if len(selected) == goal else "candidates_exhausted")


def self_check():
    from types import SimpleNamespace
    from tempfile import TemporaryDirectory
    native_prompt = qwen_system_prompt()
    assert "<think>" not in native_prompt and "</think>" not in native_prompt
    assert TRAVEL_SYSTEM_PROMPT.split("<tools>", 1)[1].split("</tools>", 1)[0] in native_prompt
    valid = '<think>Find flights.</think><tool_call>{"name":"SearchFlight","arguments":{"departure":"Boston","destination":"Dallas","date":"2022-03-01"}}</tool_call>'
    assert protocol_error(valid) is None
    assert protocol_error("<think>x</think><answer>plan</answer>") is None
    assert protocol_error("<think>x</think><answer>unclosed")
    assert protocol_error('<think>x</think><tool_call>{"name":"Unknown","arguments":{}}</tool_call>') is None
    assert protocol_error("<think>x</think><answer>a</answer><tool_call>{}</tool_call>")
    message = {"role": "assistant", "content": "<think>reason</think><answer>plan</answer>"}
    normalized = normalize_messages([message])[0]
    assert normalized["reasoning_content"] == "reason" and normalized["content"] == "<answer>plan</answer>"
    assert "reasoning_content" not in message
    assert extract_answer('<think>Use <answer> tags.</think><answer>actual plan</answer>') == "actual plan"
    assert parse_plan('{"day":1},{"day":2}') == [{"day": 1}, {"day": 2}]
    agent = TravelReActAgent.__new__(TravelReActAgent)
    agent.generation_cfg = {"enable_thinking": True}
    agent.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1]})
    assert agent.count_tokens([message]) == 3
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "resume.json"
        prior = [{"role": "system", "content": "travel"}, {"role": "user", "content": "question"}, {"role": "assistant", "content": valid}, {"role": "user", "content": "<tool_response>flight</tool_response>"}]
        checkpoint.write_text(json.dumps({"query": "question", "model": "fake", "generation_cfg": agent.generation_cfg, "messages": prior, "api_calls": []}))
        agent._local = threading.local()
        agent._local.checkpoint_path = checkpoint
        agent.model, agent.system_prompt, agent.max_turns = "fake", "travel", 3
        calls = []
        agent.call_server = lambda messages: calls.append(len(messages)) or "<think>done</think><answer>plan</answer>"
        result = agent.run("question")
        assert calls == [4] and result["plan"] == "plan" and result["messages"][:4] == prior
        agent.system_prompt = "changed prompt"
        try:
            agent.run("question")
        except ValueError as exc:
            assert "system prompt" in str(exc)
        else:
            raise AssertionError("Changed prompt must not resume an old checkpoint")
        agent.system_prompt = "travel"
        agent.call_server = lambda messages: "<think>done</think><answer>plan"
        result = agent.run("question", response_validator=protocol_error)
        assert result["termination"] == "invalid response" and len(result["messages"]) == 5
        invalid = prior + [{"role": "assistant", "content": "invalid"}]
        checkpoint.write_text(json.dumps({"query": "question", "model": "fake", "generation_cfg": agent.generation_cfg, "messages": invalid, "api_calls": []}))
        agent.call_server = lambda messages: (_ for _ in ()).throw(AssertionError("Invalid checkpoint must stop before calling teacher"))
        assert agent.run("question", response_validator=protocol_error)["termination"] == "invalid response"
    assert protocol_error("<think>reason</think>Let me check.<tool_call>{\"name\":\"SearchCity\",\"arguments\":{\"state\":\"Texas\"}}</tool_call>") is None
    assert protocol_error('<think>reason</think><tool_call>{}</tool_call><tool_call>{}</tool_call>')
    fake = TravelReActAgent.__new__(TravelReActAgent)
    fake.generation_cfg = {"enable_thinking": True, "response_format": {"type": "json_object"}}
    fake.server_cfg = {"raw_response_tokens": True}
    fake._local = threading.local()
    raw = 'Mention <tool_call> in thought.\n</think>\n<tool_call>{"name":"SearchCity","arguments":{"state":"Texas"}}</tool_call>'
    fake.tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=98, decode=lambda ids, **kw: raw if ids == [10, 11] else (_ for _ in ()).throw(AssertionError(ids)))
    fake.model = "fake"
    choice = SimpleNamespace(token_ids=[10, 11, 99], finish_reason="stop", message=SimpleNamespace(content="corrupted", reasoning="wrong split"))
    forwarded = []
    fake.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: forwarded.append(kw) or SimpleNamespace(choices=[choice], usage=None))))
    decoded = TravelReActAgent.call_server(fake, [{"role":"system","content":"test"},{"role":"user","content":"question"}])
    assert forwarded[-1]["extra_body"]["response_format"] == fake.generation_cfg["response_format"]
    assert decoded == "<think>\n" + raw and protocol_error(decoded) is None
    assert fake._local.requests[-1]["content"].strip().startswith("<tool_call>")
    assert "Mention <tool_call>" in fake._local.requests[-1]["reasoning_content"]
    choice.token_ids = [10, 11, 98]
    assert TravelReActAgent.call_server(fake, [{"role":"system","content":"test"},{"role":"user","content":"question"}]) == decoded
    agent._local = threading.local()
    agent.max_context = 1000
    seen = []
    agent.tools = {"SearchCity": SimpleNamespace(call=lambda args: seen.append(args) or "Texas")}
    responses = iter([decoded, "<think>done</think><answer>plan</answer>"])
    agent.call_server = lambda messages: next(responses)
    result = agent.run("question", response_validator=protocol_error)
    assert result["termination"] == "answer" and seen == [{"state": "Texas"}]

    # Exercise the real dispatcher/writer with two fake servers, including resume.
    from contextlib import nullcontext
    saved = {key: globals()[key] for key in ("TravelReActAgent", "OpenAI", "eval_one_sample")}
    original_urlopen = urllib.request.urlopen
    guard = threading.Lock()
    inflight, peaks, assignments = collections.Counter(), collections.Counter(), {}

    class FakeAgent:
        def __init__(self, server_cfg, system_prompt, **kwargs):
            self.url, self.system_prompt = server_cfg["url"], system_prompt
            self._local = threading.local()

        def run(self, query, **kwargs):
            with guard:
                assert query not in assignments
                assignments[query] = self.url
                inflight[self.url] += 1
                peaks[self.url] = max(peaks[self.url], inflight[self.url])
            time.sleep(0.02)
            with guard:
                inflight[self.url] -= 1
            text = "<think>done</think><answer>plan</answer>"
            return {"plan": "plan", "termination": "answer", "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": query}, {"role": "assistant", "content": text}], "api_calls": [{"content": "<answer>plan</answer>", "reasoning_content": "done", "raw_response": text, "finish_reason": "stop"}]}

        def count_tokens(self, messages):
            return 10

    def fake_format(url, **kwargs):
        query = re.search(r"Original user query: (q\d+)", kwargs["messages"][1]["content"]).group(1)
        assert assignments[query] == url, "Formatting must stay on the question's server"
        day = {key: "-" for key in ["current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps([{**day, "day": 1}])), finish_reason="stop")], usage=None)

    try:
        globals()["TravelReActAgent"] = FakeAgent
        globals()["OpenAI"] = lambda base_url, **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **request: fake_format(base_url, **request))))
        globals()["eval_one_sample"] = lambda item, plan: {"success_pass": 1}
        urllib.request.urlopen = lambda *args, **kwargs: nullcontext()
        with TemporaryDirectory() as directory:
            run = Path(directory)
            endpoints = [{"url": "http://fake-a/v1", "concurrency": 2}, {"url": "http://fake-b/v1", "concurrency": 2}]
            write_json(run / "config.json", {"protocol_version": 4, "endpoints": endpoints, "repo": str(Path(__file__).resolve().parents[1]), "served_model": "fake", "teacher_model": "fake", "seed": 42, "target_successes": 8, "concurrency": 4, "max_context": 100, "max_tokens": 10, "max_turns": 3, "generation": {}, "formatter": {"temperature": 0, "top_p": 1, "max_tokens": 10, "presence_penalty": 0, "top_k": -1, "repetition_penalty": 1, "enable_thinking": False}})
            (run / "candidates.jsonl").write_text("".join(json.dumps({"query": f"q{i}", "days": 1, "level": "easy", "id": str(i)}) + "\n" for i in range(8)))
            collect(run, workers=4)
            assert len(assignments) == 8 and set(peaks.values()) == {2}
            assert len(rows(run / "successful_trajectories.jsonl")) == 8
            assert {x["endpoint_url"] for x in rows(run / "attempts.jsonl")} == {x["url"] for x in endpoints}
            collect(run, workers=4)
            assert len(assignments) == 8 and len(rows(run / "attempts.jsonl")) == 8
    finally:
        globals().update(saved)
        urllib.request.urlopen = original_urlopen
    print("SELF_CHECK_PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required")
        self_check()
        collect(args.run_dir, args.limit, args.workers)
        status = json.loads((args.run_dir / "status.json").read_text())
        if status["state"] == "complete":
            config = json.loads((args.run_dir / "config.json").read_text())
            endpoints = config.get("endpoints") or [{"url": config["url"], "pid_file": "server.pid"}]
            for endpoint in endpoints:
                pid_name = endpoint.get("pid_file", "server.pid")
                if Path(pid_name).name != pid_name:
                    raise ValueError("Server PID file must be inside the run directory")
                pid_file = args.run_dir / pid_name
                if not pid_file.exists():
                    continue
                pid = int(pid_file.read_text())
                process = Path("/proc") / str(pid)
                try:
                    if not process.exists() or process.stat().st_uid != os.getuid() or os.getpgid(pid) != pid:
                        continue
                    argv = (process / "cmdline").read_bytes().split(b"\0")
                    port = str(urllib.parse.urlsplit(endpoint["url"]).port).encode()
                    if b"vllm.entrypoints.openai.api_server" in argv and b"--port" in argv and argv[argv.index(b"--port") + 1] == port and b"--model" in argv and argv[argv.index(b"--model") + 1] == config["teacher_model"].encode():
                        os.killpg(pid, signal.SIGTERM)
                        print("OWNED_TEACHER_SERVER_STOPPED", pid, flush=True)
                except (FileNotFoundError, ProcessLookupError):
                    pass
