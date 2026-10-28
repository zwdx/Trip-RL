from tools import *
from prompt import TRAVEL_SYSTEM_PROMPT
from openai import OpenAI  
import json 
from transformers import AutoTokenizer
from utils import format_planning
import time 
import threading
from pathlib import Path


TOOL_MAP = {
    "SearchFlight": SearchFlight(),
    "SearchAccommodation": SearchAccommodation(),
    "SearchRestaurant": SearchRestaurant(),
    "SearchAttraction": SearchAttraction(),
    "SearchCity": SearchCity(),
    "GoogleDistanceMatrix": GoogleDistanceMatrix(),
}


def split_native_thinking(response: str):
    stripped = response.lstrip()
    if stripped.startswith("<think>"):
        reasoning, marker, action = stripped[len("<think>"):].partition("</think>")
        return reasoning, action if marker else ""
    return "", response


def extract_answer(response: str) -> str: 
    response = split_native_thinking(response)[1]
    if "<answer>" in response and "</answer>" in response: 
        lpos = response.find("<answer>")
        rpos = response.rfind("</answer>")
        return response[lpos+len("<answer>"):rpos].strip()
    return ""


def normalize_messages(messages):
    result = []
    for original in messages:
        message = original.copy()
        if message["role"] == "assistant":
            if message["content"].lstrip().startswith("<think>"):
                message["reasoning_content"], message["content"] = split_native_thinking(message["content"])
        result.append(message)
    return result


class TravelReActAgent: 
    def __init__(self, 
                 system_prompt: str = TRAVEL_SYSTEM_PROMPT, 
                 generation_cfg: dict = {}, 
                 server_cfg: dict = {},
                 max_turns: int = 50, 
                 max_context: int = 64 * 1024,
                 model: str = "Qwen/Qwen3-4B-Instruct-2507"):
        self.system_prompt = system_prompt 
        self.tools = TOOL_MAP 
        
        self.server_cfg = server_cfg
        self.generation_cfg = generation_cfg
        self.max_turns = max_turns
        self.max_context = max_context
        self.model = model
        self._local = threading.local()
        self.client = OpenAI(base_url=server_cfg.get("url", "http://localhost:8001/v1"), api_key=server_cfg.get("api_key", "EMPTY"), timeout=1800, max_retries=0)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(server_cfg.get("tokenizer_path", self.model))
        except Exception as e:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    
    def call_server(self, messages: list[dict], max_retries: int = 3) -> str:
        cfg = self.generation_cfg
        checkpoint = getattr(self._local, "checkpoint_path", None)
        if checkpoint:
            # ponytail: rewrite full history per turn; use append-only events if checkpoint IO grows significant.
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps({"model": self.model, "generation_cfg": cfg, "query": messages[1]["content"], "messages": messages, "api_calls": self._local.requests}, ensure_ascii=False))
            temporary.replace(checkpoint)
        wire_messages = normalize_messages(messages) if cfg.get("enable_thinking") else messages
        output_budget = getattr(self._local, "max_tokens", cfg.get("max_tokens", 8192))
        if self.server_cfg.get("bounded_context"):
            output_budget = min(output_budget, self.max_context - self.count_tokens(messages))
            if output_budget <= 0:
                raise ValueError("Native prompt exceeds model context")
        extra = {k: cfg[k] for k in ("top_k", "min_p", "repetition_penalty", "response_format", "stop_token_ids") if k in cfg}
        template = {k: cfg[k] for k in ("enable_thinking", "preserve_thinking") if k in cfg}
        if template:
            extra["chat_template_kwargs"] = template
        if self.server_cfg.get("raw_response_tokens"):
            extra["return_token_ids"] = True
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=wire_messages,
                    temperature=cfg.get("temperature", 0.6), top_p=cfg.get("top_p", 0.95),
                    max_tokens=output_budget,
                    presence_penalty=cfg.get("presence_penalty", 0.0),
                    frequency_penalty=0.0,
                    reasoning_effort=cfg.get("reasoning_effort", "high"),
                    seed=getattr(self._local, "seed", 42) + len(getattr(self._local, "requests", [])),
                    extra_body=extra,
                )
                choice = response.choices[0]
                message = choice.message
                content = message.content or ""
                reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
                raw_response = None
                if self.server_cfg.get("raw_response_tokens"):
                    ids = list(choice.token_ids or [])
                    if not ids:
                        raise ValueError("Teacher did not return requested raw token IDs")
                    while ids and ids[-1] in (self.tokenizer.eos_token_id, getattr(self.tokenizer, "pad_token_id", None)):
                        ids.pop()
                    raw = self.tokenizer.decode(ids, skip_special_tokens=False)
                    raw_response = "<think>\n" + raw if cfg.get("enable_thinking") else raw
                    if cfg.get("enable_thinking"):
                        reasoning, marker, content = raw.partition("</think>")
                        if not marker:
                            reasoning, content = raw, ""
                if not content and not reasoning:
                    raise ValueError("Empty teacher response")
                if not hasattr(self._local, "requests"):
                    self._local.requests = []
                self._local.requests.append({"endpoint_url": self.server_cfg.get("url"), "finish_reason": choice.finish_reason, "stop_reason": getattr(choice, "stop_reason", None), "output_token_ids": list(getattr(choice, "token_ids", None) or []), "usage": response.usage.model_dump() if response.usage else None, "content": content, "reasoning_content": reasoning, "raw_response": raw_response})
                if raw_response is not None:
                    return raw_response
                if reasoning:
                    content = f"<think>\n{reasoning}\n</think>\n\n{content}"
                return content
            except Exception:
                if attempt + 1 == max_retries:
                    raise
                time.sleep(2 ** attempt)

    def count_tokens(self, messages: list[dict]) -> int: 
        if self.generation_cfg.get("enable_thinking"):
            encoded = self.tokenizer.apply_chat_template(normalize_messages(messages), tokenize=True, add_generation_prompt=True, enable_thinking=True, preserve_thinking=True, reasoning_effort=self.generation_cfg.get("reasoning_effort", "xhigh"))
            ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
            return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
        token_length = 0 
        
        for msg in messages: 
            token_length += len(self.tokenizer.encode(msg["content"])) 
        return token_length

    def run(self, query: str, format_result: bool = False, response_validator=None): 
        self._local.requests = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        checkpoint = getattr(self._local, "checkpoint_path", None)
        if checkpoint and checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved["query"] != query or saved["model"] != self.model or saved["generation_cfg"] != self.generation_cfg or saved["messages"][0] != messages[0]:
                raise ValueError("Checkpoint does not match question/model/generation settings/system prompt")
            messages = saved["messages"]
            self._local.requests = saved["api_calls"]
        num_round = sum(message["role"] == "assistant" for message in messages)
        num_llms_available = max(0, self.max_turns - num_round)
        termination, prediction, format_prediction = None, None, None

        if response_validator and (any(response_validator(message["content"]) for message in messages if message["role"] == "assistant") or any(call["finish_reason"] != "stop" for call in self._local.requests)):
            termination, prediction = "invalid response", "answer not found"

        while num_llms_available and termination is None:
            if self.server_cfg.get("bounded_context") and self.count_tokens(messages) >= self.max_context:
                termination, prediction = "token limits", "answer not found"
                break
            num_round += 1 
            num_llms_available -= 1 

            llm_response = self.call_server(messages) 
            messages.append({"role": "assistant", "content": llm_response}) 
            if response_validator and (response_validator(llm_response) or (self._local.requests and self._local.requests[-1]["finish_reason"] != "stop")):
                termination, prediction = "invalid response", "answer not found"
                break
            print(f"[Round {num_round}] response chars: {len(llm_response)}")

            action_response = split_native_thinking(llm_response)[1]
            if ("<tool_call>" in action_response and "</tool_call>" in action_response):
                try:
                    tool_call = action_response.split("<tool_call>")[1].split("</tool_call>")[0].strip() 
                    tool_call = json.loads(tool_call)
                    tool_name = tool_call["name"]
                    tool_params = tool_call["arguments"]
                    if tool_name in self.tools:
                        tool_response = self.tools[tool_name].call(tool_params)
                        tool_response = f"<tool_response>\n{tool_response}</tool_response>"
                        print(f"[Round {num_round}] Tool call {tool_name} invocation success with response length {len(tool_response)}")
                    else:
                        tool_response = f"<tool_response>\nTool {tool_name} not found, please rethink your tool call and try again.</tool_response>"
                        print(f"[Round {num_round}] Invalid tool name: {tool_name}")
                    messages.append({"role": "user", "content": tool_response})
                except Exception as e:
                    tool_response = f"<tool_response>\nError calling tool: {e}</tool_response>"
                    print(f"[Round {num_round}] Error calling tool: {e}")
                    messages.append({"role": "user", "content": tool_response})
            
            elif "<answer>" in action_response and "</answer>" in action_response: 
                termination = "answer"
                prediction = extract_answer(llm_response) 
                break 
            # TODO: GPT-OSS-120B has a bug, the answer is not wrapped in <answer> tags, so we need to add it manually
            elif "<answer>" in action_response: 
                if self.server_cfg.get("require_closed_answer"):
                    termination, prediction = "unclosed answer", "answer not found"
                    break
                termination = "answer" 
                messages[-1]["content"] += "</answer>"
                prediction = action_response.split("<answer>")[1].strip()
                break 

            cur_tokens = self.count_tokens(messages)  
            print(f"[Round {num_round}] Count tokens: {cur_tokens}")
            if cur_tokens > self.max_context: 
                print(f"Token limit exceeded: {cur_tokens} > {self.max_context}, terminating ...")
                if self.server_cfg.get("bounded_context"):
                    termination, prediction = "token limits", "answer not found"
                    break
                messages[-1]["content"] = "The conversation has reached the token limit. Please summarize the conversation to generate a travel plan."  

                prediction = self.call_server(messages) 
                if "</answer>" not in prediction:
                    prediction += "</answer>"
                messages.append({"role": "assistant", "content": prediction}) 
                termination = "token limits"
                break 

        last_action = split_native_thinking(messages[-1]["content"])[1]
        if not ("<answer>" in last_action and "</answer>" in last_action):
            if not prediction:
                prediction = "answer not found" 
            if not num_llms_available and termination is None:
                termination = "turn limits"

        if format_result: 
            if prediction != "answer not found":
                try:
                   format_prediction = format_planning(prediction, query)
                except Exception as e:
                    format_prediction = None

        return {
            "query": query,
            "plan": prediction,
            "format_plan": format_prediction,
            "termination": termination,
            "messages": messages,
            "api_calls": self._local.requests,
        }


if __name__ == "__main__":
    agent = TravelReActAgent(model="Qwen/Qwen3-30B-A3B-Thinking-2507")

    questions = [
        "Please create a travel plan departing from Minneapolis and heading to Seattle for 3 days, from March 29th to March 31st, 2022, with a budget of $1,800.",
    ]

    for question in questions:
        result = agent.run(question)
        print(result)
        print("-" * 40)
