import uuid
import torch
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager


class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, **kwargs):
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.rollout_manager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, rollout_manager.async_llm_servers)
        self.tokenizer = tokenizer
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, disable_thinking=config.get("rllm", {}).get("disable_thinking", False))

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.accumulate_reasoning = config.get("rllm", {}).get("accumulate_reasoning", False)

        self.train_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.do_sample is False else config.actor_rollout_ref.rollout.temperature,
            top_k=config.actor_rollout_ref.rollout.top_k,
            top_p=config.actor_rollout_ref.rollout.top_p,
        )

        self.val_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else config.actor_rollout_ref.rollout.val_kwargs.temperature,
            top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
            top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
        )

        print(f"train_sampling_params: {self.train_sampling_params}")
        print(f"val_sampling_params: {self.val_sampling_params}")

        self.validate = False  # flag enabled/disabled by AgentWorkflowEngine.execute_tasks_verl

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        sampling_params = self.val_sampling_params.copy() if self.validate or validate else self.train_sampling_params.copy()
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))

        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_ids)

        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        # === 1. 调用后端，拿到原始 completion token_ids ===
        completion_ids: list[int] = await self.server_manager.generate(
            request_id=application_id, prompt_ids=prompt_ids, sampling_params=sampling_params
        )

        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"
            completion_ids = completion_ids[:max_tokens]

        # === 2. 按 OpenAIEngine 的方式：decode → 清理特殊 token → 重新 encode → parse_completion ===
        # 2.1 解码：必须 skip_special_tokens=False，保留所有特殊符号，供 parse_completion 使用
        raw_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        # 2.2 文本级清理，使其尽量等价于 OpenAIEngine 返回的 text
        cleaned_text = raw_text

        # 去掉结尾的 eos_token / eot_token（vLLM 常会把它们一起吐出来，而 OpenAI 的 text 一般不包含）
        eos_token = getattr(self.tokenizer, "eos_token", None)
        eot_token = getattr(self.chat_parser, "eot_token", None)

        # 循环剥离多余的结尾标记（防止重复）
        changed = True
        while changed and cleaned_text:
            changed = False
            if eos_token and cleaned_text.endswith(eos_token):
                cleaned_text = cleaned_text[: -len(eos_token)]
                changed = True
            if eot_token and cleaned_text.endswith(eot_token):
                cleaned_text = cleaned_text[: -len(eot_token)]
                changed = True

        cleaned_text = cleaned_text.rstrip()

        # 2.3 重新 encode，得到“干净”的 completion_ids，保证和 cleaned_text 一一对应
        clean_completion_ids = self.tokenizer.encode(cleaned_text, add_special_tokens=False)

        # 2.4 解析：完全对齐 OpenAIEngine.completion 的流程
        parsed_output = self.chat_parser.parse_completion(clean_completion_ids)

        # print(f"[DEBUG] Raw completion text   : {raw_text}")
        # print(f"[DEBUG] Cleaned completion text: {cleaned_text}")
        # print(f"[DEBUG] Parsed output          : {parsed_output}")

        return ModelOutput(
            # 和 OpenAIEngine 一样：text 等于“API 返回的文本”（这里是我们清理后的文本）
            text=cleaned_text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=clean_completion_ids,
            prompt_length=prompt_length,
            completion_length=len(clean_completion_ids),
            finish_reason=finish_reason,
        )

    async def get_model_response_with_logits(self, messages: list[dict], **kwargs) -> tuple[ModelOutput, torch.Tensor | None]:
        """
        For ARPO: returns ModelOutput + per-token logprobs (as pseudo-logits for entropy).
        """
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        sampling_params = self.val_sampling_params.copy() if self.validate or validate else self.train_sampling_params.copy()
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))

        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_ids)

        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        # 关键：return_logprobs=True 拿到每个 token 的 logprobs
        result = await self.server_manager.generate(
            request_id=application_id, prompt_ids=prompt_ids, sampling_params=sampling_params, return_logprobs=True
        )
        if isinstance(result, tuple):
            completion_ids, logprobs_data = result
        else:
            completion_ids, logprobs_data = result, None

        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"
            completion_ids = completion_ids[:max_tokens]
            if logprobs_data is not None:
                logprobs_data = logprobs_data[:max_tokens]

        raw_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        cleaned_text = raw_text
        eos_token = getattr(self.tokenizer, "eos_token", None)
        eot_token = getattr(self.chat_parser, "eot_token", None)
        changed = True
        while changed and cleaned_text:
            changed = False
            if eos_token and cleaned_text.endswith(eos_token):
                cleaned_text = cleaned_text[: -len(eos_token)]
                changed = True
            if eot_token and cleaned_text.endswith(eot_token):
                cleaned_text = cleaned_text[: -len(eot_token)]
                changed = True
        cleaned_text = cleaned_text.rstrip()

        clean_completion_ids = self.tokenizer.encode(cleaned_text, add_special_tokens=False)
        parsed_output = self.chat_parser.parse_completion(clean_completion_ids)

        output = ModelOutput(
            text=cleaned_text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=clean_completion_ids,
            prompt_length=prompt_length,
            completion_length=len(clean_completion_ids),
            finish_reason=finish_reason,
        )

        # 将 logprobs 转为 pseudo-logits for entropy: 直接用 log_prob 即可
        logits_tensor = None
        if logprobs_data is not None:
            try:
                import torch
                # logprobs_data 是 list[dict]，每个 dict 可能是 {token_id: logprob}
                # 提取每个位置上被选中 token 的 logprob
                log_probs = []
                for lp_dict in logprobs_data:
                    if lp_dict is None:
                        continue
                    # vLLM logprobs 格式：{token_id: Logprob(logprob=..., rank=..., ...)}
                    # 找到当前 token 对应的 logprob
                    for tid, lp_obj in lp_dict.items():
                        if hasattr(lp_obj, 'logprob'):
                            log_probs.append(lp_obj.logprob)
                            break
                if log_probs:
                    logits_tensor = torch.tensor(log_probs, dtype=torch.float32)
            except Exception:
                logits_tensor = None

        return output, logits_tensor

    def wake_up(self):
        self.rollout_manager.wake_up()

    def sleep(self):
        self.rollout_manager.sleep()
