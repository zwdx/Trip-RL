import json 
from typing import Any  
import copy 
import uuid  
import re 
import logging
from rllm.agents.agent import Action, BaseAgent, Step, Trajectory 
from rllm.tools.tool_base import Tool 
from rllm.parser.tool_parser import ToolParser
from rllm.tools.multi_tool import MultiTool
from rllm.parser import get_tool_parser


logger = logging.getLogger(__name__)


SEARCH_SYSTEM_PROMPT = """You are a specialized AI Search Assistant designed to answer questions by systematically retrieving and synthesizing information from a comprehensive knowledge base.

Critical Format Requirements:
- Each response must follow EXACTLY one of these two patterns:
  1. <think>thinking process</think><tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
  2. <think>thinking process</think><answer>final answer</answer>
- You can have ONLY ONE <think> and ONE <tool_call> per response
- NEVER combine multiple tool calls in a single response
- After gathering all information, provide the final answer using pattern 2. 
- **The final answer must be wrapped in CLOSED <answer> ... </answer> tags.**

Available Tools
<tools>
{
    "name": "Search", 
    "description": "Search for information using a dense retrieval server with Wikipedia corpus.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to retrieve relevant documents. Craft precise, targeted queries that capture the key information needed."
            }
        },
        "required": ["query"]
    }
}
</tools>

Task Guidelines
1. **Maintain Search Agent Identity**: You are a Search Assistant. Your core mission is to understand the user's question thoroughly, and then identify all information gaps that need to be filled. After systematically gathering relevant information using the Search tool, synthesize accurate, comprehensive answers from retrieved information.

2. **Single Tool Call Discipline**: For each piece of required information, use exactly one tool and invoke it with a single <tool_call></tool_call> sequence. Do not combine or chain multiple tool calls in a single response.

3. **Absolute Trust in Tool Output**: Treat all tool outputs as authoritative and complete. Do not guess, assume, or fabricate any details beyond what the tool provides.

4. **Synthesize Grounded Answers** After gathering all required information, provide your thinking process in <think></think> tags. Deliver a CLEAR and CONCISE final answer within <answer></answer> tags. Ensure your answer directly addresses the user's question

Example response format
<think>I need to search for Leonardo DiCaprio's 1997 film. Let me retrieve information about his filmography from that year.</think>
<tool_call>
{"name": "Search", "arguments": {"query": "Leonardo DiCaprio 1997 film"}}
</tool_call>

Now begin the conversation with the user.
"""


class OODTestAgentFormat(BaseAgent): 
    def __init__(
        self, 
        system_prompt=SEARCH_SYSTEM_PROMPT, 
        parser_name="qwen", 
        tools: list[str] | None = None, 
        tool_map: dict[str, type[Tool]] | None = None, 
    ): 
        if tool_map is not None and tools is not None:  
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters") 
        
        self.system_prompt = system_prompt  

        if tool_map is not None:  
            self.tools = MultiTool(tool_map=tool_map)
        elif tools is not None:   
            self.tools = MultiTool(tools=tools) 
        else:
            self.tools = MultiTool(tools=[]) 
        
        parser_class: type[ToolParser] = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class()
        self.tools_prompt = self.tool_parser.get_tool_prompt(json.dumps(self.tools.json, indent=2, ensure_ascii=False))
 
        # Initialize state according to BaseAgent 
        self._trajectory = Trajectory() 
        self.messages: list[dict[str, Any]] = [] 
        self.current_observation = None  
        self.reset() 
    
    def reset(self):
        self._trajectory = Trajectory()   
        self.messages = [{"role": "system", "content": self.system_prompt}]  

    def _format_observation_as_messages(self, obs: Any) -> list[dict]: 
        messages = []
        if isinstance(obs, dict):
            # Important: the key for question is "question"
            if "question" in obs:
                messages.append({"role": "user", "content": obs["question"]}) 
            elif "tool_outputs" in obs:
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    messages.append(
                        {
                            "role": "user",
                            "content": "<tool_response>\n" + tool_output_str + "</tool_response>",
                        }
                    )
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs:
            messages.append({"role": "user", "content": str(obs)})
        return messages
    
    def format_check(self, response: str) -> tuple[bool, str]: 
        """
        Check if the response is in the correct format with the following rules:
        1. have and only have one <think> 
        2. <answer></answer> and <tool_call></tool_call> can only appear one type of tag in the response 
        3. xml tags should be closed properly  
        4. the sequence of tags should be correct, i.e., <think>...</think><tool_call>...</tool_call> or <think>...</think><answer>...</answer> 
        5. if there is <tool_call></tool_call>, then each tool call should be a valid tool call, i.e., a valid json object  

        the final format should be within the following format:
            - <think> thinking process </think><answer> final answer </answer>
            - <think> thinking process </think><tool_call> tool call json 1 </tool_call><tool_call> tool call json 2 </tool_call> 
        """

        if response.count("<think>") != 1:
            return False, f"There should be exactly one <think> tag in the response, found {response.count('<think>')}" 
        if response.count("</think>") != 1:
            return False, f"There should be exactly one </think> tag in the response, found {response.count('</think>')}" 
        
        if not response.strip().startswith("<think>"):
            return False, "Response must start with the <think> tag" 
        
        try:
            think_end_index = response.index("</think>") + len("</think>") 
            remaining_response = response[think_end_index:].strip()  
        except ValueError:
            return False, "Mismatched <think></think> tags"
        
        if not remaining_response:
            return False, "Must have either an <answer> or <tool_call> tag after <think>" 
        
        tag_pattern = r'<(/?\w+)>'
        tag_matches = list(re.finditer(tag_pattern, remaining_response))
        if not tag_matches:
            return False, f"Content found after <think> block, but no valid tags (<answer> or <tool_call>) were found" 
        
        stack = []
        tag_sequence = [] 
        focused_tags = ["tool_call", "/tool_call", "answer", "/answer"] 

        for match in tag_matches:
            tag = match.group(1) 
            pos = match.start()  
            end_pos = match.end()  

            if tag not in focused_tags:
                print(f"Info: Tag '{tag} is not part of the standard format and will be ignored") 
                continue  

            if tag.startswith("/"):
                tag_name = tag[1:]
                if not stack:
                    return False, f"Unexpected closing tag '</{tag_name}'"
                open_tag = stack.pop() 
                if open_tag["name"] != tag_name:
                    return False, f"Mismatched tags: <{open_tag['name']}> and </{tag_name}>" 

                content_start = open_tag['end_pos'] 
                content_end = pos 
                content = remaining_response[content_start:content_end]
                tag_sequence.append({"name": tag_name, "content": content}) 
            else:
                tag_name = tag  
                if stack: 
                    return False, "Nested tags are not allowed in the section after <think>" 
                
                stack.append({"name": tag_name, "start_pos": pos, "end_pos": end_pos})
        
        if stack: 
            unclosed = [tag["name"] for tag in stack] 
            return False, f"Unclosed tags: {unclosed}"  
        
        if not tag_sequence:
            return False, "No complete tag pairs found after <think>"  
        
        answer_count = sum(1 for tag in tag_sequence if tag["name"] == "answer")
        tool_call_count = sum(1 for tag in tag_sequence if tag["name"] == "tool_call") 

        if answer_count > 0 and tool_call_count > 0: 
            return False, "Cannot have both <answer> and <tool_call> tags"

        if answer_count > 0: 
            if answer_count != 1:
                return False, f"Must have exactly one <answer> tag, found {answer_count}" 
            if len(tag_sequence) != 1:
                return False, "Only a single <answer> tag is allowed after the <think> tag" 
        elif tool_call_count > 0:
            if len(tag_sequence) != tool_call_count:
                return False, "All tags after <think> must be <tool_call> if tool calls are present" 
        else:
            return False, "Must have either an <answer> or <tool_call> tag after <think>"  
        
        for tag in tag_sequence:
            if tag["name"] == "tool_call":
                try:
                    json.loads(tag["content"].strip())
                except json.JSONDecodeError as e:
                    return False, f"Invalid JSON in <tool_call> tag: {e}. Content: {tag['content'][:100]}..." 

        return True, "" 
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs): 
        obs_messages = self._format_observation_as_messages(observation) 
        self.messages.extend(obs_messages)  
        self.current_observation = observation  
    
    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's state based on the model's response.
        Parses the response, updates messages, and the current step in the trajectory.
        """
        termination = "answer" 
        tool_calls_dict = [] 
        assistant_content = response  
        print(f"Assistant response (length: {len(assistant_content)}): {assistant_content}")

        # format checking 
        format_check, format_check_log = self.format_check(response)

        if not format_check:
            print(f"Format error occurs in the previous turn! Reason: {format_check_log}")
            termination = "Format error occurs in the previous turn!" + f"Reason: {format_check_log}" 
        else:
            try:
                tool_calls = self.tool_parser.parse(response)
                tool_calls_dict = [
                    {
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": tool_call.to_dict(),
                    } for tool_call in tool_calls 
                ]
            except Exception as e: 
                print(f"Failed to parse tool calls from string response: {e}") 
                tool_calls_dict = []  
                error_response = "Format error occurs!" + f"Reason: Failed to parse tool calls from string response: {e}" 
                assistant_content = error_response  
        
        assistant_message = {"role": "assistant", "content": assistant_content} 
        if len(tool_calls_dict) > 0:
            for call in tool_calls_dict:
                if isinstance(call.get("function", {}).get("arguments"), dict):
                    call["function"]["arguments"] = json.dumps(call["function"]["arguments"], ensure_ascii=False) 
        else:
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": {
                            "response": assistant_content,
                            "termination": termination,
                        },
                    },
                }
            ]    
        
        self.messages.append(assistant_message) 

        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=tool_calls_dict,
            model_response=response,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)

        return Action(action=tool_calls_dict)
        
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self.messages 
    
    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory 
    