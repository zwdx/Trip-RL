import json 
import queue 
import warnings 
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from rllm.environments.base.base_env import BaseEnv  
from rllm.rewards.reward_fn import RewardFunction, zero_reward 
from rllm.tools.multi_tool import MultiTool   
from rllm.tools.tool_base import Tool  
import os 
import random
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 0))
REWARD_TYPE = os.environ.get("REWARD_TYPE", "") 
TOOL_CALL_ERROR = os.environ.get("TOOL_CALL_ERROR", "false").lower() == "true" 
TOOL_CALL_ERROR_PERCENTAGE = float(os.environ.get("TOOL_CALL_ERROR_PERCENTAGE", 0.0))
TOOL_PARALLELISM = max(1, int(os.environ.get("AGENT_STAR_TOOL_PARALLELISM", "4")))


class TravelToolEnvironment(BaseEnv):
    def __init__(self, task: dict | None = None, tools: list[str] | None = None, tool_map: dict[str, type[Tool]] | None = None, reward_fn: RewardFunction | None = None, max_steps=60, current_training_step=0, validation=False):
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        self.step_count = 0 
        self.max_steps = int(max_steps) 
        self.current_training_step = current_training_step
        self.validation = validation
        if REWARD_TYPE == "PLANNER_R1" and TOTAL_STEPS:
            print(f"Current training step: {self.current_training_step}; Total training steps: {TOTAL_STEPS}")

        if tool_map is not None: 
            self.tools = MultiTool(tool_map=tool_map)
        elif tools is not None:
            self.tools = MultiTool(tools=tools)
        else:
            self.tools = MultiTool(tools=[])
        
        self.task = task
        if reward_fn is None:
            warnings.warn("No reward function specified, will get 0 reward.", stacklevel=2)
            self.reward_fn = zero_reward 
        else:
            self.reward_fn = reward_fn  
    
    def reset(self):
        self.step_count = 0  
        return self.task, {}   
    
    def step(self, action: list[dict] | str | dict): 
        if isinstance(action, dict):
            action = [action] 

        self.step_count += 1  

        reward = 0 

        done = self.step_count >= self.max_steps or isinstance(action, str)  
        reach_limit = done 

        if isinstance(action, list) and action: 
            for tool_call in action:
                if tool_call.get("function", {}).get("name") == "finish":
                    done = True 
                    break 
        if done:
            if isinstance(action, str): 
                llm_response = action  
            elif isinstance(action, list):
                finish_action = None 
                for tool_call in action:
                    if tool_call.get("function", {}).get("name") == "finish":
                        finish_action = tool_call 
                        break 
                if finish_action:
                    arguments = finish_action.get("function", {}).get("arguments", {}) 
                    llm_response = arguments.get("response", "") 
                    termination = arguments.get("termination", "")  
                    if "Format error occurs in the previous turn!" in termination: 
                        llm_response = llm_response + f"\n{termination}" 
                else:
                    llm_response = str(action)  
            
            task_info = self.task if self.task is not None else {}  
            if REWARD_TYPE == "PLANNER_R1" and TOTAL_STEPS:
                reward_output = self.reward_fn(task_info=task_info, action=llm_response, current_step=self.current_training_step, total_steps=TOTAL_STEPS)
            else:
                reward_output = self.reward_fn(task_info=task_info, action=llm_response) 
            return {}, reward_output.reward, done, {"response": action, "metadata": reward_output.metadata, "reach_limit": reach_limit}
        
        tool_calls = action 
        assert isinstance(tool_calls, list)  
        tool_outputs = self._execute_tool_calls(tool_calls)  
        next_obs = {"tool_outputs": tool_outputs}  

        return next_obs, reward, done, {"response": action, "metadata": {}, "reach_limit": reach_limit} 

    def _execute_tool_call(self, tool_call: dict[Any, Any]) -> tuple[str, str]:
        tool_name = tool_call.get("function", {}).get("name", "unknown")
        tool_call_id = tool_call.get("id")
        try:
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_output = self.tools(tool_name=tool_name, **tool_args)

            tool_output_str = tool_output.to_string() if not isinstance(tool_output, str) else tool_output
            if not self.validation and TOOL_CALL_ERROR and random.random() < TOOL_CALL_ERROR_PERCENTAGE:
                tool_output_str = f"Error: Current tool {tool_name} with execution args {tool_args} is not available."
                print(f"Tool call {tool_name} invocation failed with injected error")
            else:
                print(f"Tool call {tool_name} invocation succ with length {len(tool_output_str)}")
            return tool_call_id, tool_output_str
        except Exception as e: 
            print(f"Error executing tool call {tool_name}: {e}") 
            return tool_call_id, f"Error: {str(e)}"

    def _execute_tool_calls(self, tool_calls: list[dict[Any, Any]]) -> dict[str, str]:
        if len(tool_calls) <= 1 or TOOL_PARALLELISM == 1:
            results = [self._execute_tool_call(tool_call) for tool_call in tool_calls]
        else:
            workers = min(TOOL_PARALLELISM, len(tool_calls))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(self._execute_tool_call, tool_calls))
        return {tool_call_id: output for tool_call_id, output in results}

    @staticmethod 
    def from_dict(env_args: dict) -> "TravelToolEnvironment":
        tools = env_args.pop("tools", None)
        tool_map = env_args.pop("tool_map", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 40)
        current_training_step = env_args.pop("current_training_step", 0)
        validation = env_args.pop("validation", False)
        return TravelToolEnvironment(task=env_args, tools=tools, tool_map=tool_map, max_steps=max_steps, reward_fn=reward_fn, current_training_step=current_training_step, validation=validation)
