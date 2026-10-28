import os 
import requests 
from pathlib import Path
from urllib.parse import urlsplit
from requests.adapters import HTTPAdapter
import json 
import threading
import math
import time 
from uuid import uuid4
from rllm.rewards.travel_utils.eval_commonsense import check_commonsense
from rllm.rewards.travel_utils.eval_hardconstraint import check_hardconstraint 
from rllm.rewards.reward_types import RewardOutput
from rllm.utils import convert_numpy_types 
from rllm.rewards.travel_utils.strict_transcription import (
    STRICT_SYSTEM_PROMPT, bind_source, finalize_plan,
)


_THREAD_LOCAL = threading.local()
_FORMATTER_SEMAPHORE = None
_FORMATTER_SEMAPHORE_LOCK = threading.Lock()


def _formatter_dns_adapter(servers):
    """Use explicit DNS only for this HTTPS session; preserve TLS host verification."""
    import dns.resolver
    from urllib3 import PoolManager
    from urllib3.connection import HTTPSConnection
    from urllib3.connectionpool import HTTPSConnectionPool
    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = servers
    resolver.lifetime = 5
    resolver.cache = dns.resolver.Cache()

    class ResolvedConnection(HTTPSConnection):
        def _new_conn(self):
            hostname = self._dns_host
            self._dns_host = str(resolver.resolve(hostname, 'A')[0])
            try:
                return super()._new_conn()
            finally:
                self._dns_host = hostname

    class ResolvedPool(HTTPSConnectionPool):
        ConnectionCls = ResolvedConnection

    class ResolvedAdapter(HTTPAdapter):
        def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
            self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block, **kwargs)
            self.poolmanager.pool_classes_by_scheme = {
                **self.poolmanager.pool_classes_by_scheme, 'https': ResolvedPool}
    return ResolvedAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)


def _formatter_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, 'session', None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        servers = os.getenv('AGENT_STAR_FORMATTER_DNS_SERVERS', '').split()
        if servers:
            session.mount('https://integrate.api.nvidia.com/', _formatter_dns_adapter(servers))
        _THREAD_LOCAL.session = session
    return session


def _formatter_timeout() -> tuple[float, float]:
    connect = float(os.getenv('AGENT_STAR_FORMATTER_CONNECT_TIMEOUT', '10'))
    read = float(os.getenv('AGENT_STAR_FORMATTER_TIMEOUT', '300'))
    return connect, read


def _formatter_retries() -> int:
    return max(1, int(os.getenv('AGENT_STAR_FORMATTER_RETRIES', '1')))


def _formatter_max_concurrency() -> int:
    return max(1, int(os.getenv('AGENT_STAR_FORMATTER_MAX_CONCURRENCY', '1')))


def _formatter_semaphore() -> threading.BoundedSemaphore:
    global _FORMATTER_SEMAPHORE
    if _FORMATTER_SEMAPHORE is None:
        with _FORMATTER_SEMAPHORE_LOCK:
            if _FORMATTER_SEMAPHORE is None:
                _FORMATTER_SEMAPHORE = threading.BoundedSemaphore(_formatter_max_concurrency())
    return _FORMATTER_SEMAPHORE


FORMAT_PLANNING = """Please assist me in extracting valid information from a given natural language text and reconstructing it in JSON format, as demonstrated in the following example. 

**IMPORTANT: You must output ONLY valid JSON format. Do not include any explanatory text, markdown code blocks, or additional formatting.**

**GUIDELINES**
1. each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation']. Replace non-specific information like 'eat at home/on the road' with '-'. Additionally, delete any '$' symbols.
2. transportation: Preserve explicit source city/route fields even when transportation contradicts them, including every intermediate stop. Never overwrite an explicit city to match transportation. Normalize flight labels to 'Flight Number:' and 'Cost:'. For unambiguous driving modes use 'Self-driving'. Do not change the actual transport choice, identifiers, times or costs.
3. attraction: Use a ';' to separate different attractions, with each attraction strictly formatted as 'Name, City'. 
4. day: start from 1.

**OUTPUT FORMAT: Return ONLY the JSON array, nothing else.**
 
**EXAMPLES**
[{
    "day": 1,
    "current_city": "from Dallas to Peoria",
    "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01",
    "breakfast": "-",
    "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;",
    "lunch": "-",
    "dinner": "Tandoor Ka Zaika, Peoria",
    "accommodation": "Bushwick Music Mansion, Peoria"
},
{
    "day": 2,
    "current_city": "Peoria",
    "transportation": "-",
    "breakfast": "Tandoor Ka Zaika, Peoria",
    "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;",
    "lunch": "Cafe Hashtag LoL, Peoria",
    "dinner": "The Curzon Room - Maidens Hotel, Peoria",
    "accommodation": "Bushwick Music Mansion, Peoria"
},
{
    "day": 3,
    "current_city": "from Peoria to Dallas",
    "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20",
    "breakfast": "-",
    "attraction": "-",
    "lunch": "-",
    "dinner": "-",
    "accommodation": "-"
}]


Now please help me extract valid information from the following text and reconstruct it in a strict JSON format. 

**CRITICAL: Your response must be ONLY the JSON array. Do not include any markdown code blocks (```json), explanatory text, or other formatting. Just the raw JSON array starting with [ and ending with ].**

Original user query: {{{user_query}}}

Planning text: {{{planning_text}}}

"""


def extract_answer(response: str) -> str: 
    if response.lstrip().startswith("<think>"):
        if "</think>" not in response:
            raise ValueError("Unclosed thinking is not an itinerary")
        response = response.rsplit("</think>", 1)[1]
    if "<answer>" in response or "</answer>" in response:
        if "<answer>" not in response or "</answer>" not in response.rsplit("<answer>", 1)[1]:
            raise ValueError("Unclosed answer is not an itinerary")
        response = response.rsplit("<answer>", 1)[1].split("</answer>", 1)[0]
    return response.strip()


def _wait_formatter_rate_limit():
    rpm = float(os.getenv('AGENT_STAR_FORMATTER_RPM', '0'))
    if not math.isfinite(rpm) or rpm < 0:
        raise ValueError('Formatter RPM must be finite and nonnegative')
    if not rpm:
        return
    import fcntl
    path = Path(os.environ['AGENT_STAR_FORMATTER_RATE_FILE'])
    path.parent.mkdir(parents=True, exist_ok=True)
    # ponytail: one-node pacing; use a shared rate service for multi-node training.
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        previous = float(stream.read() or '0')
        delay = max(0., 60./rpm-max(0., time.monotonic()-previous))
        if delay:
            time.sleep(delay)
        stream.seek(0)
        stream.truncate()
        stream.write(str(time.monotonic()))
        stream.flush()


def _record_formatter_failure(**record):
    directory = os.getenv('AGENT_STAR_FORMATTER_DIAGNOSTICS_DIR', '')
    if not directory:
        return
    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        record['timestamp_unix'] = time.time()
        with (path / f'{uuid4().hex}.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
    except OSError as error:
        print(f'Cannot save formatter failure diagnostic: {error}')


def call_format_model(messages: list[dict], format_model=None, max_retries: int | None = None):
    base_url = os.getenv("AGENT_STAR_FORMATTER_URL", "").strip()
    if not base_url:
        raise RuntimeError("Set AGENT_STAR_FORMATTER_URL to the formatter /v1 endpoint")
    provider_key = {"openrouter.ai": "OPENROUTER_API_KEY",
                    "integrate.api.nvidia.com": "NVIDIA_API_KEY"}.get(urlsplit(base_url).hostname)
    remote = provider_key is not None
    api_key = "EMPTY"
    if remote:
        api_key = os.getenv(provider_key, "").strip()
        key_file = os.getenv("AGENT_STAR_FORMATTER_API_KEY_FILE", "").strip()
        if not api_key and key_file and Path(key_file).is_file():
            api_key = Path(key_file).read_text().strip()
        if not api_key:
            raise RuntimeError(f"Formatter requires {provider_key} or AGENT_STAR_FORMATTER_API_KEY_FILE")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    base_url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": format_model or os.getenv("AGENT_STAR_FORMATTER_MODEL", "Qwen3.8-27B-FP8"),
        "messages": messages,
        "max_tokens": 8192, 
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "top_k": -1,
        "min_p": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "repetition_penalty": 1,
        "stop_token_ids": [248046, 248044],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if remote:
        for key in ("top_k", "min_p", "repetition_penalty", "stop_token_ids", "chat_template_kwargs"):
            payload.pop(key)
        if provider_key == "OPENROUTER_API_KEY":
            payload["reasoning"] = {"enabled": False}
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
    retries = _formatter_retries() if max_retries is None else max(1, int(max_retries))
    attempts = retries
    timeout = _formatter_timeout()
    session = _formatter_session()
    semaphore = _formatter_semaphore()

    while retries:
        response = None
        started = time.monotonic()
        try:
            with semaphore:
                if remote:
                    _wait_formatter_rate_limit()
                response = session.post(base_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Formatter output truncated or missing stop reason")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Formatter returned empty content")
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            # Save the exact request data and response body, never authorization headers.
            _record_formatter_failure(phase='request', error=f'{type(e).__name__}: {e}',
                attempt=attempts-retries+1, elapsed_seconds=time.monotonic()-started,
                request=payload, response_status=getattr(response, 'status_code', None),
                response_body=getattr(response, 'text', None),
                response_headers={key: value for key, value in getattr(response, 'headers', {}).items()
                                  if key.lower() in ('x-request-id', 'retry-after', 'content-type')})
            retryable = isinstance(e, requests.exceptions.ConnectionError)
            if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                retryable = e.response.status_code == 429 or 500 <= e.response.status_code < 600
            if isinstance(e, requests.exceptions.ReadTimeout):
                retryable = False
            retries -= 1
            if retries == 0 or not retryable:
                print(f"Failed to call format model after {attempts - retries} attempts. Error: {e}")
                raise
            time.sleep(min(30., 2. ** (attempts-retries)))


def invoke_format_model_post_process(query: str, action: str, format_model=None):
    action = extract_answer(action)
    if not action or action.lower() == "answer not found":
        return []
    format_prompt = FORMAT_PLANNING.replace("{{{user_query}}}", query).replace("{{{planning_text}}}", bind_source(action))
    messages = [
        {"role": "system", "content": STRICT_SYSTEM_PROMPT},
        {"role": "user", "content": format_prompt}
    ]

    response = call_format_model(messages, format_model)
    
    try:
        return finalize_plan(action, response)
    except ValueError as error:
        _record_formatter_failure(phase='fidelity', error=f'{type(error).__name__}: {error}',
            source=action, converted_plan=response, model=format_model or os.getenv('AGENT_STAR_FORMATTER_MODEL'))
        raise


def eval_one_sample(question: dict, tested_data: list[dict]) -> dict: 
    # TODO: important! convert numpy types to python types
    question = convert_numpy_types(question)
    if not tested_data or len(tested_data) == 0:
        return {
            "commonsense_macro_score": 0,
            "commonsense_micro_score": 0,
            "hard_macro_score": 0,
            "hard_micro_score": 0,
            "success_pass": 0,
        }
    
    commonsense_results = check_commonsense(question, tested_data) 

    hard_results = None 
    try:
        question["local_constraint"] = eval(question.get("local_constraint", "{}"))
    except:
        pass 

    if commonsense_results and commonsense_results["is_not_absent"][0] and commonsense_results["is_valid_information_in_sandbox"][0]:
        hard_results = check_hardconstraint(question, tested_data)
    
    commonsense_all_pass, micro_pass = True, 0
    for constraint, (is_valid, message) in commonsense_results.items():  
        if is_valid is not None:
            if not is_valid: 
                commonsense_all_pass = False 
            else: # Pass the commonsense constraint
                micro_pass += 1 

    commonsense_macro_score, commonsense_micro_score = int(commonsense_all_pass), micro_pass * 1.0 / 8
    
    hard_macro_score, hard_micro_score = None, None
    if hard_results:
        # Check if all hard constraints are None (not applicable)
        all_none = all(is_valid is None for is_valid, _ in hard_results.values())
        
        if not all_none:
            hard_all_pass, micro_pass, micro_total = True, 0, 0
            for constraint, (is_valid, message) in hard_results.items():  
                if is_valid is not None and not is_valid:
                    hard_all_pass = False
                if is_valid is not None:
                    micro_pass += 1 if is_valid else 0
                    micro_total += 1
            hard_macro_score, hard_micro_score = int(hard_all_pass), micro_pass * 1.0 / micro_total if micro_total > 0 else 0

    # Final pass rate: commonsense must pass, and if hard constraints exist and are applicable, they must also pass
    if hard_macro_score is None:
        # All hard constraints are not applicable, only consider commonsense
        success_pass = int(commonsense_all_pass)
        hard_macro_score, hard_micro_score = 0., 0.
    else:
        success_pass = int(commonsense_all_pass and hard_macro_score)

    return {
        "commonsense_macro_score": commonsense_macro_score,
        "commonsense_micro_score": commonsense_micro_score,
        "hard_macro_score": hard_macro_score,
        "hard_micro_score": hard_micro_score,
        "success_pass": success_pass,
        "details": {
            "commonsense_results": commonsense_results,
            "hard_results": hard_results
        }
    }
    

def fine_grained_travel_reward_fn(task_info: dict, action: str, format_model=None):
    # 0 - Format error with 0 reward
    if "Format error occurs in the previous turn!" in action: 
      return False, "Format error occurs in the previous turn!", [] 

    # 1 - format planning 
    try:
        formatted_plan = invoke_format_model_post_process(task_info["query"], action, format_model)
    except Exception as e:
        return False, f"Error formatting planning: {e}", []

    # 2 - calculate reward  
    try:
        reward_dict = eval_one_sample(task_info, formatted_plan)
        return True, reward_dict, formatted_plan
    except Exception as e:
        return False, f"Error evaluating one sample: {e}", []


def sum_travel_reward_fn(task_info: dict, action: str, format_model=None) -> RewardOutput: 
    flg, res, formatted_plan = fine_grained_travel_reward_fn(task_info, action, format_model) 
    if not flg:
        return RewardOutput(reward=0.0, metadata={"error": res})
    
    reward_value = res.get("commonsense_macro_score", 0) + res.get("commonsense_micro_score", 0) + res.get("hard_macro_score", 0) + res.get("hard_micro_score", 0) + res.get("success_pass", 0)
    return RewardOutput(reward=reward_value, metadata={"judgement": res, "formatted_plan": formatted_plan})
 

def success_travel_reward_fn(task_info: dict, action: str, format_model=None) -> RewardOutput: 
    flg, res, formatted_plan = fine_grained_travel_reward_fn(task_info, action, format_model) 
    if not flg:
        return RewardOutput(reward=0.0, metadata={"error": res})
    
    return RewardOutput(reward=res.get("success_pass", 0), metadata={"judgement": res, "formatted_plan": formatted_plan})


def macro_travel_reward_fn(task_info: dict, action: str, format_model=None) -> RewardOutput: 
    flg, res, formatted_plan = fine_grained_travel_reward_fn(task_info, action, format_model) 
    if not flg:
        return RewardOutput(reward=0.0, metadata={"error": res})
    
    reward_value = res.get("commonsense_macro_score", 0) + res.get("hard_macro_score", 0) + res.get("success_pass", 0)
    return RewardOutput(reward=reward_value, metadata={"judgement": res, "formatted_plan": formatted_plan})


def planner_r1_reward_fn(task_info: dict, action: str, current_step: int, total_steps: int, format_model=None) -> RewardOutput: 
    flg, res, formatted_plan = fine_grained_travel_reward_fn(task_info, action, format_model) 
    if not flg:
        return RewardOutput(reward=0.0, metadata={"error": res})
    
    if total_steps and current_step / total_steps <= 0.33:
        reward_value = res.get("commonsense_macro_score", 0) + res.get("commonsense_micro_score", 0) + res.get("hard_macro_score", 0) + res.get("hard_micro_score", 0) + res.get("success_pass", 0) 
    elif total_steps and current_step / total_steps <= 0.67:  
        reward_value = res.get("commonsense_macro_score", 0) + res.get("hard_macro_score", 0) + res.get("success_pass", 0) 
    else:
        reward_value = res.get("success_pass", 0) 
    
    return RewardOutput(reward=reward_value, metadata={"judgement": res, "formatted_plan": formatted_plan})


if __name__ == "__main__":
    task_info = {
        "query": "Can you design a 3-day travel itinerary for 2 people, departing from Ithaca and heading to Newark from March 18th to March 20th, 2022? Our budget is set at $1,200, and we require our accommodations to be entire rooms and visitor-friendly. Please note that we prefer not to drive ourselves during this trip.",
        "org": "Ithaca", 
        "dest": "Newark", 
        "days": 3, 
        "date": ["2022-03-18", "2022-03-19", "2022-03-20"], 
        "visiting_city_number": 1, 
        "people_number": 2, 
        "local_constraint": {
            "house_rule": "visitors",
            "cuisine": None, 
            "room type": "entire room",  
            "transportation": "no self-driving", 
        },
        "budget": 1000, 
        "level": "hard",
        "id": "hard_jsjsy2", 
    } 

    action = """ "# 3-Day Newark Travel Itinerary from Ithaca
**Travel Dates: March 18-20, 2022 | For 2 People | Budget: $1,200**

## Day 1: March 18, 2022 (Travel Day)
- **Morning**: Depart from Ithaca by car.
- **Transportation**: Drive from Ithaca to Newark (3 hours 38 minutes, 343 km, cost: $17 one way).
- **Afternoon**: Arrive in Newark and check into accommodation: **"1 Bedroom in UWS Manhattan"** ($117/night, 2 nights total $234).
- **Activity**: Visit **Military Park** (free, contact: (973) 900-5800).
- **Dinner**: At **Artistry** (avg cost $79/person, 2 meals $158 total, cuisines: Cafe, Pizza, Desserts, Seafood).
- **Accommodation**: "1 Bedroom in UWS Manhattan" ($117/night).

## Day 2: March 19, 2022 (Full Day in Newark)
- **Breakfast**: At **Harmann Restaurant** (avg cost $57/person, 2 meals $114 total, cuisines: Tea, Cafe, American, Fast Food).
- **Morning**: Visit **The Newark Museum of Art** (contact: (973) 596-6550).
- **Lunch**: At **Jaguar** (avg cost $38/person, 2 meals $76 total, cuisines: French, Bakery, Seafood).
- **Afternoon**: Visit **Branch Brook Park** (contact: (973) 268-3500).
- **Dinner**: At **Escape Terrace Bar Kitchen** (avg cost $100/person, 2 meals $200 total, cuisines: Tea, Cafe, Bakery, Chinese, Seafood).
- **Accommodation**: "1 Bedroom in UWS Manhattan" ($117/night).

## Day 3: March 20, 2022 (Travel Day)
- **Morning**: Check out from accommodation.
- **Transportation**: Drive from Newark to Ithaca (3 hours 38 minutes, 343 km, cost: $17 one way).
- **Afternoon**: Arrive back in Ithaca.

## Budget Breakdown
- Transportation (round trip): $17 + $17 = $34
- Accommodation (2 nights): $117 × 2 = $234
- Meals (Day 1 dinner, Day 2 breakfast, lunch, dinner): $158 + $114 + $76 + $200 = $548
- Attractions (assumed minimal or free): $0 (based on tool data)
- **Total cost**: $34 + $234 + $548 + $0 = $786

## Additional Notes
- All accommodations are entire rooms and visitor-friendly.
- Meal costs are averages based on restaurant data.
- Attractions may have entry fees; check individual websites for details.
- Total cost is within the $1,200 budget.

Enjoy your trip to Newark!
    """
    start_time = time.time()
    flg, res, plan = fine_grained_travel_reward_fn(task_info, action)
    print(flg, res, plan)
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
