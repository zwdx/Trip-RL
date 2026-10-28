from rllm.rewards.reward_types import RewardOutput
import requests 
import os 
import time 


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY") 


SEARCH_JUDGE_PROMPT = """
Based on the given question, standard answer, and model-predicted answer, evaluate whether the model's response is correct. Your task is to classify the result as: [CORRECT] or [INCORRECT].

First, we'll list examples for each category, then you'll evaluate a new question's predicted answer.
Here are examples of [CORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most would say Malia and Sasha, but I'm not sure, I should verify
Model Prediction 4: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama.
```
These responses are all [CORRECT] because they:
    - Fully include the important information from the standard answer.
    - Don't contain any information that contradicts the standard answer.
    - Focus only on semantic content; language, capitalization, punctuation, grammar, and order aren't important.
    - Vague statements or guesses are acceptable as long as they include the standard answer and don't contain incorrect information or contradictions.

Here are examples of [INCORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha and Susan or Sasha Obama or Malia Obama, or Natasha Marian, or Einstein
Model Prediction 3: While I don't know their exact names, I can tell you Barack Obama has two children.
Model Prediction 4: You might be thinking of Betsy and Olivia. But you should verify the details with the latest references. Is that the correct answer?
Model Prediction 5: Barack Obama's children
```
These responses are all [INCORRECT] because they:
    - Contain factual statements that contradict the standard answer.
    - Are empty or merely repeat the question.
    - Enumerate multiple answers or repeat the answer.

Pay special attention to the following:
- The standard answer may contain responses to multiple aspects of the question, and within the same aspect, there might be different descriptions, all of which are correct and are given in the same bracket, connected by commas. For example, for the question "What is the name of ByteDance's AI model?", the standard answer is "[[Doubao, Skylark]]":
    - Predicted answers "Doubao", "Doubao, Skylark", "Skylark", etc. are all [CORRECT].
- For standard answers containing responses to different aspects, the model needs to provide answers to all aspects to be considered correct; otherwise, it's directly judged as [INCORRECT]. There is no [PARTIALLY CORRECT] output option. These answers will be given in different brackets. For example, for the question "Who are the members of TFBOYS?", the standard answer is "[[Wang Junkai][Wang Yuan][Yi Yangqianxi]]":
    - Predicted answers like "Wang Junkai, Wang Yuan, Yi Yangqianxi" that include all answers are [CORRECT].
    - Predicted answers like "Wang Junkai, Yi Yangqianxi" that don't include all answers are [INCORRECT].

Also note the following points:
- For questions with numerical standard answers, the predicted answer should match the standard answer. For example, for the question "What is the total length in meters of the Huangpu River Bridge on the Jinshan Railway?", the standard answer is "3518.17":
    - Predicted answers "3518", "3518.1", "3518.17" are all [CORRECT].
    - Predicted answers "3520" and "3600" are [INCORRECT].
- If the model prediction doesn't directly answer the question, attempts to circumvent or fails to directly provide the standard answer, it's considered an [INCORRECT] answer.
    - For example, for the question "Who is JJ Lin's wife?", with the standard answer "Ding Wenqi", model predictions like "JJ Lin's wife", "JJ Lin's wife should be excellent", "JJ Lin's wife might be a public figure" are all [INCORRECT].
- If the standard answer contains more information than the question asks for, the predicted answer only needs to include the information mentioned in the question.
    - For example, for the question "What is the main chemical component of magnesite?", with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT] answers.
- If information omitted in the predicted answer can be clearly inferred from the question, it's considered correct.
    - For example, for the question "The Nuragic ruins of Barumini were listed as a World Cultural Heritage by UNESCO in 1997, so where is this site located?", with the standard answer "Sardinia, Italy", the predicted answer "Sardinia" is considered [CORRECT].
- If it's clear that different translations of a name refer to the same person, it's considered correct.
    - For example, if the standard answer is "Robinson", answers like "Lubinson" or "Lubinsun" are both correct.
- You should focus more on the match between the standard answer and the model prediction, rather than whether the standard answer itself is correct.

Below is a new question example. Please reply with only [CORRECT] or [INCORRECT], without apologies or corrections to your own errors, just evaluate the answer.
```
Question: {question}
Standard Answer: {correct_answer}
Predicted Answer: {response}
```

Evaluate this new question's predicted answer as one of the following:
A. [CORRECT]
B. [INCORRECT]

Return only the option representing [CORRECT] or [INCORRECT], i.e., just return A or B, without adding any other text.
""".strip()


def extract_answer_from_response(response: str) -> str:
    if "<answer>" in response and "</answer>" in response: 
        lpos = response.find("<answer>")
        rpos = response.rfind("</answer>")
        return response[lpos+len("<answer>"):rpos].strip()
    
    return None


def call_judge_model(messages: list[dict], max_retries=20):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    base_url = "https://api.deepseek.com/chat/completions"
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": 16, 
        "temperature": 0.6,
        "top_p": 0.95,
    }
    
    while max_retries:
        try:
            response = requests.post(base_url, headers=headers, json=payload, timeout=180)
            content = response.json()["choices"][0]["message"]["content"] 

            if content:
                
                return content 
        except Exception as e:
            max_retries -= 1
            time.sleep(0.2)

            if max_retries == 0:
                print(f"Failed to call format model after {max_retries} attempts. Error: {e}")
                raise e


def my_search_reward_fn(task_info: dict, action: str) -> RewardOutput:
    if  "Format error occurs in the previous turn!" in action: 
        return RewardOutput(reward=0.0, is_correct=False, metadata={"error": "Format error occurs in the previous turn!"})

    question, gt_answer = task_info.get("question"), task_info.get("answer") 

    prediction = extract_answer_from_response(action) 
    if not prediction:
        return RewardOutput(reward=0.0, is_correct=False, metadata={"error": "No answer found in the response!"})

    messages = [
        {"role": "system", "content": "You are a helpful assistant that can judge the correctness of the answer."},
        {"role": "user", "content": SEARCH_JUDGE_PROMPT.format(question=question, correct_answer=gt_answer, response=prediction)}
    ] 

    try:
        response = call_judge_model(messages) 
        response = response.strip()[:2]
        if response in ["A", "a", "A.", "a."]:
            print(f"Judgement correct for prediction {prediction} and standard answer {gt_answer}")
            return RewardOutput(reward=1.0, is_correct=True, metadata={"judgetment": "The response is correct!"})
        else:
            return RewardOutput(reward=0.0, is_correct=False, metadata={"error": "The response is incorrect!"})
    except Exception as e:
        print(f"Failed to judge the response: {e}")
        return RewardOutput(reward=0.0, is_correct=False, metadata={"error": f"Failed to judge the response: {e}"})


if __name__ == "__main__":
    task = {"question": "浙江省会是哪里", "answer": "杭州"}
    action = "<answer>是杭州</answer>"
    reward = my_search_reward_fn(task, action)
    print(reward)
