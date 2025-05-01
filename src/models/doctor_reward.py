from typing import Any, Dict, List
import os
import re
import json
from collections import defaultdict

from langchain_openai import ChatOpenAI
from tqdm import tqdm
from openai import OpenAI

from src.data.doctor_patient_prompts import *
from src.utils.utils import call_gpt


import re
from typing import List, Dict

def parse_dialog(completion: str) -> List[Dict[str, str]]:
    """
    Parses a model completion string into a list of dialog turns with roles and content,
    retaining the <|im_end|> marker at the end of each message.

    Each turn is represented as a dict:
    {
        "role": "user" or "assistant",
        "content": the actual content (including <|im_end|>)
    }

    Args:
        completion: A single completion string with <|im_start|>role and <|im_end|> markers.

    Returns:
        A list of dicts with keys "role" and "content", in order of appearance.
    """
    dialog = []
    pattern = re.compile(r"<\|im_start\|>(user|assistant)\s*(.*?)(<\|im_end\|>)", re.DOTALL)

    # Optional: handle initial assistant segment before first user block
    initial_parts = completion.split("<|im_start|>", 1)
    if initial_parts[0].strip():
        dialog.append({
            "role": "assistant",
            "content": initial_parts[0].strip()
        })

    for match in pattern.finditer(completion):
        role = match.group(1).strip()
        content = match.group(2).rstrip()
        content += match.group(3)  # append <|im_end|>
        dialog.append({"role": role, "content": content})

    return dialog

def format_dialog(dialog):
    result=""
    for message in dialog[1:]:
        if message['role']=='assistant':
            result+='doctor: '+message['content']+"\n"
        else:
            result+='patient: '+message['content']+'\n'
    return result


def get_fact_score(facts, context):
    fact_checker_client = OpenAI(api_key="8cefb70606f3472d8731bd65661ce409",
                            base_url="http://8289.model.mingxingtech.com:10032/v1")
    fact_checker_model = 'qwen2.5:72b'
    fact_num = len(facts)
    correct_facts = 0
    for fact in facts:
        prompt = check_fact_prompt.format(context=context, fact=fact)
        fact_check_messages = [{"role": "user", "content": prompt}]
        ans = call_gpt(fact_checker_client,fact_checker_model,fact_check_messages)
        if "True" in ans:
            correct_facts += 1
    fact_score = correct_facts / fact_num
    return fact_score


def match_choice(text,options_dict):
    option = ["A", "B", "C", "D", "E", "F", "G"]
    res = re.search(r"(answer: |答案|正确选项)(?:是|：|为|应该是|应该为)\s*(.*)", text, re.S) #(.*?)(。|\.|$)
    if res:
        return "".join([x for x in res.group(2) if x in option])
    else:
        tmp=[]
        for op_letter, op_text in options_dict.items():
            if not op_text:
                continue
            if op_text in text:
                #print(f"Found {op_letter}:{op_text} in response line: {text}")
                tmp.append(op_letter)
        return "".join(tmp)
    return "".join([i for i in text if i in option])



def correctness_reward(completions: List[List[Dict[str, Any]]],options:List[Dict[str,str]], answers: List[str]) -> List[float]:
    """
    Assigns a reward based on the correctness of the model's answers.

    For each prompt, compares the model's final answer to the expected answer
    using a match_choice to get the output option and compare to the true answer
    Returns 3.0 for correct model answer, 0.0 otherwise.

    Args:
        prompts: List of prompt strings to evaluate.
        completions: Nested list of completion dicts from the model; we use the first element's "content".
        answers: List of expected answer strings.

    Returns:
        A list of floats, one per prompt, where each value is either 3.0 (correct) or 0.0 (incorrect).
    """
    rewards = []

    for i,completion_group in enumerate(completions):
        content = completion_group[0]["content"]

        # 提取最后一个assistant响应段
        last_response = content.split("<|im_start|>assistant")[-1].strip()

        model_answer = match_choice(last_response,options[i])
        correct_answer = answers[i].strip()
        print(model_answer,correct_answer)

        reward = 3.0 if model_answer == correct_answer else 0.0
        rewards.append(reward)

    return rewards


def format_reward(completions: List[List[Dict[str, Any]]]) -> List[float]:
    """
    Computes a formatting reward based on the presence of specific tags
    in each model response.

    Tag scoring:
      - "question:" at the beginning: 1; present: 0.5
      - "answer:" at the beginning: 1; present: 0.5
    If a single response contains both or multiple of either, reward is 0.
    Final reward is the average over all responses.

    Args:
        completions: Nested list of completion dicts from the model;
                     we use the "content" field of the first dict in each sublist.

    Returns:
        A list of floats, one per completion, representing the format score.
    """
    scores = []

    for completion_group in completions:
        content = completion_group[0]["content"]

        dialog=parse_dialog(content)

        total_score = 0.0
        valid_count = 0

        for response in dialog:
            if response['role']=='user':
                continue
            response=response['content']

            # Check how many times each keyword appears
            q_count = response.count("question:")
            a_count = response.count("answer:")

            # Invalid if more than one or both present
            if q_count + a_count != 1:
                score = 0.0
            else:
                if response.startswith("question:") or response.startswith("answer:"):
                    score = 1.0
                else:
                    score = 0.5

            total_score += score
            valid_count += 1

        avg_score = total_score / valid_count if valid_count > 0 else 0.0
        scores.append(avg_score)

    return scores

def fact_score_reward(model,tokenizer,facts: List[List[str]],completions: List[List[Dict[str, Any]]]
) -> List[float]:
    """
    Computes a fact score reward based on the number of acquired facts in each model response.

    For each prompt and corresponding completion:
    - Use doctor_understanding_prompt to first prompt the model using partial facts (initial understanding).
    - Then use full prompt + completion to get updated understanding.
    - Use get_fact_score to compute fact match score from both stages.
    - Reward is: final_score - initial_score.

    Returns:
        A list of float lists; one per prompt, with fact score deltas per completion.
    """
    all_rewards = []

    for i in range(len(completions)):
        fact_list = facts[i]
        completion = completions[i][0]["content"]
        dialog=parse_dialog(completion)
        formatted_dialog=format_dialog(dialog)

        # Get initial fact score (before model response)
        understanding_prompt = doctor_understanding_prompt.format(
            patient_information='，'.join(fact_list[:max(1, len(fact_list) // 2)]) + '。',
            dialogue=formatted_dialog
        )
        understanding_prompt = [(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + understanding_prompt + "\n<|im_end|>\n<|im_start|>assistant\n"
        )]
        inputs = tokenizer(understanding_prompt, return_tensors="pt").to(model.device)
        # 生成文本
        output = model.generate(
            **inputs,
            max_new_tokens=128,  # 控制生成长度
            do_sample=True,  # 是否采样（True 为采样，False 为贪婪搜索）
            temperature=0.7,  # 控制生成多样性
            top_p=0.9,  # nucleus sampling
            eos_token_id=tokenizer.eos_token_id  # 可指定结束符
        )
        # 解码输出
        context = tokenizer.decode(output[0], skip_special_tokens=True)
        score = get_fact_score(fact_list, context)
        all_rewards.append(score*2)

    return all_rewards




def overall_reward(model,tokenizer,facts: List[str], completions: List[List[Dict[str, Any]]],
                   options:List[Dict[str, str]] ,answers: List[str]) -> Dict[
    str, List[float]]:
    """
    Combines correctness, format, and fact score rewards into a comprehensive score set.

    Args:
        prompts: List of prompt strings.
        completions: Nested list of completion dicts from the model.
        answers: List of expected answer strings.

    Returns:
        A dict with keys:
          - 'total_scores': Combined scores (correctness + format + RAG).
          - 'correctness_scores': Individual correctness rewards.
          - 'format_scores': Individual format rewards.
          - 'rag_scores': Individual RAG rewards.

    Raises:
        ValueError: If the lengths of inputs do not align.
    """
    # Validation
    n = len(facts)
    if not (n == len(completions) == len(answers)):
        raise ValueError("facts, completions, and answers must have the same length.")

    correctness_scores = correctness_reward(completions, options,answers)
    format_scores = format_reward(completions)
    fact_scores = fact_score_reward(model,tokenizer,facts, completions)

    total_scores: List[float] = [c + f + r for c, f, r in zip(correctness_scores, format_scores, fact_scores)]

    return {
        "total_scores": total_scores,
        "correctness_scores": correctness_scores,
        "format_scores": format_scores,
        "fact_scores": fact_scores,
    }