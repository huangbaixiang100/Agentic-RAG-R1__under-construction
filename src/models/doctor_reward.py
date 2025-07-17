from typing import Any, Dict, List, Tuple
import random
import numpy as np
import torch
from openai import OpenAI
from src.data.doctor_patient_prompts import *
from src.utils.utils import call_gpt
import re


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

#旧的正则表达式
# def match_choice(text,options_dict):
#     option = ["A", "B", "C", "D", "E", "F", "G"]
#     res = re.search(r"(answer: |答案|正确选项|正确结论|正确判断|正确 答案)(?:是|：|为|应该是|应该为)\s*(.*)", text, re.S) #(.*?)(。|\.|$)
#     if res:
#         return "".join([x for x in res.group(2) if x in option])
#     else:
#         tmp=[]
#         for op_letter, op_text in options_dict.items():
#             if not op_text:
#                 continue
#             if op_text in text:
#                 #print(f"Found {op_letter}:{op_text} in response line: {text}")
#                 tmp.append(op_letter)
#         return "".join(tmp)
#     return "".join([i for i in text if i in option])


#更新之后的正则表达式
def match_choice(text,options_dict):
    option = ["A", "B", "C", "D", "E", "F", "G"]
    #res = re.search(r"(answer: |答案|正确选项)(?:是|：|为|应该是|应该为)(.*?)(。|\.|$)", text, re.S)
    res = re.search(r"(answer: |答案|正确选项|正确结论|正确判断|正确 答案)(?:是|：|为|应该是|应该为)\s*(.*)", text, re.S) #(.*?)(。|\.|$)
    pattern = r"(?:正确答案|answer|正确选项|正确结论|正确判断|正确 答案)[：:是为应该是应该为\s]*[【]?\s*([A-Ga-g]{1,7})\s*[】]?"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        # 多个匹配只取第一个；去重排序标准化
        answer = matches[0].upper()
        answer = "".join(sorted(set(answer)))
        if res:
            res_answer="".join([x for x in res.group(2) if x in option])
            #if res_answer!= answer:
                #print(text)
                #print(answer,res_answer)
                #print('*'*30)
        return answer
    else:
        tmp=[]
        for op_letter, op_text in options_dict.items():
            # 添加空值检查，防止 None 类型错误
            if op_text is not None and isinstance(op_text, str) and op_text in text:
                #print(f"Found {op_letter}:{op_text}")
                tmp.append(op_letter)
        return "".join(tmp)


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

# def correctness_reward(completions: List[List[Dict[str, Any]]], options: List[Dict[str, str]], answers: List[str]) -> List[float]:
#     """
#     改进的正确性奖励函数，提供渐进式奖励而非二元奖励。
    
#     奖励策略：
#     - 完全正确: 2.0分
#     - 部分匹配或接近正确: 1.0分  
#     - 有答案格式但错误: 0.5分
#     - 无有效答案: 0.0分
#     """
#     rewards = []

#     for i, completion_group in enumerate(completions):
#         content = completion_group[0]["content"]
        
#         # 提取最后一个assistant响应段
#         last_response = content.split("<|im_start|>assistant")[-1].strip()
        
#         model_answer = match_choice(last_response, options[i])
#         correct_answer = answers[i].strip()
        
#         # 渐进式奖励设计
#         if model_answer == correct_answer:
#             reward = 2.0  # 完全正确
#         elif model_answer and len(model_answer) == 1 and model_answer in ["A", "B", "C", "D", "E", "F", "G"]:
#             # 有有效的选项格式，但答案错误
#             reward = 0.5
#         elif any(ans_pattern in last_response.lower() for ans_pattern in ["answer:", "答案", "选择", "选项"]):
#             # 包含答案相关关键词，但没有明确选项
#             reward = 0.3
#         else:
#             reward = 0.0
            
#         print(f"模型答案: {model_answer}, 正确答案: {correct_answer}, 奖励: {reward}")
#         rewards.append(reward)

#     return rewards






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

        print(f"🔍 Format Reward调试 - 原始content: {content[:100]}...")

        # 首先尝试用parse_dialog解析（标准对话格式）
        dialog = parse_dialog(content)

        if len(dialog) > 0:
            print(f"📋 解析出{len(dialog)}轮对话")
            # 标准对话格式处理
            total_score = 0.0
            valid_count = 0

            for response in dialog:
                if response['role'] == 'user':
                    continue
                response_content = response['content']

                # Check how many times each keyword appears
                q_count = response_content.count("question:")
                a_count = response_content.count("answer:")
                
                print(f"  📝 Assistant回复: {response_content[:50]}...")
                print(f"  📊 question标记数: {q_count}, answer标记数: {a_count}")

                # Invalid if more than one or both present
                if q_count + a_count != 1:
                    score = 0.0
                    print(f"  ❌ 标记数量异常，分数=0")
                else:
                    if response_content.startswith("question:") or response_content.startswith("answer:"):
                        score = 1.0
                        print(f"  ✅ 以标记开头，分数=1.0")
                    else:
                        score = 0.5
                        print(f"  ⚠️  包含标记但不开头，分数=0.5")

                total_score += score
                valid_count += 1

            avg_score = total_score / valid_count if valid_count > 0 else 0.0
        else:
            # 如果parse_dialog解析失败，直接分析原始content
            print(f"⚠️ 对话解析失败，直接分析原始文本")
            
            # 直接分析整个content
            content_clean = content.strip()
            q_count = content_clean.count("question:")
            a_count = content_clean.count("answer:")
            
            print(f"📊 原始文本 - question标记数: {q_count}, answer标记数: {a_count}")
            
            # 检查是否符合格式要求
            if q_count + a_count != 1:
                avg_score = 0.0
                print(f"❌ 标记数量异常({q_count + a_count})，分数=0")
            else:
                if content_clean.startswith("question:") or content_clean.startswith("answer:"):
                    avg_score = 1.0
                    print(f"✅ 以标记开头，分数=1.0")
                elif "question:" in content_clean or "answer:" in content_clean:
                    avg_score = 0.5
                    print(f"⚠️ 包含标记但不开头，分数=0.5")
                else:
                    avg_score = 0.0
                    print(f"❌ 无有效标记，分数=0")

        print(f"🏆 最终Format Reward: {avg_score}")
        scores.append(avg_score)

    return scores

# ================ Shapley值计算相关函数 ================

@torch.no_grad()
def get_answer_logprob(model, tokenizer, full_prompt, target_answer, past_key_values=None):
    """
    计算目标答案在给定prompt下的对数概率
    """
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
    target_ids = tokenizer(target_answer, return_tensors="pt").input_ids.to(model.device)

    if past_key_values is not None:
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    else:
        outputs = model(input_ids=input_ids, use_cache=True)

    logits = outputs.logits[:, -target_ids.size(-1):, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_for_targets = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    avg_logprob = log_probs_for_targets.mean().item()

    return avg_logprob, outputs.past_key_values


def compute_equal_shapley_values(
    model, tokenizer,
    all_facts: List[str],
    atomic_question: str,
    target_answer: str,
    max_samples: int = 50,
    min_samples: int = 3,
    tol: float = 1e-2
) -> np.ndarray:
    """
    所有事实完全平等地计算Shapley值，使用蒙特卡洛方法
    
    Args:
        model: 医生模型
        tokenizer: 分词器
        all_facts: 所有事实列表（所有事实平等对待）
        atomic_question: 原子问题
        target_answer: 目标答案
        max_samples: 最大采样次数
        min_samples: 最小采样次数
        tol: 收敛容忍度
        
    Returns:
        所有事实的Shapley值数组
    """
    n = len(all_facts)
    if n == 0:
        return np.array([])
        
    print(f"🧮 使用蒙特卡洛方法计算{n}个事实的Shapley值（所有事实平等）...")
    print(f"📊 所有事实: {all_facts}")
    print(f"🎯 原子问题: {atomic_question}")
    
    shapley_scores = np.zeros(n)
    shapley_scores_prev = np.zeros(n)

    def build_medical_prompt(selected_facts):
        """构建医疗问诊prompt - 不使用基础事实，纯粹基于选中的事实"""
        info_text = '，'.join(selected_facts) + '。' if selected_facts else ''
        
        prompt = f"""你是一名专业的医生，具备丰富的医疗知识。请根据以下患者信息回答问题：

患者信息：{info_text}

问题：{atomic_question}
答案："""
        return prompt

    # 计算空集分数（没有任何事实）
    empty_prompt = build_medical_prompt([])
    v_empty, _ = get_answer_logprob(model, tokenizer, empty_prompt, target_answer)
    
    # 计算完整分数（所有事实）
    full_prompt = build_medical_prompt(all_facts)
    full_score, _ = get_answer_logprob(model, tokenizer, full_prompt, target_answer)

    print(f"📊 空集分数（无事实）: {v_empty:.4f}")
    print(f"📊 完整分数（所有事实）: {full_score:.4f}")

    # 蒙特卡洛采样计算Shapley值
    for t in range(1, max_samples + 1):
        # 随机排列所有事实（完全平等）
        perm = list(range(n))
        random.shuffle(perm)
        
        shapley_scores_prev = shapley_scores.copy()
        
        # 从空集开始逐步添加事实
        v_prev = v_empty
        active_facts = []

        for j, idx in enumerate(perm):
            # 添加当前事实
            active_facts.append(all_facts[idx])
            cur_prompt = build_medical_prompt(active_facts)
            v_j, _ = get_answer_logprob(model, tokenizer, cur_prompt, target_answer)

            # 计算边际贡献（完全基于当前联盟vs前一个联盟）
            marginal_contrib = v_j - v_prev
            phi_old = shapley_scores[idx]
            shapley_scores[idx] = (t - 1)/t * phi_old + (1/t) * marginal_contrib
            v_prev = v_j

            # 早停条件
            if abs(v_j - full_score) < 1e-2:
                break
        
        # 检查收敛
        if t >= min_samples:
            shapley_diffs = np.abs(shapley_scores - shapley_scores_prev)
            avg_diff = np.mean(shapley_diffs)
            if avg_diff < tol:
                print(f"✅ Shapley值在第{t}次迭代后收敛")
                break
    
    print(f"📈 最终Shapley值（平等计算）: {shapley_scores}")
    return shapley_scores


def normalize_shapley_weights(shapley_scores: np.ndarray, method: str = "softmax", temperature: float = 2.0) -> np.ndarray:
    """
    对Shapley值进行归一化得到权重，支持多种归一化方法
    
    流程第二步：
    "然后归一化得到每个未知信息的权重"
    
    Args:
        shapley_scores: 原始Shapley值
        method: 归一化方法，可选：
                - "z_score": Z-score归一化，基于标准化后的绝对值进行softmax
                - "softmax": Softmax归一化，基于原始值进行softmax
        temperature: 温度参数（仅在softmax方法中使用），控制分布的尖锐程度
                    - temperature < 1: 使分布更尖锐，突出最重要的信息
                    - temperature > 1: 使分布更平滑，权重更均匀
                    - temperature = 1: 标准softmax
        
    Returns:
        归一化后的权重
    """
    if len(shapley_scores) == 0:
        return np.array([])
    
    # 如果只有一个事实，权重为1
    if len(shapley_scores) == 1:
        weights = np.array([1.0])
        print(f"📊 {method}归一化 (单个事实): {weights}")
        return weights
    
    # 处理所有Shapley值相同的情况
    if np.std(shapley_scores) < 1e-8:
        # 如果所有Shapley值相同，使用均匀权重
        weights = np.ones(len(shapley_scores)) / len(shapley_scores)
        print(f"📊 {method}归一化 (均匀分布): {weights}")
        return weights
    
    try:
        if method == "z_score":
            # Z-score归一化方法（默认）
            print("📊 使用Z-score归一化方法...")
            
            # 步骤1: 计算Z-score标准化
            mean_score = np.mean(shapley_scores)
            std_score = np.std(shapley_scores)
            z_scores = (shapley_scores - mean_score) / std_score
            
            print(f"  原始Shapley值: {shapley_scores}")
            print(f"  均值: {mean_score:.4f}, 标准差: {std_score:.4f}")
            print(f"  Z-score: {z_scores}")
            
            # 步骤2: 取绝对值（重要性不分正负）
            abs_z_scores = np.abs(z_scores)
            print(f"  |Z-score|: {abs_z_scores}")
            
            # 步骤3: 对绝对值进行softmax归一化
            # 为了数值稳定性，先减去最大值
            shifted_abs_z = abs_z_scores - np.max(abs_z_scores)
            exp_scores = np.exp(shifted_abs_z)
            weights = exp_scores / np.sum(exp_scores)
            
            # 确保权重和为1
            weights = weights / np.sum(weights)
            
            print(f"📊 Z-score归一化结果: {weights}")
            
        elif method == "softmax":
            # Softmax归一化方法
            print(f"📊 使用Softmax归一化方法 (temperature={temperature:.2f})...")
            
            # 为了数值稳定性，先减去最大值
            shifted_scores = shapley_scores - np.max(shapley_scores)
            
            # 应用温度参数并计算指数
            exp_scores = np.exp(shifted_scores / temperature)
            
            # Softmax归一化
            weights = exp_scores / np.sum(exp_scores)
            
            # 确保权重和为1
            weights = weights / np.sum(weights)
            
            print(f"📊 Softmax归一化结果: {weights}")
            
        else:
            raise ValueError(f"不支持的归一化方法: {method}。支持的方法: 'z_score', 'softmax'")
        
        # 验证权重的有效性
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            print(f"⚠️ {method}计算出现数值问题，回退到均匀权重")
            weights = np.ones(len(shapley_scores)) / len(shapley_scores)
        
        return weightscc
        
    except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
        print(f"⚠️ {method}计算出错: {e}，回退到均匀权重")
        weights = np.ones(len(shapley_scores)) / len(shapley_scores)
        return weights


def evaluate_weighted_fact_acquisition(
    model, tokenizer,
    all_facts: List[str],
    known_facts: List[str], 
    formatted_dialog: str,
    shapley_weights: np.ndarray
) -> float:
    """
    流程：先平等计算Shapley值，再在奖励阶段使用前50%已知事实和Shapley加权
    
    Args:
        model: 医生模型
        tokenizer: 分词器  
        all_facts: 所有事实列表
        known_facts: 前50%已知事实（仅在奖励阶段用于生成医生理解）
        formatted_dialog: 格式化的对话内容
        shapley_weights: 所有事实的Shapley权重（平等计算得出）
        
    Returns:
        基于Shapley权重的加权奖励分数
    """
    print(f"🔍 开始加权事实奖励计算...")
    print(f"📋 总事实数: {len(all_facts)}, 已知事实数: {len(known_facts)}")
    
    # 使用前50%事实作为医生已知信息，生成医生理解
    understanding_prompt = doctor_understanding_prompt.format(
        patient_information='，'.join(known_facts) + '。' if known_facts else '',
        dialogue=formatted_dialog
    )
    understanding_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + understanding_prompt + "\n<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tokenizer(understanding_prompt, return_tensors="pt").to(model.device)
    
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )
    
    context = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 统计每个事实是否出现，然后使用Shapley权重加权
    print("📝 检查各事实的出现情况并进行Shapley加权:")
    weighted_score = 0.0
    
    for i, fact in enumerate(all_facts):
        try:
            # 使用fact checker检查事实是否出现
            fact_checker_client = OpenAI(api_key="8cefb70606f3472d8731bd65661ce409",
                                        base_url="http://8289.model.mingxingtech.com:10032/v1")
            fact_checker_model = 'qwen2.5:72b'
            
            prompt = check_fact_prompt.format(context=context, fact=fact)
            fact_check_messages = [{"role": "user", "content": prompt}]
            ans = call_gpt(fact_checker_client, fact_checker_model, fact_check_messages)
            
            fact_appeared = 1.0 if "True" in ans else 0.0
            
        except Exception as e:
            print(f"⚠️ 事实检查API失败: {e}, 使用字符串匹配")
            fact_lower = fact.lower()
            context_lower = context.lower()
            
            if (fact_lower in context_lower or 
                any(key_word in context_lower for key_word in fact_lower.split() if len(key_word) > 2)):
                fact_appeared = 1.0
            else:
                fact_appeared = 0.0
        
        # 使用Shapley权重进行加权
        weighted_contribution = shapley_weights[i] * fact_appeared
        weighted_score += weighted_contribution
        
        print(f"  📋 事实 {i+1}: {fact[:30]}...")
        print(f"    出现: {'✅' if fact_appeared > 0 else '❌'} ({fact_appeared})")
        print(f"    Shapley权重: {shapley_weights[i]:.3f}")
        print(f"    加权贡献: {weighted_contribution:.3f}")
    
    final_score = weighted_score * 3  # 保持与原始分数相同的scale
    
    print(f"🎯 总加权分数: {weighted_score:.3f}")
    print(f"🏆 最终分数: {final_score:.3f}")
    print("=" * 60)
    
    return final_score


def compute_shapley_weighted_fact_score(
    model, tokenizer, fact_list, formatted_dialog, 
    atomic_question, target_answer, use_shapley=True
):
    """
    改为旧版本的整体评分方式：对所有事实进行Shapley计算和整体评分
    
    完整流程：
    1. 将前50%事实作为"医生已知事实"
    2. 对所有事实进行Shapley值计算和加权
    3. 对所有事实进行整体评分
    """
    if not use_shapley:
        # 传统模式：完全按旧版本方式
        understanding_prompt = doctor_understanding_prompt.format(
            patient_information='，'.join(fact_list[:max(1, len(fact_list) // 2)]) + '。',
            dialogue=formatted_dialog
        )
        understanding_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + understanding_prompt + "\n<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = tokenizer(understanding_prompt, return_tensors="pt").to(model.device)
        
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
        
        context = tokenizer.decode(output[0], skip_special_tokens=True)
        score = get_fact_score(fact_list, context)
        return score * 3
    
    else:
        # Shapley整体评分模式：改为旧版本方式，全部都计算
        split_point = max(1, len(fact_list) // 2)
        known_facts = fact_list[:split_point]  # 前50%作为医生已知事实
        
        print(f"🔍 Shapley整体评分模式（旧版本方式）")
        print(f"📊 总事实数: {len(fact_list)}, 前50%事实数: {len(known_facts)}")
        
        try:
            # 步骤1: 对所有事实平等地计算Shapley值（蒙特卡洛方法）
            print("🚀 开始执行Shapley流程...")
            print("📊 步骤1: 所有事实平等计算Shapley值")
            shapley_scores = compute_equal_shapley_values(
                model, tokenizer,
                fact_list,  # 所有事实平等对待，不区分已知/未知
                atomic_question, target_answer,
                max_samples=50,
                min_samples=3
            )
            
            # 步骤2: 归一化得到权重
            print("⚖️ 步骤2: 归一化得到权重")
            shapley_weights = normalize_shapley_weights(shapley_scores, method="softmax", temperature=2.0)
            
            # 步骤3: 多轮对话已在外部完成(formatted_dialog)
            print("💬 步骤3: 多轮对话已完成")
            
            # 步骤4&5: 统计事实出现情况并基于Shapley权重加权奖励
            print("🎯 步骤4&5: 统计事实出现并计算加权奖励")
            final_score = evaluate_weighted_fact_acquisition(
                model, tokenizer,
                fact_list, known_facts,  # 所有事实 + 前50%已知事实
                formatted_dialog, shapley_weights
            )
            
            return final_score
            
        except Exception as e:
            print(f"❌ 平等Shapley值计算出错: {e}, 回退到传统模式")
            # 出错时回退到传统模式
            return compute_shapley_weighted_fact_score(
                model, tokenizer, fact_list, formatted_dialog, 
                atomic_question, target_answer, use_shapley=False
            )

# ================ 修改后的奖励函数 ================

def fact_score_reward(model, tokenizer, facts: List[List[str]], 
                     completions: List[List[Dict[str, Any]]],
                     use_shapley: bool = False,
                     atomic_questions: List[str] = None,
                     target_answers: List[str] = None) -> List[float]:
    """
    计算基于事实的奖励分数，支持传统模式和Shapley值加权模式
    
    Args:
        model: 策略模型
        tokenizer: 分词器  
        facts: 事实列表
        completions: 模型完成的对话
        use_shapley: 是否使用Shapley值加权
        atomic_questions: 原子问题列表（Shapley模式需要）
        target_answers: 目标答案列表（Shapley模式需要）
    
    Returns:
        事实分数奖励列表
    """
    all_rewards = []

    for i in range(len(completions)):
        fact_list = facts[i]
        completion = completions[i][0]["content"]
        dialog = parse_dialog(completion)
        formatted_dialog = format_dialog(dialog)

        if use_shapley and atomic_questions and target_answers:
            # 使用Shapley值加权模式
            atomic_question = atomic_questions[i]
            target_answer = target_answers[i]
            
            score = compute_shapley_weighted_fact_score(
                model, tokenizer, fact_list, formatted_dialog,
                atomic_question, target_answer, use_shapley=True
            )
        else:
            # 传统模式
            score = compute_shapley_weighted_fact_score(
                model, tokenizer, fact_list, formatted_dialog,
                "", "", use_shapley=False
            )
        
        all_rewards.append(score)

    return all_rewards




def overall_reward(model, tokenizer, facts: List[str], 
                  completions: List[List[Dict[str, Any]]],
                  options: List[Dict[str, str]], answers: List[str],
                  use_shapley: bool = False,
                  atomic_questions: List[str] = None) -> Dict[str, List[float]]:
    """
    结合正确性、格式和事实分数奖励的综合评分
    
    Args:
        model: 策略模型
        tokenizer: 分词器
        facts: 事实列表
        completions: 模型完成的对话
        options: 选项字典列表
        answers: 答案列表  
        use_shapley: 是否使用Shapley值加权fact score
        atomic_questions: 原子问题列表
    Returns:
        包含各种分数的字典
    """
    # 参数验证
    n = len(facts)
    if not (n == len(completions) == len(answers)):
        raise ValueError("facts, completions, and answers must have the same length.")

    correctness_scores = correctness_reward(completions, options, answers)
    format_scores = format_reward(completions)
    
    # 构建目标答案（用于Shapley值计算）
    target_answers = None
    if use_shapley and atomic_questions:
        target_answers = []
        for i, answer_keys in enumerate(answers):
            target_answer = '，'.join([options[i][c] for c in answer_keys]).strip('，')
            target_answers.append(target_answer)
    
    fact_scores = fact_score_reward(
        model, tokenizer, facts, completions,
        use_shapley=use_shapley,
        atomic_questions=atomic_questions,
        target_answers=target_answers
    )

    total_scores: List[float] = [c + f + r for c, f, r in 
                               zip(correctness_scores, format_scores, fact_scores)]

    return {
        "total_scores": total_scores,
        "correctness_scores": correctness_scores,
        "format_scores": format_scores,
        "fact_scores": fact_scores,
    }


# ================ Token级奖励分配系统 ================

def extract_question_boundaries(completion_text: str, tokenizer) -> List[
    Tuple[int, int]
]:
    """
    识别对话中每个问题的token边界
    
    Args:
        completion_text: 完整的对话文本
        tokenizer: 分词器
        
    Returns:
        List[Tuple[int, int]]: 每个问题的(start_token_idx, end_token_idx)
    """
    # 解析对话
    dialog = parse_dialog(completion_text)
    
    boundaries = []
    current_pos = 0
    
    for turn in dialog:
        if turn['role'] == 'assistant':  # 医生的问题
            turn_text = turn['content']
            
            # 分词并找到在完整文本中的位置
            turn_tokens = tokenizer.encode(
                turn_text, add_special_tokens=False
            )
            turn_length = len(turn_tokens)
            
            # 在完整文本中找到对应位置
            full_tokens = tokenizer.encode(
                completion_text, add_special_tokens=False
            )
            
            # 寻找匹配的token序列
            for i in range(current_pos, len(full_tokens) - turn_length + 1):
                if full_tokens[i:i + turn_length] == turn_tokens:
                    boundaries.append((i, i + turn_length))
                    current_pos = i + turn_length
                    break
    
    return boundaries


def compute_question_shapley_gains(
    model, tokenizer, fact_list: List[str], dialog: List[Dict[str, str]], 
    atomic_question: str, target_answer: str, shapley_weights: np.ndarray
) -> List[float]:
    """
    计算每个问题带来的Shapley信息增益
    
    Args:
        model: 医生模型
        tokenizer: 分词器
        fact_list: 完整事实列表
        dialog: 解析后的对话
        atomic_question: 原子问题
        target_answer: 目标答案
        shapley_weights: 预计算的Shapley权重
        
    Returns:
        List[float]: 每个问题的Shapley增益
    """
    question_gains = []
    
    # 创建事实检查器
    fact_checker_client = OpenAI(api_key="8cefb70606f3472d8731bd65661ce409",
                                base_url="http://8289.model.mingxingtech.com:10032/v1")
    fact_checker_model = 'qwen2.5:72b'
    
    for turn in dialog:
        if turn['role'] == 'assistant':  # 医生的问题
            question_gain = 0.0
            turn_content = turn['content']
            
            # 检查这个问题获取了哪些事实
            for i, fact in enumerate(fact_list):
                try:
                    # 检查事实是否在该轮对话中被获取
                    prompt = check_fact_prompt.format(context=turn_content, fact=fact)
                    fact_check_messages = [{"role": "user", "content": prompt}]
                    ans = call_gpt(fact_checker_client, fact_checker_model, fact_check_messages)
                    
                    if "True" in ans:
                        # 如果获取了该事实，累加其Shapley值
                        question_gain += shapley_weights[i]
                        
                except Exception as e:
                    print(f"⚠️ 检查事实时出错: {e}")
                    # 回退到字符串匹配
                    if fact.lower() in turn_content.lower():
                        question_gain += shapley_weights[i]
            
            question_gains.append(question_gain)
    
    return question_gains


def compute_token_level_rewards(
    model, tokenizer, facts: List[str], 
    completions: List[List[Dict[str, Any]]],
    options: List[Dict[str, str]], answers: List[str],
    use_shapley: bool = False,
    atomic_questions: List[str] = None,
    alpha: float = 1.0,  # 过程奖励权重
    beta: float = 1.0,   # 结果奖励权重
    gamma: float = 3.0,  # 最终答案奖励权重
    max_completion_length: int = None,  # 新增：最大completion长度限制
    **kwargs
) -> Dict[str, Any]:
    """
    计算token级奖励分配 - 真正的问题级精细分配
    
    实现逻辑：
    1. 识别每个"question:"句子的token边界
    2. 为每个问题计算其Shapley信息增益
    3. 每个问题的token获得: α × 该问题Shapley增益 + β × 该问题Shapley增益 × answer_correct
    4. 最终答案token获得: γ × answer_correct
    5. 其他token获得: 0
    
    Args:
        max_completion_length: 最大completion长度，与completion_mask保持一致
    
    Returns:
        Dict包含:
        - token_rewards: List[List[float]] - 每个样本的token级奖励
        - total_scores: List[float] - 兼容性的总分
        - question_shapley_gains: List[List[float]] - 每个问题的Shapley增益
    """
    print(" ===== 真正的Token级问题奖励分配 =====")
    print(f" 输入参数: max_completion_length={max_completion_length}")
    print(f" completions数量: {len(completions)}")
    
    try:
        # 先计算基础的奖励分数（用于兼容性）
        base_rewards = overall_reward(
            model, tokenizer, facts, completions, options, answers,
            use_shapley=use_shapley, atomic_questions=atomic_questions
        )
        
        token_rewards_list = []
        question_gains_list = []
        
        for i, completion_list in enumerate(completions):
            print(f" 处理completion组 {i}: 包含{len(completion_list)}个样本")
            for j, completion_dict in enumerate(completion_list):
                # 获取完整的对话文本
                completion_text = completion_dict.get('content', '')
                print(f"📄 Sample {i}-{j}: 原始文本长度={len(completion_text)}字符")
                
                # 检查最终答案是否正确
                base_idx = i * len(completion_list) + j
                answer_correct = 0.0
                if base_idx < len(base_rewards.get('correctness_scores', [])):
                    answer_correct = 1.0 if base_rewards['correctness_scores'][base_idx] > 0 else 0.0
                
                print(f" Sample {i}-{j}: answer_correct={answer_correct}")
                
                # 将文本tokenize并应用截断
                tokens = tokenizer.encode(completion_text, add_special_tokens=False)
                original_token_length = len(tokens)
                
                if max_completion_length is not None and len(tokens) > max_completion_length:
                    tokens = tokens[:max_completion_length]
                    completion_text = tokenizer.decode(tokens, skip_special_tokens=False)
                    print(f" Sample {i}-{j}: Token序列从{original_token_length}截断到{len(tokens)}")
                else:
                    print(f" Sample {i}-{j}: 无需截断，保持长度={len(tokens)}")
                
                # 初始化所有token奖励为0
                token_rewards = [0.0] * len(tokens)
                
                #  关键：识别question和answer的token边界
                question_boundaries, answer_boundaries = extract_question_answer_boundaries(completion_text, tokenizer)
                
                print(f"🎭 Sample {i}-{j}: 识别到{len(question_boundaries)}个问题, {len(answer_boundaries)}个答案")
                
                # 如果使用Shapley，计算每个问题的信息增益
                question_gains = []
                if use_shapley and atomic_questions and i < len(facts):
                    fact_list = facts[i]
                    atomic_question = atomic_questions[base_idx] if base_idx < len(atomic_questions) else "默认问题"
                    
                    print(f" Sample {i}-{j}: 开始计算每个问题的Shapley增益...")
                    
                    # 解析对话
                    dialog = parse_dialog(completion_text)
                    
                    try:
                        # 计算整体Shapley权重
                        shapley_scores = compute_equal_shapley_values(
                            model, tokenizer, fact_list, atomic_question, 
                            answers[i] if i < len(answers) else "默认答案",
                            max_samples=15, min_samples=3
                        )
                        shapley_weights = normalize_shapley_weights(shapley_scores, method="softmax", temperature=2.0)
                        
                        # 计算每个问题的Shapley增益
                        question_gains = compute_question_shapley_gains(
                            model, tokenizer, fact_list, dialog, atomic_question,
                            answers[i] if i < len(answers) else "默认答案", shapley_weights
                        )
                        
                        print(f" Sample {i}-{j}: 问题Shapley增益: {question_gains}")
                        
                    except Exception as e:
                        print(f" Sample {i}-{j}: Shapley计算失败: {e}, 使用默认增益")
                        question_gains = [1.0] * len(question_boundaries)
                else:
                    # 不使用Shapley时，所有问题增益相等
                    question_gains = [1.0] * len(question_boundaries)
                
                question_gains_list.append(question_gains)
                
                #  为每个问题的token分配奖励
                for q_idx, (start_idx, end_idx) in enumerate(question_boundaries):
                    if q_idx < len(question_gains):
                        shapley_gain = question_gains[q_idx]
                        
                        # 问题获得的奖励：过程奖励 + 结果奖励（如果答案正确）
                        question_reward = alpha * shapley_gain + beta * shapley_gain * answer_correct
                        
                        print(f"💰 Sample {i}-{j} Question {q_idx}: Shapley增益={shapley_gain:.3f}, 奖励={question_reward:.3f}")
                        
                        # 将奖励分配给该问题的所有token
                        for token_idx in range(start_idx, min(end_idx, len(token_rewards))):
                            token_rewards[token_idx] = question_reward
                
                #  为最终答案的token分配奖励
                for start_idx, end_idx in answer_boundaries:
                    answer_reward = gamma * answer_correct
                    print(f"🏆 Sample {i}-{j}: 答案部分获得奖励={answer_reward:.3f}")
                    
                    # 将奖励分配给答案的所有token
                    for token_idx in range(start_idx, min(end_idx, len(token_rewards))):
                        token_rewards[token_idx] = answer_reward
                
                # 统计奖励分配情况
                nonzero_rewards = sum(1 for r in token_rewards if r != 0)
                total_reward = sum(token_rewards)
                print(f"✅ Sample {i}-{j}: 非零奖励token数={nonzero_rewards}/{len(token_rewards)}, 总奖励={total_reward:.3f}")
                
                token_rewards_list.append(token_rewards)
        
        print(f" 总共生成了{len(token_rewards_list)}个token_rewards")
        
        # 返回包含token级奖励的结果
        result = {
            'token_rewards': token_rewards_list,
            'total_scores': base_rewards.get('total_scores', []),
            'fact_scores': base_rewards.get('fact_scores', []),
            'correctness_scores': base_rewards.get('correctness_scores', []),
            'format_scores': base_rewards.get('format_scores', []), 
            'question_shapley_gains': question_gains_list  # 新增：每个问题的Shapley增益
        }
        
        print(f" 返回结果包含字段: {list(result.keys())}")
        print(f" Format scores: {result['format_scores']}")
        
        if 'shapley_scores' in base_rewards:
            result['shapley_scores'] = base_rewards['shapley_scores']
            
        return result
        
    except Exception as e:
        print(f"❌ Token级奖励计算失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回空的token奖励
        total_samples = sum(len(comp_list) for comp_list in completions)
        return {
            'token_rewards': [[] for _ in range(total_samples)],
            'total_scores': [0.0] * total_samples,
            'fact_scores': [0.0] * total_samples,
            'correctness_scores': [0.0] * total_samples,
            'format_scores': [0.0] * total_samples,  
            'question_shapley_gains': [[] for _ in range(total_samples)]
        }


def extract_question_answer_boundaries(completion_text: str, tokenizer) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    识别对话中每个"question:"和"answer:"的token边界
    
    Args:
        completion_text: 完整的对话文本  
        tokenizer: 分词器
        
    Returns:
        Tuple[question_boundaries, answer_boundaries]: 
        - question_boundaries: List[Tuple[int, int]] - 每个问题的(start_token_idx, end_token_idx)
        - answer_boundaries: List[Tuple[int, int]] - 每个答案的(start_token_idx, end_token_idx)
    """
    print(f" 开始识别question和answer边界...")
    
    # 将完整文本tokenize
    full_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
    full_text = tokenizer.decode(full_tokens, skip_special_tokens=False)
    
    question_boundaries = []
    answer_boundaries = []
    
    # 使用正则表达式找到所有question:和answer:的位置
    question_pattern = r'question\s*:'
    answer_pattern = r'answer\s*:'
    
    question_matches = list(re.finditer(question_pattern, full_text, re.IGNORECASE))
    answer_matches = list(re.finditer(answer_pattern, full_text, re.IGNORECASE))
    
    print(f" 找到{len(question_matches)}个question标记, {len(answer_matches)}个answer标记")
    
    # 为每个question找到其token边界
    for i, match in enumerate(question_matches):
        start_char = match.start()
        
        # 找到下一个question或answer的开始位置作为结束点
        end_char = len(full_text)
        for next_match in question_matches[i+1:] + answer_matches:
            if next_match.start() > start_char:
                end_char = next_match.start()
                break
        
        # 将字符位置转换为token位置
        prefix_text = full_text[:start_char]
        question_text = full_text[start_char:end_char]
        
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
        question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
        
        start_token = len(prefix_tokens)
        end_token = start_token + len(question_tokens)
        
        question_boundaries.append((start_token, end_token))
        print(f" Question {i}: 字符[{start_char}:{end_char}] -> token[{start_token}:{end_token}]")
    
    # 为每个answer找到其token边界
    for i, match in enumerate(answer_matches):
        start_char = match.start()
        
        # 找到下一个question或answer的开始位置作为结束点
        end_char = len(full_text)
        for next_match in question_matches + answer_matches[i+1:]:
            if next_match.start() > start_char:
                end_char = next_match.start()
                break
        
        # 将字符位置转换为token位置
        prefix_text = full_text[:start_char]
        answer_text = full_text[start_char:end_char]
        
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        
        start_token = len(prefix_tokens)
        end_token = start_token + len(answer_tokens)
        
        answer_boundaries.append((start_token, end_token))
        print(f" Answer {i}: 字符[{start_char}:{end_char}] -> token[{start_token}:{end_token}]")
    
    return question_boundaries, answer_boundaries


def overall_reward_with_token_allocation(
    model, tokenizer, facts: List[str], 
    completions: List[List[Dict[str, Any]]],
    options: List[Dict[str, str]], answers: List[str],
    use_shapley: bool = False,
    atomic_questions: List[str] = None,
    use_token_level: bool = False,  # 参数：是否启用token级奖励
    max_completion_length: int = None,  # 新增：截断长度参数
    **kwargs
) -> Dict[str, List[float]]:
    """
    增强版的overall_reward，支持token级奖励分配
    
    Args:
        use_token_level: 是否使用token级奖励分配
        其他参数同original overall_reward
        
    Returns:
        如果use_token_level=True，返回包含token_rewards的详细结果
        否则返回标准格式以保持兼容性
    """
    if use_token_level:
        # 使用新的token级奖励计算
        return compute_token_level_rewards(
            model, tokenizer, facts, completions, options, answers,
            use_shapley=use_shapley, atomic_questions=atomic_questions, 
            max_completion_length=max_completion_length, **kwargs
        )
    else:
        # 回退到原始的overall_reward，不使用token级奖励
        return overall_reward(
            model, tokenizer, facts, completions, options, answers,
            use_shapley=use_shapley, atomic_questions=atomic_questions
        )


