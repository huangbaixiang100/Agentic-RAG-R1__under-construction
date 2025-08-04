import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json
import os
import pandas as pd

doctor_prompt_shapley = """You are a professional doctor with extensive medical knowledge. Based on the patient information provided, please answer the following question using your medical expertise. Give your answer directly.

Question: {question}
Patient Information: {information}
Answer: """

@torch.no_grad()
def get_answer_logprob(model, tokenizer, full_prompt, target_answer, past_key_values=None):
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
    target_ids = tokenizer(target_answer, return_tensors="pt").input_ids.to(model.device)

    if past_key_values is not None:
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    else:
        outputs = model(input_ids=input_ids, use_cache=True)

    logits = outputs.logits[:, -target_ids.size(-1):, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_for_targets = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    #total_logprob = log_probs_for_targets.sum().item()
    avg_logprob = log_probs_for_targets.mean().item()

    return avg_logprob, outputs.past_key_values

def normalize_shapley_values(shapley_scores):
    """
    归一化Shapley值，参考doctor_reward.py中的normalize_shapley_weights
    """
   
    
    return 1.0*shapley_scores

def compute_recall_at_k(shapley_scores, atom_infos, noise_infos, k):
    """
    计算Recall@k，使用改进的计算方法
    
    Args:
        shapley_scores: 每条信息的shapley值
        atom_infos: 所有信息列表
        noise_infos: 噪声信息列表
        k: 取前k个进行评估
    
    Returns:
        recall@k: 前k个中真实信息的比例
        detailed_info: 包含计算过程的详细信息字典
    """
    print(f"\n📊 计算Recall@{k}...")
    
    # 1. 基础统计
    total_infos = len(atom_infos)
    total_real_infos = total_infos - len(noise_infos)
    
    if total_real_infos <= 0:
        print(f"⚠️ 无真实信息! (总信息={total_infos}, 噪声信息={len(noise_infos)})")
        return 0.0, {
            'k': k,
            'total_infos': total_infos,
            'total_real_infos': total_real_infos,
            'real_info_count': 0,
            'denominator': 0,
            'recall': 0.0,
            'top_k_info': [],
            'shapley_stats': {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        }
    
    print(f"📈 基础统计:")
    print(f"  - 总信息数: {total_infos}")
    print(f"  - 真实信息数: {total_real_infos}")
    print(f"  - 噪声信息数: {len(noise_infos)}")
    
    # 2. 归一化Shapley值
    normalized_scores = normalize_shapley_values(shapley_scores)
    
    shapley_stats = {
        'mean': np.mean(shapley_scores),
        'std': np.std(shapley_scores),
        'min': np.min(shapley_scores),
        'max': np.max(shapley_scores)
    }
    
    print(f"📊 Shapley值统计:")
    print(f"  - 原始均值: {shapley_stats['mean']:.4f}")
    print(f"  - 标准差: {shapley_stats['std']:.4f}")
    print(f"  - 最小值: {shapley_stats['min']:.4f}")
    print(f"  - 最大值: {shapley_stats['max']:.4f}")
    
    # 3. 配对并排序
    info_scores = list(zip(atom_infos, normalized_scores))
    sorted_info_scores = sorted(info_scores, key=lambda x: x[1], reverse=True)
    
    # 4. 获取top-k信息
    effective_k = min(k, len(sorted_info_scores))
    top_k_infos = [info for info, score in sorted_info_scores[:effective_k]]
    
    print(f"🔝 Top-{effective_k}信息:")
    for idx, (info, score) in enumerate(sorted_info_scores[:effective_k]):
        is_real = info not in noise_infos
        print(f"  {idx+1}. {'✅' if is_real else '❌'} [{score:.4f}] {info[:50]}...")
    
    # 5. 统计真实信息
    real_info_count = sum(1 for info in top_k_infos if info not in noise_infos)
    denominator = min(effective_k, total_real_infos)
    
    if denominator <= 0:
        print(f"⚠️ 无效分母! (k={effective_k}, total_real_infos={total_real_infos})")
        return 0.0, {
            'k': k,
            'total_infos': total_infos,
            'total_real_infos': total_real_infos,
            'real_info_count': real_info_count,
            'denominator': denominator,
            'recall': 0.0,
            'top_k_info': top_k_infos,
            'shapley_stats': shapley_stats
        }
    
    # 6. 计算recall
    recall = real_info_count / denominator
    
    print(f"📈 Recall@{k}计算结果:")
    print(f"  - 有效k值: {effective_k}")
    print(f"  - Top-k中真实信息数: {real_info_count}")
    print(f"  - 分母(min(k,total_real)): {denominator}")
    print(f"  - Recall@{k}: {recall:.4f}")
    
    # 7. 返回结果和详细信息
    detailed_info = {
        'k': k,
        'total_infos': total_infos,
        'total_real_infos': total_real_infos,
        'real_info_count': real_info_count,
        'denominator': denominator,
        'recall': recall,
        'top_k_info': top_k_infos,
        'shapley_stats': shapley_stats
    }
    
    return recall, detailed_info

def estimate_atomic_shapley_with_tolerance(
    model, tokenizer,
    doctor_prompt, atomic_question, atomic_information,
    target_answer,
    max_samples=1,       # Maximum number of samples
    tol=1e-2,             # Tolerance value for convergence
    min_samples=3,        # Minimum number of samples
    early_stop_eps=1e-2   # Early stopping threshold for a single permutation
):
    """
    为所有信息计算Shapley值，不使用基础信息分割
    """
    all_facts = atomic_information
    n = len(all_facts)
    
    if n == 0:
        return np.array([]), [], all_facts

    print(f"🧮 使用蒙特卡洛方法计算{n}个信息的Shapley值（全部信息）...")
    print(f"📊 所有信息: {all_facts}")
    print(f"🎯 原子问题: {atomic_question}")

    shapley_scores = np.zeros(n)
    prev_shapley_scores = np.zeros(n)

    def build_prompt(selected_facts):
        """构建医疗问诊prompt"""
        if selected_facts:
            info_text = '\n'.join(selected_facts)
        else:
            info_text = ''
        
        return doctor_prompt.format(
            question=atomic_question,
            information=info_text
        )

    # 计算空集分数（没有任何信息）
    empty_prompt = build_prompt([])
    v_empty, _ = get_answer_logprob(model, tokenizer, empty_prompt, target_answer)
    
    # 计算完整分数（所有信息）
    full_prompt = build_prompt(all_facts)
    full_score, _ = get_answer_logprob(model, tokenizer, full_prompt, target_answer)

    print(f"📊 空集分数（无信息）: {v_empty:.4f}")
    print(f"📊 完整分数（所有信息）: {full_score:.4f}")

    convergence_count = 0
    
    # 蒙特卡洛采样计算Shapley值
    for t in tqdm(range(1, max_samples + 1)):
        print(f"\n==== Sample {t}/{max_samples} ====")
        
        # 随机排列所有信息索引
        perm = list(range(n))
        random.shuffle(perm)
        print(f"Permutation: {perm}")
        
        prev_shapley_scores = shapley_scores.copy()
        
        # 从空集开始逐步添加信息
        v_prev = v_empty
        active_facts = []

        for j, idx in enumerate(perm):
            # 添加当前信息到联盟
            active_facts.append(all_facts[idx])
            cur_prompt = build_prompt(active_facts)
            v_j, _ = get_answer_logprob(model, tokenizer, cur_prompt, target_answer)

            # 计算边际贡献（当前联盟 vs 前一个联盟）
            marginal_contrib = v_j - v_prev
            
            # 更新Shapley值（滑动平均）
            phi_old = shapley_scores[idx]
            shapley_scores[idx] = (t - 1)/t * phi_old + (1/t) * marginal_contrib
            v_prev = v_j

            print(f"[Step {j+1}] Add info #{idx} \"{all_facts[idx][:50]}...\"")
            print(f"→ LogProb: {v_j:.4f}, Marginal: {marginal_contrib:.4f}, Shapley = {shapley_scores[idx]:.4f}")

            # 早停条件：如果接近完整分数
            if abs(v_j - full_score) < early_stop_eps:
                print(f"→ Early stopping (|{v_j:.4f} - {full_score:.4f}| < {early_stop_eps})")
                break
        
        # 检查收敛
        if t >= min_samples:
            shapley_diffs = np.abs(shapley_scores - prev_shapley_scores)
            avg_diff = np.mean(shapley_diffs)
            print(f"\nAverage Shapley value change: {avg_diff:.6f}")
            
            if avg_diff < tol:
                convergence_count += 1
                print(f"Convergence detected ({convergence_count}/2)")
                if convergence_count >= 2:
                    print(f"✅ Shapley值在第{t}次迭代后收敛")
                    break
            else:
                convergence_count = 0
    
    print(f"📈 最终Shapley值: {shapley_scores}")
    return shapley_scores, [], all_facts  # 返回空的基础信息，所有信息作为shapley_info


def process_all_data(model, tokenizer, data_path, output_path):
    """处理所有数据并保存结果"""
    # 初始化结果存储
    all_results = []
    running_recalls = {1: [], 3: [], 5: [], 10: []}
    k_values = [1, 3, 5, 10]
    
    # 读取所有数据
    datas = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            datas.append(json.loads(line))
    
    print(f"📊 Total samples to process: {len(datas)}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(columns=[
        'sample_id', 'question', 'target_answer', 
        'recall@1', 'recall@3', 'recall@5', 'recall@10',
        'shapley_scores', 'context_info',
        'shapley_mean', 'shapley_std', 'shapley_min', 'shapley_max',
        'top_1_info', 'top_3_info', 'top_5_info', 'top_10_info'
    ])
    
    # 处理每个样本
    for idx, data in enumerate(datas, 1):
        print(f"\n{'='*50}")
        print(f"Processing sample {idx}/{len(datas)}")
        print(f"{'='*50}")
        
        # 提取上下文信息
        atom_infos = data['atomic_facts'] if isinstance(data['atomic_facts'], list) else [data['atomic_facts']]
        
        # 添加噪声信息
        noise_infos = [
            "The patient's zodiac sign is Aquarius",
            "The patient's Chinese zodiac is Dragon",
            "The patient has red hair",
            "The patient is 178cm tall",
            "The patient's hometown is in Zhejiang",
            "The patient was born in March"
        ]
        atom_infos.extend(noise_infos)
        
        # 获取问题和目标答案
        atomic_question = data['question']
        target_answer = data['answer_idx']
        
        print(f"Question: {atomic_question}")
        print(f"Target answer: {target_answer}")
        
        try:
            # 计算Shapley值
            shapley, basic_info, shapley_info = estimate_atomic_shapley_with_tolerance(
                model, tokenizer,
                doctor_prompt_shapley, atomic_question, atom_infos,
                target_answer,
                max_samples=1,
                min_samples=3,
                tol=1e-2,
                early_stop_eps=1e-2
            )
            
            # 计算各个k值的Recall
            recalls = {}
            recall_details = {}
            print("\n--- Recall@k Analysis ---")
            for k in k_values:
                recall, details = compute_recall_at_k(shapley, atom_infos, noise_infos, k)
                recalls[k] = recall
                recall_details[k] = details
                running_recalls[k].append(recall)
                print(f"Recall@{k}: {recall:.4f}")
            
            # 计算并显示当前平均Recall
            print("\n--- Current Average Recall@k ---")
            for k in k_values:
                avg_recall = np.mean(running_recalls[k])
                print(f"Average Recall@{k}: {avg_recall:.4f}")
            
            # 保存结果
            result_row = {
                'sample_id': data.get('id', idx),
                'question': atomic_question,
                'target_answer': target_answer,
                'recall@1': recalls[1],
                'recall@3': recalls[3],
                'recall@5': recalls[5],
                'recall@10': recalls[10],
                'shapley_scores': shapley.tolist(),
                'context_info': atom_infos,
                'shapley_mean': recall_details[1]['shapley_stats']['mean'],
                'shapley_std': recall_details[1]['shapley_stats']['std'],
                'shapley_min': recall_details[1]['shapley_stats']['min'],
                'shapley_max': recall_details[1]['shapley_stats']['max'],
                'top_1_info': recall_details[1]['top_k_info'],
                'top_3_info': recall_details[3]['top_k_info'],
                'top_5_info': recall_details[5]['top_k_info'],
                'top_10_info': recall_details[10]['top_k_info']
            }
            
            # 更新DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
            
            # 实时保存结果
            results_df.to_csv(output_path, index=False)
            print(f"\n💾 Results saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing sample {idx}: {str(e)}")
            continue
    
    # 计算最终统计
    print("\n" + "="*50)
    print("📊 Final Statistics")
    print("="*50)
    
    for k in k_values:
        avg_recall = np.mean(running_recalls[k])
        std_recall = np.std(running_recalls[k])
        print(f"Recall@{k}: {avg_recall:.4f} ± {std_recall:.4f}")
    
    return results_df

if __name__ == '__main__':
    model_name_or_path = "/ssd/xiaobei/common_llm/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        trust_remote_code=True,
        mirror="hf-mirror"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        device_map="auto", 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        mirror="hf-mirror"
    )
    model.eval()

    # 设置输入输出路径
    data_path = '/home/xiaobei/Agentic-RAG-R1__under-construction/acc_testnew/dataset/medqa_test_convo.jsonl'
    output_path = '/home/xiaobei/Agentic-RAG-R1__under-construction/evaluation_results/medqa_shapley_results4.csv'
    
    # 处理所有数据
    results_df = process_all_data(model, tokenizer, data_path, output_path)
