import os
import sys
import json
import torch
import logging
import argparse
import re
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader

from src.data.prepare_dataset import prepare_dataset
from src.utils.patient_model import PatientModel
from src.models.doctor_reward import match_choice

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_medical_question(prompt: str) -> str:
    """
    从prompt中提取医学问题
    """
    # 尝试从提示中提取问题
    question_match = re.search(r'问题：(.+?)(?:选项|$)', prompt, re.DOTALL)
    if question_match:
        return question_match.group(1).strip()
    
    # 或者查找"question_type"之后的内容
    type_match = re.search(r'question_type=(.+?),\s*question=(.+?),', prompt, re.DOTALL)
    if type_match and len(type_match.groups()) >= 2:
        return f"{type_match.group(1).strip()} - {type_match.group(2).strip()}"
    
    return "未能提取到医学问题"


def extract_answer_from_text(text: str, options: Dict[str, str]) -> str:
    """
    参考doctor_trainer.py中match_choice函数实现，
    从文本中提取答案，返回选项字母（如A、B、C、D、E）
    """
    option_letters = ["A", "B", "C", "D", "E", "F", "G"]
    
    # 尝试匹配"answer: "后的选项字母
    res = re.search(r"(answer:|答案|正确选项)(?:是|：|为|应该是|应该为)\s*([A-G])", text, re.IGNORECASE)
    if res:
        # 只返回单个匹配的字母
        matched = res.group(2).upper()
        if matched in option_letters:
            return matched
    
    # 如果上面没有匹配到，尝试找到文本中提到的选项内容
    # 只有当选项内容完全出现在文本中才考虑
    matched_options = []
    for op_letter, op_text in options.items():
        if not op_text:
            continue
        if op_text in text:
            logger.debug(f"Found option content {op_letter}:{op_text} in response")
            matched_options.append(op_letter)
    
    if len(matched_options) == 1:
        return matched_options[0]
    
    # 最后尝试直接找文本中的A-G单独字母提及
    # 只取最后一个提及的字母
    all_letters = re.findall(r'\b([A-G])\b', text)
    if all_letters:
        last_letter = all_letters[-1]
        if last_letter in option_letters:
            return last_letter
    
    # 如果都未匹配到，返回空字符串
    return ""


def format_options_text(options: Dict[str, str]) -> str:
    """
    格式化选项文本，用于提示模型
    """
    options_text = "\n选项：\n"
    for letter, content in options.items():
        if content:  # 只添加非空选项
            options_text += f"{letter}. {content}\n"
    return options_text


def create_completion_mask(
        completion_ids: torch.LongTensor,
        start_ids: List[int],
        end_ids: List[int],
) -> torch.LongTensor:
    """
    Create a binary mask for the completion part.
    Only tokens between <|im_start|>assistant and the next <|im_start|>, or after the last assistant, are counted.

    Args:
        completion_ids: (seq_len,) completion token ids
        start_ids: token ids for "<|im_start|>assistant"
        end_ids: token ids for "<|im_start|>"

    Returns:
        mask: (seq_len,) 0/1 tensor
    """
    seq_len = completion_ids.size(0)
    mask = torch.zeros(seq_len, dtype=torch.long, device=completion_ids.device)

    i = 0
    while i < seq_len:
        # 找到下一个 "<|im_start|>assistant"
        if i + len(start_ids) <= seq_len and torch.all(
                completion_ids[i:i + len(start_ids)] == torch.tensor(start_ids, device=completion_ids.device)):
            i += len(start_ids)
            start_pos = i

            # 向后找下一个 "<|im_start|>"
            while i < seq_len:
                if i + len(end_ids) <= seq_len and torch.all(
                        completion_ids[i:i + len(end_ids)] == torch.tensor(end_ids, device=completion_ids.device)):
                    break
                i += 1
            end_pos = i

            # 标记[start_pos, end_pos)之间是1
            mask[start_pos:end_pos] = 1
        else:
            i += 1

    return mask


def generate_completions_multi_round(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    num_generations: int = 4,
    max_new_tokens: int = 128,
    max_length_for_gather: int = 2048,
    temperature: float = 0.7,
    do_sample: bool = True,
    max_generate_iterations: int = 8,
    patient_models: List['PatientModel'] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-round generation with patient_models per sample.
    """
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"

    # Step 1: Tokenize initial prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)

    # Repeat for multiple generations
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

    # Expand patient_models if needed
    if patient_models is not None and num_generations > 1:
        expanded_patient_models = []
        for model_i in patient_models:
            expanded_patient_models.extend([model_i] * num_generations)
        patient_models = expanded_patient_models

    batch_size = prompt_ids.size(0)

    current_ids = prompt_ids.clone()
    current_mask = prompt_mask.clone()

    should_gen = torch.ones(batch_size, dtype=torch.bool, device=device)
    final_outputs: List[Optional[torch.LongTensor]] = [None] * batch_size #完整的prompt+所有生成内容
    completion_texts=[""] * batch_size

    #同一个batch里不同sample一起decode，需要注意padding，每轮生成完之后重新pad
    for round_idx in range(max_generate_iterations):
        print("=" * 80)
        print(f"[Round {round_idx + 1}/{max_generate_iterations}] Start")
        print(f"  should_gen: {should_gen.tolist()}")
        print(f"  current_ids shape: {current_ids.shape}")

        if not should_gen.any():
            break

        active = torch.nonzero(should_gen).squeeze(1) #获得需要继续生成的sample id
        print(f"[Generation] active batch indices: {active.tolist()}")

        #对active samples做生成
        outputs = model.generate(
            input_ids=current_ids,
            attention_mask=current_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        old_len = current_ids.size(1)
        history_ids = outputs[:, :old_len]
        new_generated_ids = outputs[:, old_len:]

        history_texts = tokenizer.batch_decode(history_ids, skip_special_tokens=False)
        generated_texts = tokenizer.batch_decode(new_generated_ids, skip_special_tokens=False)
        history_texts = [
            text.replace(tokenizer.pad_token, "").strip()
            for text in history_texts
        ]
        generated_texts = [
            text.replace(tokenizer.pad_token, "").strip()
            for text in generated_texts
        ]

        next_prompts = []

        for idx, text in enumerate(generated_texts):
            b = active[idx].item()
            print(f"\n[Sample {b}] Generated text: {repr(text)}")

            merged_text = history_texts[idx] #merged_text中没有额外的padding

            # Step2: Check answer or no question
            if "answer:" in text.lower():
                completion_texts[b] += text
                merged_text += text
                final_outputs[b] = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                    device).squeeze(0)
                should_gen[b] = False
                continue

            # Step3: Handle question
            if "question:" in text.lower() and (round_idx < max_generate_iterations-1):
                start = text.lower().index("question:")
                question = text[start:].strip()
                try:
                    # 每个样本用自己的 patient_models[b]
                    answer = patient_models[b].get_answer(question) if patient_models is not None else "No answer available."
                except Exception as exc:
                    answer = f"Get Patient Answer Error: {exc}"

                new_text=text + '\n<|im_start|>user\n' + answer + '<|im_end|>\n<|im_start|>assistant\n'
                completion_texts[b] += new_text
                merged_text += new_text

                next_prompt_ids = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                    device).squeeze(0)
                next_prompts.append(next_prompt_ids)
            else:
                # 不是question也不是answer，继续生成
                # 在最后一轮，强制要求生成answer
                if round_idx == max_generate_iterations - 1:
                    # 强制生成answer
                    merged_text += text
                    merged_text += "\n答题结束，请根据以上信息，给出最终答案，并以【正确答案是XX】的形式给出选项。\nanswer:"
                    next_prompt_ids = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                        device).squeeze(0)
                    next_prompts.append(next_prompt_ids)
                else:
                    # 普通继续生成
                    merged_text += text
                    completion_texts[b] += text
                    next_prompt_ids = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                        device).squeeze(0) 
                    next_prompts.append(next_prompt_ids)

        if next_prompts:
            texts = [tokenizer.decode(t, skip_special_tokens=False) for t in next_prompts]
            tokenizer.padding_side = "left"
            enc = tokenizer(texts, add_special_tokens=False,return_tensors="pt", padding=True)
            current_ids = enc.input_ids.to(device)
            current_mask = enc.attention_mask.to(device)

    tokenizer.padding_side = "right"
    completion_ids=tokenizer(completion_texts,add_special_tokens=False,return_tensors="pt", padding=True).input_ids.to(
                    device)
    completion_masks=[]
    prompt_len = prompt_ids.size(1)  # 统一固定的prompt长度
    allowed_completion_len = max_length_for_gather - prompt_len

    if completion_ids.size(1) > allowed_completion_len:
        # 统一裁剪到 allowed_completion_len
        completion_ids = completion_ids[:, :allowed_completion_len]

    for b in range(batch_size):
        mask = create_completion_mask(
            completion_ids[b],
            start_ids=tokenizer.encode("<|im_start|>assistant", add_special_tokens=False),
            end_ids=tokenizer.encode("<|im_start|>", add_special_tokens=False),
        )
        completion_masks.append(mask)
    completion_masks = torch.stack(completion_masks, dim=0)

    return prompt_ids, prompt_mask, completion_ids,  completion_masks


def custom_collate_fn(batch):
    """
    Collate a batch of dicts with potentially non-tensor and variable-length fields.
    This version preserves lists and dicts as-is without stacking.
    """
    collated = {key: [sample[key] for sample in batch] for key in batch[0]}
    return collated


def parse_args():
    parser = argparse.ArgumentParser(description="Test CMB model on medical QA tasks")
    parser.add_argument("--model_path", type=str, default="/root/小北健康-模型", help="Path to the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to test")
    parser.add_argument("--results_dir", type=str, default="experiments/evaluation/cmb_test", help="Directory for results")
    parser.add_argument("--max_generate_iterations", type=int, default=4, help="Maximum number of generation iterations")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 准备加速器
    accelerator = Accelerator()
    device = accelerator.device
    
    # 加载模型
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # 准备数据集
    logger.info("Preparing dataset")
    test_dataset = prepare_dataset("test", "cmb", eval_size=0)
    
    # 限制样本数量
    if args.max_samples is not None:
        test_dataset = test_dataset[:args.max_samples]
        logger.info(f"Testing on first {args.max_samples} samples")
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # 准备模型和加速器
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    
    # 评估结果
    results = {
        "accuracy": 0,
        "correct_answers": 0,
        "total_samples": 0,
        "detailed_results": []
    }
    
    # 开始评估
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            prompts = batch["prompt"]
            correct_answers = batch["answer"]
            batch_facts = batch["facts"]
            batch_options = batch["option"]
            questions = batch["question"]
            
            logger.info(f"Batch {batch_idx+1}/{len(test_dataloader)}")
            
            # 打印样例信息用于调试
            for i, (prompt, question) in enumerate(zip(prompts, questions)):
                logger.info(f"Sample {i} question: {question}")
                logger.info(f"Sample {i} options: {batch_options[i]}")
            
            # 创建患者模型
            patient_model_list = []
            for facts in batch_facts:
                patient_model = PatientModel(facts)
                patient_model_list.append(patient_model)
            
            # 多轮生成对话
            _, _, completion_ids, _ = generate_completions_multi_round(
                model,
                tokenizer,
                prompts,
                num_generations=1,
                max_new_tokens=512,
                max_length_for_gather=2048,
                temperature=args.temperature,
                do_sample=True,
                max_generate_iterations=args.max_generate_iterations,
                patient_models=patient_model_list
            )
            
            completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=False)
            completion_texts = [text.replace(tokenizer.pad_token, "") for text in completion_texts]
            
            # 处理每个样本
            for i, (text, correct_answer) in enumerate(zip(completion_texts, correct_answers)):
                # 提取回答
                full_text = text
                model_answer = ""
                
                # 使用match_choice从doctor_reward.py
                options_dict = batch_options[i]
                extracted_answer = match_choice(full_text, options_dict)
                
                is_correct = extracted_answer == correct_answer
                if is_correct:
                    results["correct_answers"] += 1
                
                # 构建对话日志
                dialog_log = {
                    "original_question": questions[i],
                    "options": batch_options[i],
                    "rounds": []
                }
                
                # 解析对话轮次
                parts = re.split(r'<\|im_start\|>|<\|im_end\|>', full_text)
                round_idx = 1
                for j in range(0, len(parts)-1, 2):
                    role = parts[j].strip()
                    content = parts[j+1].strip() if j+1 < len(parts) else ""
                    
                    if role == "assistant":
                        dialog_log["rounds"].append({
                            "round": round_idx,
                            "role": "doctor",
                            "content": content
                        })
                        round_idx += 1
                    elif role == "user":
                        dialog_log["rounds"].append({
                            "round": round_idx - 1,
                            "role": "patient",
                            "content": content
                        })
                
                # 添加结果
                sample_result = {
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "medical_question": questions[i],
                    "facts": batch_facts[i],
                    "model_output": full_text,
                    "model_answer": extracted_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "full_generated_text": full_text,
                    "options": batch_options[i],
                    "dialog_log": {
                        **dialog_log,
                        "extracted_answer": extracted_answer
                    }
                }
                
                results["detailed_results"].append(sample_result)
                results["total_samples"] += 1
    
    # 计算准确率
    if results["total_samples"] > 0:
        results["accuracy"] = results["correct_answers"] / results["total_samples"]
    
    # 保存结果
    results_path = os.path.join(args.results_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {results_path}")
    logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['correct_answers']}/{results['total_samples']})")


if __name__ == "__main__":
    main() 