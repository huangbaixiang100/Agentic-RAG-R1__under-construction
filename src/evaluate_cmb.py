import argparse
import os
import json
import sys
import torch
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data.prepare_dataset import prepare_dataset
from src.models.doctor_trainer import generate_completions_multi_round, parse_dialog_simple
from src.models.doctor_reward import parse_dialog, format_dialog, match_choice
from src.utils.patient_model import PatientModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation_logs.log')
    ]
)

def evaluate_model_on_cmb(
    model_path: str,
    dataset_split: str = "test",
    num_samples: int = 100,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    max_generate_iterations: int = 6,
    temperature: float = 0.7,
    do_sample: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "evaluation_results",
    is_local_model: bool = False,
    train_file: str = None,
    test_file: str = None,
):
    """
    评估模型在CMB数据集上的表现
    
    Args:
        model_path: 模型路径
        dataset_split: 指定使用哪个数据集，可以是"train"或"test"
        num_samples: 评估样本数量，如果为None则使用全部样本
        batch_size: 批处理大小
        max_new_tokens: 最大生成token数
        max_generate_iterations: 最大对话轮数
        temperature: 生成温度
        do_sample: 是否采样生成
        device: 设备
        output_dir: 输出目录
        is_local_model: 是否为本地模型
        train_file: 训练集文件路径，如果为None则使用默认路径
        test_file: 测试集文件路径，如果为None则使用默认路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 强制将模型视为本地模型
    if is_local_model:
        logging.info(f"模型路径类型: 本地模型路径 (由命令行参数指定)")
    else:
        # 检查是否为绝对路径
        is_absolute_path = os.path.isabs(model_path)
        # 检查是否包含文件夹分隔符
        has_path_separator = '/' in model_path or '\\' in model_path
        # 判断是否为本地路径
        is_local_path = is_absolute_path or has_path_separator or os.path.exists(model_path)
        logging.info(f"模型路径类型: {'本地模型路径' if is_local_path else 'Hugging Face模型ID'}")
        is_local_model = is_local_path
    
    # 加载模型和分词器
    print(f"加载模型: {model_path}")
    try:
        # 检查模型目录是否包含必要文件
        if is_local_model:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                logging.info(f"找到模型配置文件: {config_path}")
            else:
                logging.warning(f"未找到模型配置文件: {config_path}")
        
        # 强制设置为本地文件模式
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=is_local_model
        )
        
        # 尝试直接从配置文件加载
        if is_local_model:
            logging.info("尝试从配置文件加载模型...")
            config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
            logging.info(f"模型配置加载成功, 模型类型: {config.model_type}")
        
        try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
            local_files_only=is_local_model
        )
        logging.info("模型加载成功!")
    except Exception as e:
        logging.error(f"加载模型时发生错误: {str(e)}")
        
        # 错误处理：尝试备用加载方法
        logging.error("尝试备用加载方法...")
        try:
            # 用正斜杠规范化路径
            normalized_path = model_path.replace('\\', '/')
            logging.info(f"规范化路径: {normalized_path}")
            
            # 尝试加载tokenizer
            logging.info("尝试加载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                normalized_path,
                trust_remote_code=True,
                local_files_only=True,
                use_fast=False
            )
            logging.info("tokenizer加载成功!")
            
            # 尝试加载模型
            logging.info("尝试加载模型...")
            # 使用更多参数以确保加载成功
            model = AutoModelForCausalLM.from_pretrained(
                normalized_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                revision=None,
                from_tf=False,
                force_download=False,
                cache_dir=None,
                resume_download=False,
            )
            logging.info("备用方法成功加载模型!")
        except Exception as e2:
            logging.error(f"备用加载方法也失败: {str(e2)}")
            raise RuntimeError(f"无法加载模型，请检查路径是否正确，并确保包含所有必要的模型文件。\n原始错误: {str(e)}\n备用错误: {str(e2)}")
            
    except Exception as e:
        logging.error(f"加载模型时发生错误: {str(e)}")
            
    model.eval()
    
    # 记录数据集文件路径信息
    if train_file:
        logging.info(f"使用自定义训练集文件: {train_file}")
    if test_file:
        logging.info(f"使用自定义测试集文件: {test_file}")
    
    # 加载数据集 - 使用修改后的prepare_dataset函数直接加载指定数据集
    print(f"加载CMB数据集 (split={dataset_split})")
    train_dataset, eval_dataset = prepare_dataset(
        dataset_split, 
        "cmb", 
        eval_size=0,  # eval_size参数不再使用
        train_file=train_file,
        test_file=test_file
    )
    
    # 根据split参数选择对应的数据集
    if dataset_split == "train" or dataset_split == "all":
        dataset = train_dataset
    else:  # test或eval
        dataset = eval_dataset
    
    # 如果指定了样本数量限制，则只使用前num_samples个样本
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    print(f"数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    def custom_collate_fn(batch):
        collated = {key: [sample[key] for sample in batch] for key in batch[0]}
        return collated
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn
    )
    
    # 评估结果
    all_results = []
    correct_count = 0
    total_count = 0
    
    # 对每个批次进行评估
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="评估批次")):
        prompts = batch["prompt"]
        answers = batch["answer"]
        facts = batch["facts"]
        options = batch["option"]
        
        # 为每个样本创建PatientModel
        patient_models = [PatientModel(f) for f in facts]
        
        logging.info("="*80)
        logging.info(f"批次 {batch_idx+1}/{len(dataloader)}")
        
        # 使用generate_completions_multi_round生成对话
        with torch.no_grad():
            _, _, completion_ids, _ = generate_completions_multi_round(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                num_generations=1,  # 每个样本只生成一次
                max_new_tokens=max_new_tokens,
                max_length_for_gather=2048,
                temperature=temperature,
                do_sample=do_sample,
                max_generate_iterations=max_generate_iterations,
                patient_models=patient_models,
            )
        
        # 解码生成的完成内容
        completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=False)
        
        # 分析结果
        for i, (completion, answer, option) in enumerate(zip(completion_texts, answers, options)):
            # 将批处理索引转换为总体索引
            sample_idx = batch_idx * batch_size + i
            
            logging.info("-"*80)
            logging.info(f"样本 {sample_idx}, 正确答案: {answer}")
            
            # 解析对话
            try:
                # 使用更健壮的对话解析方法
                dialog = parse_dialog_simple(completion)
                formatted_dialog = format_dialog(dialog)
                
                # 记录详细的对话内容
                logging.info(f"样本 {sample_idx} 的完整对话:")
                for turn_idx, turn in enumerate(dialog):
                    role = "医生" if turn["role"] == "assistant" else "患者"
                    content = turn["content"]
                    logging.info(f"[轮次 {turn_idx+1}] {role}: {content}")
                
                # 提取最后一个assistant响应
                last_responses = [msg for msg in dialog if msg['role'] == 'assistant']
                if last_responses:
                    last_response = last_responses[-1]['content']
                else:
                    last_response = completion
                
                # 从响应中提取选择
                model_choice = match_choice(last_response, option)
                correct = model_choice == answer.strip()
                
                if correct:
                    correct_count += 1
                total_count += 1
                
                # 记录模型选择和正确性
                logging.info(f"模型选择: {model_choice}, 正确答案: {answer}, 是否正确: {'✓' if correct else '✗'}")
                logging.info(f"总对话轮数: {len(dialog)}")
                
                # 收集结果
                result = {
                    "sample_idx": sample_idx,
                    "model_choice": model_choice,
                    "correct_answer": answer,
                    "is_correct": correct,
                    "dialog": formatted_dialog,
                    "raw_completion": completion,
                    "num_turns": len(dialog),
                }
                all_results.append(result)
                
                # 分析对话中的问题和回答标记
                has_question = any("question:" in turn["content"].lower() or "问题:" in turn["content"].lower() 
                                 for turn in dialog if turn["role"] == "assistant")
                has_answer = any("answer:" in turn["content"].lower() or "回答:" in turn["content"].lower() or "答案:" in turn["content"].lower() 
                               for turn in dialog if turn["role"] == "assistant")
                
                logging.info(f"对话分析: 包含问题标记={has_question}, 包含回答标记={has_answer}")
                
            except Exception as e:
                logging.error(f"处理样本 {sample_idx} 时出错: {e}")
                logging.error(f"原始生成内容: {completion[:200]}...")
                
                result = {
                    "sample_idx": sample_idx,
                    "error": str(e),
                    "completion": completion,
                }
                all_results.append(result)
        
        # 每个批次后记录当前准确率
        current_accuracy = correct_count / total_count if total_count > 0 else 0
        logging.info(f"当前进度: 已评估 {total_count} 样本, 正确数: {correct_count}, 准确率: {current_accuracy:.4f}")
    
    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"正确数: {correct_count}/{total_count}, 准确率: {accuracy:.4f}")
    logging.info("="*80)
    logging.info(f"最终结果: 正确数: {correct_count}/{total_count}, 准确率: {accuracy:.4f}")
    
    # 保存结果
    # 从文件路径中提取数据集标识
    train_name = os.path.basename(train_file).split('.')[0] if train_file else "default_train"
    test_name = os.path.basename(test_file).split('.')[0] if test_file else "default_test"
    dataset_identifier = train_name if dataset_split == "train" else test_name
    
    results_file = os.path.join(output_dir, f"cmb_{dataset_split}_{dataset_identifier}_results_{len(dataset)}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": total_count,
                "samples": all_results,
                "config": {
                    "model_path": model_path,
                    "dataset_split": dataset_split,
                    "num_samples": num_samples,
                    "batch_size": batch_size,
                    "max_new_tokens": max_new_tokens,
                    "max_generate_iterations": max_generate_iterations,
                    "temperature": temperature,
                    "do_sample": do_sample,
                    "is_local_model": is_local_model,
                    "train_file": train_file,
                    "test_file": test_file,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    
    print(f"评估结果已保存到: {results_file}")
    logging.info(f"评估结果已保存到: {results_file}")
    
    # 返回总准确率
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="评估模型在CMB数据集上的表现")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--dataset_split", type=str, default="test", choices=["train", "test", "eval", "all"], 
                        help="指定数据集类型: train-训练集, test/eval-测试集, all-全部数据")
    parser.add_argument("--num_samples", type=int, default=None, help="评估样本数量，None表示使用全部样本")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--max_generate_iterations", type=int, default=6, help="最大对话轮数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--no_sample", action="store_true", help="不使用采样生成")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--log_file", type=str, default="evaluation_logs.log", help="日志文件路径")
    parser.add_argument("--local_model", action="store_true", help="指定加载本地模型而非Hugging Face模型")
    parser.add_argument("--train_file", type=str, default=None, help="自定义训练集文件路径")
    parser.add_argument("--test_file", type=str, default=None, help="自定义测试集文件路径")
    
    args = parser.parse_args()
    
    # 配置日志文件
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info("="*80)
    logging.info(f"开始评估模型: {args.model_path}")
    logging.info(f"配置: 数据集={args.dataset_split}, 样本数={args.num_samples or '全部'}, 批大小={args.batch_size}")
    if args.train_file:
        logging.info(f"自定义训练集文件: {args.train_file}")
    if args.test_file:
        logging.info(f"自定义测试集文件: {args.test_file}")
    logging.info(f"生成配置: max_tokens={args.max_new_tokens}, iterations={args.max_generate_iterations}, temp={args.temperature}, do_sample={not args.no_sample}")
    
    evaluate_model_on_cmb(
        model_path=args.model_path,
        dataset_split=args.dataset_split,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_generate_iterations=args.max_generate_iterations,
        temperature=args.temperature,
        do_sample=not args.no_sample,
        device=args.device,
        output_dir=args.output_dir,
        is_local_model=args.local_model,
        train_file=args.train_file,
        test_file=args.test_file,
    )

if __name__ == "__main__":
    main() 