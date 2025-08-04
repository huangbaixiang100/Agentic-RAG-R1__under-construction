#!/usr/bin/env python3
"""
使用正确的方法分析MedQA和CMB结果文件
分别使用各自对应的方法计算准确率和bootstrap统计
"""

import json
import sys
import os
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple
from pathlib import Path

def load_jsonl_data(file_path: str) -> List[Dict]:
    """加载JSONL文件数据，处理重复ID"""
    data = []
    seen_ids = set()  # 用于跟踪已见过的ID
    duplicate_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line.strip())
                
                # 检查ID是否重复
                sample_id = item.get('id')
                if sample_id is not None:
                    if sample_id in seen_ids:
                        duplicate_count += 1
                        continue  # 跳过重复ID
                    seen_ids.add(sample_id)
                
                data.append(item)
                
            except json.JSONDecodeError as e:
                print(f"⚠️  文件 {file_path} 第{line_num}行JSON解析错误: {e}")
                continue
    
    if duplicate_count > 0:
        print(f"📄 文件 {os.path.basename(file_path)}: 跳过 {duplicate_count} 个重复ID")
    
    return data

def calculate_medqa_accuracy(data: List[Dict]) -> Tuple[float, List[bool]]:
    """使用MedQA方法计算准确率"""
    correct_list = []
    total_samples = 0
    correct_samples = 0
    
    for item in data:
        total_samples += 1
        
        # 获取结果信息
        interactive_system = item.get('interactive_system', {})
        is_correct = interactive_system.get('correct', False)
        
        # 统计正确数
        if is_correct:
            correct_samples += 1
            correct_list.append(True)
        else:
            correct_list.append(False)
    
    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    return accuracy, correct_list

def calculate_cmb_accuracy(data: List[Dict]) -> Tuple[float, List[bool]]:
    """使用CMB方法计算准确率"""
    correct_list = []
    total_samples = 0
    correct_samples = 0
    
    for item in data:
        total_samples += 1
        
        # 获取结果信息
        interactive_system = item.get('interactive_system', {})
        is_correct = interactive_system.get('correct', False)
        
        # 统计正确数
        if is_correct:
            correct_samples += 1
            correct_list.append(True)
        else:
            correct_list.append(False)
    
    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    return accuracy, correct_list

def bootstrap_variance(correct_list: List[bool], n_bootstrap: int = 100) -> Tuple[float, float, float, float]:
    """
    使用bootstrap方法计算准确率的方差和置信区间
    返回: (方差, 95%置信区间下界, 95%置信区间上界, bootstrap平均准确率)
    """
    if not correct_list:
        return 0, 0, 0, 0
    
    n_samples = len(correct_list)
    bootstrap_accuracies = []
    
    for _ in range(n_bootstrap):
        # 有放回抽样
        bootstrap_sample = [random.choice(correct_list) for _ in range(n_samples)]
        bootstrap_accuracy = sum(bootstrap_sample) / len(bootstrap_sample)
        bootstrap_accuracies.append(bootstrap_accuracy)
    
    # 计算统计量
    variance = np.var(bootstrap_accuracies)
    confidence_lower = np.percentile(bootstrap_accuracies, 2.5)
    confidence_upper = np.percentile(bootstrap_accuracies, 97.5)
    bootstrap_mean = np.mean(bootstrap_accuracies)
    
    return variance, confidence_lower, confidence_upper, bootstrap_mean

def analyze_file(file_path: str) -> Dict:
    """分析单个文件的结果"""
    print(f"\n📊 分析文件: {os.path.basename(file_path)}")
    
    try:
        # 加载数据
        data = load_jsonl_data(file_path)
        if not data:
            print("   ❌ 没有找到有效数据")
            return None
        
        # 根据文件名判断是MedQA还是CMB
        is_medqa = 'medqa' in file_path.lower()
        dataset_type = 'MedQA' if is_medqa else 'CMB'
        
        # 使用对应方法计算准确率
        if is_medqa:
            accuracy, correct_list = calculate_medqa_accuracy(data)
        else:
            accuracy, correct_list = calculate_cmb_accuracy(data)
        
        # Bootstrap计算方差
        variance, conf_lower, conf_upper, bootstrap_mean = bootstrap_variance(correct_list)
        variance=np.sqrt(variance)
        # 提取模型信息
        if '8b2' in file_path:
            model = 'Qwen3_8b2'
        elif '8b' in file_path:
            model = 'Qwen3_8b'
        else:
            model = 'Qwen3_1.7b'
        
        result = {
            'file_name': os.path.basename(file_path),
            'dataset': dataset_type,
            'model': model,
            'accuracy': accuracy,
            'bootstrap_mean': bootstrap_mean,
            'variance': variance,
            'confidence_lower': conf_lower,
            'confidence_upper': conf_upper,
            'sample_count': len(correct_list)
        }
        
        print(f"   ✅ 原始准确率: {accuracy:.4f}")
        print(f"   📊 Bootstrap平均: {bootstrap_mean:.4f}")
        print(f"   📏 方差: {variance:.6f}")
        print(f"   🔍 95%置信区间: [{conf_lower:.4f}, {conf_upper:.4f}]")
        print(f"   📈 样本数: {len(correct_list)}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ 分析失败: {e}")
        return None

def main():
    """主函数"""
    print("🔍 特定结果文件分析工具 - 使用正确的方法")
    print("=" * 60)
    
    # 要分析的文件列表
    files_to_analyze = [
        "/home/xiaobei/Agentic-RAG-R1__under-construction/prompt-base/mediQ/results/vllm_8701_Qwen3_8b2/QwenVLLM_local_cmb_results.jsonl",
        "/home/xiaobei/Agentic-RAG-R1__under-construction/prompt-base/mediQ/results/vllm_8701_Qwen3_8b2/QwenVLLM_local_medqa_results.jsonl",
        "/home/xiaobei/Agentic-RAG-R1__under-construction/prompt-base/mediQ/results/vllm_8702_Qwen3_8b/QwenVLLM_local_cmb_results.jsonl",
        "/home/xiaobei/Agentic-RAG-R1__under-construction/prompt-base/mediQ/results/vllm_8702_Qwen3_8b/QwenVLLM_local_medqa_results.jsonl",
        "/home/xiaobei/Agentic-RAG-R1__under-construction/prompt-base/mediQ/results/qwen3-1.7b_cmb_results.jsonl",
        "/home/xiaobei/Agentic-RAG-R1__under-construction/prompt-base/mediQ/results/qwen3-1.7b_medqa_results.jsonl"
    ]
    
    # 分析所有文件
    results = []
    for file_path in files_to_analyze:
        if os.path.exists(file_path):
            result = analyze_file(file_path)
            if result:
                results.append(result)
        else:
            print(f"\n⚠️  文件不存在: {file_path}")
    
    if not results:
        print("\n❌ 没有成功分析任何文件")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 格式化数值列
    df_display = df.copy()
    df_display['accuracy'] = df_display['accuracy'].apply(lambda x: f"{x:.4f}")
    df_display['bootstrap_mean'] = df_display['bootstrap_mean'].apply(lambda x: f"{x:.4f}")
    df_display['variance'] = df_display['variance'].apply(lambda x: f"{x:.6f}")
    df_display['confidence_lower'] = df_display['confidence_lower'].apply(lambda x: f"{x:.4f}")
    df_display['confidence_upper'] = df_display['confidence_upper'].apply(lambda x: f"{x:.4f}")
    
    # 保存结果
    output_file = "/home/xiaobei/Agentic-RAG-R1__under-construction/acc_test/correct_method_analysis_results.csv"
    df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.6f')
    
    # 保存Excel格式
    excel_file = "/home/xiaobei/Agentic-RAG-R1__under-construction/acc_test/correct_method_analysis_results.xlsx"
    df.to_excel(excel_file, index=False, engine='openpyxl')
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    print("📊 分析结果摘要")
    print("=" * 60)
    
    # 按数据集和模型分组显示结果
    print("\n📈 按数据集和模型的性能比较:")
    print("-" * 80)
    print(f"{'数据集':<8} {'模型':<12} {'准确率':<10} {'Bootstrap均值':<14} {'方差':<12} {'样本数':<8}")
    print("-" * 80)
    
    for _, row in df_display.iterrows():
        print(f"{row['dataset']:<8} {row['model']:<12} {row['accuracy']:<10} "
              f"{row['bootstrap_mean']:<14} {row['variance']:<12} {row['sample_count']:<8}")
    
    print("\n💾 详细结果已保存到:")
    print(f"   CSV: {output_file}")
    print(f"   Excel: {excel_file}")

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    main()