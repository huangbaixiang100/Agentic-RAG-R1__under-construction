#!/usr/bin/env python3
"""
UoT结果文件准确率分析脚本
分析predicted和target字段的一致性
"""

import json
import os
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple
from pathlib import Path

def load_json_data(file_path: str) -> List[Dict]:
    """加载JSON文件数据"""
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        # 如果数据是列表格式
        if isinstance(json_data, list):
            data = json_data
        # 如果数据是字典格式，可能包含results字段
        elif isinstance(json_data, dict):
            if 'results' in json_data:
                data = json_data['results']
            else:
                # 尝试找到包含样本数据的字段
                for key, value in json_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict) and 'predicted' in value[0] and 'target' in value[0]:
                            data = value
                            break
        
        print(f"📄 文件 {os.path.basename(file_path)}: 加载了 {len(data)} 个样本")
        return data
        
    except Exception as e:
        print(f"❌ 加载文件 {file_path} 失败: {e}")
        return []

def calculate_accuracy_from_data(data: List[Dict]) -> Tuple[float, List[bool]]:
    """从数据计算准确率，返回准确率和每个样本的正确性列表"""
    correct_list = []
    processed_count = 0
    
    for item in data:
        # 检查必要字段是否存在
        if 'predicted' not in item or 'target' not in item:
            continue
        
        processed_count += 1
        
        # 比较predicted和target
        predicted = str(item['predicted']).strip().upper()
        target = str(item['target']).strip().upper()
        
        is_correct = (predicted == target)
        correct_list.append(is_correct)
    
    accuracy = sum(correct_list) / len(correct_list) if correct_list else 0
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
        data = load_json_data(file_path)
        if not data:
            print("   ❌ 没有找到有效数据")
            return None
        
        # 计算准确率
        accuracy, correct_list = calculate_accuracy_from_data(data)
        
        # Bootstrap计算方差
        variance, conf_lower, conf_upper, bootstrap_mean = bootstrap_variance(correct_list)
        variance=np.sqrt(variance)
        # 提取模型信息
        model_name = os.path.basename(file_path).split('_')[0]
        
        result = {
            'file_name': os.path.basename(file_path),
            'model': model_name,
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
    print("🔍 UoT结果文件分析工具")
    print("=" * 60)
    
    # 要分析的文件列表
    files_to_analyze = [
        "/home/xiaobei/Agentic-RAG-R1__under-construction/prompt-base/UoT/uot_results/vllm_8702_cmb_full_uot.json",
        "/home/xiaobei/Agentic-RAG-R1__under-construction/prompt-base/UoT/uot_results/vllm_8701_cmb_full_uot.json",
        "/home/xiaobei/Agentic-RAG-R1__under-construction/prompt-base/UoT/uot_results/qwen1.7b_cmb_full_uot.json"
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
    df['accuracy'] = df['accuracy'].apply(lambda x: f"{x:.4f}")
    df['bootstrap_mean'] = df['bootstrap_mean'].apply(lambda x: f"{x:.4f}")
    df['variance'] = df['variance'].apply(lambda x: f"{x:.6f}")
    df['confidence_lower'] = df['confidence_lower'].apply(lambda x: f"{x:.4f}")
    df['confidence_upper'] = df['confidence_upper'].apply(lambda x: f"{x:.4f}")
    
    # 保存结果
    output_file = "/home/xiaobei/Agentic-RAG-R1__under-construction/acc_test/uot_results_analysis.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 保存Excel格式
    excel_file = "/home/xiaobei/Agentic-RAG-R1__under-construction/acc_test/uot_results_analysis.xlsx"
    df.to_excel(excel_file, index=False, engine='openpyxl')
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    print("📊 UoT分析结果摘要")
    print("=" * 60)
    
    # 按模型显示结果
    print("\n📈 各模型性能比较:")
    print("-" * 80)
    print(f"{'模型':<15} {'准确率':<10} {'Bootstrap均值':<14} {'方差':<12} {'样本数':<8}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['model']:<15} {row['accuracy']:<10} "
              f"{row['bootstrap_mean']:<14} {row['variance']:<12} {row['sample_count']:<8}")
    
    print("\n💾 详细结果已保存到:")
    print(f"   CSV: {output_file}")
    print(f"   Excel: {excel_file}")

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    main() 