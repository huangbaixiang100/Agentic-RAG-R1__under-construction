#!/usr/bin/env python3
"""
批量计算CMB结果的准确率和bootstrap方差
分析所有训练结果并生成统计表格
"""

import json
import re
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random

# def match_choice(text,options_dict):
#     option = ["A", "B", "C", "D", "E", "F", "G"]
#     res = re.search(r"(answer: |答案|正确选项)(?:是|：|为|应该是|应该为)(.*?)(。|\.|$)", text, re.S)
#     #res = re.search(r"(answer: |答案|正确选项)(?:是|：|:|为|应该是|应该为)\s*(.*)", text, re.S) #(.*?)(。|\.|$)
#     #res = re.search(r"(?:answer|答案|正确答案|正确选项)[：:是为应该是应该为\s]*[【]?\s*([A-Fa-f]{1,6})\s*[】]?", text,
#     #                re.IGNORECASE)
#     if res:
#         #print(res)
#         #print(res.group(2))
#         #print("".join([x for x in res.group(2) if x in option]))
#         return "".join([x for x in res.group(2) if x in option])
#     else:
#         tmp=[]
#         for op_letter, op_text in options_dict.items():
#             if op_text in text:
#                 print(f"Found {op_letter}:{op_text}")
#                 tmp.append(op_letter)
#         return "".join(tmp)
#     return "".join([i for i in text if i in option])

def match_choice(text, options_dict):
    """使用正则表达式提取答案选项"""
    option = ["A", "B", "C", "D", "E", "F", "G"]
    
    # 首先尝试匹配带星号的选项格式: **A**, **BC** 等
    star_pattern = r"\*\*([A-Ga-g]{1,7})\*\*"
    star_matches = re.findall(star_pattern, text, re.IGNORECASE)
    if star_matches:
        # 多个匹配只取第一个；去重排序标准化
        answer = star_matches[0].upper()
        answer = "".join(sorted(set(answer)))
        return answer
    
    # 更新之后的正则表达式
    res = re.search(
        r"(answer: |答案|正确选项|正确结论|正确判断|正确 答案)(?:是|：|为|应该是|应该为)\s*(.*)", 
        text, re.S
    )
    pattern = (r"(?:正确答案|answer|正确选项|正确结论|正确判断|正确 答案|正确的选项|答案)"
               r"[：:是为应该是应该为\s]*[【]?\s*([A-Ga-g]{1,7})\s*[】]?")
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        # 多个匹配只取第一个；去重排序标准化
        answer = matches[0].upper()
        answer = "".join(sorted(set(answer)))
        if res:
            res_answer = "".join([x for x in res.group(2) if x in option])
        return answer
    else:
        # 尝试匹配"正确答案是[文本]"格式
        answer_text_patterns = [
            r"【正确答案是(.*?)】",        # 【正确答案是xxx】
            r"正确答案是(.*?)】",          # 正确答案是xxx】
            r"【正确答案是(.*)",           # 【正确答案是xxx (没有结束括号)
            r"正确答案是(.*?)[,，。\n]",    # 正确答案是xxx，或正确答案是xxx。
            r"正确答案是\s*([^,，。\n]+)",  # 正确答案是xxx (没有标点符号)
            r"答案是\s*([^,，。\n]+)"       # 答案是xxx (简化版本)
        ]
        
        answer_text = None
        for pattern in answer_text_patterns:
            text_matches = re.findall(pattern, text)
            if text_matches:
                answer_text = text_matches[0].strip()
                break
        
        if answer_text:
            # 找到了"正确答案是[文本]"格式的答案
            # 将答案文本与选项内容进行比对
            matched_options = []
            
            # 记录最佳匹配的选项和相似度
            best_match = None
            best_similarity = 0
            
            for op_letter, op_content in options_dict.items():
                if op_content is not None and isinstance(op_content, str):
                    # 检查选项内容是否包含答案文本，或答案文本是否包含选项内容
                    if answer_text in op_content or op_content in answer_text:
                        matched_options.append(op_letter)
                    else:
                        # 计算文本相似度 - 简单实现
                        # 这里使用一个简单的方法：共同单词数量 / 总单词数量
                        answer_words = set(answer_text.split())
                        option_words = set(op_content.split())
                        common_words = answer_words.intersection(option_words)
                        
                        if answer_words and option_words:  # 避免除零错误
                            similarity = len(common_words) / max(len(answer_words), len(option_words))
                            
                            # 更新最佳匹配
                            if similarity > best_similarity and similarity > 0.5:  # 设置阈值
                                best_similarity = similarity
                                best_match = op_letter
            
            # 如果找到了精确匹配
            if matched_options:
                return "".join(sorted(matched_options))
            # 如果找到了相似匹配
            elif best_match:
                return best_match
        
        # 如果上述方法都未匹配到，尝试在文本中查找选项内容
        tmp = []
        for op_letter, op_text in options_dict.items():
            # 添加空值检查，防止 None 类型错误
            if op_text is not None and isinstance(op_text, str) and op_text in text:
                tmp.append(op_letter)
        return "".join(tmp)


def load_jsonl_data(file_path: str) -> List[Dict]:
    """加载JSONL文件数据"""
    data = []
    seen_ids = set()  # 用于去重
    duplicate_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                    
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


def calculate_accuracy_from_data(data: List[Dict]) -> Tuple[float, List[bool]]:
    """从数据计算准确率，返回准确率和每个样本的正确性列表"""
    correct_list = []
    processed_count = 0
    
    for item in data:
        # 跳过错误样本
        if (item.get('final_answer') == 'error' or 
                not item.get('final_answer')):
            continue
        
        processed_count += 1
        content = item['final_answer']
        
        # 处理选项数据
        if 'options' in item:
            options_dict = {k: v for k, v in item['options'].items() 
                           if v and v.strip() != ""}
        elif 'option' in item:
            import ast
            if isinstance(item['option'], str):
                options_dict = ast.literal_eval(item['option'])
            else:
                options_dict = item['option']
        else:
            options_dict = {}
        
        # 提取预测答案
        pred = item.get('extracted_choice', '')
        if not pred:
            pred = match_choice(content, options_dict)
        
        # 获取正确答案
        answer = item.get('answer_idx', '') or item.get('answer', '')
        
        # 检查是否正确
        is_correct = (answer == pred)
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


def analyze_single_file(file_path: str) -> Dict:
    """分析单个文件的结果"""
    print(f"📊 分析文件: {os.path.basename(file_path)}")
    
    try:
        # 加载数据
        data = load_jsonl_data(file_path)
        if not data:
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'accuracy': 0,
                'bootstrap_mean': 0,
                'variance': 0,
                'confidence_lower': 0,
                'confidence_upper': 0,
                'sample_count': 0,
                'status': 'no_data'
            }
        
        # 计算准确率
        accuracy, correct_list = calculate_accuracy_from_data(data)
        
        # Bootstrap计算方差
        variance, conf_lower, conf_upper, bootstrap_mean = bootstrap_variance(correct_list)
        variance=np.sqrt(variance)
        result = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'accuracy': accuracy,
            'bootstrap_mean': bootstrap_mean,
            'variance': variance,
            'confidence_lower': conf_lower,
            'confidence_upper': conf_upper,
            'sample_count': len(correct_list),
            'status': 'success'
        }
        
        print(f"   ✅ 准确率: {accuracy:.4f}, 样本数: {len(correct_list)}")
        return result
        
    except Exception as e:
        print(f"   ❌ 分析失败: {e}")
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'accuracy': 0,
            'bootstrap_mean': 0,
            'variance': 0,
            'confidence_lower': 0,
            'confidence_upper': 0,
            'sample_count': 0,
            'status': f'error: {str(e)}'
        }


def find_all_jsonl_files(directories: List[str]) -> List[str]:
    """查找所有目录下的.jsonl文件"""
    all_files = []
    
    for directory in directories:
        if os.path.exists(directory):
            pattern = os.path.join(directory, "*.jsonl")
            files = glob.glob(pattern)
            all_files.extend(files)
            print(f"📁 目录 {directory}: 找到 {len(files)} 个JSONL文件")
        else:
            print(f"⚠️  目录不存在: {directory}")
    
    return sorted(all_files)


def main():
    """主函数"""
    print("🏥 CMB批量结果分析工具")
    print("=" * 60)
    
    # 定义要分析的目录
    directories = [
        "/home/xiaobei/Agentic-RAG-R1__under-construction/acc_test/cmb_result",
        "/home/xiaobei/Agentic-RAG-R1__under-construction/acc_testnew/cmb_result"
    ]
    
    # 查找所有JSONL文件
    all_files = find_all_jsonl_files(directories)
    
    if not all_files:
        print("❌ 没有找到任何JSONL文件")
        return
    
    print(f"\n🔍 总共找到 {len(all_files)} 个文件")
    print("-" * 40)
    
    # 分析所有文件
    results = []
    for file_path in all_files:
        result = analyze_single_file(file_path)
        results.append(result)
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 重新排列列的顺序
    columns_order = [
        'file_name', 'file_path', 'accuracy', 'bootstrap_mean', 'variance', 
        'confidence_lower', 'confidence_upper', 'sample_count', 'status'
    ]
    df = df[columns_order]
    
    # 格式化数值列
    df['accuracy'] = df['accuracy'].apply(lambda x: f"{x:.4f}")
    df['bootstrap_mean'] = df['bootstrap_mean'].apply(lambda x: f"{x:.4f}")
    df['variance'] = df['variance'].apply(lambda x: f"{x:.6f}")
    df['confidence_lower'] = df['confidence_lower'].apply(lambda x: f"{x:.4f}")
    df['confidence_upper'] = df['confidence_upper'].apply(lambda x: f"{x:.4f}")
    
    # 保存结果
    output_file = "/home/xiaobei/Agentic-RAG-R1__under-construction/acc_test/cmb_batch_analysis_results.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 保存Excel格式
    excel_file = "/home/xiaobei/Agentic-RAG-R1__under-construction/acc_test/cmb_batch_analysis_results.xlsx"
    df.to_excel(excel_file, index=False, engine='openpyxl')
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    print("📊 批量分析结果摘要")
    print("=" * 60)
    
    successful_results = df[df['status'] == 'success']
    if len(successful_results) > 0:
        # 转换回数值类型用于统计
        successful_results_numeric = successful_results.copy()
        successful_results_numeric['accuracy'] = successful_results_numeric['accuracy'].astype(float)
        successful_results_numeric['bootstrap_mean'] = successful_results_numeric['bootstrap_mean'].astype(float)
        successful_results_numeric['variance'] = successful_results_numeric['variance'].astype(float)
        
        print(f"✅ 成功分析文件数: {len(successful_results)}")
        print(f"📈 平均准确率: {successful_results_numeric['accuracy'].mean():.4f}")
        print(f"📊 准确率标准差: {successful_results_numeric['accuracy'].std():.4f}")
        print(f"🎯 最高准确率: {successful_results_numeric['accuracy'].max():.4f}")
        print(f"🎯 最低准确率: {successful_results_numeric['accuracy'].min():.4f}")
        print(f"📐 平均方差: {successful_results_numeric['variance'].mean():.6f}")
        
        # 显示前5个最佳结果
        print(f"\n🏆 准确率前5名:")
        top5 = successful_results_numeric.nlargest(5, 'accuracy')
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"   {i}. {row['file_name']}: {row['accuracy']:.4f}")
    
    error_count = len(df[df['status'] != 'success'])
    if error_count > 0:
        print(f"\n❌ 分析失败文件数: {error_count}")
    
    print(f"\n💾 结果已保存到:")
    print(f"   CSV: {output_file}")
    print(f"   Excel: {excel_file}")
    
    # 显示详细结果表格
    print(f"\n📋 详细结果表格:")
    print(df.to_string(index=False, max_rows=20))
    
    if len(df) > 20:
        print(f"... 显示前20行，完整结果请查看保存的文件")


if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    main()