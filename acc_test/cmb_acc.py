#!/usr/bin/env python3
"""
计算CMB数据集的正确率
"""
import json
import re


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


def calculate_accuracy(result_file_path):
    """计算CMB测试结果的正确率"""
    cnt = 0
    correct = 0
    error_cases = []
    no_pred_cases = []  # 存储没有提取到答案的案例ID
    
    print("📊 开始计算正确率...")
    print(f"📁 结果文件: {result_file_path}")
    
    with open(result_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                cnt += 1
                
                # 跳过错误样本
                if (data.get('final_answer') == 'error' or 
                        not data.get('final_answer')):
                    print(f"⚠️  样本 {data.get('id', cnt)} 处理错误，跳过")
                    continue
                
                content = data['final_answer']
                
                # 处理选项数据 - 支持字符串格式和字典格式
                if 'options' in data:
                    options_dict = {k: v for k, v in data['options'].items() 
                                    if v and v.strip() != ""}
                elif 'option' in data:
                    import ast
                    if isinstance(data['option'], str):
                        options_dict = ast.literal_eval(data['option'])
                    else:
                        options_dict = data['option']
                else:
                    options_dict = {}
                
                # 提取预测答案
                pred = data.get('extracted_choice', '')
                if not pred:
                    pred = match_choice(content, options_dict)
                
                # 获取正确答案 - 支持不同的键名
                answer = data.get('answer_idx', '') or data.get('answer', '')
                
                # 检查是否正确
                if answer == pred:
                    correct += 1
                else:
                    # 记录错误案例
                    error_cases.append({
                        'id': data.get('id', cnt),
                        'question': data.get('question', '')[:100] + '...',
                        'correct_answer': answer,
                        'predicted_answer': pred,
                        'final_answer': content[:200] + '...'
                    })
                
                if not pred:
                    print(f"⚠️  样本 {data.get('id', cnt)} 无法提取答案:")
                    print(f"   {content[:100]}...")
                    print()
                    # 记录没有提取到答案的案例
                    no_pred_cases.append(data.get('id', cnt))
    
    # 输出统计结果
    print("\n" + "="*50)
    print("📊 CMB测试结果统计")
    print("="*50)
    print(f"总样本数: {cnt}")
    print(f"正确数量: {correct}")
    print(f"正确率: {correct/cnt:.4f} ({correct/cnt*100:.2f}%)")
    
    # 输出没有提取到答案的统计
    no_pred_count = len(no_pred_cases)
    print(f"无法提取答案数量: {no_pred_count}")
    if no_pred_count > 0:
        print(f"无法提取答案的ID: {', '.join(map(str, no_pred_cases))}")
    print("="*50)
    
    # 输出部分错误案例
    if error_cases:
        print("\n❌ 前5个错误案例:")
        for i, case in enumerate(error_cases[:5]):
            print(f"\n{i+1}. ID: {case['id']}")
            print(f"   问题: {case['question']}")
            print(f"   正确答案: {case['correct_answer']}")
            print(f"   预测答案: {case['predicted_answer']}")
            print(f"   模型回答: {case['final_answer']}")
    
    return correct/cnt if cnt > 0 else 0


if __name__ == "__main__":
    # CMB结果文件路径
    result_file = "/home/xiaobei/Agentic-RAG-R1__under-construction/acc_testnew/cmb_result/llma3bcmb2.jsonl"
    
    try:
        accuracy = calculate_accuracy(result_file)
        print(f"\n🎯 最终准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到结果文件 {result_file}")
        print("请先运行测试脚本生成结果文件")
    except Exception as e:
        print(f"❌ 计算过程中出错: {e}") 