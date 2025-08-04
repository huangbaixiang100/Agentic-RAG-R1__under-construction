import json
import re
from collections import defaultdict
from utils import *

def analyze_no_answer_questions(modelans_path):
    """分析没有提取到答案的题目"""
    
    no_answer_questions = []  # 存储没有答案的题目
    empty_choice_questions = []  # 存储choice为空的题目
    all_questions = []
    
    print(f"正在分析文件: {modelans_path}")
    print("=" * 60)
    
    with open(modelans_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                llmans = json.loads(line)
                all_questions.append(llmans)
                
                # 检查是否有final_answer字段
                if 'final_answer' not in llmans or llmans['final_answer'] == "":
                    no_answer_questions.append({
                        'id': llmans.get('id', f'line_{line_num}'),
                        'line_num': line_num,
                        'reason': 'no_final_answer',
                        'question': llmans.get('question', 'N/A')[:100] + '...' if len(llmans.get('question', '')) > 100 else llmans.get('question', 'N/A'),
                        'exam_type': llmans.get('exam_type', 'N/A'),
                        'question_type': llmans.get('question_type', 'N/A')
                    })
                    continue
                
                # 检查choice提取是否为空
                if 'option' in llmans:
                    options_dict = {key: value for key, value in llmans['option'].items() if value != ""}
                    choice = match_choice(llmans['final_answer'], options_dict)
                    
                    if choice == "":
                        empty_choice_questions.append({
                            'id': llmans.get('id', f'line_{line_num}'),
                            'line_num': line_num,
                            'reason': 'empty_choice',
                            'final_answer': llmans['final_answer'][:200] + '...' if len(llmans['final_answer']) > 200 else llmans['final_answer'],
                            'question': llmans.get('question', 'N/A')[:100] + '...' if len(llmans.get('question', '')) > 100 else llmans.get('question', 'N/A'),
                            'exam_type': llmans.get('exam_type', 'N/A'),
                            'question_type': llmans.get('question_type', 'N/A'),
                            'options': options_dict
                        })
                        
            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析错误: {e}")
                continue
    
    # 打印统计结果
    print(f"📊 统计结果:")
    print(f"   总题目数量: {len(all_questions)}")
    print(f"   没有final_answer的题目: {len(no_answer_questions)}")
    print(f"   choice提取为空的题目: {len(empty_choice_questions)}")
    print(f"   总计有问题的题目: {len(no_answer_questions) + len(empty_choice_questions)}")
    print(f"   问题题目占比: {(len(no_answer_questions) + len(empty_choice_questions)) / len(all_questions) * 100:.2f}%")
    
    # 详细列出没有final_answer的题目
    if no_answer_questions:
        print(f"\n❌ 没有final_answer的题目 ({len(no_answer_questions)}个):")
        print("-" * 60)
        for i, q in enumerate(no_answer_questions[:10], 1):  # 只显示前10个
            print(f"{i}. ID: {q['id']} (第{q['line_num']}行)")
            print(f"   考试类型: {q['exam_type']} | 题型: {q['question_type']}")
            print(f"   题目: {q['question']}")
            print()
        
        if len(no_answer_questions) > 10:
            print(f"   ... 还有 {len(no_answer_questions) - 10} 个题目")
    
    # 详细列出choice为空的题目
    if empty_choice_questions:
        print(f"\n⚠️  choice提取为空的题目 ({len(empty_choice_questions)}个):")
        print("-" * 60)
        for i, q in enumerate(empty_choice_questions[:10], 1):  # 只显示前10个
            print(f"{i}. ID: {q['id']} (第{q['line_num']}行)")
            print(f"   考试类型: {q['exam_type']} | 题型: {q['question_type']}")
            print(f"   题目: {q['question']}")
            print(f"   模型回答: {q['final_answer']}")
            print(f"   选项: {q['options']}")
            print()
        
        if len(empty_choice_questions) > 10:
            print(f"   ... 还有 {len(empty_choice_questions) - 10} 个题目")
    
    # 按考试类型分组统计
    print(f"\n📈 按考试类型分组统计:")
    print("-" * 60)
    
    exam_stats = defaultdict(lambda: {'no_answer': 0, 'empty_choice': 0, 'total': 0})
    
    for q in all_questions:
        exam_type = q.get('exam_type', 'Unknown')
        exam_stats[exam_type]['total'] += 1
    
    for q in no_answer_questions:
        exam_stats[q['exam_type']]['no_answer'] += 1
    
    for q in empty_choice_questions:
        exam_stats[q['exam_type']]['empty_choice'] += 1
    
    for exam_type, stats in exam_stats.items():
        problem_count = stats['no_answer'] + stats['empty_choice']
        problem_rate = problem_count / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{exam_type}:")
        print(f"   总数: {stats['total']}, 有问题: {problem_count} ({problem_rate:.2f}%)")
        print(f"   - 没有答案: {stats['no_answer']}")
        print(f"   - 提取失败: {stats['empty_choice']}")
    
    # 返回详细信息
    return {
        'total_questions': len(all_questions),
        'no_answer_questions': no_answer_questions,
        'empty_choice_questions': empty_choice_questions,
        'exam_stats': dict(exam_stats)
    }

if __name__ == "__main__":
    # 分析CMB结果文件
    modelans_path = "/home/xiaobei/Agentic-RAG-R1__under-construction/acc_testnew/cmb_result/promed4.jsonl"
    
    print("🔍 分析CMB测试结果中的问题题目")
    print("=" * 60)
    
    try:
        result = analyze_no_answer_questions(modelans_path)
        
        # 生成问题题目ID列表文件
        problem_ids = []
        problem_ids.extend([q['id'] for q in result['no_answer_questions']])
        problem_ids.extend([q['id'] for q in result['empty_choice_questions']])
        
        if problem_ids:
            with open('problem_question_ids.txt', 'w', encoding='utf-8') as f:
                f.write("问题题目ID列表:\n")
                f.write("=" * 30 + "\n")
                for i, qid in enumerate(problem_ids, 1):
                    f.write(f"{i}. {qid}\n")
            
            print(f"\n💾 问题题目ID已保存到: problem_question_ids.txt")
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {modelans_path}")
        print("请确认文件路径是否正确")
    except Exception as e:
        print(f"❌ 错误: {e}") 