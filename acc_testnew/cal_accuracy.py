import json
import re
from collections import defaultdict
from utils import *

def cal_cmb_accuracy(modelans_path):
    lengths=[]
    llmanswers=[]
    cnt=0
    with open(modelans_path, 'r', encoding='utf-8') as f:
        #llmanswers=json.load(f)
        for line in f:
            #print(cnt)
            llmans=json.loads(line)
            cnt += 1
            if 'answer' not in llmans.keys():
                continue
            else:
                llmanswers.append(json.loads(line))
    print(len(llmanswers))
    
    exam_main_key_set = set()
    for item in llmanswers:
        exam_main_key_set.add(item['exam_type'])

    #full_correct_answer_count  = {}
    partial_correct_answer_count = {}
    all_answer_count = {}
    for key in exam_main_key_set:
        #full_correct_answer_count[key] = 0
        partial_correct_answer_count[key] = 0
        all_answer_count[key] = 0
    cnt=0
    correct_cnt = 0
    
    # 添加一个计数器来记录已打印的错误多选题数量
    wrong_multi_printed = 0
    print("\n前5个错误的多选题示例：")
    print("-" * 50)
    
    for llmans in llmanswers:
        lengths.append(len(llmans['dialogue']))
        ty = llmans["question_type"]

        ress = defaultdict(int)
        options_dict={key:value for key, value in llmans['option'].items() if value!=""}
        choice = match_choice(llmans['final_answer'],options_dict)
        if choice=="":
            cnt+=1
            print(llmans["final_answer"])
        if len(choice) > 1 and ty != "多项选择题":
            choice = choice[0]
        if len(choice) > 0:
            ress[choice] += 1
        if len(ress) > 0:
            partial_ans = sorted(ress.items(), key=lambda x: x[1], reverse=True)[0][0]
        else:
            partial_ans = ""

        # 打印错误的多选题
        if ty == "多项选择题" and partial_ans != llmans['answer'] and wrong_multi_printed < 5:
            print(f"问题 {wrong_multi_printed + 1}:")
            print(f"题目: {llmans['question']}")
            print("选项:")
            for k, v in llmans['option'].items():
                print(f"{k}: {v}")
            print(f"正确答案: {llmans['answer']}")
            print(f"模型回答: {llmans['final_answer']}")
            print(f"提取的答案: {partial_ans}")
            print("-" * 50)
            wrong_multi_printed += 1

        all_answer_count[llmans['exam_type']] +=1
        option_str = "\n".join([f"{key}: {value}" for key, value in llmans['option'].items()])
        if option_str== '':
            print(llmans['id'])
            all_answer_count[llmans['exam_type']] -=1
        else:
            if partial_ans == llmans['answer']:
                partial_correct_answer_count[llmans['exam_type']] +=1
                correct_cnt+=1

    print('correct',correct_cnt)
    print(max(lengths),len(lengths))
    print((sum(lengths) / len(lengths)))
            
    print('all',cnt)
    print('accuracy',correct_cnt/len(lengths))
    #full_accuray_dict = {}
    partial_accuray_dict = {}
    #full_average_accuracy = 0.0
    partial_average_accuracy = 0.0
    for key in exam_main_key_set:
        #full_accuray_dict[key]=full_correct_answer_count[key] / all_answer_count[key]
        partial_accuray_dict[key]=partial_correct_answer_count[key]/all_answer_count[key]
        #full_average_accuracy += full_accuray_dict[key]
        partial_average_accuracy +=partial_accuray_dict[key]
    #full_average_accuracy /= len(exam_main_key_set)
    #full_accuray_dict['average_accuracy'] = full_average_accuracy
    partial_average_accuracy /= len(exam_main_key_set)
    partial_accuray_dict['average_accuracy'] = partial_average_accuracy
    
    # 分别统计单选和多选正确率
    single_total_count = 0
    #full_single_right_count = 0
    partial_single_right_count = 0
    multi_total_count = 0
    #full_multi_right_count = 0
    partial_multi_right_count = 0
    
    for llmans in llmanswers:
        if llmans['question_type'] == '多项选择题':
            multi_total_count+=1
            #if llmans['answer'] == full_ans:
            #    full_multi_right_count+=1
            if llmans['answer'] == partial_ans:
                partial_multi_right_count+=1
        else:
            single_total_count+=1
            #if llmans['answer'] == full_ans:
            #    full_single_right_count+=1
            if llmans['answer'] == partial_ans:
                partial_single_right_count+=1
    
    #full_accuray_dict['single_acc'] = full_single_right_count/single_total_count
    #full_accuray_dict['multi_acc'] = full_multi_right_count/multi_total_count
    partial_accuray_dict['single_acc'] =partial_single_right_count/single_total_count
    partial_accuray_dict['multi_acc'] =partial_multi_right_count/multi_total_count

    return {},partial_accuray_dict

modelans_path = ("cmb_result/llma3bcmb1.jsonl")
full_accuray_dict,partial_accuracy_dict = cal_cmb_accuracy(modelans_path)
print("partial:",partial_accuracy_dict)
