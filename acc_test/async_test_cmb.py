#!/usr/bin/env python3
"""
使用CMB数据集和中文prompt测试GRPO训练后的医生模型
"""
from call_llm import call_gpt
from openai import OpenAI
import os
import json
from multiprocessing import Pool, Value, Lock, Manager
from tqdm import tqdm
import re
from collections import defaultdict
from prompts import *
import time

# 患者模型配置 (保持原有配置)
patient_client = OpenAI(
    api_key="8cefb70606f3472d8731bd65661ce409",
    base_url="http://8289.model.mingxingtech.com:10032/v1"
)
patient_model = 'qwen2.5:72b'

# 医生模型配置 (使用本地vLLM API)
doctor_client = OpenAI(
    api_key="EMPTY",  # vLLM不需要真实的API key
    base_url="http://localhost:8702/v1"
)
doctor_model = "qwen3_1.7b"  # 修改为1.7b模型


# Define the function to process data
def process_data(data):
    # 修改：处理没有context字段的情况
    initial_info = ""
    if 'context' in data:
        if isinstance(data['context'], list) and len(data['context']) > 0:
            initial_info = data['context'][0]
        elif isinstance(data['context'], str):
            # Assuming sentences are separated by periods, taking the first sentence
            initial_info = data['context'].split(". ")[0]
    elif 'facts' in data:
        # 如果有facts字段，使用facts的前几条作为初始信息
        initial_info = "\n".join(data['facts'][:3]) if isinstance(data['facts'], list) else ""
    
    partial_question = initial_info + '\n' + data['question']
    
    # 处理选项数据 - 支持不同的键名
    options_dict = {}
    if 'options' in data:
        options_dict = data['options']
    elif 'option' in data:
        options_dict = data['option']
    
    option_str = "\n".join([f"{key}: {value}" for key, value in options_dict.items()])
    doctor_prompt = doctor_system_prompt.format(question_type='multiple choice question', question=partial_question,
                                                option_str=option_str)

    # 使用atomic_facts字段，如果不存在则使用facts字段
    atomic_facts = data.get('atomic_facts', data.get('facts', []))
    atomic_facts_str = '\n'.join(atomic_facts) if isinstance(atomic_facts, list) else str(atomic_facts)
    patient_prompt = patient_system_prompt.format(atomic_facts=atomic_facts_str)

    doctor_messages = [{'role': 'user', 'content': doctor_prompt}]
    patient_messages = [{'role': 'system', 'content': patient_prompt}]
    flag = 0

    for i in range(10):
        # 你需要根据自己的情况进行 API 调用
        doctor_question = call_gpt(doctor_client, doctor_model, doctor_messages)  # 同步调用
        if not doctor_question or '!model error:' in doctor_question:
            print(doctor_question)
            flag = 1
            break

        doctor_messages.append({'role': 'assistant', 'content': doctor_question})
        if 'answer:' in doctor_question:
            data['final_answer'] = doctor_question
            break
        patient_messages.append({'role': 'user', 'content': doctor_question})
        patient_reply = call_gpt(patient_client, patient_model, patient_messages)  # 同步调用
        if '!model error' in patient_reply:
            print('patient error:'+patient_reply)
            flag = 1
            break
        doctor_messages.append({'role': 'user', 'content': patient_reply})
        patient_messages.append({'role': 'assistant', 'content': patient_reply})

    if "final_answer" not in data.keys():
        data["final_answer"] = ""
        if flag:
            data["final_answer"] = "error"

    data["dialogue"] = doctor_messages

    return data


def worker(args):
    data, out_path, lock = args

    result = process_data(data)

    if result['final_answer']=='error':
        return False
    with lock:
        with open(out_path, 'a', encoding='utf-8') as outfile:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
    return True


if __name__ == "__main__":
    out_path = "cmb_result/qwen3_1.7b_cmb.jsonl"  # 修改输出文件名，使用模型名称

    # 修改JSON加载方式：从逐行加载改为整体加载JSON数组
    with open("/home/xiaobei/Agentic-RAG-R1__under-construction/src/data/cmb_atomic_patient_test.json", 'r', encoding='utf-8') as f:
        datas = json.load(f)  # 整体加载JSON数组

    test_datas = []
    continue_id = set()
    if os.path.exists(out_path):
        with open(out_path, "r+", encoding='utf-8') as f:
            out_data = f.readlines()
            for line in out_data:
                line = json.loads(line)
                continue_id.add(line['id'])

    for data in datas:
        if data['id'] in continue_id:
            continue
        else:
            test_datas.append(data)

    manager = Manager()
    lock = manager.Lock()

    with Pool(200) as p:
        # Use imap_unordered to ensure it handles the data without strict ordering, allowing for better parallelism
        for result in tqdm(p.imap_unordered(worker, [(data, out_path, lock) for data in test_datas]),
                           total=len(test_datas)):
            if not result:
                print("Error in processing data")
                with open('log.txt', 'a') as log_file:
                    log_file.write("Error during processing\n")