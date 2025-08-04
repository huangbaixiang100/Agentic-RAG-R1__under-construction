from call_llm import call_gpt
from openai import OpenAI
import os
import json
from multiprocessing import Pool, Value, Lock, Manager
from tqdm import tqdm
import re
from collections import defaultdict
#from utils import *
from prompts import *
import time


patient_client = OpenAI(api_key="8cefb70606f3472d8731bd65661ce409",base_url="http://8289.model.mingxingtech.com:10032/v1")
patient_model='qwen2.5:72b'
#doctor_client = OpenAI(api_key="8cefb70606f3472d8731bd65661ce409",base_url="http://8289.model.mingxingtech.com:10032/v1")
#doctor_model='qwen2.5:72b'
doctor_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8702/v1")
doctor_model = 'Qwen3_8b'  # 使用本地部署的模型名称
#doctor_client = OpenAI(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjZmMGE2YWM2LTc1ZjMtNDg4MS1hMmM0LTJhMmVkNTM5OTQ3MyJ9.VorTaQQtldDKfgAHC7FYuS9T3p4-9N1YCGO6GB98E6k", base_url="http://162.105.88.35:3000/api") #deepseek-r1-local
#doctor_model="deepseek-r1-64k-local"

# Define the function to process data
def process_data(data):
    if isinstance(data['context'], list) and len(data['context']) > 0:
        initial_info = data['context'][0]
    elif isinstance(data['context'], str):
        # Assuming sentences are separated by periods, taking the first sentence
        initial_info = data['context'].split(". ")[0]
    else:
        initial_info = ""  # Default fallback
    partial_question = initial_info +'\n' +data['question']
    option_str = "\n".join([f"{key}: {value}" for key, value in data['options'].items()])
    doctor_prompt = doctor_system_prompt_en.format(question_type='multiple choice question', question=partial_question,
                                                option_str=option_str)

    patient_prompt = patient_system_prompt_en.format(atomic_facts='\n'.join(data['atomic_facts']))

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
    out_path = "/home/xiaobei/Agentic-RAG-R1__under-construction/acc_testnew/medqa_result/llama8b_grpo_lastlast.jsonl"  # 设置输出路径

    datas=[]
    with open("dataset/medqa_test_convo.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            datas.append(json.loads(line))

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