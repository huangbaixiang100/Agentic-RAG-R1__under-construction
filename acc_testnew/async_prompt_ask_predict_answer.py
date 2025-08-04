from call_llm import call_gpt
from openai import OpenAI
import os
import json
import random
from multiprocessing import Pool, Value, Lock, Manager
from tqdm import tqdm
import re
from collections import defaultdict
#from utils import *
from prompts import *
import numpy as np
import time
SEED = 81  # 你可以根据需求更换种子值

random.seed(SEED)
np.random.seed(SEED)

patient_client = OpenAI(api_key="8cefb70606f3472d8731bd65661ce409",base_url="http://8289.model.mingxingtech.com:10032/v1")
patient_model='qwen2.5:72b'
#doctor_client = OpenAI(api_key="8cefb70606f3472d8731bd65661ce409",base_url="http://8289.model.mingxingtech.com:10032/v1")
#doctor_model='qwen2.5:72b'
doctor_client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8703/v1")
doctor_model = 'Qwen3_8b'  # 使用本地部署的模型名称
#doctor_client = OpenAI(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjZmMGE2YWM2LTc1ZjMtNDg4MS1hMmM0LTJhMmVkNTM5OTQ3MyJ9.VorTaQQtldDKfgAHC7FYuS9T3p4-9N1YCGO6GB98E6k", base_url="http://162.105.88.35:3000/api") #deepseek-r1-local
#doctor_model="deepseek-r1-64k-local"
#dataset_name='medqa'
dataset_name='cmb'


# Define the function to process data
def process_data(data):
    # 添加重试机制的装饰器
    max_retries = 3
    retry_delay = 2  # 秒
    
    if dataset_name=='medqa':
        option_str = "\n".join([f"{key}: {value}" for key, value in data['options'].items()])
        question_type='multiple choice question'
        if isinstance(data['context'], list) and len(data['context']) > 0:
            initial_info = data['context'][0]
        elif isinstance(data['context'], str):
            # Assuming sentences are separated by periods, taking the first sentence
            initial_info = data['context'].split(". ")[0]
        else:
            initial_info = ""  # Default fallback
        partial_question = initial_info + '\n' + data['atomic_question']
    else:
        option_str = "\n".join([f"{key}: {value}" for key, value in data['option'].items()])
        question_type=data['question_type']
        partial_question = '，'.join(data['facts'][:int(len(data['facts']) / 2)]) + '。' + data['atomic_question']

    doctor_prompt = doctor_system_prompt.format(question_type=question_type, question=partial_question,
                                                option_str=option_str)

    patient_prompt = patient_system_prompt.format(atomic_facts='\n'.join(data['facts']))

    doctor_messages = [{'role': 'user', 'content': doctor_prompt}]
    patient_messages = [{'role': 'system', 'content': patient_prompt}]
    flag = 0

    for i in range(10):
        # 添加重试机制
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 在重试之间添加延迟
                if retry_count > 0:
                    time.sleep(retry_delay)
                
                doctor_question = call_gpt(doctor_client, doctor_model, doctor_messages)  # 同步调用
                
                if not doctor_question:
                    raise Exception("Empty response")
                
                if 'model error:' in doctor_question:
                    print(f"Attempt {retry_count + 1}/{max_retries}: {doctor_question}")
                    retry_count += 1
                    continue
                
                # 如果成功获取到响应，跳出重试循环
                break
                
            except Exception as e:
                print(f"Attempt {retry_count + 1}/{max_retries}: Error - {str(e)}")
                retry_count += 1
        
        # 如果所有重试都失败了
        if retry_count >= max_retries:
            print("All retry attempts failed")
            flag = 1
            break

        doctor_messages.append({'role': 'assistant', 'content': doctor_question})
        if 'answer:' in doctor_question:
            data['final_answer'] = doctor_question
            break
        patient_messages.append({'role': 'user', 'content': doctor_question})
        # 对patient模型的调用也添加重试机制
        retry_count = 0
        while retry_count < max_retries:
            try:
                if retry_count > 0:
                    time.sleep(retry_delay)
                
                patient_reply = call_gpt(patient_client, patient_model, patient_messages)  # 同步调用
                
                if not patient_reply:
                    raise Exception("Empty response")
                
                if 'error' in patient_reply:
                    print(f"Patient model attempt {retry_count + 1}/{max_retries}: {patient_reply}")
                    retry_count += 1
                    continue
                
                break
                
            except Exception as e:
                print(f"Patient model attempt {retry_count + 1}/{max_retries}: Error - {str(e)}")
                retry_count += 1
        
        if retry_count >= max_retries:
            print("All patient model retry attempts failed")
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
    out_path = "cmb_result/llama3b_cmb_promednew.jsonl"  # 设置输出路径

    file_name = 'dataset/cmb_atomic_patient_test.json'
    with open(file_name, 'r', encoding='utf-8') as f:
        if 'jsonl' in file_name:
            datas = []
            for line in f:
                datas.append(json.loads(line))
        else:
            datas = json.load(f)

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

    # 减少并发进程数到5个
    with Pool(5) as p:
        # 使用imap而不是imap_unordered来保持顺序并减少竞争
        for result in tqdm(p.imap(worker, [(data, out_path, lock) for data in test_datas], chunksize=1),
                          total=len(test_datas)):
            if not result:
                print("Error in processing data")
                with open('log.txt', 'a') as log_file:
                    log_file.write("Error during processing\n")