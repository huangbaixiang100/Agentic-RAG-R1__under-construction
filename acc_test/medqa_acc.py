import json
import re

from collections import Counter, defaultdict

def match_choice(text,options_dict):
    option = ["A", "B", "C", "D", "E", "F", "G"]
    #res = re.search(r"(answer: |答案|正确选项)(?:是|：|为|应该是|应该为)(.*?)(。|\.|$)", text, re.S)
    res = re.search(r"(answer: |答案|正确选项)(?:是|：|为|应该是|应该为)\s*(.*)", text, re.S) #(.*?)(。|\.|$)
    pattern = r"(?:正确答案|answer|正确选项)[：:是为应该是应该为\s]*[【]?\s*([A-Ga-g]{1,7})\s*[】]?"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        # 多个匹配只取第一个；去重排序标准化
        answer = matches[0].upper()
        answer = "".join(sorted(set(answer)))
        if res:
            res_answer="".join([x for x in res.group(2) if x in option])
            #if res_answer!= answer:
                #print(text)
                #print(answer,res_answer)
                #print('*'*30)
        return answer
    else:
        tmp=[]
        for op_letter, op_text in options_dict.items():
            if op_text in text:
                #print(f"Found {op_letter}:{op_text}")
                tmp.append(op_letter)
        return "".join(tmp)
    return "".join([i for i in text if i in option])



datas=[]
cnt=0
correct=0
with open('/home/xiaobei/Agentic-RAG-R1__under-construction/acc_testnew/medqa_result/llama8b_grpo_lastlast.jsonl',encoding='utf-8')as f:
    for line in f:
        data=json.loads(line)
        cnt+=1
        content=data['final_answer']
        options_dict = {k: v for k, v in data['options'].items() if v.strip() != ""}
        pred = match_choice(content, options_dict)
        if not pred:
            print(content)
        answer = data['answer_idx']
        if answer==pred:
            correct+=1

print(cnt)
print(correct)
print(correct/cnt)
