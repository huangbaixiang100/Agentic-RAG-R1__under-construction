import re
import json
from collections import defaultdict

def match_choice_(text,options_dict):
    option = ["A", "B", "C", "D", "E", "F", "G"]
    res = re.search(r"(answer: |答案|正确选项)(?:是|：|为|应该是|应该为)(.*?)(。|\.|$)", text, re.S)
    #res = re.search(r"(answer: |答案|正确选项)(?:是|：|:|为|应该是|应该为)\s*(.*)", text, re.S) #(.*?)(。|\.|$)
    #res = re.search(r"(?:answer|答案|正确答案|正确选项)[：:是为应该是应该为\s]*[【]?\s*([A-Fa-f]{1,6})\s*[】]?", text,
    #                re.IGNORECASE)
    if res:
        #print(res)
        #print(res.group(2))
        #print("".join([x for x in res.group(2) if x in option]))
        return "".join([x for x in res.group(2) if x in option])
    else:
        tmp=[]
        for op_letter, op_text in options_dict.items():
            if op_text in text:
                print(f"Found {op_letter}:{op_text}")
                tmp.append(op_letter)
        return "".join(tmp)
    return "".join([i for i in text if i in option])


def match_choice(text,options_dict):
    option = ["A", "B", "C", "D", "E", "F", "G"]
    #res = re.search(r"(answer: |答案|正确选项)(?:是|：|为|应该是|应该为)(.*?)(。|\.|$)", text, re.S)
    res = re.search(r"(answer: |答案|正确选项)(?:是|：|为|应该是|应该为)\s*(.*)", text, re.S) #(.*?)(。|\.|$)
    pattern = r"(?:正确答案|answer|正确选项)[：:是为应该是应该为\s]*[【]?\s*([A-Ga-g]{1,7})\s*[】]?"
    pattern = r"(?:正确答案|答案|正确选项|answer)[：:\s]*(?:is|是|为|应该是|应该为)?\s*[【]?\s*([A-Ga-g]{1,7})\s*[】]?"

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
