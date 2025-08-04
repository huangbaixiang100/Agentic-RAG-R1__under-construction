import re
import json
from collections import defaultdict

def match_choice(text,options_dict):
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


# def match_choice(text, options_dict):
#     option = ["A", "B", "C", "D", "E", "F", "G"]
    
#     # 专门处理【正确答案是...】格式
#     pattern_bracket = r"【正确答案是([A-Ga-g](?:[和、]?[A-Ga-g])*)】"
#     matches_bracket = re.findall(pattern_bracket, text, re.IGNORECASE)
    
#     if matches_bracket:
#         # 提取所有字母，忽略分隔符
#         answer_text = matches_bracket[0]
#         answer = "".join([c for c in answer_text.upper() if c in option])
#         answer = "".join(sorted(set(answer)))
#         return answer
    
#     # 首先尝试匹配标准格式的答案
#     pattern = r"(?:正确答案|答案|正确选项|answer)[：:\s]*(?:is|是|为|应该是|应该为)?\s*[【]?\s*([A-Ga-g]{1,7})\s*[】]?"
#     matches = re.findall(pattern, text, re.IGNORECASE)
    
#     if matches:
#         # 多个匹配只取第一个；去重排序标准化
#         answer = matches[0].upper()
#         answer = "".join(sorted(set(answer)))
#         return answer
    
#     # 如果没有匹配到标准格式，尝试匹配中文分隔的格式
#     pattern_cn = r"(?:正确答案|答案|正确选项|answer)[：:\s]*(?:is|是|为|应该是|应该为)?\s*([A-Ga-g](?:[和、\s]*[A-Ga-g])*)"
#     matches_cn = re.findall(pattern_cn, text, re.IGNORECASE)
    
#     if matches_cn:
#         # 提取所有字母，忽略分隔符
#         answer = "".join([c for c in matches_cn[0].upper() if c in option])
#         answer = "".join(sorted(set(answer)))
#         return answer
    
#     # # 如果还是没有匹配到，尝试在文本中查找选项内容
#     # tmp = []
#     # for op_letter, op_text in options_dict.items():
#     #     if op_text in text:
#     #         tmp.append(op_letter)
#     # if tmp:
#         return "".join(sorted(set(tmp)))
    
#     # 最后的后备方案：直接提取所有字母
#     return "".join(sorted(set([i.upper() for i in text if i.upper() in option])))
