import requests
import json


class InputLengthError(requests.RequestException):
    """The length of input exceeds the max length"""


class InvalidKeyError(requests .RequestException):
    """The key is invalid."""


def call_model(message,url='http://0.0.0.0:8001/v1/chat/completions'):
    headers = {
        "Content-Type": "application/json"
    }
    if type(message) is str:
        messages=[{ "role": "user","content": message}]
    elif type(message) is list:
        messages=message
    else:
        return (f"error: type of message must be string or list")
    data = json.dumps({
        "model": "qwen2.5",
        "messages": messages
    })

    try:
        raw_response = requests.post(url, headers=headers, data=data)
        raw_response.raise_for_status()  # Raises stored HTTPError, if one occurred.

        res = json.loads(raw_response.text)
        words = res['choices'][0]['message']['content']
        #print(words)
        return words
    except requests.HTTPError as e:
        if e.response.status_code in (400, 401):
            # Handle specific error codes if needed
            error_info = e.response.json()
            error_message = error_info.get('error', {}).get('message', 'Unknown error')
            return f"error: {error_message}"
        else:
            return "error: Unexpected error"
    except Exception as e:
        return f"error: {str(e)}"


def call_gpt(client,model,messages,temperature=0.3):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False
        )
        words=response.choices[0].message.content
        print(words)
        return words
    except requests.HTTPError as e:
        if e.response.status_code in (400, 401):
            # Handle specific error codes if needed
            error_info = e.response.json()
            error_message = error_info.get('error', {}).get('message', 'Unknown error')
            return f"{model} !model error: {error_message}"
        else:
            return f"{model} !model error: Unexpected error"
    except Exception as e:
        return f"{model} !model error: {str(e)}"


def call_qwq_stream(client,model,messages):
    try:
        reasoning_content=""
        answer_content=""
        is_answering = False
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        for chunk in completion:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # 打印思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    #print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                else:
                    # 开始回复
                    if delta.content != "" and is_answering is False:
                        #print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True
                    # 打印回复过程
                    #print(delta.content, end='', flush=True)
                    answer_content += delta.content
        print(reasoning_content,answer_content)
        return reasoning_content,answer_content
    except requests.HTTPError as e:
        if e.response.status_code in (400, 401):
            # Handle specific error codes if needed
            error_info = e.response.json()
            error_message = error_info.get('error', {}).get('message', 'Unknown error')
            return f"{model} error: {error_message}"
        else:
            return f"{model} error: Unexpected error"
    except Exception as e:
        return "error",f"{model} error: {str(e)}"


#通过中转接口调用gpt-4
def call_gpt_mid(model,messages):
    # 使用中转链接可以和特定的API可以不必向openai发起请求，且请求无须魔法
    # 调用方式与openai官网一致，仅需修改baseurl
    Baseurl = "https://api.claudeshop.top"
    Skey = "sk-lhPEYmEdLU7mNtxBZxVYTclqeNJzPH0HfNbcb53Hd1X6cU3A"
    payload = json.dumps({
        "model": model,#"gpt-4o-2024-05-13",
        "messages": messages
    })
    url = Baseurl + "/v1/chat/completions"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {Skey}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    # 解析 JSON 数据为 Python 字典
    data = response.json()
    # 获取 content 字段的值
    content = data
    answer=content['choices'][0]['message']['content']
    print(answer)
    return answer