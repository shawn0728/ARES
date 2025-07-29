from tqdm import tqdm
import os
import requests
# from PIL import Image
import base64
import uuid
import json
import time
import re
from openai import OpenAI
import openai

# 保存原始的 __init__ 方法
original_init_func = tqdm.__init__

def custom_tqdm_init(self, *args, **kwargs):
    kwargs.setdefault('ncols', 135) # 修改 tqdm 的默认进度条长度
    # kwargs.setdefault('disable', True)
    original_init_func(self, *args, **kwargs)

# 用自定义的方法替换 tqdm 的 __init__ 方法
tqdm.__init__ = custom_tqdm_init

def _get_xhs_openai_url():
    def can_connect_to_url(url, timeout=5):
        try:
            requests.head(url, timeout=timeout, allow_redirects=True)
            return True
        except requests.RequestException as e:
            return False
        
    openai_url = os.environ.get("XHS_OPENAI_URL", "")
    if not openai_url:
        for url in ["http://agi-serving.devops.xiaohongshu.com/eval/v1", "http://agi-serving.devops.rednote.life/eval/v1"]:
            if can_connect_to_url(url, timeout=5):
                return url
        raise ValueError(
            '连接不上 "http://agi-serving.devops.xiaohongshu.com/eval/v1" 或者 "http://agi-serving.devops.rednote.life/eval/v1"，'
            '请设置环境变量 XHS_OPENAI_URL="xxxxxxx"')
    else:
        return openai_url

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def is_base64_encoded(s):
    # Check if the string is a valid Base64 encoded string
    base64_pattern = re.compile(r'^[A-Za-z0-9+/=]+$')
    
    if not base64_pattern.match(s):
        return False
    
    try:
        # Check if the string can be decoded as Base64
        # It should not raise an exception
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False

def invoke_by_base64(base64_image, prompt, max_tokens=512, temperature=None, max_retries=10):

    if not is_base64_encoded(base64_image):
        base64_image = encode_image(base64_image)

    client = OpenAI(api_key="demo_api_1", base_url=_get_xhs_openai_url()) 
    response = 200
    result = None
    for attempt in range(max_retries):
        print(f"===> 调用GPT-4o中 ... (尝试 {attempt + 1}/{max_retries})")
        try:
            completion = client.chat.completions.create(
                model = "gpt4o-ptu-agi-offline-shiqing",
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                                "finish_reason": "stop",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    #"detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens = max_tokens,
                temperature = temperature if temperature is not None else None
            )

            result = completion.choices[0].message.content[0]['text']
            return response, result
        except Exception as e:
            print(f'Failed to invoke GPT-4V API: {e}')
            if attempt < max_retries - 1:  # If not the last attempt, wait a bit before retrying
                time.sleep(1)  # You can adjust the sleep time if necessary
            else:
                print('Max retries reached. Exiting.')
                return None, None

def interleaved_gpt4o(prompt_prefix, prompt_subfix, data, max_tokens=512, temperature=None, max_retries=10):
    # if not is_base64_encoded(base64_image):
    #     base64_image = encode_image(base64_image)

    interleaved_list = data['interleaved_list']
    meta = dict(data['meta'])

    messages = [
                    {
                        "role": "user",
                        "content": []
                    }
                ]

    messages[0]['content'].append({
                                "type": "text",
                                "text": prompt_prefix,
                                "finish_reason": "stop",
                            })

    for item in interleaved_list:
        img_pattern0 = "<image>"
        img_pattern1 = r"<image_(\d+)>"
        img_key0 = re.findall(img_pattern0, item)
        img_key1 = re.findall(img_pattern1, item)

        if len(img_key0) > 0 or len(img_key1) > 0:
            # print(item)
            # import pdb;pdb.set_trace()
            image_num = None
            if len(img_key1) > 0:
                match = re.search(img_pattern1, item)
                if match:
                    image_num = match.group(1)  # 提取捕获组中的数字
            messages[0]['content'].append({
                                "type": "text",
                                "text": f"<img_{image_num}>",
                                "finish_reason": "stop",
                            })
            item_warpper = item.replace("<", "").replace(">", "")
            messages[0]['content'].append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{meta[item_warpper]}",
                                }
                            })
            messages[0]['content'].append({
                                "type": "text",
                                "text": f"<\img_{image_num}>",
                                "finish_reason": "stop",
                            })
        else:
            messages[0]['content'].append({
                                "type": "text",
                                "text": item,
                                "finish_reason": "stop",
                            })
    messages[0]['content'].append({
                                "type": "text",
                                "text": prompt_subfix,
                                "finish_reason": "stop",
                            })
    # import pdb;pdb.set_trace()
    # print(messages[0])


    client = OpenAI(api_key="demo_api_1", base_url=_get_xhs_openai_url()) 
    response = 200
    result = None
    for attempt in range(max_retries):
        print(f"===> 调用GPT-4o中 ... (尝试 {attempt + 1}/{max_retries})")
        try:
            completion = client.chat.completions.create(
                model = "gpt4o-ptu-agi-offline-shiqing",
                messages = messages,
                max_tokens = max_tokens,
                temperature = temperature if temperature is not None else None
            )

            result = completion.choices[0].message.content[0]['text']
            return response, result
        except Exception as e:
            print(f'Failed to invoke GPT-4V API: {e}')
            if attempt < max_retries - 1:  # If not the last attempt, wait a bit before retrying
                time.sleep(1)  # You can adjust the sleep time if necessary
            else:
                print('Max retries reached. Exiting.')
                return None, None

def puretext_gpt4o(prompt, max_tokens=512, temperature=None, max_retries=10):

    messages = [
                    {
                        "role": "user",
                        "content": []
                    }
                ]

    messages[0]['content'].append({
                                "type": "text",
                                "text": prompt,
                                "finish_reason": "stop",
                            })

    client = OpenAI(api_key="demo_api_1", base_url=_get_xhs_openai_url()) 
    response = 200
    result = None
    for attempt in range(max_retries):
        print(f"===> 调用GPT-4o中 ... (尝试 {attempt + 1}/{max_retries})")
        try:
            completion = client.chat.completions.create(
                model = "gpt4o-ptu-agi-offline-shiqing",
                messages = messages,
                max_tokens = max_tokens,
                temperature = temperature if temperature is not None else None
            )

            result = completion.choices[0].message.content[0]['text']
            return response, result
        except Exception as e:
            print(f'Failed to invoke GPT-4V API')
            # print(f'Failed to invoke GPT-4V API: {e}')
            if attempt < max_retries - 1:  # If not the last attempt, wait a bit before retrying
                time.sleep(1)  # You can adjust the sleep time if necessary
            else:
                print('Max retries reached. Exiting.')
                return None, None