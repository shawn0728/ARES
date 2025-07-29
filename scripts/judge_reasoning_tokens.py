import json
import requests
from openai import OpenAI
# from utilities.gpt_api import puretext_gpt4o
import random
import time
import glob
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import base64
import io
import re
import sys
from tqdm import tqdm
from get_api import puretext_gpt4o


pre_prompt_en = """
You are a specialist in multimodal reasoning, entropy analysis, and linguistics. 
I will provide you with a single token at a time. For each token, you should judge 
whether it is a **semantically meaningful high-entropy token that likely serves as 
a reasoning trigger in a rollout process**.

Specifically, this includes:
- Transition words (e.g., but, however)
- Reasoning initiation words (e.g., therefore, thus)
- Key structural tokens that signal a shift in reasoning or steps
- Tokens with clearly interpretable semantics (not noise, numbers, or placeholders)

Respond with only True or False.
Here is the token:
"""


def get_gpt_response():
    """Process a single row of the parquet file."""

    json_input_path = "/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/mess/token_entropy_log_0724_cs_v02_total_geometry3k_filtered_readable.json"
    json_output_path = "/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/mess/gpt_filtered_0724_tokens_cs_v02_total_geometry3k_filtered_readable.jsonl"
    with open(json_input_path, 'r', encoding='utf-8') as f:
        json_input_dict = json.load(f)
    key_set = set(json_input_dict.keys())
    if os.path.exists(json_output_path):
        try:
            with open(json_output_path, 'r', encoding='utf-8') as f:
                exists_key_dict = json.load(f)
                exists_key_set = set(exists_key_dict.keys())
                key_set = key_set - exists_key_set
        except Exception as e:
            print(f"Error reading existing JSON file: {e}")
            exists_key_set = set()
    pre_prompt = pre_prompt_en
    import pdb
    # post_prompt = post_prompt_zh

    model_name = "model"
    result_json = {
        "messages": [{"role": "user", "content": []}],
        'tokens_to_generate': 1024,
        'top_k': 1,
        "no_log": True,
        'eos_word': '<|endofassistant|>',
        "max_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.9,
        "model": 'model',
    }

    for key in tqdm(key_set):
    #    result_json["messages"][0]["content"].append({"type": "text", "text": pre_prompt.strip()})
    #    result_json["messages"][0]["content"].append({"type": "text", "text": key.strip()})
       result_prompt = f"{pre_prompt.strip()}\n{key.strip()}"
       status,response = puretext_gpt4o(result_prompt)
       # import pdb; pdb.set_trace()
       item = {key:response}
       with open(json_output_path, "a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    

def post_process_filtered_gpt_response():
    with open("/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/mess/gpt_filtered_0724_tokens_cs_v02_total_geometry3k_filtered_readable.jsonl", "r") as f:
        lines = f.readlines()
    filtered_items = []
    for line in lines:
        item = json.loads(line)
        for key, value in item.items():
            if value.lower() == "true":
                filtered_items.append(key)
    filtered_items = set(filtered_items)
    # 写出一个set
    with open("/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/mess/crucial_token_final_version.json", "w") as f:
        json.dump(list(filtered_items), f, ensure_ascii=False, indent=2)

def main():
    # get_gpt_response()
    post_process_filtered_gpt_response()

if __name__ == "__main__": 
    main()