# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer

DEFAULT_TARGET_LENGTH = 64  # fallback value
target_length_dict = {}     # difficulty → avg token len
def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def laser_d_length_reward(gen_len, target_L,alpha=0.5):
    # print(f"[Debug] gen_len = {gen_len}, target_L = {target_L}")
    if target_L > gen_len:
        return alpha
    else:
        return 0.0

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        length_bonus = laser_d_length_reward(gen_len=reward_input["response_length"], target_L = reward_input.get("target_length", DEFAULT_TARGET_LENGTH), alpha=0.5)
        format_score = format_reward(response)
        if "accuracy" in reward_input and reward_input["accuracy"] is not None:
            accuracy_score = reward_input["accuracy"]
        else:
            accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        # overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
        C = None
        if format_score == 0.0:
            C = -1.0
        elif accuracy_score == 1.0:
            C = 1.0
        elif accuracy_score == 0.0:
            C = -0.5

        # Step-based length reward (S)
        S = length_bonus if C == 1.0 else 0.0

        overall_score = C + S
        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "length_bonus": length_bonus,
            }
        )
    return scores
