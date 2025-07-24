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
DEFAULT_TARGET_ENTROPY = 0.2

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

def entropy_reward(gen_entropy, target_entropy, difficulty,alpha=0.5):
    """
    Compute the entropy reward based on the generated entropy, target entropy, and difficulty level.
    """
    entropy_reward = 0
    if difficulty == "easy":
        if gen_entropy <= target_entropy:
            entropy_reward = 1.0
        else:
            entropy_reward = 0.0

    elif difficulty == "medium":
        if gen_entropy <= target_entropy:
            entropy_reward = 1.0
        else:
            entropy_reward = 0.0
    
    elif difficulty == "hard":
        if gen_entropy >= target_entropy:
            entropy_reward = 1.0
        else:
            entropy_reward = 0.0
    return entropy_reward * alpha


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1,alpha_length_bonus: float = 0.5, alpha_entropy: float = 0.5) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        length_bonus = laser_d_length_reward(
            gen_len=reward_input["response_length"],
            target_L=reward_input.get("target_length", DEFAULT_TARGET_LENGTH),
            alpha=alpha_length_bonus
        )
        entropy_penalty_score = 0.0
        gen_high_entropy_token_num = reward_input.get("high_entropy_token_num",1)
        target_high_entropy_token_num = reward_input.get("target_high_entropy_token_num", 1)


        format_score = format_reward(response)
        accuracy_score = reward_input.get("accuracy") or accuracy_reward(response, reward_input["ground_truth"])
        # print(f"[Debug]format_score = {format_score}, gen_entropy = {gen_entropy}, target_entropy = {target_entropy}")
        # Difficulty weight
        difficulty = reward_input.get("difficulty", "easy")
        overall_score = 0
        entropy_score = 0.0
        if difficulty == "easy":
            if accuracy_score == 1.0:
                if gen_high_entropy_token_num <= target_high_entropy_token_num:
                    entropy_score = alpha_entropy
                    overall_score += entropy_score

        elif difficulty == "medium":
            if accuracy_score == 1.0:
                if gen_high_entropy_token_num <= target_high_entropy_token_num:
                    entropy_score = alpha_entropy
                    overall_score += entropy_score

        elif difficulty == "hard":
            if accuracy_score == 1.0:
                entropy_score = alpha_entropy
                overall_score += entropy_score
            
            if accuracy_score == 0.0:
                if gen_high_entropy_token_num <= target_high_entropy_token_num:
                    entropy_score = -alpha_entropy
                    overall_score += entropy_score


        overall_score += format_weight * format_score + accuracy_score
        scores.append({
            "overall": overall_score,
            "accuracy": accuracy_score,
            "format": format_score,
            "high_entropy_token_num_score": entropy_score
        })

    return scores