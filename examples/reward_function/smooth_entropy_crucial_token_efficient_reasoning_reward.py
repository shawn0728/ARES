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
import numpy as np
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


def laser_d_length_reward(gen_len, target_L, alpha=0.5):
    if target_L > gen_len:
        return alpha
    else:
        return 0.0


def smooth_entropy_score(gen: float, target: float, mode: str = "lt", scale: float = 1.0) -> float:
    """
    平滑的熵 token 数奖励函数，基于 sigmoid 曲线。
    - mode='lt'：越小越好（easy/medium）
    - mode='gt'：越大越好（hard）
    返回值 ∈ [0, scale]，或用于负向惩罚
    """
    diff = gen - target
    if mode == "lt":
        score = scale * (1.0 / (1.0 + np.exp(diff)))
    elif mode == "gt":
        score = scale * (1.0 / (1.0 + np.exp(-diff)))
    else:
        raise ValueError("Invalid mode: must be 'lt' or 'gt'")
    return np.clip(score, 0.0, scale)


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1, alpha_length_bonus: float = 0.5, alpha_entropy: float = 0.5) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    scores = []

    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # 清理格式
        format_score = format_reward(response)
        accuracy_score = reward_input.get("accuracy") or accuracy_reward(response, reward_input["ground_truth"])
        difficulty = reward_input.get("difficulty", "easy")

        gen_token_num = reward_input.get("high_entropy_token_num", 1)
        target_token_num = reward_input.get("target_high_entropy_token_num", 1)

        entropy_score = 0.0

        if difficulty in ["easy", "medium"]:
            if accuracy_score == 1.0:
                entropy_score = smooth_entropy_score(
                    gen=gen_token_num,
                    target=target_token_num,
                    mode="lt",
                    scale=alpha_entropy
                )

        if difficulty == "hard":
            if accuracy_score == 1.0:
                entropy_score = alpha_entropy
            elif accuracy_score == 0.0:
                entropy_score = -smooth_entropy_score(
                    gen=gen_token_num,
                    target=target_token_num,
                    mode="lt",
                    scale=alpha_entropy
                )
        overall_score = accuracy_score + entropy_score
        # 可选开启格式奖励
        # overall_score += format_score * format_weight

        scores.append({
            "overall": overall_score,
            "accuracy": accuracy_score,
            "format": format_score,
            "high_entropy_token_num_score": entropy_score,
        })

    return scores
