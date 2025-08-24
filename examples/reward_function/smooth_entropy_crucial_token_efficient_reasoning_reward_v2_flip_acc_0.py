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
import math

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


def _huber_penalty(x: float, kappa: float) -> float:
    """Smooth penalty for x>=0; quadratic near 0, linear in tail."""
    x = max(float(x), 0.0)
    return 0.5 * x * x / kappa if x <= kappa else x - 0.5 * kappa

def _sigmoid_saturate(x: float, temp: float) -> float:
    """Saturating [0,1] growth; temp controls softness."""
    t = max(float(temp), 1e-6)
    return 1.0 / (1.0 + math.exp(-x / t))

def _margin_from_target(target: float, frac: float, min_margin: float = 1.0) -> float:
    """Difficulty-aware tolerance band around target."""
    tgt = max(float(target), 1.0)
    return max(min_margin, frac * tgt)

def smooth_entropy_score(gen: float, target: float, mode: str = "lt", scale: float = 1.0) -> float:
    """
    平滑的熵 token 数奖励函数，基于 sigmoid 曲线。
    - mode='lt'：越小越好（easy/medium）
    - mode='gt'：越大越好（hard）
    返回值 ∈ [0, scale]，或用于负向惩罚
    """
    diff = gen - target
    diff = np.clip(diff, -50, 50) 
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

        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score   = format_reward(response)
        accuracy_score = reward_input.get("accuracy") or accuracy_reward(response, reward_input["ground_truth"])
        difficulty     = reward_input.get("difficulty", "easy")

        gen_token_num    = float(reward_input.get("high_entropy_token_num", 1))
        target_token_num = float(reward_input.get("target_high_entropy_token_num", 1))
        alpha_entropy_sample = float(reward_input.get("alpha_entropy", alpha_entropy))
        # New: soft reasoning cost (continuous) with per-bucket lambda (dual ascent)
        lambda_entropy = float(reward_input.get("lambda_entropy", 0.0))
        soft_cost = float(reward_input.get("reasoning_soft_cost", 0.0))


          
        MARGIN_FRAC = {"easy": 0.15, "medium": 0.25, "hard": 0.35} 
        KAPPA       = {"easy": 2.0,  "medium": 3.0,   "hard": 4.0}  
        TEMP        = {"easy": 2.0,  "medium": 2.5,   "hard": 3.0}  
        CAP_SCALE   = {"easy": 1.0,  "medium": 1.0,   "hard": 1.2}


        delta   = gen_token_num - target_token_num
        margin  = _margin_from_target(target_token_num, MARGIN_FRAC.get(difficulty, 0.2), min_margin=1.0)
        kappa   = KAPPA.get(difficulty, 2.0)
        temp    = TEMP.get(difficulty, 2.0)
        cap     = alpha_entropy_sample * CAP_SCALE.get(difficulty, 1.0)

        entropy_score = 0.0

        # If lambda and soft_cost exist, prioritize constrained formulation
#           if (reward_input.get("lambda_entropy") is not None) and (reward_input.get("reasoning_soft_cost") is not None):
#             entropy_score = - lambda_entropy * soft_cost`
#           else:
        if difficulty in ("easy", "medium"):
            if accuracy_score == 1.0:
                if difficulty == "easy":
                    over = max(delta - margin, 0.0)                     
                    pen  = _huber_penalty(over, kappa)
                    entropy_score = -min(cap, pen / (margin + kappa) * cap)
                else:  # medium
                    dev  = max(abs(delta) - margin, 0.0)                
                    pen  = _huber_penalty(dev, kappa)
                    entropy_score = -min(cap, pen / (margin + kappa) * cap)
            else:
                # lack = max((target_token_num - gen_token_num) - margin, 0.0)
                over = max(delta - margin, 0.0)
                gain = _sigmoid_saturate(over, temp) * cap * (0.6 if difficulty == "easy" else 0.8)
                entropy_score = gain

        elif difficulty == "hard":
            if accuracy_score == 1.0:
                if delta >= -margin:
                    bonus = _sigmoid_saturate(delta - (-margin), temp) * cap 
                    entropy_score = bonus
                else:
                    pen = _huber_penalty((-delta) - margin, kappa)
                    entropy_score = -min(cap, 0.5 * pen / (margin + kappa) * cap)
            else:

                # lack = max((target_token_num - gen_token_num) - margin, 0.0)
                over = max(delta - margin, 0.0)
                gain = _sigmoid_saturate(over, temp) * cap
                entropy_score = gain

        overall_score = float(accuracy_score) + float(entropy_score)
        # overall_score += format_score * format_weight

        scores.append({
            "overall": overall_score,
            "accuracy": float(accuracy_score),
            "format": float(format_score),
            "high_entropy_token_num_score": float(entropy_score),
        })

    return scores