import re
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer

DEFAULT_TARGET_LENGTH = 64
target_length_dict = {}  # difficulty -> target length (int)

def format_reward(response: str) -> float:
    """
    Reward 1.0 if response follows CoT format: <think>...</think> ... \\boxed{...}
    """
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response.strip()) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Reward 1.0 if boxed answer matches ground truth.
    """
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def laser_d_length_reward(gen_len: int, difficulty: str, delta: float = 0.1) -> float:
    """
    Reward 1.0 if generated length is within delta * target length.
    """
    target_L = target_length_dict.get(difficulty, DEFAULT_TARGET_LENGTH)
    return 1.0 if abs(gen_len - target_L) <= delta * target_L else 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1, length_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    Compute Laser-D reward scores in batch mode.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    import pdb; pdb.set_trace()  # Debugging breakpoint
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # fix Qwen's output format
        difficulty = reward_input.get("difficulty", "medium")
        ground_truth = reward_input["ground_truth"]

        accuracy_score = accuracy_reward(response, ground_truth)
        format_score = format_reward(response)
        gen_len = reward_input.get("response_length", len(response.split()))
        length_score = laser_d_length_reward(gen_len, difficulty)

        overall_score = (
            (1 - format_weight - length_weight) * accuracy_score
            + format_weight * format_score
            + length_weight * length_score
        )

        scores.append({
            "overall": overall_score,
            "format": format_score,
            "accuracy": accuracy_score,
            "length": length_score,
        })

    return scores