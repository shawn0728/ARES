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

from typing import Any, Dict, List

import numpy as np
import torch

from ..protocol import DataProto
from collections import defaultdict


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> Dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    difficulty_list = batch.non_tensor_batch["difficulty"]  # List[str] of same length as batch size
    response_length_np = response_length.cpu().numpy()  # (batch_size,) -> NumPy for easier indexing

    # 分桶统计 response_length
    bucketed_stats = defaultdict(list)
    for i, difficulty in enumerate(difficulty_list):
        bucketed_stats[difficulty].append(response_length_np[i])

    bucketed_metrics = {}
    for difficulty, lengths in bucketed_stats.items():
        lengths_tensor = torch.tensor(lengths, dtype=torch.float32)
        bucketed_metrics[f"response_length/{difficulty}/mean"] = torch.mean(lengths_tensor).detach().item()
        bucketed_metrics[f"response_length/{difficulty}/max"] = torch.max(lengths_tensor).detach().item()
        bucketed_metrics[f"response_length/{difficulty}/min"] = torch.min(lengths_tensor).detach().item()
        bucketed_metrics[f"response_length/{difficulty}/count"] = len(lengths)

    # === [NEW METRIC] accuracy vs high_entropy_token_num for all difficulty ===
    entropy_acc_metrics = defaultdict(list)
    for target_difficulty in ["easy", "medium", "hard"]:
        entropy_list = []
        accuracy_list = []

        for i, difficulty in enumerate(difficulty_list):
            if difficulty == target_difficulty:
                entropy = batch.non_tensor_batch["high_entropy_token_num"][i]
                acc = batch.non_tensor_batch["accuracy"][i]
                entropy_list.append(entropy)
                accuracy_list.append(acc)

        if len(entropy_list) >= 3:
            entropy_arr = np.array(entropy_list)
            acc_arr = np.array(accuracy_list)

            min_e = int(np.min(entropy_arr))
            max_e = int(np.max(entropy_arr))
            bin_edges = np.linspace(min_e, max_e, 4, dtype=int)  # 3段 -> 4个边界（int）

            for i in range(3):
                start = bin_edges[i]
                end = bin_edges[i + 1]
                # 注意最后一段包含 end
                if i < 2:
                    mask = (entropy_arr >= start) & (entropy_arr < end)
                else:
                    mask = (entropy_arr >= start) & (entropy_arr <= end)
                bin_tag = f"bin{i+1}"
                if mask.sum() > 0:
                    avg_acc = float(np.mean(acc_arr[mask]))
                else:
                    avg_acc = float("nan")

                entropy_acc_metrics[f"entropy vs acc/{target_difficulty}/accuracy/{bin_tag}"] = avg_acc
                entropy_acc_metrics[f"entropy vs acc/{target_difficulty}/count/{bin_tag}"] = int(mask.sum())

    # === [NEW METRIC] valid / invalid reasoning token count per difficulty ===
    reasoning_token_metrics = defaultdict(list)

    for target_difficulty in ["easy", "medium", "hard"]:
        valid_list = []
        invalid_list = []

        for i, difficulty in enumerate(difficulty_list):
            if difficulty == target_difficulty:
                valid = batch.non_tensor_batch.get("valid_reasoning_token_num", [0] * len(difficulty_list))[i]
                invalid = batch.non_tensor_batch.get("invalid_reasoning_token_num", [0] * len(difficulty_list))[i]
                valid_list.append(valid)
                invalid_list.append(invalid)

        if len(valid_list) >= 1:
            reasoning_token_metrics[f"reasoning_tokens_statistics/{target_difficulty}/valid_mean"] = float(np.mean(valid_list))
            reasoning_token_metrics[f"reasoning_tokens_statistics/{target_difficulty}/invalid_mean"] = float(np.mean(invalid_list))
            reasoning_token_metrics[f"reasoning_tokens_statistics/{target_difficulty}/count"] = len(valid_list)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    metrics.update(bucketed_metrics)
    metrics.update(entropy_acc_metrics)
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], num_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }
