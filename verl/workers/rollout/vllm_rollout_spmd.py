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

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union, Set
import re
import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig
import json


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[Dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: Dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> Dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None

def compute_entropy_from_logprobs(logprobs_per_token_list: List[List[Dict[int, Any]]]) -> List[float]:
    entropy_list = []
    for logprobs_per_token in logprobs_per_token_list:
        token_entropies = []
        for logprob_dict in logprobs_per_token:
            log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
            # Tail-mass corrected entropy using raw (truncated) probs + a tail bucket
            raw_probs = np.exp(log_probs)
            sum_raw = float(np.sum(raw_probs))
            tail = max(0.0, 1.0 - sum_raw)
            entropy = -float(np.sum(raw_probs * log_probs))
            if tail > 0.0:
                entropy -= float(tail * np.log(max(tail, 1e-12)))
            token_entropies.append(entropy)
        mean_entropy = np.mean(token_entropies) if token_entropies else 0.0
        entropy_list.append(float(mean_entropy))
    return entropy_list

def collect_global_token_entropy_from_logprobs(logprobs_per_token_list: List[List[Dict[int, Any]]]) -> List[float]:
    all_token_entropies_list = []
    for logprobs_per_token in logprobs_per_token_list:
        token_entropies = []
        for logprob_dict in logprobs_per_token:
            log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
            raw_probs = np.exp(log_probs)
            sum_raw = float(np.sum(raw_probs))
            tail = max(0.0, 1.0 - sum_raw)
            entropy = -float(np.sum(raw_probs * log_probs))
            if tail > 0.0:
                entropy -= float(tail * np.log(max(tail, 1e-12)))
            token_entropies.append(entropy)
        all_token_entropies_list.extend(token_entropies)
    return all_token_entropies_list



def compute_high_entropy_threshold(entropy_values: List[float], percentile: float = 95.0):
    return np.percentile(entropy_values, percentile)


def compute_high_entropy_token_num_from_logprobs_dynamic(logprobs_per_token_list, percentile=95.0):
    all_token_entropies = []
    # 先统计所有 token entropy
    for logprobs_per_token in logprobs_per_token_list:
        for logprob_dict in logprobs_per_token:
            log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
            raw_probs = np.exp(log_probs)
            sum_raw = float(np.sum(raw_probs))
            tail = max(0.0, 1.0 - sum_raw)
            entropy = -float(np.sum(raw_probs * log_probs))
            if tail > 0.0:
                entropy -= float(tail * np.log(max(tail, 1e-12)))
            all_token_entropies.append(entropy)

    threshold = compute_high_entropy_threshold(all_token_entropies, percentile)

    # 再统计每个 sample 的高熵 token 数
    high_entropy_token_num_list = []
    for logprobs_per_token in logprobs_per_token_list:
        count = 0
        for logprob_dict in logprobs_per_token:
            log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
            raw_probs = np.exp(log_probs)
            sum_raw = float(np.sum(raw_probs))
            tail = max(0.0, 1.0 - sum_raw)
            entropy = -float(np.sum(raw_probs * log_probs))
            if tail > 0.0:
                entropy -= float(tail * np.log(max(tail, 1e-12)))
            if entropy > threshold:
                count += 1
        high_entropy_token_num_list.append(count)

    return high_entropy_token_num_list, threshold

def normalize_token(token: str) -> str:
    """
    Normalize token string by removing leading special characters and converting to lowercase.
    """
    token = re.sub(r"^[▁Ġ]+", "", token)  # remove leading special characters
    token = token.strip().lower()
    return token

# def compute_high_entropy_token_num_from_logprobs(
#     logprobs_per_token_list: List[List[Dict[int, Any]]],
#     tokenizer,
#     threshold: float = 0.5,
#     reasoning_token_set: Set[str] = None
# ) -> List[int]:
#     """
#     Count high entropy tokens that are also in reasoning_token_set.
#     """
#     high_entropy_token_num_list = []

#     for logprobs_per_token in logprobs_per_token_list:
#         high_entropy_count = 0
#         for logprob_dict in logprobs_per_token:
#             log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
#             probs = np.exp(log_probs)
#             entropy = -np.sum(probs * log_probs)

#             if entropy > threshold:
#                 # get the token string of top-1 token
#                 top_token_id = max(logprob_dict.items(), key=lambda x: x[1].logprob)[0]
#                 token_str = tokenizer.decode([top_token_id])
#                 token_str = normalize_token(token_str)

#                 if reasoning_token_set is None or token_str in reasoning_token_set:
#                     high_entropy_count += 1

#         high_entropy_token_num_list.append(high_entropy_count)

#     return high_entropy_token_num_list


# def compute_high_entropy_token_num_from_logprobs(logprobs_per_token_list: List[List[Dict[int, Any]]], threshold: float = 0.5) -> List[int]:
#     high_entropy_token_num_list = []
#     for logprobs_per_token in logprobs_per_token_list:
#         high_entropy_count = 0
#         for logprob_dict in logprobs_per_token:
#             log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
#             probs = np.exp(log_probs)
#             entropy = -np.sum(probs * log_probs)
#             # if entropy > threshold:
#             #     high_entropy_count += 1
#         high_entropy_token_num_list.append(high_entropy_count)
#     return high_entropy_token_num_list

def compute_reasoning_token_window_entropy(
    logprobs_per_token_list: List[List[Dict[int, Any]]],
    tokenizer,
    threshold: float,
    reasoning_token_set: Set[str],
    window_size: int = 16
) -> List[List[Dict[str, Any]]]:
    """
    For each reasoning trigger token (matched by token string),
    calculate the average entropy of the following `window_size` tokens.

    Return a list of reasoning token info per sample.
    """
    valid_high_entropy_token_num_list = []
    invalid_high_entropy_token_num_list = []
    soft_cost_sum_list = []
    
    for logprobs_per_token in logprobs_per_token_list:
        valid_high_entropy_token_num = 0
        invalid_high_entropy_token_num = 0
        soft_cost_sum = 0.0
        token_entropies = []
        token_ids = []
        for logprob_dict in logprobs_per_token:
            log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
            raw_probs = np.exp(log_probs)
            sum_raw = float(np.sum(raw_probs))
            tail = max(0.0, 1.0 - sum_raw)
            entropy = -float(np.sum(raw_probs * log_probs))
            if tail > 0.0:
                entropy -= float(tail * np.log(max(tail, 1e-12)))
            token_entropies.append(entropy)
            top_token_id = max(logprob_dict.items(), key=lambda x: x[1].logprob)[0]
            token_ids.append(top_token_id)

        token_strs = [normalize_token(tokenizer.decode([tid])) for tid in token_ids]

        for i, tok in enumerate(token_strs):
            if tok in reasoning_token_set:
                # 检查是否有足够的后续 token 来形成完整的窗口
                remaining_length = len(token_entropies) - i - 1
                if remaining_length >= window_size:
                    next_window = token_entropies[i + 1 : i + 1 + window_size]
                    mean_following_entropy = float(np.mean(next_window))
                    # 连续软成本：越高越贵
                    soft_cost = max(0.0, mean_following_entropy - threshold)
                    soft_cost_sum += soft_cost
                    if mean_following_entropy >= threshold:
                        valid_high_entropy_token_num += 1
                    else:
                        invalid_high_entropy_token_num += 1
        valid_high_entropy_token_num_list.append(valid_high_entropy_token_num)
        invalid_high_entropy_token_num_list.append(invalid_high_entropy_token_num)
        soft_cost_sum_list.append(float(soft_cost_sum))

        # result.append(sample_result)
    return valid_high_entropy_token_num_list, invalid_high_entropy_token_num_list, soft_cost_sum_list

def get_percentile_threshold_list(logprobs_per_token_list: List[List[Dict[int, Any]]],percentile: float = 99.0) -> List[float]:
    # get 99 percentile threshold for each sample and return a list
    percentile_threshold_list = []
    for logprobs_per_token in logprobs_per_token_list:
        token_entropies = []
        for logprob_dict in logprobs_per_token:  # limit token count if needed
            log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
            raw_probs = np.exp(log_probs)
            sum_raw = float(np.sum(raw_probs))
            tail = max(0.0, 1.0 - sum_raw)
            entropy = -float(np.sum(raw_probs * log_probs))
            if tail > 0.0:
                entropy -= float(tail * np.log(max(tail, 1e-12)))
            token_entropies.append(entropy)
        if token_entropies:  # avoid crash on empty list
            sample_percentile = np.percentile(token_entropies, percentile)
        else:
            sample_percentile = 0.0  # or np.nan if you prefer signaling missing data
        percentile_threshold_list.append(sample_percentile)

    return percentile_threshold_list


def get_window_average_entropy_percentile_threshold_list(
    logprobs_per_token_list: List[List[Dict[int, Any]]], 
    percentile: float = 98.0, 
    window_size: int = 8
) -> List[float]:
    """
    计算基于窗口平均熵的 percentile threshold
    1. 对每个 token，计算后续 window_size 个 token 的平均熵
    2. 收集所有窗口平均熵，计算 percentile
    3. 只有当剩余 token 数量足够时才计算窗口平均熵
    """
    percentile_threshold_list = []
    
    for sample_idx, logprobs_per_token in enumerate(logprobs_per_token_list):
        window_average_entropies = []
        
        # 计算每个 token 的熵
        token_entropies = []
        for logprob_dict in logprobs_per_token:
            log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
            probs = np.exp(log_probs)
            entropy = -np.sum(probs * log_probs)
            token_entropies.append(entropy)
        
        # 对每个 token，计算后续窗口的平均熵
        for i in range(len(token_entropies)):
            # 检查是否有足够的后续 token 来形成完整的窗口
            remaining_length = len(token_entropies) - i - 1
            if remaining_length >= window_size:
                next_window = token_entropies[i + 1 : i + 1 + window_size]
                mean_window_entropy = float(np.mean(next_window))
                window_average_entropies.append(mean_window_entropy)
            else:
                # 剩余长度不足，添加之前的平均值以保持 batch 长度一致
                if window_average_entropies:
                    window_average_entropies.append(np.mean(window_average_entropies))
                else:
                    # 如果还没有任何窗口平均值，使用当前 token 的熵
                    window_average_entropies.append(token_entropies[i])
        
        # 计算窗口平均熵的 percentile
        if window_average_entropies:
            sample_percentile = np.percentile(window_average_entropies, percentile)
            print(f"[Debug] Sample {sample_idx}: window_average_entropies count={len(window_average_entropies)}, {percentile}th percentile={sample_percentile:.4f}")
        else:
            sample_percentile = 0.0
            print(f"[Debug] Sample {sample_idx}: No window average entropies found")
        
        percentile_threshold_list.append(sample_percentile)
    
    return percentile_threshold_list

def load_reasoning_token_set(json_path = "/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/crucial_token_list/crucial_token_final_version_0731.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        token_list = json.load(f)
    print("[Debug] Loaded reasoning token set from:", json_path)
    return set(token_list)



def get_percentile_threshold_list(logprobs_per_token_list: List[List[Dict[int, Any]]],percentile: float = 99.0) -> List[float]:
    # get 99 percentile threshold for each sample and return a list
    percentile_threshold_list = []
    for logprobs_per_token in logprobs_per_token_list:
        token_entropies = []
        for logprob_dict in logprobs_per_token:  # limit token count if needed
            log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
            raw_probs = np.exp(log_probs)
            sum_raw = float(np.sum(raw_probs))
            tail = max(0.0, 1.0 - sum_raw)
            entropy = -float(np.sum(raw_probs * log_probs))
            if tail > 0.0:
                entropy -= float(tail * np.log(max(tail, 1e-12)))
            token_entropies.append(entropy)
        if token_entropies:  # avoid crash on empty list
            sample_percentile = np.percentile(token_entropies, percentile)
        else:
            sample_percentile = 0.0  # or np.nan if you prefer signaling missing data
        percentile_threshold_list.append(sample_percentile)

    return percentile_threshold_list


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        self.tokenizer = tokenizer
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)
        high_entropy_threshold = prompts.meta_info.get("high_entropy_threshold", 1.5)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        window_size = 4
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        # with self.update_sampling_params(**prompts.meta_info):
        with self.update_sampling_params(logprobs=10,**prompts.meta_info):
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )

            logprobs_per_token_list = []
            for completion in completions:
                for output in completion.outputs:
                    logprobs_per_token = output.logprobs  # list of dict
                    logprobs_per_token_list.append(logprobs_per_token)
            
            entropy_list = compute_entropy_from_logprobs(logprobs_per_token_list)
            # tokenwise_entropy_list = collect_global_token_entropy_from_logprobs(logprobs_per_token_list)
            # 使用基于窗口平均熵的 percentile 计算
            percentile_threshold_list = get_window_average_entropy_percentile_threshold_list(
                logprobs_per_token_list, 
                percentile=95.0, 
                window_size=window_size
            )
            # import ipdb;ipdb.set_trace()
            reasoning_token_set = load_reasoning_token_set("/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/crucial_token_list/crucial_token_final_version_0731.json")
            
            valid_trigger_count_list, invalid_trigger_count_list, soft_cost_sum_list = compute_reasoning_token_window_entropy(
                logprobs_per_token_list,
                tokenizer=self.tokenizer,
                threshold=high_entropy_threshold,
                reasoning_token_set=reasoning_token_set,
                window_size=window_size
            )
            high_entropy_token_num_list = valid_trigger_count_list
            # print("[Debug] Calculated high entropy token num list:", high_entropy_token_num_list)
            print("[Debug] Calculated high entropy threshold:", high_entropy_threshold)

            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data,
                                "entropies": entropy_list,
                                "high_entropy_token_num": high_entropy_token_num_list,
                                "invalid_reasoning_trigger_list": invalid_trigger_count_list,
                                "entropy_percentile_thresholds": percentile_threshold_list,
                                "reasoning_soft_cost": soft_cost_sum_list,
                                }
        else:
            non_tensor_batch = {"entropies": entropy_list,
                                "high_entropy_token_num": high_entropy_token_num_list,
                                "invalid_reasoning_trigger_list": invalid_trigger_count_list,
                                "entropy_percentile_thresholds": percentile_threshold_list,
                                "reasoning_soft_cost": soft_cost_sum_list,
                                }
            

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def record_high_entropy_tokens(self, logprobs_per_token_list, threshold):
        """
        Record tokens with entropy greater than threshold into a log file.
        """
        high_entropy_tokens = []
        for logprobs_per_token in logprobs_per_token_list:
            token_entropies = []
            token_ids = []
            for logprob_dict in logprobs_per_token:
                log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
                probs = np.exp(log_probs)
                entropy = -np.sum(probs * log_probs)
                token_entropies.append(entropy)

                max_token_id = max(logprob_dict.items(), key=lambda kv: kv[1].logprob)[0]
                token_ids.append(max_token_id)

            # decoded_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            decoded_tokens = [tok if isinstance(tok, str) else tok.decode('utf-8', errors='replace') for tok in self.tokenizer.convert_ids_to_tokens(token_ids)]

            for tok, ent in zip(decoded_tokens, token_entropies):
                if ent > threshold:
                    high_entropy_tokens.append({"token": tok, "entropy": float(ent)})

        with open("/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/mess/token_entropy_log_0724_cs_v02_total_geometry3k.jsonl", "a") as f:
            for item in high_entropy_tokens:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # @torch.no_grad()
    # def generate_sequences_with_tokenwise(self, prompts: DataProto):
    #     input_ids: torch.Tensor = prompts.batch["input_ids"]
    #     attention_mask: torch.Tensor = prompts.batch["attention_mask"]
    #     position_ids: torch.Tensor = prompts.batch["position_ids"]
    #     eos_token_id: int = prompts.meta_info["eos_token_id"]
    #     batch_size = input_ids.size(0)
    #     high_entropy_threshold = prompts.meta_info.get("high_entropy_threshold", 1.5)

    #     non_tensor_batch = prompts.non_tensor_batch
    #     batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
    #     batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
    #     if batch_size != len(batch_raw_prompt_ids):
    #         raise RuntimeError("vllm sharding manager is not working properly.")

    #     if batch_multi_modal_data is not None:
    #         vllm_inputs = []
    #         for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
    #             vllm_inputs.append({
    #                 "prompt_token_ids": list(raw_prompt_ids),
    #                 "multi_modal_data": _process_multi_modal_data(
    #                     multi_modal_data,
    #                     prompts.meta_info["min_pixels"],
    #                     prompts.meta_info["max_pixels"],
    #                     prompts.meta_info["video_fps"],
    #                 ),
    #             })
    #     else:
    #         vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

    #     with self.update_sampling_params(logprobs=10, **prompts.meta_info):
    #         completions: List[RequestOutput] = self.inference_engine.generate(
    #             prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
    #         )

    #         logprobs_per_token_list = []
    #         for completion in completions:
    #             for output in completion.outputs:
    #                 logprobs_per_token_list.append(output.logprobs)

    #         # per-sample mean entropy
    #         entropy_list = compute_entropy_from_logprobs(logprobs_per_token_list)

    #         # per-token entropy flat list for global percentile
    #         tokenwise_entropy_list = []
    #         tokenwise_entropy_flat = []
    #         for logprobs_per_token in logprobs_per_token_list:
    #             token_entropies = []
    #             for logprob_dict in logprobs_per_token:
    #                 log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
    #                 probs = np.exp(log_probs)
    #                 entropy = -np.sum(probs * log_probs)
    #                 token_entropies.append(entropy)
    #             tokenwise_entropy_list.append(token_entropies)
    #             tokenwise_entropy_flat.extend(token_entropies)
    #         percentile_threshold_list = get_percentile_threshold_list(logprobs_per_token_list, percentile=99.0)

    #         # high_entropy_token_num_list = compute_high_entropy_token_num_from_logprobs(logprobs_per_token_list, tokenizer = self.tokenizer,threshold=high_entropy_threshold)
    #         high_entropy_token_num_list = compute_reasoning_token_window_entropy(logprobs_per_token_list, tokenizer=self.tokenizer, threshold=high_entropy_threshold, reasoning_token_set=load_reasoning_token_set("/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/mess/crucial_token_final_version.json"), window_size=16)

    #         print(f"[Debug] Calculated high entropy threshold used in filtering: {high_entropy_threshold}")

    #         response_ids = [output.token_ids for completion in completions for output in completion.outputs]
    #         response_ids = VF.pad_2d_list_to_length(
    #             response_ids, self.pad_token_id, max_length=self.config.response_length
    #         ).to(input_ids.device)

    #         if self.sampling_params.n > 1:
    #             batch_size *= self.sampling_params.n
    #             input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
    #             attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
    #             position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
    #             if batch_multi_modal_data is not None:
    #                 batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

    #     sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
    #     response_length = response_ids.size(1)
    #     delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
    #     delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
    #     if position_ids.dim() == 3:  # qwen2vl mrope case
    #         delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

    #     response_position_ids = position_ids[..., -1:] + delta_position_id
    #     position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

    #     response_mask = VF.get_response_mask(
    #         response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
    #     )
    #     attention_mask = torch.cat((attention_mask, response_mask), dim=-1)
    #     tokenwise_entropy_threshold = np.percentile(tokenwise_entropy_flat, 99.0)

    #     # self.record_high_entropy_tokens(logprobs_per_token_list, tokenwise_entropy_threshold)
        
    #     batch = TensorDict(
    #         {
    #             "prompts": input_ids,
    #             "responses": response_ids,
    #             "input_ids": sequence_ids,
    #             "attention_mask": attention_mask,
    #             "response_mask": response_mask,
    #             "position_ids": position_ids,
    #         },
    #         batch_size=batch_size,
    #     )

    #     non_tensor_batch_out = {
    #         "entropies": entropy_list,
    #         "high_entropy_token_num": high_entropy_token_num_list,
    #         "tokenwise_entropy_threshold": np.array([tokenwise_entropy_threshold] * batch_size, dtype=np.float32),
    #         "entropy_percentile_thresholds": percentile_threshold_list,
    #     }
    #     if batch_multi_modal_data is not None:
    #         non_tensor_batch_out["multi_modal_data"] = batch_multi_modal_data

    #     return DataProto(batch=batch, non_tensor_batch=non_tensor_batch_out, meta_info=prompts.meta_info)
