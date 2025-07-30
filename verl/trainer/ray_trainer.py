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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from ..utils.dataset import RLHFDataset, collate_fn, FiltereRLHFdDataset
from .core_algos import AdvantageEstimator, FixedKLController, KLController, compute_kl, get_kl_controller
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from mathruler.grader import extract_boxed_content, grade_answer
from .data_loader import create_filtered_train_dataloader


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        base_dataset: RLHFDataset = None,   
        bootstrap_train_dataloder: Optional[StatefulDataLoader] = None,
        bootstrap_val_dataloader: Optional[StatefulDataLoader] = None,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.data_iterator = iter(self.train_dataloader)
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None
        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.difficulty_dict_train = {}
        self.target_length_dict = {}
        self.target_high_entropy_token_num_dict = {}
        self.skip_gid_set = set()
        self.bootstrap_train_dataloader = bootstrap_train_dataloder
        self.bootstrap_val_dataloader = bootstrap_val_dataloader
        self.base_dataset = base_dataset
        self.epoch_update_iter = 0
        self.epoch_id = 0
        self.drop_allpass_entropy_threshold = -1
        self.high_entropy_token_threshold = -1
        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        # 需要更改
        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)
        skip_gid_path = os.path.join(folder_path, "skip_gid_set.json")
        with open(skip_gid_path, "w") as f:
            json.dump(list(self.skip_gid_set), f)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            print("[Checkpoint] No load_checkpoint_path specified, skipping load.")
            return
        if not os.path.exists(self.config.trainer.load_checkpoint_path):
            print(f"[Checkpoint] Path not found: {self.config.trainer.load_checkpoint_path}, skipping load.")
            return

        tracker_json_path = os.path.join(self.config.trainer.load_checkpoint_path, CHECKPOINT_TRACKER)
        with open(tracker_json_path, "r") as f:
            tracker_info = json.load(f)
        load_path = os.path.dirname(tracker_info["last_actor_path"])
        self.global_step = tracker_info["last_global_step"]
        print(f"[Checkpoint] Automatically loading latest checkpoint from: {load_path}")
        # if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
        #     raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        # print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        # self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        # actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        actor_path = os.path.join(load_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            # critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            critic_path = os.path.join(load_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        # dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        # skip_gid_path = os.path.join(self.config.trainer.load_checkpoint_path, "skip_gid_set.json")
        dataloader_path = os.path.join(load_path, "dataloader.pt")
        skip_gid_path = os.path.join(load_path, "skip_gid_set.json")
        if os.path.exists(skip_gid_path):
            with open(skip_gid_path, "r") as f:
                self.skip_gid_set = set(json.load(f))
        else:
            print("No skip_gid_set found, initializing empty set.")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            self.data_iterator = iter(self.train_dataloader)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        uid2rollouts = defaultdict(list)
        print("Start validation...")
        all_token_entropy_list = []
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps
            test_gen_batch.meta_info["high_entropy_threshold"] = self.high_entropy_token_threshold

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            # test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences_with_tokenwise(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            accuracies = reward_metrics["accuracy"]
            lengths = test_batch.batch["response_mask"].sum(dim=-1).cpu().tolist()
            global_ids = test_batch.non_tensor_batch["global_id"]
            entropies = test_batch.non_tensor_batch["entropies"]
            tokenwise_entropy_list = test_batch.non_tensor_batch["tokenwise_entropy_threshold"].tolist()
            high_entropy_token_num = test_batch.non_tensor_batch["high_entropy_token_num"]
            all_token_entropy_list.extend(tokenwise_entropy_list)

            for gid, acc, entropy,high_ety_token_num, length in zip(global_ids, accuracies,entropies,high_entropy_token_num, lengths):
                uid2rollouts[gid].append({'length': length,'entropy':entropy, "high_ety_token_num":high_ety_token_num,'is_correct': acc > 0.5})
            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        # 聚合 accuracy，判定 difficulty，如果 high_entropy_token_threshold 未设置(初始validation），则使用所有 token 的平均 entropy
        if self.high_entropy_token_threshold < 0:
            self.high_entropy_token_threshold = np.mean(all_token_entropy_list)
        difficulty_dict_val = {}
        difficulty_rollouts = defaultdict(list)

        for gid, rollouts in uid2rollouts.items():
            acc_rate = sum(r["is_correct"] for r in rollouts) / len(rollouts)
            if acc_rate >= 2 / 3:
                diff = "easy"
            elif acc_rate >= 1 / 3:
                diff = "medium"
            else:
                diff = "hard"
            difficulty_dict_val[gid] = diff
            difficulty_rollouts[diff].extend(rollouts)

        self.difficulty_dict_val = difficulty_dict_val
        # 使用 LASER-D 的 ECR 方式计算 target length
        val_difficulty_statistics = {}
        target_entropy_dict = {}
        target_high_entropy_num_dict = {}

        # 统计所有 entropy
        all_high_entropy_token_num = []
        for rollouts in difficulty_rollouts.values():
            all_high_entropy_token_num.extend([r["high_ety_token_num"] for r in rollouts])

        min_entropy = np.min(all_high_entropy_token_num)
        max_entropy = np.max(all_high_entropy_token_num)

        difficulty_entropy_coef = {"easy": 0.35, "medium": 0.5, "hard": 0.65}

        for diff in ["easy", "medium", "hard"]:
            target_high_entropy_num_dict[diff] = int(min_entropy + difficulty_entropy_coef[diff] * (max_entropy - min_entropy))

        for diff in ["easy", "medium", "hard"]:
            rollouts = difficulty_rollouts[diff]
            val_difficulty_statistics[diff] = len(rollouts)

            if not rollouts:
                self.target_length_dict[diff] = 64
                continue
            
            # calculate expected_entropy
            entropies = [r["entropy"] for r in rollouts]
            expected_entropy = np.percentile(entropies, 75)  # 例如 p75
            target_entropy_dict[diff] = expected_entropy

            min_len, max_len, interval = 96, self.config.data.max_response_length, 32
            C_d = {"easy": 6, "medium": 3, "hard": 1}[diff]
            found = False
            for l in range(min_len, max_len + 1, interval):
                # 从全局样本sample
                all_rollout_num = len(rollouts)
                correct = [r for r in rollouts if r["is_correct"]]
                covered = [r for r in correct if r["length"] <= l]
                P_ld = len(covered) / max(1, all_rollout_num)
                ECR_d = P_ld * C_d
                if ECR_d >= 1.0:
                    self.target_length_dict[diff] = l
                    found = True
                    break
            if not found:
                self.target_length_dict[diff] = max_len  # fallback
        

        # self.target_high_entropy_token_num_dict = target_high_entropy_num_dict
        # drop_ratio = 0.8  # 或者 0.4, 0.3
        # self.drop_allpass_entropy_threshold = min_entropy + drop_ratio * (self.target_high_entropy_token_num_dict['easy'] - min_entropy)
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        # val_reward_metrics.update(
        #     {
        #         "val/difficulty_statistics": val_difficulty_statistics,
        #         "val/target_length": self.target_length_dict,
        #     }
        # )
        val_reward_metrics.update({"val/self.expected_high_entropy_token_num": self.target_high_entropy_token_num_dict})
        val_reward_metrics.update({"val/self.high_entropy_token_threshold": self.high_entropy_token_threshold})
        # import pdb; pdb.set_trace()  # Debugging breakpoint
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics}

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: Dict[str, Any]) -> DataProto:
        # get the ref response together with corresponding prompts,but the actor rollout hasn't been generated yet
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            # except StopIteration:
            #     self.data_iterator = iter(self.train_dataloader)
            #     batch_dict = next(self.data_iterator)
            except StopIteration:
                print("[_make_batch_data] Iterator exhausted.")
                return None
            except Exception as e:
                print(f"[_make_batch_data] Other exception: {e}")
                import traceback; traceback.print_exc()
                return None
            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
                "high_entropy_threshold": self.high_entropy_token_threshold,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps", "high_entropy_threshold"],
            )

            # generate a batch
            # gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences_with_tokenwise(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            # import pdb;pdb.set_trace()
            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)
            # import pdb;pdb.set_trace()

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise ValueError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]


    def bootstrap(self):
        """
        Bootstrap difficulty_dict and target_length_dict before training using rollouts.
        """
        print("[Bootstrap] Start bootstrap epoch for Laser-D difficulty + length estimation.")
        rollout_n = self.config.worker.rollout.n
        # max_batches = getattr(self.config.trainer, "bootstrap_max_batches", 20)

        uid2acc = defaultdict(list)
        uid2len = defaultdict(list)

        self.actor_rollout_ref_wg.prepare_rollout_engine()

        self.data_iterator = iter(self.bootstrap_train_dataloader)

        # self.actor_rollout_ref_wg.release_rollout_engine()
        for batch_dict in tqdm(self.bootstrap_train_dataloader,desc="[Bootstrap] Processing batches"):
            try:
                # 构造 DataProto
                test_batch = DataProto.from_single_dict(batch_dict)
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
                test_gen_batch.meta_info = {
                    "n": rollout_n,
                    "min_pixels": self.config.data.min_pixels,
                    "max_pixels": self.config.data.max_pixels,
                    "video_fps": self.config.data.video_fps,
                }

                test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)

                test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)

                test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * rollout_n)

                # concatnate to final result
                test_batch = test_batch.repeat(repeat_times=rollout_n, interleave=True)
                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate reward
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(test_batch))
                accuracies = reward_metrics["accuracy"]
                lengths = test_batch.batch["response_mask"].sum(dim=-1).cpu().tolist()
                global_ids = test_batch.non_tensor_batch["global_id"]
                entropies = test_batch.non_tensor_batch["entropies"]

                self.update_difficulty_and_skip_gid_set(test_batch)

            except Exception as e:
                print(f"[Bootstrap] Failed to process batch: {e}")
                continue

        self.actor_rollout_ref_wg.release_rollout_engine()

        self.logger.log(
            data={
                "epoch/id": self.epoch_id,
                "epoch/skip_gid_count": len(self.skip_gid_set),
                "epoch/iter_num":len(self.train_dataloader) 
            },
            step=self.global_step
        )

        # print(f"[Bootstrap] difficulty_dict entries: {len(self.difficulty_dict)}")

    def compute_batch_accuracy(self, batch: DataProto) -> List[float]:
        responses = batch.batch["responses"]
        response_lengths = batch.batch["response_mask"].sum(dim=-1)
        ground_truths = batch.non_tensor_batch["ground_truth"]
        accuracies = []

        for response_ids, gt, response_length in zip(responses, ground_truths, response_lengths):
            valid_response_ids = response_ids[: response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )
            answer = extract_boxed_content(response_str)
            accuracies.append(1.0 if grade_answer(answer, gt) else 0.0)

        return accuracies
    
    def update_difficulty_and_skip_gid_set(self, batch: DataProto) -> None:
        global_ids = batch.non_tensor_batch["global_id"]
        accuracies = batch.non_tensor_batch["accuracy"]
        entropies = batch.non_tensor_batch["entropies"]
        high_entropy_token_nums = batch.non_tensor_batch["high_entropy_token_num"]
        difficulty_coefficient = {
            "easy": 0.75,
            "medium": 1.0,
            "hard": 1.25,
        }
        uid_info = defaultdict(lambda: {"acc": [], "entropy": [], "high_entropy_token_num": []})
        difficulty_bucket = defaultdict(list)

        for gid, acc, entropy, tokennum in zip(global_ids, accuracies, entropies, high_entropy_token_nums):
            uid_info[gid]["acc"].append(acc)
            uid_info[gid]["entropy"].append(entropy)
            uid_info[gid]["high_entropy_token_num"].append(tokennum)

        for gid, info in uid_info.items():
            acc_rate = np.mean(info["acc"])
            mean_token_num = np.mean(info["high_entropy_token_num"])
            # dynamic sampling here
            # if acc_rate == 1.0 and mean_entropy < self.drop_allpass_entropy_threshold:
            #     self.skip_gid_set.add(gid)
            if acc_rate >= 2 / 3:
                difficulty = "easy"
            elif acc_rate >= 1 / 3:
                difficulty = "medium"
            else:
                difficulty = "hard"

            self.difficulty_dict_train[gid] = difficulty
            difficulty_bucket[difficulty].append(mean_token_num)

        for diff, values in difficulty_bucket.items():
            self.target_high_entropy_token_num_dict[diff] = round(difficulty_coefficient[diff] * np.mean(values))
            
        drop_ratio = 0.8
        if difficulty_bucket["easy"]:
            min_entropy = min(difficulty_bucket["easy"])
            self.drop_allpass_entropy_threshold = min_entropy + drop_ratio * (
                self.target_high_entropy_token_num_dict["easy"] - min_entropy
            )
        self.drop_allpass_entropy_threshold = min_entropy + drop_ratio * (self.target_high_entropy_token_num_dict['easy'] - min_entropy)
        print(f"[Train] Updated difficulty_dict, size={len(self.difficulty_dict_train)}")

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step",position=0)
        val_metrics: Optional[Dict[str, Any]] = None
        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)


        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        if self.config.data.bootstrap and self.global_step == 0:
            self.bootstrap()
            self.train_dataloader = create_filtered_train_dataloader(
            config=self.config.data,
            base_dataset=self.base_dataset,
            skip_gid_set=self.skip_gid_set
            )
            self.data_iterator = iter(self.train_dataloader)
            self.logger.log(
                data={
                    "epoch/id": self.epoch_id,
                    "epoch/skip_gid_count": len(self.skip_gid_set),
                    "epoch/iter_num":len(self.train_dataloader) 
                },
                step=self.global_step,
            )
        # perform validation before training
        # currently, we only support validation using the reward_function.
        # if self.val_reward_fn is not None and self.config.trainer.val_before_train:
        #     val_metrics = self._validate()
        #     self.logger.log(data=val_metrics, step=self.global_step)
        #     if self.config.trainer.val_only:
        #         return
        


        while self.global_step < self.training_steps:
            self.global_step += 1
            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    if batch is None:
                        self.epoch_id += 1
                        self.train_dataloader = create_filtered_train_dataloader(
                        config=self.config.data,
                        base_dataset=self.base_dataset,
                        skip_gid_set=self.skip_gid_set
                        )
                        self.data_iterator = iter(self.train_dataloader)
                        # import pdb;pdb.set_trace()
                        batch = self._make_batch_data(metrics=metrics)
                        # ✅ Log current epoch + skip gid count to wandb
                        self.logger.log(
                            data={
                                "epoch/id": self.epoch_id,
                                "epoch/skip_gid_count": len(self.skip_gid_set),
                                "epoch/iter_num":len(self.train_dataloader) 
                            },
                            step=self.global_step,
                        )
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                # import pdb;pdb.set_trace()
                self._balance_batch(batch, metrics=metrics)
                
                batch.non_tensor_batch['accuracy'] = self.compute_batch_accuracy(batch)


                # use training batch to determine the high_entropy_token_threshold
                global_ids = batch.non_tensor_batch["global_id"]
                self.high_entropy_token_threshold = np.mean(batch.non_tensor_batch["entropy_percentile_thresholds"])
                self.update_difficulty_and_skip_gid_set(batch)
                batch.non_tensor_batch["dynamic_token_length"] = [self.target_length_dict[self.difficulty_dict_train[gid]] for gid in global_ids]
                batch.non_tensor_batch["difficulty"] = [self.difficulty_dict_train[gid] for gid in global_ids]
                batch.non_tensor_batch["target_high_entropy_token_num"] = [self.target_high_entropy_token_num_dict[self.difficulty_dict_train[gid]] for gid in global_ids]


                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
