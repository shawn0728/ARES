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
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.modeling_flash_attention_utils import index_first_axis, pad_input, unpad_input

from ...protocol import DataProto
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
import numpy as np
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

        # Adaptive KL (per difficulty bucket) following PPO target-KL heuristic
        def _get(name: str, default):
            return getattr(self.config, name, default)

        dtarg_default = _get("kl_target_default", 0.01)
        beta_init = _get("kl_beta_init", 1.0)
        self._kl_state = {
            "easy": {"beta": beta_init, "dtarg": _get("kl_target_easy", dtarg_default), "last_kl": None},
            "medium": {"beta": beta_init, "dtarg": _get("kl_target_medium", dtarg_default), "last_kl": None},
            "hard": {"beta": beta_init, "dtarg": _get("kl_target_hard", dtarg_default), "last_kl": None},
        }
        # New: smoother log-space update controls
        self._kl_tol = _get("kl_update_tol", 1.5)
        self._beta_min = _get("kl_beta_min", 1e-4)
        self._beta_max = _get("kl_beta_max", 1e2)
        self._ema_decay = _get("kl_ema_decay", 0.97)
        self._kl_lr = _get("kl_beta_lr", 0.05)  # step in log-space
        self._kl_clip = _get("kl_beta_clip", 0.2)  # |delta| cap per update in error domain
        self._kl_update_interval = _get("kl_update_interval", 100)
        self._kl_deadzone = _get("kl_deadzone", 0.02)  # small error not updated
        # track EMA of error per bucket and update counters
        self._e_ema = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
        self._update_count = 0

    def _beta_for_levels(self, difficulty_levels, device):
        # map per-sample difficulty -> current beta
        betas = [self._kl_state.get(str(level), self._kl_state["medium"]).get("beta", 1.0) for level in difficulty_levels]
        return torch.tensor(betas, device=device, dtype=torch.float32).unsqueeze(-1)

    def _update_beta_per_bucket(self, difficulties, seq_mean_kl: torch.Tensor):
        # group per-bucket mean KL, then apply smoother log-space update
        self._update_count += 1
        bucket_values: Dict[str, list] = {"easy": [], "medium": [], "hard": []}
        for diff, val in zip(difficulties, seq_mean_kl.detach().cpu().tolist()):
            key = str(diff)
            if key in bucket_values:
                bucket_values[key].append(val)
        for diff, vals in bucket_values.items():
            if not vals:
                continue
            cur = float(sum(vals) / len(vals))
            st = self._kl_state[diff]
            last = st.get("last_kl")
            if last is None:
                last = cur
            mean_kl = (1.0 - self._ema_decay) * cur + self._ema_decay * last
            st["last_kl"] = mean_kl
            # error signal vs target (within tolerance band)
            target = max(1e-8, st["dtarg"])  # avoid zero
            e = (mean_kl / target) - 1.0
            # deadzone
            if abs(e) < self._kl_deadzone:
                continue
            # only update at intervals
            if (self._update_count % self._kl_update_interval) != 0:
                continue
            # EMA over error then capped small step in log-space
            self._e_ema[diff] = (1.0 - self._ema_decay) * e + self._ema_decay * self._e_ema[diff]
            de = float(np.clip(self._e_ema[diff], -self._kl_clip, self._kl_clip))
            # update log(beta)
            cur_beta = max(self._beta_min, min(self._beta_max, st["beta"]))
            logb = float(np.log(cur_beta))
            logb_new = np.clip(logb + self._kl_lr * de, np.log(self._beta_min), np.log(self._beta_max))
            st["beta"] = float(np.exp(logb_new))

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            for input_dict in micro_batch["multi_modal_inputs"]:
                for key, value in input_dict.items():
                    multi_modal_inputs[key].append(value)

            for key, value in multi_modal_inputs.items():
                if len(value) != 0:
                    multi_modal_inputs[key] = torch.cat(value, dim=0)
                else:
                    multi_modal_inputs[key] = None

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs


    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs", "difficulty"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]

                        # Base KL per token (unscaled), then adaptive per-difficulty beta
                        kld_base = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )  # (bsz, response_length)

                        if getattr(self.config, "dynamic_kl_loss", True) and ("difficulty" in model_inputs):
                            difficulty_levels = model_inputs["difficulty"]
                            # compute per-seq mean KL for beta update (mask-aware)
                            with torch.no_grad():
                                mask = response_mask.float()
                                valid = torch.clamp(mask.sum(dim=1), min=1.0)
                                seq_mean_kl = (kld_base * mask).sum(dim=1) / valid
                                self._update_beta_per_bucket(difficulty_levels, seq_mean_kl)
                            beta_per_seq = self._beta_for_levels(difficulty_levels, device=kld_base.device)
                            kld = kld_base * beta_per_seq
                        else:
                            kld = kld_base

                        # --- Second-chance KL relaxation (local, per-sample) ---
                        if getattr(self.config, "enable_second_chance_kl", True):
                            try:
                                # optional signals
                                difficulties = model_inputs.get("difficulty", None)
                                accuracies = model_inputs.get("accuracy", None)
                                soft_costs = model_inputs.get("reasoning_soft_cost", None)
                                relax_vec = torch.ones((kld.shape[0], 1), device=kld.device, dtype=kld.dtype)
                                if soft_costs is not None:
                                    # build mask: (easy-only?) & (incorrect-only?) & (soft_cost high)
                                    soft_cost_t = torch.tensor(soft_costs, device=kld.device, dtype=kld.dtype).view(-1, 1)
                                    mask = (soft_cost_t >= self.config.second_chance_soft_cost_min)
                                    if difficulties is not None and getattr(self.config, "second_chance_easy_only", True):
                                        diff_mask = torch.tensor([1.0 if str(d)=="easy" else 0.0 for d in difficulties], device=kld.device, dtype=kld.dtype).view(-1,1)
                                        mask = mask & (diff_mask > 0.5)
                                    if accuracies is not None and getattr(self.config, "second_chance_incorrect_only", True):
                                        acc_t = torch.tensor(accuracies, device=kld.device, dtype=kld.dtype).view(-1,1)
                                        mask = mask & (acc_t < 0.5)
                                    # apply per-sample scale
                                    scale = self.config.second_chance_kl_relax
                                    relax_vec = torch.where(mask, torch.full_like(relax_vec, float(scale)), relax_vec)
                                # multiply tokenwise KL by per-sample factor
                                kld = kld * relax_vec
                            except Exception:
                                pass

                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        pg_loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef

                    loss = pg_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_metrics["pg_clipfrac_higher"],
                        "actor/pg_clipfrac_lower": pg_metrics["pg_clipfrac_lower"],
                        "actor/entropy_loss": pg_metrics["entropy_loss"],
                        "actor/ppo_kl": pg_metrics["ppo_kl"],
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics


