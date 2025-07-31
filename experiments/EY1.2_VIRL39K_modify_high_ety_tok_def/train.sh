#!/bin/bash

set -x


export PYTHONUNBUFFERED=1
# MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
export WANDB_BASE_URL=https://api.wandb.ai


# lennonye的api key
# export WANDB_API_KEY=5b12869314286239ee1d4f6888a9c2c767e24ef5

# fufu的api key
export WANDB_API_KEY="9c62933f71d04da55b4bc9e7f61f72356dfbf071"
PROJECT_NAME=Adaptive_thinking
RUN_ID=EY1.2_VIRL39K_modify_high_ety_tok_def

export PYTHONUNBUFFERED=1

EXPERIMENT_NAME=EY1.2_VIRL39K_modify_high_ety_tok_def
MODEL_PATH=/cpfs/user/yym/models/adaptive-mm-v02  # replace it with your local file path

python3 -m verl.trainer.main \
    config=/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/experiments/${EXPERIMENT_NAME}/config.yaml \
    data.train_files=/cpfs/user/yym/hugging_face/datasets/VIRL39K/geometry3k_final_w_global_id.parquet \
    data.val_files=/cpfs/user/yym/hugging_face/datasets/MMK12_geometry3k/MMK12_geometry3k_w_global_id.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.load_checkpoint_path=/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.wandb_run_id=${RUN_ID} \
    # worker.actor.micro_batch_size_per_device_for_update=1 \
    # worker.actor.micro_batch_size_per_device_for_experience=8 \
    # worker.rollout.tensor_parallel_size=1 \