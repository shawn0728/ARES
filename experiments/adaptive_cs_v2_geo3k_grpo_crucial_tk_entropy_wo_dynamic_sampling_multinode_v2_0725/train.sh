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
RUN_ID=adaptive_new_data_0726

export PYTHONUNBUFFERED=1

EXPERIMENT_NAME=adaptive_new_data_0726
MODEL_PATH=/cpfs/user/yym/models/adaptive-mm-v02  # replace it with your local file path

python3 -m verl.trainer.main \
    config=/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/experiments/${EXPERIMENT_NAME}/config.yaml \
    data.train_files=/cpfs/data/hude/dataset/data_curation/ViRL39K_dataset/only@train \
    data.val_files=/cpfs/data/chenshuang/datasets/hard_validation/data@train \
    data.max_response_length=16000 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.wandb_run_id=${RUN_ID} \
    # trainer.load_checkpoint_path=/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    # worker.actor.micro_batch_size_per_device_for_update=1 \
    # worker.actor.micro_batch_size_per_device_for_experience=8 \
    # worker.rollout.tensor_parallel_size=1 \