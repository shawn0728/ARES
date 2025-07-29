#!/bin/bash

set -x


export WANDB_API_KEY=5b12869314286239ee1d4f6888a9c2c767e24ef5

export PYTHONUNBUFFERED=1
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
EXPERIMENT_NAME=ETN_01.1_crucial_token_entropy_test_new
PROJECT_NAME=Adaptive_thinking
RUN_ID=crucial_token_entropy_trial_three

python3 -m verl.trainer.main \
    config=/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/experiments/${EXPERIMENT_NAME}/config.yaml \
    data.train_files=/cpfs/user/yym/hugging_face/datasets/hiyouga_geometry3k_train/geometry3k_with_paths.jsonl \
    data.val_files=/cpfs/user/yym/hugging_face/datasets/hiyouga_geometry3k_test/geometry3k_with_paths.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=${EXPERIMENT_NAME}\
    trainer.load_checkpoint_path=/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/checkpoints/easy_r1/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.wandb_run_id=${RUN_ID} \

