#!/bin/bash

set -x


export WANDB_API_KEY=5b12869314286239ee1d4f6888a9c2c767e24ef5

export PYTHONUNBUFFERED=1
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
EXPERIMENT_NAME=ETN_01_qwen_3b_test_new_reward_denominator_all_samples

python3 -m verl.trainer.main \
    config=/cpfs/user/yym/projects/Adaptive-VL/experiments/${EXPERIMENT_NAME}/config.yaml \
    data.train_files=/cpfs/user/yym/hugging_face/datasets/hiyouga_geometry3k_train/geometry3k_with_paths.jsonl \
    data.val_files=/cpfs/user/yym/hugging_face/datasets/hiyouga_geometry3k_test/geometry3k_with_paths.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=${EXPERIMENT_NAME}\
    trainer.load_checkpoint_path=/cpfs/user/yym/projects/Adaptive-VL/checkpoints/easy_r1/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=1

