#!/bin/bash
set -x

export PYTHONUNBUFFERED=1
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY="9c62933f71d04da55b4bc9e7f61f72356dfbf071" # replace with your own wandb api key
wandb_run_id=xxx # replace with your own wandb run id
PROJECT_NAME=xxx # replace with your own project name
EXPERIMENT_NAME=xxx # replace with your own experiment name
MODEL_PATH=xxx  # replace it with your local file path

python3 -m verl.trainer.main \
    config=./experiments/${EXPERIMENT_NAME}/config.yaml \
    data.train_files="train_dataset"@train \
    data.val_files="value_dataset"@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.load_checkpoint_path=./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.wandb_run_id=${wandb_run_id} \
    trainer.val_freq=5 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \