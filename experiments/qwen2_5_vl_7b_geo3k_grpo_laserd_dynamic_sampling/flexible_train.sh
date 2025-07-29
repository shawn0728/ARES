
#!/bin/bash

set -x

cd /cpfs/user/yym/projects/EasyR1_duplicated

pip install -r /cpfs/user/yym/projects/EasyR1/requirements_old.txt

pip install /cpfs/user/yym/projects/packages/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl --no-cache-dir

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

# wandb setting

# lennonye的wandb api key
# export WANDB_API_KEY=5b12869314286239ee1d4f6888a9c2c767e24ef5

# fufu的wandb api key
export WANDB_API_KEY="9c62933f71d04da55b4bc9e7f61f72356dfbf071"

export WANDB_BASE_URL=https://api.wandb.ai
EXPERIMENT_NAME=qwen2_5_vl_7b_geo3k_grpo_laserd_dynamic_sampling
PROJECT_NAME=Adaptive_thinking
RUN_ID=grpo_laserd_trial_1

export PYTHONUNBUFFERED=1

python3 -m verl.trainer.main \
    config=/cpfs/user/yym/projects/EasyR1_duplicated/experiments/${EXPERIMENT_NAME}/config.yaml \
    data.train_files=/cpfs/user/yym/hugging_face/datasets/hiyouga_geometry3k_train/geometry3k_with_paths.jsonl \
    data.val_files=/cpfs/user/yym/hugging_face/datasets/hiyouga_geometry3k_test/geometry3k_with_paths.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.load_checkpoint_path=/cpfs/user/yym/projects/EasyR1_duplicated/checkpoints/${project_name}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.wandb_run_id=${RUN_ID}