
#!/bin/bash

set -x

cd /cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_mean_entropy

pip install -r /cpfs/user/yym/projects/EasyR1/requirements_old.txt

pip install /cpfs/user/yym/projects/packages/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl --no-cache-dir

# wandb setting

# lennonye的wandb api key
# export WANDB_API_KEY=5b12869314286239ee1d4f6888a9c2c767e24ef5

# fufu的wandb api key
# export WANDB_API_KEY="9c62933f71d04da55b4bc9e7f61f72356dfbf071"

export WANDB_BASE_URL=https://api.wandb.ai
# fufu的api key
export WANDB_API_KEY="9c62933f71d04da55b4bc9e7f61f72356dfbf071"
PROJECT_NAME=Adaptive_thinking
RUN_ID=mean_entropy_trial_one

export PYTHONUNBUFFERED=1

EXPERIMENT_NAME=revisual_7b_geo3k_grpo_mean_entropy_dynamic_sampling
MODEL_PATH=/cpfs/user/yym/models/revisual-r1  # replace it with your local file path

python3 -m verl.trainer.main \
    config=/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_mean_entropy/experiments/${EXPERIMENT_NAME}/config.yaml \
    data.train_files=/cpfs/user/yym/hugging_face/datasets/hiyouga_geometry3k_train/geometry3k_with_paths.jsonl \
    data.val_files=/cpfs/user/yym/hugging_face/datasets/hiyouga_geometry3k_test/geometry3k_with_paths.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.load_checkpoint_path=/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_mean_entropy/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.wandb_run_id=${RUN_ID} \
    # worker.rollout.tensor_parallel_size=1 \