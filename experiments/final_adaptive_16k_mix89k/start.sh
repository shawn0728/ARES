eval "$(/cpfs/data/chenshuang/tool/bin/conda shell.bash hook)"
conda activate csgo
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd /cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy
# 强制使用软文件锁
export FILELOCK_SOFT=1

export HF_DATASETS_FILELOCK_TIMEOUT=1
export HF_DATASETS_FILELOCK_TYPE=soft


# 禁用各种缓存
export DATASETS_DISABLE_CACHING=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 使用本地目录而不是网络文件系统
mkdir -p /tmp/hf_cache_$(whoami)
export HF_DATASETS_CACHE=/tmp/hf_cache_$(whoami)
export HF_HOME=/tmp/hf_cache_$(whoami)
export TRANSFORMERS_CACHE=/tmp/hf_cache_$(whoami)

ray stop

bash /cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/experiments/final_adaptive_16k_mix89k/train.sh