#!/usr/bin/env bash
set -euo pipefail

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

# ===== 基本配置（按需改） =====
export RAY_PORT=6379
# export RAY_DASHBOARD_PORT=8265         # 可不需要
export NUM_GPUS_PER_NODE=8              # 每节点 GPU 数


# 修改TRRAIN_SH为训练脚本路径
export TRAIN_SH="/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/experiments/final_adaptive_16k_mix89k/train.sh"
export WORKDIR="/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy"

# ===== 自动生成任务ID（基于实验名称）=====
# 从TRAIN_SH路径中提取experiments文件夹后的实验名称
export TASK_ID=$(echo "${TRAIN_SH}" | sed -n 's/.*\/experiments\///p' | sed 's/\/train\.sh$//')
echo "[INFO] Task ID: ${TASK_ID}"

# 所有节点都能读写的共享目录，每个任务使用独立子目录
export SHARED_DIR="/cpfs/user/yym/ray/${TASK_ID}"
export RAY_ADDR_FILE="$SHARED_DIR/ray_head_addr.txt"
export RAY_LOCK_DIR="$SHARED_DIR/ray_head_lock"   # 用于head抢占的原子锁目录

# 创建任务专用目录
mkdir -p "${SHARED_DIR}"
echo "[INFO] Task directory: ${SHARED_DIR}"


# ===== 获取本机内网 IP =====
get_ip() {
  # 使用 hostname -I 获取IP地址，取第一个（通常是内网IP）
  hostname -I | awk '{print $1}' || true
}
LOCAL_IP="$(get_ip)"
if [[ -z "${LOCAL_IP}" ]]; then
  echo "[FATAL] Cannot determine local IP." >&2
  exit 1
fi
echo "[INFO] Local IP = ${LOCAL_IP}"

# ===== 清理残留的 ray（可选）=====
ray stop || true

# ===== 抢占为 head（原子创建目录：成功 => 本机是 head）=====
if mkdir "${RAY_LOCK_DIR}" 2>/dev/null; then
  echo "[INFO] This node becomes HEAD."

  # 启动 head
  ray start --head \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --num-gpus="${NUM_GPUS_PER_NODE}"

  # 写出地址供 worker 读取
  echo "${LOCAL_IP}:${RAY_PORT}" > "${RAY_ADDR_FILE}"
  echo "[INFO] Head address written to ${RAY_ADDR_FILE}: $(cat ${RAY_ADDR_FILE})"

  # （可选）等几秒让 worker 加入
  sleep 60

  # 仅在 head 执行你的训练脚本
  cd "${WORKDIR}"
  export RAY_ADDRESS=auto
  bash "${TRAIN_SH}"

  # 训练结束后，如需自动停掉集群，取消注释：
  # ray stop
else
  echo "[INFO] This node will join as WORKER. Waiting for ${RAY_ADDR_FILE} ..."

  # 等待 head 写出地址
  for i in {1..180}; do
    if [[ -s "${RAY_ADDR_FILE}" ]]; then
      break
    fi
    sleep 1
  done
  if [[ ! -s "${RAY_ADDR_FILE}" ]]; then
    echo "[FATAL] Timeout waiting for head address file: ${RAY_ADDR_FILE}" >&2
    exit 1
  fi

  HEAD_ADDR="$(cat "${RAY_ADDR_FILE}")"
  echo "[INFO] HEAD_ADDR = ${HEAD_ADDR}"

  # 启动 worker 并指向 head
  ray start \
    --address="${HEAD_ADDR}" \
    --num-gpus="${NUM_GPUS_PER_NODE}"

  # 如需让脚本常驻（防止容器退出），可加：
  # tail -f /dev/null
fi
