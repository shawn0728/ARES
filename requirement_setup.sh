# 安装 requirements

# pip install -r /cpfs/user/yym/projects/EasyR1/wandb_safe.txt --no-deps

# pip install ray==2.43.0 --no-deps

apt update && apt install -y tmux

cd /cpfs/user/yym/projects/EasyR1

pip install -r /cpfs/user/yym/projects/EasyR1/requirements_old.txt

pip install /cpfs/user/yym/projects/packages/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl --no-cache-dir

export WANDB_API_KEY=5b12869314286239ee1d4f6888a9c2c767e24ef5

export PYTHONUNBUFFERED=1

cd /cpfs/user/yym/projects/EasyR1


# bash /cpfs/user/yym/projects/EasyR1/examples/qwen2_5_vl_3b_geo3k_grpo_test.sh
# ABI_FLAG=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
# FILENAME="flash_attn-2.8.0.post2+cu12torch2.7cxx11abi${ABI_FLAG}-cp311-cp311-linux_x86_64.whl"
# URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/${FILENAME}"

# wget -O "$FILENAME" "$URL"
# wget -O /cpfs/user/yym/projects/packages/"$FILENAME" "$URL"