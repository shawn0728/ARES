cd /cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy
while true
do
    bash /cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/experiments/mimo_7b_geo3k_grpo_crucial_tk_entropy_dynamic_sampling/train.sh
    exit_code=$?
    echo "Training crashed with exit code $exit_code, restarting in 10 seconds..."
    sleep 10
done