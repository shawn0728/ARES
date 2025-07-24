import json
from collections import defaultdict

input_file = "/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/mess/token_entropy_log.jsonl"
output_file = "/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/mess/mean_entropy_token_dict_no_G.json"

token_entropy_list = defaultdict(list)

with open(input_file, "r") as f:
    for line in f:
        item = json.loads(line)
        token = item["token"].lstrip("Ġ")  # 去除开头的 Ġ
        entropy = item["entropy"]
        token_entropy_list[token].append(entropy)

token_entropy_mean = {token: sum(entropies) / len(entropies) for token, entropies in token_entropy_list.items()}

with open(output_file, "w") as f:
    json.dump(token_entropy_mean, f, ensure_ascii=False, indent=2)

print(f"去除 'Ġ' 前缀且按平均值聚合完成，共 {len(token_entropy_mean)} 个token。输出文件：{output_file}")