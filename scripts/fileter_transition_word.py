import json

with open("/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/mess/mean_entropy_token_dict_no_G.json", "r") as f:
    token_entropy = json.load(f)

turning_words = {"but", "however", "though", "although", "nevertheless", "nonetheless",
                 "yet", "still", "even", "instead", "whereas", "while", "otherwise"}

filtered = {token: entropy for token, entropy in token_entropy.items() if token.lower() in turning_words}

print("筛选出的转折词：", filtered.keys())
print(f"共 {len(filtered)} 个转折词")