import json
from collections import defaultdict


import re
import unicodedata

import re
import unicodedata
from typing import Tuple

def clean_and_check_token(token: str) -> Tuple[str, bool]:
    cleaned = re.sub(r"^[▁Ġ]+", "", token)
    # 原始清洗：去前缀、转小写
    cleaned = cleaned.lower()

    # 排除空字符串
    if not cleaned:
        return cleaned, False

    # 纯标点或控制字符（Z=space separator, P=punctuation, C=other control）
    if all(unicodedata.category(c).startswith(("P", "Z", "C")) for c in cleaned):
        return cleaned, False

    # 全部是非单词字符（\w = [a-zA-Z0-9_]）
    if re.fullmatch(r"[^\w]+", cleaned):
        return cleaned, False

    # 特殊token样式，如 <pad>, <|endoftext|>
    if re.match(r"^<.*>$", cleaned):
        return cleaned, False

    # 不可打印字符
    if not all(c.isprintable() for c in cleaned):
        return cleaned, False

    # 可选：太短而无意义
    if len(cleaned) == 1 and not cleaned.isalnum():
        return cleaned, False

    return cleaned, True



def main():
    input_file = "/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/mess/token_entropy_log_0724_cs_v02_total_geometry3k.jsonl"
    output_file = "/cpfs/user/yym/projects/Adaptive-VL-worktrees/exp_crucial_token_entropy/mess/token_entropy_log_0724_cs_v02_total_geometry3k_filtered_readable.json"

    token_entropy_list = defaultdict(list)

    with open(input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            cleaned_token, is_valid = clean_and_check_token(item["token"])
            entropy = item["entropy"]
            if is_valid:
                token_entropy_list[cleaned_token].append(entropy)

    token_entropy_mean = {token: sum(entropies) / len(entropies) for token, entropies in token_entropy_list.items()}

    with open(output_file, "w") as f:
        json.dump(token_entropy_mean, f, ensure_ascii=False, indent=2)

    print(f"，共 {len(token_entropy_mean)} 个token。输出文件：{output_file}")

if __name__ == "__main__":
    main()