import pandas as pd
import re
import re
import json

def fix_parquet():
    df = pd.read_parquet("/cpfs/user/yym/hugging_face/datasets/MMK12_geometry3k/MMK12_geometry3k_w_global_id.parquet")  # 原 parquet 路径

    def fix_problem_text(text):
        if isinstance(text, str):
            return re.sub(r"<image_(\d+)>", r"<image\1>", text)
        return text

    df["problem"] = df["problem"].apply(fix_problem_text)
    df.to_parquet("/cpfs/user/yym/hugging_face/datasets/MMK12_geometry3k/MMK12_geometry3k_w_global_id_fixed_question.parquet")  # 修复后输出路径


def fix_jsonl():
    # 输入输出路径
    input_path = "/cpfs/user/yym/hugging_face/datasets/VIRL39K/ViRL39K_with_paths.jsonl"
    output_path = "/cpfs/user/yym/hugging_face/datasets/VIRL39K/ViRL39K_with_paths_fixed.jsonl"

    def fix_image_tags(text):
        if isinstance(text, str):
            return re.sub(r"<image_(\d+)>", r"<image\1>", text)
        return text

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            example = json.loads(line)
            if "problem" in example:
                example["problem"] = fix_image_tags(example["problem"])
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"✅ Fixed JSONL saved to: {output_path}")

def main():
    fix_jsonl()
main()