import os
import json
import random
from datasets import load_dataset
from PIL import Image

def save_pil_images_and_rewrite(
    dataset_name="hiyouga/geometry3k",
    split="train",
    save_dir="/cpfs/user/yym/hugging_face/datasets/hiyouga_geometry3k_train_seq/geometry3k_images",
    output_path="/cpfs/user/yym/hugging_face/datasets/hiyouga_geometry3k_train_seq/geometry3k_with_paths.jsonl"
):
    os.makedirs(save_dir, exist_ok=True)
    ds = load_dataset(dataset_name, split=split)

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, ex in enumerate(ds):
            image_paths = []
            for i, img in enumerate(ex["images"]):
                if isinstance(img, Image.Image):
                    filename = f"img_{idx}_{i}.png"
                    full_path = os.path.join(save_dir, filename)
                    img.convert("RGB").save(full_path)
                    image_paths.append(full_path)

            fout.write(
                json.dumps({
                    "problem": ex["problem"],
                    "answer": ex["answer"],
                    "images": image_paths,
                    "global_id": f"{dataset_name.replace('/', '_')}-{split}-{idx:06d}"
                }, ensure_ascii=False) + "\n"
            )

def split_train_val(
    file_path="/cpfs/user/yym/hugging_face/datasets/hiyouga_geometry3k/geometry3k_with_paths.jsonl",
    test_sample_num=1000
):
    train_output_path = file_path.replace(".jsonl", "_train.jsonl")
    val_output_path = file_path.replace(".jsonl", "_val.jsonl")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.seed(42)
    val_lines = random.sample(lines, k=min(test_sample_num, len(lines)))
    train_lines = [line for line in lines if line not in val_lines]

    with open(train_output_path, "w", encoding="utf-8") as f_train:
        f_train.writelines(train_lines)
    with open(val_output_path, "w", encoding="utf-8") as f_val:
        f_val.writelines(val_lines)

    print(f"✅ 写入完成：训练集 {len(train_lines)} 条，验证集 {len(val_lines)} 条")

def main():
    save_pil_images_and_rewrite()
    split_train_val()

if __name__ == "__main__":
    main()