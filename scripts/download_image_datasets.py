import os
import json
import base64
from io import BytesIO
from PIL import Image
from datasets import load_dataset
import random

def save_pil_images_and_rewrite(dataset_name="hiyouga/geometry3k", split="train", save_dir="/cpfs/user/yym/hugging_face/datasets/hiyouga_geometry3k_train_new/geometry3k_images", output_path="/cpfs/user/yym/hugging_face/datasets/csfufu_mmrl/geometry3k_with_paths.jsonl"):
    os.makedirs(save_dir, exist_ok=True)

    ds = load_dataset(dataset_name, split=split)

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, ex in enumerate(ds):
            image_paths = []
            for i, img in enumerate(ex["images"]):
                if not isinstance(img, Image.Image):
                    continue  # 跳过非 PIL 图片（以防万一）

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

def save_base64_images(
    dataset_name="csfufu/mmrl",
    split="train",
    save_dir="/cpfs/user/yym/hugging_face/datasets/csfufu_mmrl/images",
    output_path="/cpfs/user/yym/hugging_face/datasets/csfufu_mmrl/csfufu_mmrl.jsonl"
):
    os.makedirs(save_dir, exist_ok=True)

    ds = load_dataset(dataset_name, split=split)

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, ex in enumerate(ds):
            image_paths = []
            for i, img_obj in enumerate(ex["images"]):
                try:
                    # import pdb;pdb.set_trace()
                    bytesstr = img_obj["bytes"]
                    # img_data = base64.b64decode(b64str)
                    img_data = bytesstr
                    img = Image.open(BytesIO(img_data)).convert("RGB")

                    filename = f"img_{idx}_{i}.png"
                    full_path = os.path.join(save_dir, filename)
                    img.save(full_path)
                    image_paths.append(full_path)
                except Exception as e:
                    print(f"❌ Failed to process image {idx}_{i}: {e}")
                    continue

            fout.write(
                json.dumps({
                    "problem": ex.get("problem", ""),
                    "answer": ex.get("answer", ""),
                    "images": image_paths,
                    "pass_rate": ex.get("pass_rate", None),
                    "difficulty_score": ex.get("difficulty_score", None),
                    "extracted_answers": ex.get("extracted_answers", []),
                    "global_id": f"{dataset_name.replace('/', '_')}-{split}-{idx:06d}"
                }, ensure_ascii=False) + "\n"
            )

def split_train_test_jsonl(file_path="/cpfs/user/yym/hugging_face/datasets/csfufu_mmrl/csfufu_mmrl.jsonl",test_ratio=0.1,test_sample_num=1000):

    # 输入输出路径
    input_path = file_path  # 替换成你的源文件路径
    train_output_path = input_path.split('.')[0] + "_train.jsonl"
    val_output_path = input_path.split('.')[0] + "_val.jsonl"

    # 设置验证集大小
    val_size = test_sample_num

    # 读取所有行
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 随机划分
    random.seed(42)  # 固定随机种子，确保可复现
    val_lines = random.sample(lines, k=val_size)
    val_set = set(val_lines)
    train_lines = [line for line in lines if line not in val_set]

    # 写入文件
    with open(train_output_path, "w", encoding="utf-8") as f_train:
        f_train.writelines(train_lines)

    with open(val_output_path, "w", encoding="utf-8") as f_val:
        f_val.writelines(val_lines)

    print(f"✅ 写入完成：训练集 {len(train_lines)} 条，验证集 {len(val_lines)} 条")
def main():
    save_pil_images_and_rewrite()
    # save_base64_images()
    # split_train_test_jsonl()

if __name__ == "__main__":
    main()