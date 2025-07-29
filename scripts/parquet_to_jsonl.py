import pandas as pd
import json
from pathlib import Path

input_path = "/cpfs/user/yym/hugging_face/datasets/MMK12_geometry3k/MMK12_geometry3k_w_global_id.parquet"
output_jsonl = "/cpfs/user/yym/hugging_face/datasets/MMK12_geometry3k/MMK12_geometry3k_with_paths.jsonl"
image_dir = "/cpfs/user/yym/hugging_face/datasets/MMK12_geometry3k/images"
Path(image_dir).mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(input_path)

with open(output_jsonl, "w", encoding="utf-8") as fout:
    for idx, row in enumerate(df.to_dict(orient="records")):
        images = row.get("images", [])
        if not isinstance(images, list):
            images = list(images)

        image_paths = []
        for i, img in enumerate(images):
            if isinstance(img, dict) and isinstance(img.get("bytes"), bytes):
                img_path = f"{image_dir}/img_{idx}_{i}.jpg"
                with open(img_path, "wb") as img_file:
                    img_file.write(img["bytes"])
                image_paths.append(img_path)

        row["images"] = image_paths
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")