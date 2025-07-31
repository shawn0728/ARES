from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files="/cpfs/user/yym/hugging_face/datasets/mix_geo3k_like/mix_geo3k_like_w_global_id.parquet",
    split="train",
)

import pdb;pdb.set_trace()
print(dataset[0])