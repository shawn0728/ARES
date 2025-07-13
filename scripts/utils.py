import os
import json
from dotenv import load_dotenv
import os
from datasets import load_dataset


def download_hf_datasets():
    ds = load_dataset("hiyouga/geometry3k", split="train", trust_remote_code=True)
    with open("/cpfs/user/yym/hugging_face/datasets/geometry3k_train_with_id.jsonl", "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            ex["global_id"] = f"geometry-train-{i:06d}"
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def download_hf_model(repo_id,local_dir):
    from huggingface_hub import snapshot_download
    snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,  
    local_dir_use_symlinks=False,
    token="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"
    )
def reindex_wrong_tensor_index():
    from safetensors import safe_open
    model_dir = "/cpfs/user/yym/models/text-coldstart-adaptive-v01"
    output_index_file = os.path.join(model_dir, "model.safetensors.index.json")

    weight_map = {}
    metadata = {}

    # 遍历所有分片
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".safetensors") and "-of-" in fname:
            full_path = os.path.join(model_dir, fname)
            with safe_open(full_path, framework="pt") as f:
                for key in f.keys():
                    weight_map[key] = fname

    # 写入新的 index 文件
    with open(output_index_file, "w") as f:
        json.dump({"metadata": metadata, "weight_map": weight_map}, f, indent=2)

    print(f"✅ 新 index 文件写入成功: {output_index_file}")

def check_params():
    from transformers import AutoModelForVision2Seq
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
    from PIL import Image
    model = AutoModelForVision2Seq.from_pretrained(
    "/cpfs/user/yym/models/text-coldstart-adaptive-v01",
    trust_remote_code=True
    ).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数数量: {total_params}")

def check_lm_head(model, tokenizer):
    print("=== 🔍 检查 lm_head 状态 ===")
    
    # 检查是否有 lm_head 属性
    if not hasattr(model, "lm_head"):
        print("❌ 模型中没有 lm_head！这通常说明权重未正确保存或加载。")
        return

    lm_head_weight = model.lm_head.weight
    vocab_size = tokenizer.vocab_size
    head_shape = tuple(lm_head_weight.shape)

    print(f"✅ lm_head.weight.shape: {head_shape}")
    print(f"✅ tokenizer.vocab_size: {vocab_size}")

    if head_shape[0] != vocab_size:
        print("⚠️ vocab_size 与 lm_head 第一维不匹配！可能使用了不同 tokenizer 或权重不完整。")
    else:
        print("🎯 lm_head 与 tokenizer 匹配，没问题！")

def count_all_pass_number(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f) 
        print("current number of all pass examples",len(data))



def main():
    # local_path = "/cpfs/user/yym/models/huggingface_base_models"
    # model = AutoModelForVision2Seq.from_pretrained(
    #     local_path,
    #     torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # ).cuda()
    # processor = AutoProcessor.from_pretrained(local_path)
    # tokenizer = processor.tokenizer
    # check_lm_head(model, tokenizer)
    # download_hf_model(repo_id="MM-R1-HH/text-coldstart-adaptive-v01",local_dir="/cpfs/user/yym/models/text-coldstart-adaptive-v01-original")
    # download_hf_datasets()
    count_all_pass_number("/cpfs/user/yym/projects/EasyR1/checkpoints/easy_r1/EY01.5_qwen7b_csfufu_original_rwd_remove_easy_rank_8_bootstrap/global_step_60/skip_gid_set.json")

if __name__ == "__main__":
    main()

