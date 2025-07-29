from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from PIL import Image
import torch
from dotenv import load_dotenv
import os



def main():
    # local_path = "/cpfs/user/yym/projects/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo/global_step_16/actor/huggingface"
    local_path = "/cpfs/user/yym/models/text-coldstart-adaptive-v01"
    model = AutoModelForVision2Seq.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ).cuda()
    processor = AutoProcessor.from_pretrained(local_path)
    tokenizer = processor.tokenizer
    # 示例图片和文本（根据你任务类型调整）
    image = Image.open("/cpfs/user/yym/projects/EasyR1/assets/easyr1_grpo.png").convert("RGB")
    prompt = "请描述图中大致流程。"
    # import pdb; pdb.set_trace()
    # 第一步：构建对话模版
    conversation = [
        {"role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]

    # 第二步：使用 apply_chat_template 得到格式化文本（带 <|vision_start|> 之类）
    templated_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False  # 这一步只生成带 tag 的纯字符串
    )
    # import pdb; pdb.set_trace()
    # 第三步：tokenizer + image -> processor 构建完整输入
    inputs = processor(
        text=templated_prompt,
        # images=image,
        return_tensors="pt"
    ).to("cuda")

    # 第四步：调用 generate
    outputs = model.generate(**inputs, max_new_tokens=512)

    # 第五步：解码输出
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(response)
if __name__ == "__main__":
    main()
