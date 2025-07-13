from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

file_path = "/cpfs/user/yym/models/adaptive-mm-v01"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    file_path,
    torch_dtype="auto",
    device_map={"": 0},  # 明确分配到 GPU 0
    tp_plan=None         # 显式禁用 tensor parallel
)


min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(file_path, min_pixels=min_pixels, max_pixels=max_pixels)

question = "Which of the boxes comes next in the sequence? Select answers from A-E"
prompt = '''You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.'''

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/cpfs/user/yym/projects/EasyR1/assets/easyr1_grpo.png",
            },
            {"type": "text", "text": question + prompt},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
print("Generating...")
generated_ids = model.generate(**inputs, max_new_tokens=2048)
print("Generation finished.")
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

print("Decoding...")
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

response_token_count = len(generated_ids_trimmed[0])
print(f"Response token count: {response_token_count}")



