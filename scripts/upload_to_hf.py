from huggingface_hub import HfApi,whoami
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
print(whoami(token=hf_token)["name"])
local_dir = "/cpfs/user/yym/projects/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo/global_step_16/actor/huggingface"  # 你的 huggingface 文件夹路径
repo_id = "Lennonye123/easy_r1_test_qwen_3b"         # 你要创建/上传的 repo 名称，例如：xiaohongshu/qwen2_5_vl_3b_epoch1
api = HfApi(token=hf_token)
user_info = whoami(token=hf_token)
api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="model"
)
