import os 
from safetensors.torch import save_file
import json
from distill_exp.gan.model_cfg import WanModelCFG
import torch

model = WanModelCFG.from_pretrained("/cv/wangxuekuan/release_model/wanx/distill_cfg_t2v/exp10_distill_cfg/checkpoint-500")
ckp_dir = "/cv/zhangpengpeng/cv/video_generation/DMD2_wanx/outputs/cache/time_0409_1448|26/checkpoint_model_000499/feedforward.bin"
model.load_state_dict(torch.load(ckp_dir,weights_only=True))

save_dir = ckp_dir.replace(".bin", "")
os.makedirs(save_dir, exist_ok=True)
# save using safetensors
weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
save_file(model.state_dict(), weight_path)
config_dict = dict(model.config)
if "dtype" in config_dict:
    del config_dict["dtype"]  # TODO
config_path = os.path.join(save_dir, "config.json")
# save dict as json
with open(config_path, "w") as f:
    json.dump(config_dict, f, indent=4)