from distill_exp.gan.model_cfg import WanModelCFG
import os

# ckp_dir = "/cv/zhangpengpeng/cv/video_generation/DMD2_wanx/outputs/cache/time_0409_1448|26/checkpoint_model_000499/feedforward"
ckp_dir = "/cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/outputs/merged/exp4.3/750_32"
model = WanModelCFG.from_pretrained(ckp_dir)

# from scripts.train.model.lora_utils import  load_lora_weights_manually    

# lora_dir = "/cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/outputs/exp4.3/checkpoint-750"
# model = load_lora_weights_manually(model,lora_dir,lora_alpha=32)

# output_dir = "/cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/outputs/merged/exp4.3/"
# os.makedirs(output_dir, exist_ok=True)
# model = model.merge_and_unload() 
# model.save_pretrained(output_dir+"750_32")