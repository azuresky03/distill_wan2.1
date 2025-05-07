from safetensors import safe_open

tensors = {}
dir = "/cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/outputs/exp2.7/checkpoint-50/adapter_model.safetensors"
with safe_open(dir, framework="pt", device="cpu") as f:
    print(f.keys())
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
        breakpoint()