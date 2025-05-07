#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from wan.modules.model import WanModel
from distill_exp.gan.model_cfg import WanModelCFG
from distill_exp.gan.model import WanGanModel

def main():
    # 直接设置变量，不使用参数解析
    checkpoint_path = "/cv/zhangpengpeng/cv/video_generation/Wan2.1/output/gan_training/checkpoints/epoch_0.pth"
    base_model_path = "/vepfs-zulution/models/Wan2.1-T2V-14B"
    num_layers = 1
    layer_idxs = [9, 19, 29, 39]
    desired_len = 16
    compressor_heads = 8
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载基础模型
    print("加载基础模型...")
    model_base = WanModel.from_pretrained(base_model_path)
    print(f"基础模型加载完成")
    
    # 2. 创建 GAN 模型
    print("创建 GAN 模型...")
    model = WanGanModel(
        model_base,
        num_layers=num_layers,
        layer_idxs=layer_idxs,
        device=device,
        dtype=torch.bfloat16,
        desired_len=desired_len,
        compressor_heads=compressor_heads
    )
    
    # 3. 加载检查点
    print(f"从检查点加载模型状态: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 4. 处理模型状态字典
    if "module" in next(iter(checkpoint['model'])):
        # 处理带有 "module." 前缀的状态字典 (DDP训练保存的模型)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀
            new_state_dict[name] = v
        model_state_dict = new_state_dict
    else:
        # 直接使用状态字典
        model_state_dict = checkpoint['model']
    
    # 5. 加载模型状态
    model.load_state_dict(model_state_dict,strict=True)
    print("模型状态加载成功")
    
if __name__ == "__main__":
    main()