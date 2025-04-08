import os
import json
import torch
from safetensors.torch import safe_open
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    PeftModel
)
from torch import nn

def create_wan_lora_model(model, lora_rank=16, lora_alpha=32, lora_dropout=0, apply_to_ffn=True):
    """
    为WanModelCFG模型应用LoRA，并返回可训练的PEFT模型
    
    Args:
        model: 原始WanModelCFG模型实例
        lora_rank: LoRA秩参数
        lora_alpha: LoRA alpha参数
        lora_dropout: LoRA dropout参数
        
    Returns:
        peft_model: 应用了LoRA的PEFT模型
    """
    # 使用正则表达式匹配所有以.q、.k、.v、.o结尾的模块
    # target_modules = [r".*\.(q|k|v|o)$"]

    module_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module_names.append(name)
    
    # print("可用的线性层模块:")
    # print("\n".join(module_names))
    
    # 匹配注意力层的q, k, v, o
    target_modules = []
    
    # 使用列表指定精确的模块名称，而不是正则表达式
    for name in module_names:
        if name.endswith('.q') or name.endswith('.k') or name.endswith('.v') or name.endswith('.o'):
            target_modules.append(name)
            
    if apply_to_ffn:
        # 添加FFN层，根据打印出的模块名进行匹配
        for name in module_names:
            if '.ffn.0' in name or '.ffn.2' in name:
                target_modules.append(name)
    
    # print(f"将应用LoRA的模块: {target_modules}")
    
    # 创建LoRA配置
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
    )
    
    # 转换为PEFT模型
    peft_model = get_peft_model(model, lora_config)
    
    return peft_model

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
import time
from safetensors.torch import save_file
def save_checkpoint(model, optimizer, scheduler, save_path, step, rank):
    """保存FSDP+PEFT模型的完整检查点"""
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f"{save_path}/checkpoint-{step}", exist_ok=True)
        
    torch.distributed.barrier()
    
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True,rank0_only=True),
    ):
        state_dict = model.state_dict()
        if rank == 0:
            print(f"rank{rank} 正在保存检查点, state_dict {len(state_dict)}")
            start_time = time.time()
            
            lora_state_dict = {}
            for name, param in state_dict.items():
                if 'lora_' in name:
                    lora_state_dict[name] = param.detach().cpu().clone()
            print(f"提取LoRA权重耗时: {time.time() - start_time:.2f}秒, 权重数量: {len(lora_state_dict)}")

            checkpoint_dir = os.path.join(save_path, f"checkpoint-{step}")
            save_file(lora_state_dict, os.path.join(checkpoint_dir, "adapter_model.safetensors"))
            
            if hasattr(model, 'peft_config'):
                peft_config = model.peft_config["default"]
                print(f"peft_config: {peft_config}")
                config_dict = peft_config.to_dict()
                    
                # 保存为adapter_config.json (PEFT标准文件名)
                with open(os.path.join(checkpoint_dir, "adapter_config.json"), "w") as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)                
 
            
            # torch.save({
            #     'step': step,
            #     'optimizer': optimizer.state_dict(),
            #     'scheduler': scheduler.state_dict() if scheduler else None,
            # }, f"{save_path}/checkpoint-{step}/optimizer.pt")
            
    torch.distributed.barrier()


def load_lora_weights(base_model, lora_path):
    """
    加载LoRA权重到基础模型
    
    Args:
        base_model: 原始WanModelCFG模型
        lora_path: LoRA权重保存路径
        
    Returns:
        lora_model: 加载了LoRA权重的模型
    """
    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA weights path {lora_path} does not exist")
        
    # 使用PEFT的PeftModel加载LoRA权重
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        is_trainable=True
    )
    print(f"LoRA weights loaded from {lora_path}")
    return lora_model

def load_lora_weights_manually(base_model, checkpoint_dir, lora_rank=128, lora_alpha=64, lora_dropout=0, apply_to_ffn=True):
    """
    手动加载LoRA权重和配置
    
    Args:
        base_model: 原始基础模型
        checkpoint_dir: 包含adapter_config.json和adapter_model.safetensors的目录
        
    Returns:
        lora_model: 加载了LoRA权重的模型
    """
    # 1. 加载adapter_config.json
    # config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    # if not os.path.exists(config_path):
    #     raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
    # with open(config_path, "r") as f:
    #     config_dict = json.load(f)
    
    # print(f"加载LoRA配置: {config_dict}")
    
    # # 2. 创建LoraConfig
    # lora_config = LoraConfig(
    #     r=config_dict.get("r", 16),
    #     lora_alpha=config_dict.get("lora_alpha", 32),
    #     target_modules=config_dict.get("target_modules", r".*\.(q|k|v|o)$"),
    #     lora_dropout=config_dict.get("lora_dropout", 0.05),
    #     task_type=config_dict.get("task_type", "CAUSAL_LM")
    # )
    
    module_names = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            module_names.append(name)
    
    # print("可用的线性层模块:")
    # print("\n".join(module_names))
    
    # 匹配注意力层的q, k, v, o
    target_modules = []
    
    # 使用列表指定精确的模块名称，而不是正则表达式
    for name in module_names:
        if name.endswith('.q') or name.endswith('.k') or name.endswith('.v') or name.endswith('.o'):
            target_modules.append(name)
            
    if apply_to_ffn:
        # 添加FFN层，根据打印出的模块名进行匹配
        for name in module_names:
            if '.ffn.0' in name or '.ffn.2' in name:
                target_modules.append(name)
    
    # print(f"将应用LoRA的模块: {target_modules}")
    
    # 创建LoRA配置
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
    )

    # 3. 创建PEFT模型
    peft_model = get_peft_model(base_model, lora_config)
    
    # 4. 加载权重
    weights_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")
    
    # 使用safetensors加载权重
    state_dict = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            new_key = key.replace("_fsdp_wrapped_module.", "").replace("._checkpoint_wrapped_module", "")
            state_dict[new_key] = f.get_tensor(key)
    
    # 5. 将权重应用到模型
    # 检查state_dict中的键是否与模型匹配
    missing_keys = []
    unexpected_keys = []
    
    # 获取模型中的LoRA参数
    model_state_dict = {k: v for k, v in peft_model.named_parameters() if 'lora_' in k}
    
    # 检查键是否匹配
    for k in state_dict.keys():
        if k not in model_state_dict:
            unexpected_keys.append(k)
    
    for k in model_state_dict.keys():
        if k not in state_dict:
            missing_keys.append(k)
            
    if missing_keys:
        print(f"警告: 模型中这些LoRA参数在加载的权重中缺失: {missing_keys}")
    if unexpected_keys:
        print(f"警告: 加载的权重中这些键在模型中不存在: {unexpected_keys}")
        assert False, f"意外的键: {unexpected_keys}"
        
    # 实际加载权重
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if name in state_dict:
                if param.shape != state_dict[name].shape:
                    print(f"警告: 参数 {name} 形状不匹配. 模型: {param.shape} vs 权重: {state_dict[name].shape}")
                    continue
                param.copy_(state_dict[name])
                
    print(f"LoRA权重已从 {weights_path} 成功加载")
    return peft_model

if __name__ == "__main__":
    from model_cfg import WanModelCFG

    # 初始化基础模型
    base_model = WanModelCFG.from_pretrained("/cv/wangxuekuan/code/algo_wanx_service/models/Wan2.1-T2V-14B-distill-v0")
    
    # # 加载LoRA配置和权重
    # checkpoint_dir = "/cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/outputs/exp1/checkpoint-0"
    # lora_model = load_lora_weights_manually(base_model, checkpoint_dir)
    
    # # 验证加载的模型
    # print("可训练参数数量:", sum(p.numel() for p in lora_model.parameters() if p.requires_grad))
