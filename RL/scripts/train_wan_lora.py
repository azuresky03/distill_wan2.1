from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from scripts.train.model.model_cfg import WanModelCFG, WanAttentionBlock
from scripts.train.model.lora_utils import create_wan_lora_model, load_lora_weights_manually
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
# from fastvideo.utils.parallel_states import (destroy_sequence_parallel_group, get_sequence_parallel_state,
#                                              initialize_sequence_parallel_state)
import torch
import os
import torch.distributed as dist
from accelerate.utils import set_seed
import random 
import RL.easyanimate.reward.reward_fn as reward_fn
from tqdm import tqdm
import time
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from datetime import datetime

import argparse
import json
from time import sleep
from copy import deepcopy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import ShardingStrategy
from fastvideo.utils.fsdp_util import (apply_fsdp_checkpointing, get_dit_fsdp_kwargs)
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state
from safetensors.torch import save_file
from wan.utils.utils import cache_video
from peft import PeftModel
from fastvideo.utils.parallel_states import nccl_info

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
            
            # if hasattr(model, 'peft_config'):
            #     peft_config = model.peft_config["default"]
            #     print(f"peft_config: {peft_config}")
            #     config_dict = peft_config.to_dict()
                    
            #     # 保存为adapter_config.json (PEFT标准文件名)
            #     with open(os.path.join(checkpoint_dir, "adapter_config.json"), "w") as f:
            #         json.dump(config_dict, f, indent=2, ensure_ascii=False)                
 
            
            # torch.save({
            #     'step': step,
            #     'optimizer': optimizer.state_dict(),
            #     'scheduler': scheduler.state_dict() if scheduler else None,
            # }, f"{save_path}/checkpoint-{step}/optimizer.pt")
            
    torch.distributed.barrier()

# 添加分布式友好的日志工具

def main_print(msg):
    if int(os.environ["RANK"])==0:
        print(msg)

def load_prompts(prompt_path, prompt_column="prompt", start_idx=None, end_idx=None):
    prompt_list = []
    if prompt_path.endswith(".txt"):
        with open(prompt_path, "r") as f:
            for line in f:
                prompt_list.append(line.strip())
    elif prompt_path.endswith(".jsonl"):
        with open(prompt_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                prompt_list.append(item[prompt_column])
    else:
        raise ValueError("The prompt_path must end with .txt or .jsonl.")
    prompt_list = prompt_list[start_idx:end_idx]

    return prompt_list

def create_distributed_tqdm(iterable=None, desc=None, total=None, disable=False, **kwargs):
    """为分布式训练创建tqdm进度条,只在rank 0上显示"""
    rank = int(os.environ.get("RANK", 0))
    # 只在主进程(rank=0)上显示进度条
    return tqdm(iterable, desc=desc, total=total, disable=(rank != 0) or disable, **kwargs)

def get_lora_parameters(model,rank):
    """
    筛选出模型中的LoRA参数
    """
    lora_params = []
    non_lora_params = []
    
    for name, param in model.named_parameters():
        if 'lora_' in name or param.requires_grad:
            if 'lora_' not in name:
                print(f"requires_grad参数: {name}, rank: {rank}")
            else:
                lora_params.append(param)
        else:
            non_lora_params.append(param)
    
    return lora_params, non_lora_params

def create_optimizer_and_scheduler(model, lr, weight_decay, num_train_steps,rank):
    """
    创建针对LoRA参数的优化器和学习率调度器
    """
    lora_params, _ = get_lora_parameters(model,rank)
    
    # 打印LoRA参数的数量，便于调试
    total_params = sum(p.numel() for p in lora_params)
    print(f"LoRA参数总量: {total_params / 1e6:.2f}M rank: {rank}")
    
    # 创建优化器，只优化LoRA参数
    optimizer = AdamW(lora_params, lr=lr, weight_decay=weight_decay, eps=1e-10)
    
    # 创建学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps)
    
    return optimizer, scheduler


def verify_lora_loaded(model):
    """验证LoRA权重是否正确加载"""
    # 检查peft_config是否存在
    if not hasattr(model, 'peft_config'):
        main_print("警告: 模型没有peft_config属性，可能加载失败")
        return False
    else:
        main_print(model.peft_config)
    
    # 检查是否存在LoRA参数
    lora_param_count = 0
    non_zero_params = 0
    
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_param_count += 1
            if torch.any(param != 0):
                non_zero_params += 1
    
    main_print(f"LoRA参数总数: {lora_param_count}, 非零参数: {non_zero_params}")
    
    if lora_param_count == 0:
        main_print("错误: 没有找到LoRA参数，加载失败")
        return False
    
    if non_zero_params == 0:
        main_print("警告: 所有LoRA参数都为零，可能加载失败")
        return False
    
    main_print("LoRA权重验证通过 ✓")
    return True

class WanLoraTrainer:
    def __init__(self, args):
        # 配置基础设置
        torch.backends.cuda.matmul.allow_tf32 = True

        self.args = args
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(self.local_rank)
        self.device = torch.cuda.current_device()

        # 设置随机种子
        if args.seed is not None:
            set_seed(args.seed + self.rank)

        self.prompt_list = load_prompts(args.prompt_path)

        self.reward_fn_kwargs = {}
        if args.reward_fn_kwargs is not None:
            self.reward_fn_kwargs = json.loads(args.reward_fn_kwargs)

        self.autocast_type = torch.bfloat16 if args.bf16 else torch.float32
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)

        eval_all_prompts = load_prompts(args.eval_prompt_path)
        self.org_len = len(eval_all_prompts)
        truncated_len = len(eval_all_prompts) // self.world_size * self.world_size
        if truncated_len < len(eval_all_prompts):
            eval_all_prompts += [eval_all_prompts[0]] * (truncated_len + self.world_size - len(eval_all_prompts))
        local_eval_prompts = []
        for i in range(len(eval_all_prompts)):
            if i % self.world_size == self.rank:
                local_eval_prompts.append(eval_all_prompts[i])
        self.local_eval_prompts = local_eval_prompts
        print(f"Rank {self.rank}: 将处理 {len(local_eval_prompts)} 个提示")           

        # 初始化步数
        self.init_steps = 0
        self.step = 0

        # 初始化模型和组件
        self._setup_model()
        self._setup_vae()
        self._setup_text_encoder()
        self._setup_optimizer_and_scheduler()
        self._setup_noise_scheduler()

        # 设置其他训练参数
        self.batch_size = args.train_batch_size
        self.latent_height = args.height // 8  # 假设VAE下采样是8倍
        self.latent_width = args.width // 8
        self.latent_channels = 16
        self.latent_frames = (args.frame_number-1)//4 + 1  # 视频帧数
        self.seq_len = self.latent_frames * self.latent_height * self.latent_width // 4

        self.eval_latent_height = 480 // 8  # 假设VAE下采样是8倍
        self.eval_latent_width = 832 // 8
        self.eval_latent_frames = 21 
        self.eval_seq_len = self.eval_latent_frames * self.eval_latent_height * self.eval_latent_width  // 4
        
        self.context_manager = torch.autocast(device_type="cuda", dtype=self.autocast_type)

        # 确定需要进行梯度传播的步骤列表
        if args.backprop_strategy == "last":
            self.backprop_step_list = [args.sample_steps - 1]
            main_print(f"仅对最后一步进行反向传播: {self.backprop_step_list}")
        elif args.backprop_strategy == "tail":
            self.backprop_step_list = list(range(args.sample_steps))[-args.backprop_num_steps:]
            main_print(f"对最后{args.backprop_num_steps}步进行反向传播: {self.backprop_step_list}")

        null_encoded = torch.load("/vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/mixkit/meta/null.pt", map_location="cpu",weights_only=True)
        self.null_encoded = null_encoded.to(self.device).unsqueeze(0)
        
        self.evaluate_noise = torch.randn(
                                (self.batch_size, self.latent_channels, self.eval_latent_frames, self.eval_latent_height, self.eval_latent_width),
                                device=self.device,
                                dtype=self.autocast_type
                            )
        if not self.args.debug and self.step==0: self.evaluate()

        main_print(f"初始化完成，显存占用 {torch.cuda.max_memory_reserved(self.device) / 1024**3 } GB")

    def _setup_model(self):
        """设置Wan模型和LoRA"""
        self.model = WanModelCFG.from_pretrained(self.args.transfromer_dir)
        if self.args.resume_from_checkpoint:
            self.model = load_lora_weights_manually(self.model, self.args.resume_from_checkpoint)
            self.step = self.init_steps = int(self.args.resume_from_checkpoint.split("-")[-1])+1
            main_print(f"从检查点恢复训练, 步骤: {self.step}")
        else:
            self.model = create_wan_lora_model(
                self.model,
                lora_rank=self.args.lora_rank,
                lora_alpha=self.args.lora_alpha
            )
        
        self.model = self.model.to(self.device).to(self.autocast_type)

        count = 0
        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                dist.broadcast(param.data, src=0)
                count += 1
        main_print(f"broadcast 统一 LoRA参数数量: {count}")

        dist.barrier()
        
        # 初始化FSDP
        main_print(f"--> 初始化FSDP,分片策略: {self.args.fsdp_sharding_strategy}")
        fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
            self.model,
            self.args.fsdp_sharding_strategy,
            False,  # 使用LoRA
            self.args.use_cpu_offload,
            self.args.master_weight_type,
            (WanAttentionBlock,)
        )

        self.model = FSDP(
            self.model,
            use_orig_params=True,
            **fsdp_kwargs,
        )

        main_print("--> 模型加载完成")
        
        # 应用梯度检查点
        if self.args.gradient_checkpointing:
            apply_fsdp_checkpointing(self.model, no_split_modules, self.args.selective_checkpointing)
        
        self.model.train()  # 设置模型为训练模式
        torch.cuda.empty_cache()  # 清理显存
        main_print(f"WanModelCFG 显存占用 {torch.cuda.max_memory_reserved(self.device) / 1024**3 } GB")

    def _setup_vae(self):
        """设置VAE模型"""
        self.vae = WanVAE(vae_pth=os.path.join(self.args.model_path, "Wan2.1_VAE.pth"))
        self.vae.model = self.vae.model.to(self.device).to(self.autocast_type)
        self.vae.model.eval()
        self.vae.model.requires_grad_(False)
        main_print(f"vae 显存占用 {torch.cuda.max_memory_reserved(self.device) / 1024**3 } GB")

    def _setup_text_encoder(self):
        """设置T5文本编码器"""
        t5_device = torch.device('cpu')  
        main_print(f"T5EncoderModel 初始化在: {t5_device}, 使用策略: {'动态迁移' if self.args.t5_on_cpu else '保持在GPU'}")
        
        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=self.autocast_type,
            device=t5_device,  # 初始化在CPU上
            checkpoint_path=os.path.join(self.args.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self.args.model_path, "google/umt5-xxl"),
            shard_fn=None,
        )
        main_print(f"T5EncoderModel 显存占用 {torch.cuda.max_memory_reserved(self.device) / 1024**3 } GB")

    def _setup_optimizer_and_scheduler(self):
        """设置优化器和学习率调度器"""
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            self.model, 
            lr=self.args.learning_rate, 
            weight_decay=self.args.weight_decay,
            num_train_steps=self.args.num_train_steps,
            rank=self.rank
        )

    def _setup_noise_scheduler(self):
        """设置噪声调度器"""
        self.noise_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False)
        self.noise_scheduler.set_timesteps(self.args.sample_steps, device=self.device, shift=self.args.shift)
        
        # 设置损失函数
        self.loss_fn = getattr(reward_fn, self.args.reward_fn)(device="cpu", dtype=self.autocast_type, **self.reward_fn_kwargs)

    def one_diffusion(self, train_prompt,backprop_step_list, eval=False):
        """执行一次完整的扩散过程"""

        self.noise_scheduler.set_timesteps(self.args.sample_steps, device=self.device, shift=self.args.shift)

        with torch.no_grad():
            if self.args.t5_on_cpu:
                self.text_encoder.model.to(self.device)
                
                context = self.text_encoder(train_prompt, self.device)
                
                self.text_encoder.model.to(torch.device('cpu'))
                torch.cuda.empty_cache()  
            else:
                context = self.text_encoder(train_prompt, self.device)
        
        if self.args.debug: 
            main_print(f"after t5 显存占用 {torch.cuda.max_memory_reserved(self.device) / 1024**3 } GB")

        # 准备时间步
        timesteps = self.noise_scheduler.timesteps
                    
        # 生成随机噪声
        if eval:
            latents = self.evaluate_noise
        else:
            latents = torch.randn(
                (self.batch_size, self.latent_channels, self.latent_frames, self.latent_height if not eval else self.eval_latent_height, self.latent_width if not eval else self.eval_latent_width),
                device=self.device,
                dtype=self.autocast_type
            )
        
        # 去噪循环
        for ind, t in enumerate(timesteps):
            if self.args.debug: 
                main_print(f"步骤 {ind}/{len(timesteps)}: t={t}")
                
            latent_model_input = latents
            
            # 扩展时间步到batch大小
            timestep = torch.tensor([t] * self.batch_size, device=self.device).to(latent_model_input.dtype)
            
            # 预测噪声
            with self.context_manager:
                guidance_tensor = torch.tensor(
                    [self.args.guidance_scale * 1000], 
                    device=latent_model_input.device,
                    dtype=self.autocast_type
                )
                
                latent_model_input = [latents[j] for j in range(self.batch_size)]
                
                if ind not in backprop_step_list:
                    with torch.no_grad():
                        noise_pred = self.model(
                            latent_model_input, 
                            context=context,
                            t=timestep, 
                            seq_len=self.seq_len if not eval else self.eval_seq_len, 
                            guidance=guidance_tensor if not self.args.no_cfg_distill else None
                        )[0]
                        if self.args.no_cfg_distill:
                            noise_pred_uncond = self.model(
                                latent_model_input, 
                                context=None,
                                batch_context=self.null_encoded,
                                t=timestep, 
                                seq_len=self.seq_len if not eval else self.eval_seq_len, 
                            )[0]
                            noise_pred = noise_pred + (noise_pred - noise_pred_uncond) * self.args.guidance_scale
                else:
                    if self.args.debug: 
                        main_print(f"反向传播步骤: {ind} {t}")
                    noise_pred = self.model(
                        latent_model_input, 
                        context=context,
                        t=timestep, 
                        seq_len=self.seq_len, 
                        guidance=guidance_tensor if not self.args.no_cfg_distill else None
                    )[0]
                    if self.args.no_cfg_distill:
                        noise_pred_uncond = self.model(
                            latent_model_input, 
                            context=None,
                            batch_context=self.null_encoded,
                            t=timestep, 
                            seq_len=self.seq_len if not eval else self.eval_seq_len, 
                        )[0]
                        noise_pred = noise_pred + (noise_pred - noise_pred_uncond) * self.args.guidance_scale
                noise_pred = torch.stack([noise_pred], dim=0)
                                
                # 执行去噪步骤
                latents = self.noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        return latents, context

    def train(self):
        """执行训练循环"""
        main_print(f"begin training, rank: {self.rank}, batch_size: {self.batch_size}, num_train_steps: {args.num_train_steps}, num_train_prompts: {len(self.prompt_list)}, memory: {torch.cuda.max_memory_reserved(self.device) / 1024**3 :02f} GB")
        main_print(f"backprop_step_list: {self.backprop_step_list}")
        with create_distributed_tqdm(total=self.args.num_train_steps, desc="训练中", initial=self.init_steps) as pbar:
            for step in range(self.init_steps, self.args.num_train_steps):

                self.step = step
                self.optimizer.zero_grad()
                                
                # 选择提示词
                train_prompt = random.choices(self.prompt_list, k=self.args.train_batch_size)
                
                # 执行一次扩散过程
                latents, context = self.one_diffusion(train_prompt, self.backprop_step_list)

                # 处理采样的潜变量
                sampled_latent_indices = list(range(self.args.num_decoded_latents)) if not self.args.random_decode else random.sample(range(self.latent_frames), self.args.num_decoded_latents)
                sampled_latents = latents[:, :, sampled_latent_indices, :, :]
                
                sampled_frames = torch.stack(self.vae.decode(sampled_latents), dim=0)
                if self.args.debug:
                    for i in range(sampled_frames.size(0)):
                        save_path = os.path.join(self.args.log_dir, f"step_{step}")
                        os.makedirs(save_path, exist_ok=True)
                        save_path = os.path.join(save_path, f"rank{self.rank}_sampled_frames_{i}.mp4")
                        cache_video(sampled_frames[i][None], save_file=save_path, fps=16, nrow=1, normalize=True, value_range=(-1, 1))

                # 归一化
                sampled_frames = sampled_frames.clamp(-1, 1)
                sampled_frames = (sampled_frames / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
                
                if self.args.num_sampled_frames is not None:
                    num_frames = sampled_frames.size(2) - 1
                    sampled_frames_indices = torch.linspace(0, num_frames, steps=self.args.num_sampled_frames).long()
                    sampled_frames = sampled_frames[:, :, sampled_frames_indices, :, :]

                # 计算奖励/损失
                loss, reward = self.loss_fn(sampled_frames, train_prompt)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 更新学习率
                current_lr = self.scheduler.get_last_lr()[0]
                self.scheduler.step()

                loss_item = loss.item()
                reward_item = reward.item()
                
                # 创建tensor来存储值
                loss_tensor = torch.tensor([loss_item], device=self.device)
                reward_tensor = torch.tensor([reward_item], device=self.device)
                
                # 执行all_reduce操作对所有进程的值求和
                torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(reward_tensor, op=torch.distributed.ReduceOp.SUM)
                
                # 计算平均值
                loss_item = loss_tensor.item() / self.world_size
                reward_item = reward_tensor.item() / self.world_size
                if self.rank == 0:
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss_item:.4f}", reward=f"{reward_item:.4f}", lr=f"{current_lr:.2e}")
                
                main_print(f"步骤 {step}/{self.args.num_train_steps}, 损失: {loss.item():.4f}, 奖励: {reward.item():.4f}, 学习率: {current_lr:.2e}")
                
                if self.args.save_steps > 0 and step % self.args.save_steps == 0 and step>0:
                    main_print(f"保存检查点 步骤 {step}")
                    
                    self.evaluate()
                    save_checkpoint(self.model, self.optimizer, self.scheduler, self.args.output_dir, step, self.rank)
                    dist.barrier()
                
                torch.cuda.empty_cache()  
                dist.barrier()
            
            # 训练结束，保存最终模型
            if self.rank == 0:
                main_print("保存最终模型")
            save_checkpoint(self.model, self.optimizer, self.scheduler, self.args.output_dir, step, self.rank)
            dist.barrier()
    
    def evaluate(self):
        main_print(f"开始评估 step {self.step}")
        save_path = os.path.join(self.args.log_dir, f"eval_step_{self.step}")
        if self.rank == 0:
            os.makedirs(save_path, exist_ok=True)
        for ind in range(len(self.local_eval_prompts)):
            eval_prompt = self.local_eval_prompts[ind]
            latents, context = self.one_diffusion(eval_prompt, [], eval=True)
            sampled_frames = torch.stack(self.vae.decode(latents), dim=0)
            for i in range(sampled_frames.size(0)):
                save_video_path = os.path.join(save_path, f"rank{self.rank}_eval_{ind}.mp4")
                if ind*self.world_size + self.rank <= self.org_len:
                    cache_video(sampled_frames[i][None], save_file=save_video_path, fps=16, nrow=1, normalize=True, value_range=(-1, 1))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sp_size", type=int, required=True, help="Sequence parallel size.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for initialization.")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt file.")
    parser.add_argument("--reward_fn", type=str, required=True, help="Reward function to use.")
    parser.add_argument("--reward_fn_kwargs", type=str, default=None, help="JSON string of kwargs for reward function.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--transfromer_dir", type=str, required=True, help="Path to the transformer model directory.")
    parser.add_argument("--lora_rank", type=int, required=True, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=float, required=True, help="LoRA alpha.")
    parser.add_argument("--shift", type=int, default=1, help="Shift value for noise scheduler.")
    parser.add_argument("--train_batch_size", type=int, required=True, help="Training batch size.")
    parser.add_argument("--num_train_steps", type=int, default=1000, help="Number of training steps.")

    # 添加优化器相关参数
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for LoRA parameters.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")

    # 添加CFG相关参数
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--no_cfg_distill", action="store_true")
    parser.add_argument("--sample_steps", type=int, default=10, 
                        help="Number of denoising steps per training sample")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--width", type=int, default=832, help="Image width")
    parser.add_argument("--frame_number", type=int, default=81, help="Number of video frames")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--eval_prompt_path", type=str, default="/cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/datasets/valid.txt")
    # parser.add_argument("--eval_output_dir", type=str, required=True, help="Path to the evaluation output directory.")
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=None,
        help="The number of sampled frames for the reward function."
    )

    parser.add_argument(
        "--num_decoded_latents",
        type=int,
        default=1,
        help="Number of latents to decode."
    )
    parser.add_argument("--random_decode", action="store_true")   

    parser.add_argument(
        "--backprop_strategy",
        choices=["last", "tail"],
        default="last",
        help="反向传播的策略: last(最后一步), tail(最后几步)"
    )
    parser.add_argument(
        "--backprop_num_steps",
        type=int,
        default=5,
        help="反向传播的步骤数量。仅当backprop_strategy为tail时使用"
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs", 
        help="输出目录，用于保存日志和检查点"
    )
    
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="日志记录间隔步数"
    )

    # 添加T5设备控制参数
    parser.add_argument(
        "--t5_on_cpu", 
        action="store_true", 
        help="将T5编码器放在CPU上运行以减少显存使用"
    )

    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=500,
        help="每多少步保存一次模型检查点，0表示不保存中间检查点"
    )

    # 添加FSDP相关参数
    parser.add_argument(
        "--fsdp_sharding_strategy", 
        type=str, 
        default="full", 
        choices=["full", "shard_grad_op", "no_shard"],
        help="FSDP分片策略"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="使用梯度检查点以节省显存，但会略微降低训练速度"
    )
    parser.add_argument(
        "--selective_checkpointing", 
        type=float, 
        default=1.0,
        help="选择性梯度检查点比例"
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="是否使用CPU卸载参数和梯度"
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="bf16",
        choices=["fp32", "bf16"],
        help="主权重类型 - fp32或bf16"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从指定的检查点恢复训练"
    )

    return parser.parse_args()

def main(args):
    trainer = WanLoraTrainer(args)
    trainer.train()

if __name__ == "__main__":
    args = get_args()
    main(args)