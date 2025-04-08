import torch
import os
import torch.distributed as dist
import argparse
import json
import random
from tqdm import tqdm
import numpy as np

from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from scripts.train.model.model_cfg import WanModelCFG, WanAttentionBlock
from scripts.train.model.lora_utils import create_wan_lora_model, load_lora_weights_manually
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.fsdp_util import (apply_fsdp_checkpointing, get_dit_fsdp_kwargs)

def main_print(msg):
    """分布式友好的日志打印，只在rank 0上显示"""
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg)

def load_prompts(prompt_path, prompt_column="prompt", start_idx=None, end_idx=None):
    """加载提示词列表"""
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
    """为分布式环境创建tqdm进度条,只在rank 0上显示"""
    rank = int(os.environ.get("RANK", 0))
    return tqdm(iterable, desc=desc, total=total, disable=(rank != 0) or disable, **kwargs)

class WanLoraEvaluator:
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
            args.seed = args.seed + self.rank
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

        self.autocast_type = torch.bfloat16 if args.bf16 else torch.float32
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)

        # 加载提示词
        all_prompts = load_prompts(args.prompt_path)
        self.org_len = len(all_prompts)
        
        # 确保提示词能被world_size整除
        truncated_len = len(all_prompts) // self.world_size * self.world_size
        if truncated_len < len(all_prompts):
            all_prompts += [all_prompts[0]] * (truncated_len + self.world_size - len(all_prompts))
        
        # 在各个卡上均匀分布prompt
        local_prompts = []
        for i in range(len(all_prompts)):
            if i % self.world_size == self.rank:
                local_prompts.append(all_prompts[i])
        self.local_prompts = local_prompts
        main_print(f"Rank {self.rank}: 将处理 {len(local_prompts)} 个提示")           

        # 初始化模型和组件
        self._setup_model()
        self._setup_vae()
        self._setup_text_encoder()
        self._setup_noise_scheduler()

        # 设置视频尺寸参数
        self.batch_size = 1  # 评估时批量大小为1
        self.latent_height = args.height // 8  # VAE下采样是8倍
        self.latent_width = args.width // 8
        self.latent_channels = 16
        self.latent_frames = args.frame_number  # 视频帧数
        self.seq_len = self.latent_frames * self.latent_height * self.latent_width // 4
        
        self.context_manager = torch.autocast(device_type="cuda", dtype=self.autocast_type)
        
        # 加载null编码
        null_encoded = torch.load("/vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/mixkit/meta/null.pt", map_location="cpu", weights_only=True)
        self.null_encoded = null_encoded.to(self.device).unsqueeze(0)
        
        # 准备用于评估的噪声种子
        self.noise_seeds = list(range(args.seed, args.seed + args.num_noise_variations))
        
        main_print(f"初始化完成，显存占用 {torch.cuda.max_memory_reserved(self.device) / 1024**3 } GB")

    def _setup_model(self):
        """设置Wan模型和LoRA"""
        self.model = WanModelCFG.from_pretrained(self.args.transfromer_dir)
        
        # 加载LoRA权重
        self.model = load_lora_weights_manually(self.model, self.args.lora_checkpoint)
        
        self.model = self.model.to(self.device).to(self.autocast_type)

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
        
        self.model.eval()  # 设置模型为评估模式
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
        t5_device = torch.device('cpu') if self.args.t5_on_cpu else self.device
        main_print(f"T5EncoderModel 初始化在: {t5_device}, 使用策略: {'动态迁移' if self.args.t5_on_cpu else '保持在GPU'}")
        
        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=self.autocast_type,
            device=t5_device,
            checkpoint_path=os.path.join(self.args.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self.args.model_path, "google/umt5-xxl"),
            shard_fn=None,
        )
        main_print(f"T5EncoderModel 显存占用 {torch.cuda.max_memory_reserved(self.device) / 1024**3 } GB")

    def _setup_noise_scheduler(self):
        """设置噪声调度器"""
        self.noise_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False)
        self.noise_scheduler.set_timesteps(self.args.sample_steps, device=self.device, shift=self.args.shift)

    def generate_video(self, prompt, noise_seed=None):
        """使用特定噪声种子生成单个视频"""

        self.noise_scheduler.set_timesteps(self.args.sample_steps, device=self.device, shift=self.args.shift)
        
        with torch.no_grad():
            # 设置噪声种子以确保可复现性
            if noise_seed is not None:
                torch.manual_seed(noise_seed)
                
            # 编码文本提示词
            if self.args.t5_on_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([prompt], self.device)
                self.text_encoder.model.to(torch.device('cpu'))
                torch.cuda.empty_cache()  
            else:
                context = self.text_encoder([prompt], self.device)
            
            # 设置时间步
            timesteps = self.noise_scheduler.timesteps
                        
            # 生成随机噪声
            latents = torch.randn(
                (self.batch_size, self.latent_channels, self.latent_frames, self.latent_height, self.latent_width),
                device=self.device,
                dtype=self.autocast_type
            )
            
            # 去噪循环
            for ind, t in enumerate(timesteps):
                latent_model_input = latents
                
                # 扩展时间步到batch大小
                timestep = torch.tensor([t] * self.batch_size, device=self.device).to(latent_model_input.dtype)
                
                main_print(f"去噪步骤 {ind + 1}/{len(timesteps)}")
                # 预测噪声
                with self.context_manager:
                    guidance_tensor = torch.tensor(
                        [self.args.guidance_scale * 1000], 
                        device=latent_model_input.device,
                        dtype=self.autocast_type
                    )
                    
                    latent_model_input = [latents[j] for j in range(self.batch_size)]
                    
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
                            seq_len=self.seq_len, 
                        )[0]
                        noise_pred = noise_pred + (noise_pred - noise_pred_uncond) * self.args.guidance_scale
                    
                    noise_pred = torch.stack([noise_pred], dim=0)
                                    
                    # 执行去噪步骤
                    latents = self.noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # 解码潜变量
            sampled_frames = torch.stack(self.vae.decode(latents), dim=0)
            
            return sampled_frames

    def evaluate(self):
        """对所有prompt生成多个具有不同noise的视频"""
        main_print(f"开始评估，将为每个提示词生成 {len(self.noise_seeds)} 个视频变种")
        
        # 创建输出目录
        if self.rank == 0:
            os.makedirs(self.args.output_dir, exist_ok=True)
        dist.barrier()
        
        # 遍历处理提示词
        for prompt_idx, prompt in enumerate(create_distributed_tqdm(self.local_prompts, desc="处理提示词")):
            # 创建提示词对应的输出目录
            global_prompt_idx = prompt_idx * self.world_size + self.rank
            prompt_dir = os.path.join(self.args.output_dir, f"prompt_{global_prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            
            # 保存提示词信息
            with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
                f.write(prompt)
            
            # 使用不同的noise seed生成多个视频
            for seed_idx, noise_seed in enumerate(self.noise_seeds):
                main_print(f"生成 prompt {global_prompt_idx}, seed {noise_seed}")
                
                # 生成视频
                sampled_frames = self.generate_video(prompt, noise_seed)
                
                # 保存视频
                save_path = os.path.join(prompt_dir, f"seed_{noise_seed}.mp4")
                cache_video(sampled_frames[0][None], save_file=save_path, fps=self.args.fps, normalize=True, value_range=(-1, 1))
                
                torch.cuda.empty_cache()  # 清理显存
            
            # 进程间同步
            dist.barrier()

def get_args():
    parser = argparse.ArgumentParser(description="使用LoRA模型评估视频生成")
    
    # 模型和checkpoint相关参数
    parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--transfromer_dir", type=str, required=True, help="Transformer模型路径")
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="LoRA checkpoint路径")
    
    # 分布式参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 提示词相关参数
    parser.add_argument("--prompt_path", type=str, required=True, help="提示词文件路径")
    
    # 生成参数
    parser.add_argument("--num_noise_variations", type=int, default=3, help="每个提示词生成的不同噪声变种数量")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG指导强度")
    parser.add_argument("--no_cfg_distill", action="store_true", help="不使用CFG蒸馏")
    parser.add_argument("--sample_steps", type=int, default=10, help="采样步数")
    parser.add_argument("--shift", type=int, default=1, help="Noise scheduler的shift参数")
    
    # 视频参数
    parser.add_argument("--height", type=int, default=480, help="视频高度")
    parser.add_argument("--width", type=int, default=832, help="视频宽度")
    parser.add_argument("--frame_number", type=int, default=21, help="视频帧数")
    parser.add_argument("--fps", type=int, default=16, help="视频FPS")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    
    # 其他配置
    parser.add_argument("--bf16", action="store_true", help="使用bfloat16精度")
    parser.add_argument("--t5_on_cpu", action="store_true", help="将T5编码器放在CPU上运行以减少显存使用")
    
    # FSDP参数
    parser.add_argument(
        "--fsdp_sharding_strategy", 
        type=str, 
        default="full", 
        choices=["full", "shard_grad_op", "no_shard"],
        help="FSDP分片策略"
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
    
    return parser.parse_args()

def main():
    args = get_args()
    evaluator = WanLoraEvaluator(args)
    evaluator.evaluate()

if __name__ == "__main__":
    main()
