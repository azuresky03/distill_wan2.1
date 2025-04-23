# !/bin/python3
# isort: skip_file
import argparse
import math
import os
import time
from collections import deque
from copy import deepcopy
import random

import torch
import torch.distributed as dist
import wandb
from accelerate.utils import set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from peft import LoraConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from fastvideo.utils.checkpoint import (resume_lora_optimizer, save_checkpoint, save_lora_checkpoint)
from fastvideo.utils.communications import broadcast
from scripts.train.util.hidden_communication_data_wrapper import sp_parallel_dataloader_wrapper
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.utils.fsdp_util import (apply_fsdp_checkpointing, get_dit_fsdp_kwargs)
from fastvideo.utils.parallel_states import (destroy_sequence_parallel_group, get_sequence_parallel_state,
                                             initialize_sequence_parallel_state)

from scripts.dataset.hidden_datasets import (LatentDataset, latent_collate_function)
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video, cache_image, str2bool

from wan.utils.utils import cache_video
from distill_exp.gan.model_cfg import WanModelCFG, WanAttentionBlock

from scripts.train.util.fm_solvers_unipc import FlowUniPCMultistepScheduler

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")


def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)


def reshard_fsdp(model):
    for m in FSDP.fsdp_modules(model):
        if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
            torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)


def get_norm(model_pred, norms, gradient_accumulation_steps):
    fro_norm = (
        torch.linalg.matrix_norm(model_pred, ord="fro") /  # codespell:ignore
        gradient_accumulation_steps)
    largest_singular_value = (torch.linalg.matrix_norm(model_pred, ord=2) / gradient_accumulation_steps)
    absolute_mean = torch.mean(torch.abs(model_pred)) / gradient_accumulation_steps
    absolute_max = torch.max(torch.abs(model_pred)) / gradient_accumulation_steps
    dist.all_reduce(fro_norm, op=dist.ReduceOp.AVG)
    dist.all_reduce(largest_singular_value, op=dist.ReduceOp.AVG)
    dist.all_reduce(absolute_mean, op=dist.ReduceOp.AVG)
    norms["fro"] += torch.mean(fro_norm).item()  # codespell:ignore
    norms["largest singular value"] += torch.mean(largest_singular_value).item()
    norms["absolute mean"] += absolute_mean.item()
    norms["absolute max"] += absolute_max.item()


class DistillTrainer:
    def __init__(self, args):
        """
        初始化蒸馏训练器
        
        Args:
            args: 包含训练配置的参数
        """
        self.args = args
        
        # 初始化分布式环境
        self.setup_distributed()
        
        # 设置随机种子
        if args.seed is not None:
            set_seed(args.seed + self.rank)
        
        # 初始化序列并行
        initialize_sequence_parallel_state(args.sp_size)
        self.sp_size = args.sp_size
        
        # 创建输出目录
        if self.rank <= 0 and args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # 加载null_encoded (如果需要)
        self.null_encoded = None
        if args.null_encoded_path:
            self.null_encoded = torch.load(args.null_encoded_path, map_location="cpu", weights_only=True)
            self.null_encoded = self.null_encoded.to(self.device)
        
        # 设置模型、优化器和学习率调度器
        self.setup_model()
        self.setup_noise_scheduler()
        self.setup_optimizer()
        
        # 初始化训练相关变量
        self.h = args.height//8
        self.w = args.width//8
        self.max_seq_len = (21*self.h*self.w)//4
        main_print(f"max_seq_len: {self.max_seq_len}")
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_grad_norm = args.max_grad_norm
        self.num_euler_timesteps = args.num_euler_timesteps
        self.ema_decay = args.ema_decay
        self.pred_decay_weight = args.pred_decay_weight
        self.pred_decay_type = args.pred_decay_type
        
        # 设置调试模式
        self.setup_debug_mode()
        
        # 设置数据加载器
        self.setup_dataloader()
        
        main_print("--> DistillTrainer initialization completed")
    
    def setup_distributed(self):
        """设置分布式训练环境"""
        if dist.is_initialized():
            main_print("分布式环境已初始化，跳过初始化")
            # 如果已经初始化，则直接获取环境变量
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.device = torch.device(f"cuda:{self.local_rank}")
            return
        
        # 否则进行初始化
        main_print("初始化分布式环境")
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl")
        torch.cuda.set_device(self.local_rank)
        
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}")

        if self.rank == 0:
            print(f"Starting training with {self.world_size} GPUs")

        dist.barrier()
    
    def setup_gan(self):
        num_layers = 1
        layer_idxs = [9, 19, 29, 39]
        desired_len = 16
        compressor_heads = 8
        
        # 2. 创建 GAN 模型
        print("创建 GAN 模型...")
        from distill_exp.gan.model import WanGanModel
        model = WanGanModel(
            self.transformer,
            num_layers=num_layers,
            layer_idxs=layer_idxs,
            device=self.device,
            dtype=torch.bfloat16,
            desired_len=desired_len,
            compressor_heads=compressor_heads
        )
        
        # 3. 加载检查点
        if self.args.gan_path:
            print(f"从检查点加载模型状态: {self.args.gan_path}")
            checkpoint = torch.load(self.args.gan_path, map_location="cpu")
            
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
        self.gan = model
        self.layer_idxs = layer_idxs

        self.gan.train()
        self.gan.requires_grad_(True)

        self.gan_optimizer = torch.optim.AdamW(
            self.gan.parameters(),
            lr=self.args.gan_learning_rate,
            weight_decay=self.args.gan_weight_decay,
        )

        self.gan_criterion = torch.nn.BCEWithLogitsLoss()
        
        main_print("GAN加载成功")

    def setup_model(self):
        """加载和设置模型"""
        main_print(f"--> Loading model from {self.args.resume_from_weight}")
        
        cfg = WAN_CONFIGS[self.args.task]
        
        # 创建模型
        self.transformer = WanModelCFG.from_pretrained(self.args.resume_from_weight).train()
        self.transformer.requires_grad_(True)
        
        self.teacher_transformer = WanModelCFG.from_pretrained(self.args.resume_from_weight)
        self.teacher_transformer.requires_grad_(False)

        main_print(f"memory before loading model: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB")
        self.setup_gan()
        main_print(f"memory after loading GAN: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB")
        
        if self.args.use_ema:
            self.ema_transformer = deepcopy(self.transformer)
        else:
            self.ema_transformer = None
        
        torch.cuda.empty_cache()
        main_print(
            f"  Total training parameters = {sum(p.numel() for p in self.transformer.parameters() if p.requires_grad) / 1e6} M")
        
        # 配置并应用FSDP
        self.setup_fsdp()
        
        # 设置训练状态
        self.transformer.train()
        self.teacher_transformer.requires_grad_(False)
        if self.args.use_ema:
            self.ema_transformer.requires_grad_(False)

        main_print(f"memory after loading WHOLE: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB")
    
    def setup_fsdp(self):
        """配置并应用FSDP分布式训练"""
        main_print(f"--> Initializing FSDP with sharding strategy: {self.args.fsdp_sharding_startegy}")
        
        # 准备FSDP参数
        fsdp_kwargs, self.no_split_modules = get_dit_fsdp_kwargs(
            self.transformer,
            self.args.fsdp_sharding_startegy,
            self.args.use_lora,
            self.args.use_cpu_offload,
            self.args.master_weight_type,
            (WanAttentionBlock,)
        )
        
        # 应用FSDP
        self.transformer = FSDP(self.transformer, **fsdp_kwargs)
        self.teacher_transformer = FSDP(self.teacher_transformer, **fsdp_kwargs)
        if self.args.use_ema:
            self.ema_transformer = FSDP(self.ema_transformer, **fsdp_kwargs)
        if self.args.gan:
            self.gan.extractor = FSDP(self.gan.extractor, **fsdp_kwargs)
        
        # 应用梯度检查点
        if self.args.gradient_checkpointing:
            apply_fsdp_checkpointing(self.transformer, self.no_split_modules, self.args.selective_checkpointing)
            apply_fsdp_checkpointing(self.teacher_transformer, self.no_split_modules, self.args.selective_checkpointing)
            if self.args.use_ema:
                apply_fsdp_checkpointing(self.ema_transformer, self.no_split_modules, self.args.selective_checkpointing)
            if self.args.gan:
                apply_fsdp_checkpointing(self.gan.extractor, self.no_split_modules, self.args.selective_checkpointing)
        
        main_print("--> FSDP initialization completed")
    
    def setup_noise_scheduler(self):
        """设置噪声调度器"""
        self.noise_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)
        self.noise_scheduler.set_timesteps(self.args.interval_steps, device=self.device, shift=self.args.shift)
        timesteps = self.noise_scheduler.timesteps
        main_print(f"noise_scheduler timesteps: {timesteps}")
    
    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        # 初始化步数
        self.init_steps = 0
        
        # 准备优化器
        params_to_optimize = self.transformer.parameters()
        params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
        
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
            eps=1e-8,
        )
        
        # 如果需要，从checkpoint恢复
        if self.args.resume_from_lora_checkpoint:
            self.transformer, self.optimizer, self.init_steps = resume_lora_optimizer(
                self.transformer, 
                self.args.resume_from_lora_checkpoint,
                self.optimizer
            )
        main_print(f"optimizer: {self.optimizer}")
        
        # 创建学习率调度器
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.world_size,
            num_training_steps=self.args.max_train_steps * self.world_size,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
            last_epoch=self.init_steps - 1,
        )
    
    def setup_debug_mode(self):
        """设置调试模式（如果启用）"""
        self.DEBUG = self.args.debug
        self.vae = None
        
        if self.DEBUG:
            from wan.modules.vae import WanVAE
            self.vae = WanVAE(vae_pth="/vepfs-zulution/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth")
            autocast_type = torch.bfloat16
            self.vae.model = self.vae.model.to(self.device).to(autocast_type)
            self.vae.model.eval()
            self.vae.model.requires_grad_(False)
    
    def setup_dataloader(self):
        """设置数据加载器"""
        train_dataset = LatentDataset(self.args.data_json_path)
        sampler = DistributedSampler(train_dataset, rank=self.rank, num_replicas=self.world_size, shuffle=False)
        
        train_dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            collate_fn=latent_collate_function,
            pin_memory=True,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
        )
        
        # 创建并行数据加载器包装器
        self.dataloader = sp_parallel_dataloader_wrapper(
            train_dataloader,
            self.device,
            self.args.train_batch_size,
            self.args.sp_size,
            self.args.train_sp_batch_size,
        )
        
        # 计算训练步数和轮次
        self.num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps * self.args.sp_size / self.args.train_sp_batch_size)
        if not hasattr(self.args, "num_train_epochs") or self.args.num_train_epochs is None:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / self.num_update_steps_per_epoch)
        
        # 打印训练信息
        total_batch_size = (self.world_size * self.args.gradient_accumulation_steps / self.args.sp_size * self.args.train_sp_batch_size)
        main_print("***** Running training *****")
        main_print(f"  Num examples = {len(train_dataset)}")
        main_print(f"  Dataloader size = {len(train_dataloader)}")
        main_print(f"  Num Epochs = {self.args.num_train_epochs}")
        main_print(f"  Resume training from step {self.init_steps}")
        main_print(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        main_print(f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}")
        main_print(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        main_print(f"  Total optimization steps = {self.args.max_train_steps}")
        main_print(
            f"  Total training parameters per FSDP shard = {sum(p.numel() for p in self.transformer.parameters() if p.requires_grad) / 1e9} B"
        )
        main_print(f"  Master weight dtype: {next(self.transformer.parameters()).dtype}")
    
    def train(self):
        """执行完整训练过程"""
        # 初始化进度条
        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            disable=self.rank > 0,  # 仅在主进程显示
        )
        
        # 跳过已完成的步骤
        for i in range(self.init_steps):
            next(self.dataloader)
        
        step_times = deque(maxlen=100)
        
        # 主训练循环
        for step in range(self.init_steps + 1, self.args.max_train_steps + 1):
            start_time = time.time()
            
            # 设置debug保存目录
            debug_save_dir = ""
            if self.DEBUG:
                from pathlib import Path
                debug_save_dir = f"/vepfs-zulution/wangxuekuan/code/fast-video/Wan2.1/outputs/exp14_5_debug/{step}"
                Path(debug_save_dir).mkdir(parents=True, exist_ok=True)
            
            # 设置guidance_cfg
            if self.rank == 0:
                mid = int(self.args.cfg)
                a = random.randint(mid-2, mid+3)
                a_tensor = torch.tensor([a], dtype=torch.float32, device=self.device)
            else:
                a_tensor = torch.tensor([-1], dtype=torch.float32, device=self.device)
            
            # 主进程广播 a_tensor 到所有其他进程
            dist.broadcast(a_tensor, src=0)
            assert a_tensor[0].item() > 0
            guidance_cfg = a_tensor[0].item()
            
            main_print(f"guidance_cfg: {guidance_cfg}")
            
            torch.cuda.empty_cache()
            main_print(f"memory before step: {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024} GB")

            # 执行一步训练
            loss, grad_norm, pred_norm = self.step(
                self.dataloader,
                guidance_cfg,
                debug_save_dir=debug_save_dir,
                vae=self.vae,
                args=self.args
            )

            torch.cuda.empty_cache()
            main_print(f"memory after step: {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024} GB")
            gan_loss = self.gan_step(self.dataloader, guidance_cfg, args=self.args)
            
            # 计算步骤时间
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
            
            # 更新进度条
            main_print(f"Step {step}/{self.args.max_train_steps} - Loss: {loss:.4f} Gan Loss: {gan_loss:.4f} - Step time: {step_time:.2f}s")
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
            })
            progress_bar.update(1)
            
            # 保存检查点
            if step % self.args.checkpointing_steps == 0:
                if self.args.use_lora:
                    # 保存 LoRA 权重
                    save_lora_checkpoint(self.transformer, self.optimizer, self.rank, self.args.output_dir, step)
                else:
                    # 保存完整模型检查点
                    if self.args.use_ema:
                        save_checkpoint(self.ema_transformer, self.rank, self.args.output_dir, step)
                    else:
                        save_checkpoint(self.transformer, self.rank, self.args.output_dir, step)
                dist.barrier()
            
            torch.cuda.empty_cache()
        
        main_print("Training completed!")

    def step(
        self,
        loader,
        guidance_cfg=0,
        debug_save_dir="",
        vae=None,
        args=None,
    ):
        total_loss = 0.0
        self.optimizer.zero_grad()
        model_pred_norm = {
            "fro": 0.0,  # codespell:ignore
            "largest singular value": 0.0,
            "absolute mean": 0.0,
            "absolute max": 0.0,
        }
        for _ in range(self.gradient_accumulation_steps):
            latents, encoder_hidden_states, attention_mask, encoder_attention_mask, y, clip_feature = next(loader)
            if not args.i2v:
                y, clip_feature = None, None

            model_input = latents # hunyuan like ??
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            timesteps_all = self.noise_scheduler.timesteps
             
            index = torch.randint(0, args.interval_steps-args.k, (bsz, ), device=model_input.device).long()
            
            if self.sp_size > 1:
                broadcast(index)
                broadcast(noise)

            # if squeeze to [], unsqueeze to [1]
            timesteps = timesteps_all[index]

            #timesteps_prev = (sigmas_prev * noise_scheduler.config.num_train_timesteps).view(-1)
            timesteps_prev = [timesteps] + [timesteps_all[index+ ind + 1] for ind in range(args.k)]

            # noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input
            noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

            # main_print(f"timesteps {timesteps}, timesteps_prev {timesteps_prev}, index {index}")
            guidance_tensor = torch.tensor([guidance_cfg*1000],
                                                device=noisy_model_input.device,
                                                dtype=torch.bfloat16)
            # Predict the noise residual
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = [noisy_model_input[i] for i in range(noisy_model_input.size(0))]
                model_pred, mid_features = self.transformer(x,timesteps,None,self.max_seq_len,batch_context=encoder_hidden_states,context_mask=encoder_attention_mask,guidance=guidance_tensor,y=y,clip_fea=clip_feature,cls=self.args.gan,cls_layers=self.layer_idxs)
                model_pred = model_pred[0]

                if self.args.gan:
                    self.gan.requires_grad_(False)
                    logit = self.gan(mid_features, timesteps, encoder_hidden_states, self.max_seq_len, w=self.w, h=self.h, y=y, clip_fea=clip_feature)
                    fake_labels = torch.zeros_like(logit, device=self.device, dtype=torch.bfloat16)
                    gan_loss = self.gan_criterion(logit, fake_labels)

            self.noise_scheduler.model_outputs = [None] * self.noise_scheduler.config.solver_order
            self.noise_scheduler.lower_order_nums = 0
            self.noise_scheduler._step_index = int(index.item())
            self.noise_scheduler.last_sample = None
            model_pred_prev, model_pred_x0 = self.noise_scheduler.step(sample=noisy_model_input, model_output=model_pred, timestep=timesteps,return_dict=False)

            with torch.no_grad():
                x_prev = x
                for ind in range(args.k):
                    timesteps = timesteps_prev[ind]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        teacher_output = self.teacher_transformer(x_prev,timesteps,None,self.max_seq_len,batch_context=encoder_hidden_states,context_mask=encoder_attention_mask,guidance=guidance_tensor,y=y,clip_fea=clip_feature)[0].float()
                    
                    # x_prev = solver.euler_step(noisy_model_input, teacher_output, index)
                    self.noise_scheduler.lower_order_nums = 0
                    self.noise_scheduler.model_outputs = [None] * self.noise_scheduler.config.solver_order
                    self.noise_scheduler._step_index = int(index.item()) + ind
                    self.noise_scheduler.last_sample = None
                    x_prev = self.noise_scheduler.step(sample=torch.stack(x_prev), model_output=teacher_output, timestep=timesteps,return_dict=False)[0]
                    x_prev = [x_prev[i] for i in range(x_prev.size(0))]

            # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if self.ema_transformer is not None:
                        target_pred = self.ema_transformer(x_prev, timesteps_prev[-1], None, self.max_seq_len, batch_context=encoder_hidden_states, context_mask=encoder_attention_mask,guidance=guidance_tensor,y=y,clip_fea=clip_feature)[0]
                    else:
                        target_pred = self.teacher_transformer(x_prev,timesteps_prev[-1],None,self.max_seq_len,batch_context=encoder_hidden_states,context_mask=encoder_attention_mask,guidance=guidance_tensor,y=y,clip_fea=clip_feature)[0]

                self.noise_scheduler.lower_order_nums = 0
                self.noise_scheduler.model_outputs = [None] * self.noise_scheduler.config.solver_order
                self.noise_scheduler._step_index = int(index.item()) + args.k
                self.noise_scheduler.last_sample = None
                target_prev, traget_x0 = self.noise_scheduler.step(sample=torch.stack(x_prev), model_output=target_pred, timestep=timesteps_prev[-1],return_dict=False)

            huber_c = 0.001
            # loss = loss.mean()
            loss = (torch.mean(torch.sqrt((model_pred_x0.float() - traget_x0.float())**2 + huber_c**2) - huber_c) /
                    self.gradient_accumulation_steps)
            # loss += (torch.mean(torch.sqrt((model_pred_prev.float() - target_prev.float())**2 + huber_c**2) - huber_c) /
            #         gradient_accumulation_steps)

            if self.gan:
                loss += gan_loss * self.args.gan_loss_weight / self.gradient_accumulation_steps
            
            if self.pred_decay_weight > 0:
                if self.pred_decay_type == "l1":
                    pred_decay_loss = (torch.mean(torch.sqrt(model_pred.float()**2)) * self.pred_decay_weight /
                                    self.gradient_accumulation_steps)
                    loss += pred_decay_loss
                elif self.pred_decay_type == "l2":
                    # essnetially k2?
                    pred_decay_loss = (torch.mean(model_pred.float()**2) * self.pred_decay_weight / self.gradient_accumulation_steps)
                    loss += pred_decay_loss
                else:
                    assert NotImplementedError("pred_decay_type is not implemented")

            # calculate model_pred norm and mean
            get_norm(model_pred.detach().float(), model_pred_norm, self.gradient_accumulation_steps)
            loss.backward()

            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()

        # update ema
        if self.ema_transformer is not None:
            reshard_fsdp(self.ema_transformer)
            for p_averaged, p_model in zip(self.ema_transformer.parameters(), self.transformer.parameters()):
                with torch.no_grad():
                    p_averaged.copy_(torch.lerp(p_averaged.detach(), p_model.detach(), 1 - self.ema_decay))

        grad_norm = self.transformer.clip_grad_norm_(self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()

        if debug_save_dir:
            torch.cuda.empty_cache()
            assert vae is not None
            with torch.no_grad():
                for value,name in [(latents, "latents"), (model_pred, "model_pred"), (target_pred, "target")]:
                    video = vae.decode([value.squeeze(0).float()])[0]
                    print(f"rank{dist.get_rank()} value shape: {value.shape}, video shape: {video[0].shape}")

                    cache_video(tensor=video[None],save_file=f"{debug_save_dir}/rank{dist.get_rank()}_{name}.mp4",fps=16,nrow=1,normalize=True,value_range=(-1, 1))
            dist.barrier()

        return total_loss, grad_norm.item(), model_pred_norm

    def gan_step(
        self,
        loader,
        guidance_cfg=0,
        args=None,
    ):
        total_loss = 0.0
        self.gan_optimizer.zero_grad()
        model_pred_norm = {
            "fro": 0.0,  # codespell:ignore
            "largest singular value": 0.0,
            "absolute mean": 0.0,
            "absolute max": 0.0,
        }
        for _ in range(self.gradient_accumulation_steps):
            latents, encoder_hidden_states, attention_mask, encoder_attention_mask, y, clip_feature = next(loader)
            if not args.i2v:
                y, clip_feature = None, None

            model_input = latents # hunyuan like ??
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            timesteps_all = self.noise_scheduler.timesteps
             
            index = torch.randint(0,int(args.interval_steps*args.gan_max_noise_ratio)-1, (bsz, ), device=model_input.device).long()
            
            if self.sp_size > 1:
                broadcast(index)
                broadcast(noise)

            # if squeeze to [], unsqueeze to [1]
            timesteps = timesteps_all[index]

            # noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input
            noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

            # main_print(f"timesteps {timesteps}, timesteps_prev {timesteps_prev}, index {index}")
            guidance_tensor = torch.tensor([guidance_cfg*1000],
                                                device=noisy_model_input.device,
                                                dtype=torch.bfloat16)
            # Predict the noise residual
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = [noisy_model_input[i] for i in range(noisy_model_input.size(0))]
                with torch.no_grad():
                    model_pred, mid_features = self.transformer(x,timesteps,None,self.max_seq_len,batch_context=encoder_hidden_states,context_mask=encoder_attention_mask,guidance=guidance_tensor,y=y,clip_fea=clip_feature,cls=self.args.gan,cls_layers=self.layer_idxs)
                model_pred = model_pred[0]

                # self.gan.requires_grad_(True)
                # logit = self.gan(mid_features, timesteps, encoder_hidden_states, self.max_seq_len, w=self.w, h=self.h, y=y, clip_fea=clip_feature)
                # fake_labels = torch.ones_like(logit, device=self.device, dtype=torch.bfloat16)
                # gan_true_loss = self.gan_criterion(logit, fake_labels)

            self.noise_scheduler.model_outputs = [None] * self.noise_scheduler.config.solver_order
            self.noise_scheduler.lower_order_nums = 0
            self.noise_scheduler._step_index = int(index.item())
            self.noise_scheduler.last_sample = None
            model_pred_prev, model_pred_x0 = self.noise_scheduler.step(sample=noisy_model_input, model_output=model_pred, timestep=timesteps,return_dict=False)

            noisy_model_input = self.noise_scheduler.add_noise(model_pred_x0, noise, timesteps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = [noisy_model_input[i] for i in range(noisy_model_input.size(0))]
                with torch.no_grad():
                    model_pred_fake, mid_features_fake = self.transformer(x,timesteps,None,self.max_seq_len,batch_context=encoder_hidden_states,context_mask=encoder_attention_mask,guidance=guidance_tensor,y=y,clip_fea=clip_feature,cls=self.args.gan,cls_layers=self.layer_idxs)
                self.gan.requires_grad_(True)
                logit = self.gan(mid_features_fake, timesteps, encoder_hidden_states, self.max_seq_len, w=self.w, h=self.h, y=y, clip_fea=clip_feature)
            fake_labels = torch.zeros_like(logit, device=self.device, dtype=torch.bfloat16)
            gan_fake_loss = self.gan_criterion(logit, fake_labels)

            loss = gan_fake_loss
            loss.backward()

            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
        
        self.gan_optimizer.step()

        return total_loss


def main(args):
    # 设置 TF32 加速 (如果可用)
    torch.backends.cuda.matmul.allow_tf32 = True

    # 创建 DistillTrainer 实例并执行训练
    trainer = DistillTrainer(args)
    trainer.train()
    
    # 训练完成后释放资源
    destroy_sequence_parallel_group()
    dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser()

    ### wan arguments
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--null_encoded_path",
        type=str,
        default=None,
        help="The path to the null encoded path.")
    parser.add_argument("--resume_from_weight", type=str, default=None)

    ###orginal training arguments

    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_latent_t", type=int, default=28, help="Number of latent timesteps.")
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=5)

    # validation & logs
    parser.add_argument("--validation_sampling_steps", type=str, default="64")
    parser.add_argument("--validation_guidance_scale", type=str, default="4.5")

    parser.add_argument("--validation_steps", type=float, default=64)
    parser.add_argument("--log_validation", action="store_true", default=False)
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=("Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
              " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
              " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=("Whether training should be resumed from a previous checkpoint. Use a path saved by"
              ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=("Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
              ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
              " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
              " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument("--lora_alpha", type=int, default=256, help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_rank", type=int, default=128, help="LoRA rank parameter. ")
    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
              ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument("--num_euler_timesteps", type=int, default=100)
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    # ["euler_linear_quadratic", "pcm", "pcm_linear_qudratic"]
    parser.add_argument("--scheduler_type", type=str, default="pcm", help="The scheduler type to use.")
    parser.add_argument(
        "--linear_quadratic_threshold",
        type=float,
        default=0.025,
        help="Threshold for linear quadratic scheduler.",
    )
    parser.add_argument(
        "--linear_range",
        type=float,
        default=0.5,
        help="Range for linear quadratic scheduler.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay to apply.")
    parser.add_argument("--use_ema", default=False, help="Whether to use EMA.",action="store_true")
    parser.add_argument("--multi_phased_distill_schedule", type=str, default=None)
    parser.add_argument("--pred_decay_weight", type=float, default=0.0)
    parser.add_argument("--pred_decay_type", default="l1")
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    parser.add_argument("--debug",action="store_true",default=False)
    parser.add_argument("--i2v",action="store_true",default=False)
    parser.add_argument("--k",type=int,default=1)
    parser.add_argument("--height",type=int,default=720)
    parser.add_argument("--width",type=int,default=1280)
    parser.add_argument("--interval_steps",type=int,default=50)
    parser.add_argument("--gan",action="store_true",default=False)
    parser.add_argument("--gan_path",type=str,default=None)
    parser.add_argument("--gan_loss_weight",type=float,default=0.01)
    parser.add_argument("--gan_max_noise_ratio",type=float,default=0.6)
    parser.add_argument(
        "--gan_learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for GAN model.",
    )
    parser.add_argument(
        "--gan_weight_decay", 
        type=float, 
        default=0.001, 
        help="Weight decay for GAN optimizer."
    )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    main(args)
