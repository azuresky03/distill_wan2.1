#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.amp as amp

import numpy as np
import random
from tqdm import tqdm
import logging
import time
from pathlib import Path
import sys
import json

# 添加父目录到sys.path以便导入相关模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.modules.model import WanModel
from distill_exp.gan.model_cfg import WanModelCFG
from distill_exp.gan.model import WanGanModel
from scripts.train.util.fm_solvers_unipc import FlowUniPCMultistepScheduler

def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)

class GanTrainer:
    """基于DDP的GAN训练器"""
    
    def __init__(self, args):
        """
        初始化训练器
        
        Args:
            args: 包含训练配置的参数
        """
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.setup_distributed()
        
        # 设置随机种子以确保可复现性
        self.set_seed(args.seed)
        
        # 创建日志目录
        self.setup_logging()
        
        # 加载模型和优化器（包含从检查点恢复的逻辑）
        self.setup_model()
        self.h = args.height//8
        self.w = args.width//8
        self.seq_len = 21*self.h*self.w // 4
        main_print(f"seq_len: {self.seq_len}")

        # 设置优化器和学习率调度器
        self.setup_optimizer()
        
        # 设置损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.context_manager = amp.autocast(device_type='cuda', dtype=torch.bfloat16)
        
        self.setup_dataloader()
    
    def setup_distributed(self):
        """设置分布式训练环境"""
        if dist.is_initialized():
            return
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.local_rank)
        
        if self.local_rank == 0:
            print(f"Starting training with {dist.get_world_size()} GPUs")

        self.device = torch.device(f"cuda:{self.local_rank}")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16

        dist.barrier()
    
    def setup_model(self):
        """加载和设置模型"""
        # 自动启用 bf16 训练的相关设置
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        
        # 加载基座模型
        base_model_path = self.args.base_model_path
        
        model_base = WanModel.from_pretrained(base_model_path)
        main_print(f"Base model loaded from {base_model_path}, max memory: {torch.cuda.max_memory_reserved(self.device) / 1024 / 1024:.2f}MB")

        self.distilled_model = WanModelCFG.from_pretrained(self.args.distilled_model_path).to(self.dtype).to(self.device)
        self.distilled_model.requires_grad_(False)
        self.distilled_model.eval()
        main_print(f"Distilled model loaded from {self.args.distilled_model_path}, max memory: {torch.cuda.max_memory_reserved(self.device) / 1024 / 1024:.2f}MB")
        
        # 创建GAN模型 (默认使用 bf16)
        self.model = WanGanModel(
            model_base, 
            num_layers=self.args.num_layers,
            layer_idxs=self.args.layer_idxs,
            device=self.device,
            dtype=torch.bfloat16,
            desired_len=self.args.desired_len,
            compressor_heads=self.args.compressor_heads
        )
        self.model.requires_grad_(True)
        self.model.train()
        main_print(f"GAN model created, max memory: {torch.cuda.max_memory_reserved(self.device) / 1024 / 1024:.2f}MB")

        del model_base
        torch.cuda.empty_cache()

        self.noise_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)
        
        # 从检查点恢复训练（如果有）
        self.start_epoch = 0
        self.best_score = float('inf')
        
        # 先包装模型为DDP模型
        self.model = DDP(self.model)
        
        # 从检查点恢复训练（如果有）- 移动到DDP封装之后
        if self.args.resume and os.path.exists(self.args.resume):
            # 只在主进程打印日志
            if self.local_rank == 0:
                self.logger.info(f"从检查点恢复训练: {self.args.resume}")
            
            # 加载检查点，确保正确映射到当前设备
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
            checkpoint = torch.load(self.args.resume, map_location=map_location)
            
            # 加载模型状态字典(与save_checkpoint对应)
            self.model.module.load_state_dict(checkpoint['model'])
            
            # 加载优化器状态(需要在setup_optimizer之后调用)
            if 'optimizer' in checkpoint and hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                
            # 加载学习率调度器状态(如果有)
            if 'scheduler' in checkpoint and hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                
            # 恢复训练进度
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_score = checkpoint.get('best_score', float('inf'))
            
            if self.local_rank == 0:
                self.logger.info(f"成功恢复模型状态，从第{self.start_epoch}轮开始训练")
        else:
            if self.args.resume and self.local_rank == 0:
                self.logger.warning(f"指定的检查点文件不存在: {self.args.resume}，将从头开始训练")
                
    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        # 对不同参数组应用不同的学习率
        param_groups = []
        
        # 特征提取器参数
        extractor_params = {'params': self.model.module.extractor.parameters(), 'lr': self.args.lr_extractor}
        param_groups.append(extractor_params)
        
        # 压缩器参数
        compressor_params = {'params': self.model.module.compressor.parameters(), 'lr': self.args.lr}
        param_groups.append(compressor_params)
        
        # 自注意力参数
        attn_params = {'params': self.model.module.self_attn.parameters(), 'lr': self.args.lr}
        param_groups.append(attn_params)
        
        # MLP参数
        mlp_params = {'params': self.model.module.mlp.parameters(), 'lr': self.args.lr}
        param_groups.append(mlp_params)
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
            betas=(self.args.beta1, self.args.beta2)
        )
        
        # 创建学习率调度器
        if self.args.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs, eta_min=self.args.min_lr
            )
        elif self.args.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.args.lr_step, gamma=self.args.lr_gamma
            )
        else:  # None或其他
            self.scheduler = None
    
    def setup_logging(self):
        """设置日志"""
        if self.local_rank == 0:
            self.log_dir = Path(self.args.output_dir) / 'logs'
            self.checkpoint_dir = Path(self.args.output_dir) / 'checkpoints'
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(self.log_dir / 'train.log'),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Arguments: {self.args}")
    
    def set_seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def setup_dataloader(self,valid_ratio=0.1):
        # 读取JSON文件
        with open(self.args.json_path, 'r') as f:
            data_list = json.load(f)
        
        fixed_seed = 42
        random.seed(fixed_seed)
        random.shuffle(data_list)
        
        # 计算验证集大小
        valid_size = int(len(data_list) * valid_ratio)
        
        # 分割为训练集和验证集
        valid_items = data_list[:valid_size]
        train_items = data_list[valid_size:]
        
        # 分布式训练: 将样本分配给不同的GPU
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = self.local_rank
            
            # 计算每个进程应该处理的样本数量
            train_per_rank = len(train_items) // world_size
            valid_per_rank = len(valid_items) // world_size
            
            # 计算每个进程的起始和结束索引
            train_start = rank * train_per_rank
            train_end = train_start + train_per_rank 
            
            valid_start = rank * valid_per_rank
            valid_end = valid_start + valid_per_rank 
            
            # 分配样本
            train_items = train_items[train_start:train_end]
            valid_items = valid_items[valid_start:valid_end]
            
            if self.local_rank == 0:
                self.logger.info(f"分布式训练: 将数据分配给 {world_size} 个GPU进程")
        
        if self.local_rank == 0:
            self.logger.info(f"数据集总大小: {len(data_list)}")
            self.logger.info(f"训练集总大小: {len(data_list[valid_size:])}, 验证集总大小: {len(data_list[:valid_size])}")
            
        print(f"本进程(rank {self.local_rank})处理: 训练样本 {len(train_items)}, 验证样本 {len(valid_items)}")
        print(f"本进程(rank {self.local_rank}) 训练集样本示例: {train_items[0]['latent_path']} 验证集样本示例: {valid_items[0]['latent_path']}")

        if self.args.debug:
            n = self.args.debug
            train_items = train_items[:n]
            valid_items = valid_items[:10]
            if self.local_rank == 0:
                self.logger.info(f"Debug模式启用: 每个epoch限制为{n}个样本")
        
        self.train_items = train_items
        self.valid_items = valid_items
        
        dist.barrier()
        return train_items, valid_items
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点 - 仅在主进程(rank 0)执行"""
        if self.local_rank != 0:
            return
            
        checkpoint = {
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_score': self.best_score,
            'args': self.args
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        # 保存当前 epoch 的检查点
        checkpoint_path = str(self.checkpoint_dir / f'epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
            
        # 如果是最佳模型，单独保存
        if is_best:
            best_path = str(self.checkpoint_dir / 'best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型: {best_path}")
            
        self.logger.info(f"检查点已保存: {checkpoint_path}")
    
    def train_one_epoch(self, epoch):
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch
            
        Returns:
            train_loss: 训练损失
        """
        self.model.train()
        train_loss = 0
        
        # 用于计算每10步的平均损失
        step_losses = []
        
        pbar = tqdm(self.train_items, disable=self.local_rank != 0)
        for batch_idx, item in enumerate(pbar):
            # 首先清除所有梯度
            self.optimizer.zero_grad()
            
            gt = torch.load(item["latent_path"],weights_only=True).to(self.device)
            fake = torch.load(item["generated_latent_path"],weights_only=True).to(self.device)
            prompt_embed = [torch.load(item["prompt_embed_path"],weights_only=False).to(self.device)]
            y = [torch.load(item["y_path"],weights_only=False).to(self.device)] if self.args.i2v else None
            clip_fea = torch.load(item["clip_feature_path"],weights_only=False).to(self.device) if self.args.i2v else None

            sigmas = random.randint(1,1000)
            t = torch.tensor([sigmas],device=self.device).to(self.dtype)
            noise = torch.randn_like(fake).to(self.device).to(self.dtype)
            gt_noised = [gt * (1 - sigmas / 1000) + noise * (sigmas / 1000)]
            fake_noised = [fake * (1 - sigmas / 1000) + noise * (sigmas / 1000)]
            cfg_tensor = torch.tensor([random.randint(3,8)*1000],device=self.device).to(self.dtype)
            # gt_noised = [self.noise_scheduler.add_noise(gt,noise,t)]
            # fake_noised = [self.noise_scheduler.add_noise(fake,noise,t)]

            # 步骤1: 计算真实样本的损失 (不立即更新梯度)
            with self.context_manager:
                with torch.no_grad():
                    gt_mid = self.distilled_model(gt_noised, t, prompt_embed, self.seq_len, y=y, clip_fea=clip_fea, cls=True, cls_layers=self.args.layer_idxs,guidance=cfg_tensor)[1]
                gt_logit = self.model(gt_mid, t, prompt_embed, self.seq_len, w=self.w, h=self.h, y=y, clip_fea=clip_fea)
                real_labels = torch.ones_like(gt_logit, device=self.device, dtype=self.dtype)
                real_loss = self.criterion(gt_logit, real_labels)
            
            # 反向传播但不更新梯度
            real_loss.backward()
            
            # 释放内存
            del gt_mid, gt_logit
            torch.cuda.empty_cache()
            
            # 步骤2: 计算生成样本的损失 (梯度会累积)
            with self.context_manager:
                with torch.no_grad():
                    fake_mid = self.distilled_model(fake_noised, t, prompt_embed, self.seq_len, y=y, clip_fea=clip_fea, cls=True, cls_layers=self.args.layer_idxs,guidance=cfg_tensor)[1]
                fake_logit = self.model(fake_mid, t, prompt_embed, self.seq_len, w=self.w, h=self.h, y=y, clip_fea=clip_fea)
                fake_labels = torch.zeros_like(fake_logit, device=self.device, dtype=self.dtype)
                fake_loss = self.criterion(fake_logit, fake_labels)
            
            # 反向传播累积梯度
            fake_loss.backward()
            
            # 在累积了两部分梯度后，一次性更新参数
            self.optimizer.step()

            # 清理缓存以减少内存使用
            del fake_mid, fake_logit
            torch.cuda.empty_cache()
            
            # 计算总损失（仅用于日志记录）
            total_loss = (real_loss.item() + fake_loss.item()) / 2.0
            
            # 添加到步骤损失列表
            step_losses.append(total_loss)
                
            # 计算平均损失
            reduced_loss = total_loss
            if dist.is_initialized():
                loss_tensor = torch.tensor([reduced_loss]).to(self.device)
                dist.all_reduce(loss_tensor)
                reduced_loss = loss_tensor.item() / dist.get_world_size()
                
            train_loss += reduced_loss
            
            # 每10步计算一次平均损失并显示
            if (batch_idx + 1) % 10 == 0 and len(step_losses) > 0:
                # 获取当前学习率
                lr_info = []
                for group in self.optimizer.param_groups:
                    name = group.get('name', 'unknown')
                    lr = group['lr']
                    lr_info.append(f"{name}: {lr:.2e}")
                lr_str = ", ".join(lr_info)

                avg_10step_loss = sum(step_losses[-10:]) / min(10, len(step_losses))
                main_print(f"Epoch {epoch}, Step {batch_idx+1}: 10-step avg loss: {avg_10step_loss:.4f}, lr: [{lr_str}]")
                step_losses = []
            
            if self.local_rank == 0:
                pbar.set_description(
                    f"Epoch {epoch} train loss: {reduced_loss:.4f} "
                    f"(real: {real_loss.item():.4f}, fake: {fake_loss.item():.4f}), "
                )
        
        # 计算平均训练损失
        train_loss = train_loss / len(self.train_items)
        
        return train_loss
    
    def validate(self):
        """
        在验证集上验证模型
        
        Returns:
            results: 包含验证结果的字典，包括损失和准确率指标
        """
        self.model.eval()
        val_loss = 0
        
        # 分别统计真实样本和生成样本的准确率
        real_correct = 0
        real_total = 0
        fake_correct = 0
        fake_total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.valid_items, disable=self.local_rank != 0)
            for batch_idx, item in enumerate(pbar):
                # 加载验证数据
                gt = torch.load(item["latent_path"],weights_only=True).to(self.device)
                fake = torch.load(item["generated_latent_path"],weights_only=True).to(self.device)
                prompt_embed = [torch.load(item["prompt_embed_path"],weights_only=False).to(self.device)]
                y = [torch.load(item["y_path"],weights_only=False).to(self.device)] if self.args.i2v else None
                clip_fea = torch.load(item["clip_feature_path"],weights_only=False).to(self.device) if self.args.i2v else None

                sigmas = random.randint(1,1000)
                t = torch.tensor([sigmas],device=self.device).to(self.dtype)
                noise = torch.randn_like(fake).to(self.device).to(self.dtype)
                gt_noised = [gt * (1 - sigmas / 1000) + noise * (sigmas / 1000)]
                fake_noised = [fake * (1 - sigmas / 1000) + noise * (sigmas / 1000)]
                cfg_tensor = torch.tensor([random.randint(3,8)*1000],device=self.device).to(self.dtype)

                # 在验证过程中，可以一次性计算真实和生成样本的损失
                with self.context_manager:
                    # 计算真实样本的输出
                    gt_mid = self.distilled_model(gt_noised, t, prompt_embed, self.seq_len, y=y, clip_fea=clip_fea, cls=True, cls_layers=self.args.layer_idxs,guidance=cfg_tensor)[1]
                    gt_logit = self.model(gt_mid, t, prompt_embed, self.seq_len, w=self.w, h=self.h, y=y, clip_fea=clip_fea)
                    real_labels = torch.ones_like(gt_logit, device=self.device, dtype=self.dtype)
                    real_loss = self.criterion(gt_logit, real_labels)
                    
                    # 计算生成样本的输出
                    fake_mid = self.distilled_model(fake_noised, t, prompt_embed, self.seq_len, y=y, clip_fea=clip_fea, cls=True, cls_layers=self.args.layer_idxs,guidance=cfg_tensor)[1]
                    fake_logit = self.model(fake_mid, t, prompt_embed, self.seq_len, w=self.w, h=self.h, y=y, clip_fea=clip_fea)
                    fake_labels = torch.zeros_like(fake_logit, device=self.device, dtype=self.dtype)
                    fake_loss = self.criterion(fake_logit, fake_labels)
                
                # 计算总损失
                total_loss = (real_loss.item() + fake_loss.item()) / 2.0
                
                # 计算各类准确率
                pred_real = (torch.sigmoid(gt_logit) > 0.5).float()
                pred_fake = (torch.sigmoid(fake_logit) <= 0.5).float()
                
                # 累加真实样本的正确预测和总数
                real_correct += pred_real.sum().item()
                real_total += gt_logit.size(0)
                
                # 累加生成样本的正确预测和总数
                fake_correct += pred_fake.sum().item()
                fake_total += fake_logit.size(0)
                
                # 计算平均损失
                reduced_loss = total_loss
                if dist.is_initialized():
                    loss_tensor = torch.tensor([reduced_loss]).to(self.device)
                    dist.all_reduce(loss_tensor)
                    reduced_loss = loss_tensor.item() / dist.get_world_size()
                    
                val_loss += reduced_loss
                
                # 释放内存
                del gt_mid, gt_logit, fake_mid, fake_logit
                torch.cuda.empty_cache()
                
                if self.local_rank == 0:
                    # 计算当前的准确率
                    real_acc = 100 * real_correct / real_total if real_total > 0 else 0
                    fake_acc = 100 * fake_correct / fake_total if fake_total > 0 else 0
                    total_acc = 100 * (real_correct + fake_correct) / (real_total + fake_total) if (real_total + fake_total) > 0 else 0
                    
                    pbar.set_description(
                        f"Val loss: {reduced_loss:.4f} (real: {real_loss.item():.4f}, fake: {fake_loss.item():.4f}), "
                        f"Acc: {total_acc:.2f}% (real: {real_acc:.2f}%, fake: {fake_acc:.2f}%)"
                    )
        
        # 同步各进程的统计数据
        if dist.is_initialized():
            metrics = torch.tensor([real_correct, real_total, fake_correct, fake_total], device=self.device)
            dist.all_reduce(metrics)
            real_correct, real_total, fake_correct, fake_total = metrics.tolist()
            
        # 计算最终准确率
        real_accuracy = 100 * real_correct / real_total if real_total > 0 else 0
        fake_accuracy = 100 * fake_correct / fake_total if fake_total > 0 else 0
        total_accuracy = 100 * (real_correct + fake_correct) / (real_total + fake_total) if (real_total + fake_total) > 0 else 0
        
        # 计算平均验证损失
        val_loss = val_loss / len(self.valid_items)
        
        # 将结果整理为字典
        results = {
            'loss': val_loss,
            'acc': total_accuracy,
            'real_acc': real_accuracy,
            'fake_acc': fake_accuracy
        }
        
        if self.local_rank == 0:
            self.logger.info(
                f"Validation - Loss: {results['loss']:.4f}, "
                f"Accuracy: {results['acc']:.2f}% (real: {results['real_acc']:.2f}%, fake: {results['fake_acc']:.2f}%)"
            )
            
        return results
    
    def train(self):
        """训练模型"""
        if self.local_rank == 0:
            self.logger.info(f"Starting training from epoch {self.start_epoch}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练一个epoch
            train_loss = self.train_one_epoch(epoch)
            
            # 验证
            val_results = self.validate()
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录并保存模型
            if self.local_rank == 0:
                self.logger.info(
                    f"Epoch {epoch}: train loss = {train_loss:.4f}, "
                    f"val loss = {val_results['loss']:.4f}, "
                    f"val acc = {val_results['acc']:.2f}% (real: {val_results['real_acc']:.2f}%, fake: {val_results['fake_acc']:.2f}%)"
                )
                
                # 保存最佳模型
                is_best = val_results['loss'] < self.best_score
                if is_best:
                    self.best_score = val_results['loss']
                
                if epoch % self.args.save_freq == 0: 
                    self.save_checkpoint(epoch, is_best)
        
        if self.local_rank == 0:
            self.logger.info("Training completed")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train WanGanModel with DDP')
    
    # 基本训练参数
    parser.add_argument('--base_model_path', type=str, default='/vepfs-zulution/models/Wan2.1-T2V-14B',
                        help='Base Wan model path')
    parser.add_argument('--distilled_model_path', type=str, default='/vepfs-zulution/models/Wan2.1-T2V-14B')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint frequency (epochs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    
    parser.add_argument('--json_path', type=str, default='/cv/zhangpengpeng/cv/video_generation/Wan2.1/data/processed/mixkit/video/wan_exp10_cfg_480/video_meta.json')
    
    # 模型参数
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in WanExtractor')
    parser.add_argument('--layer_idxs', type=int, nargs='+', default=[9,39],
                        help='Layer indices for feature extraction')
    parser.add_argument('--desired_len', type=int, default=16,
                        help='Desired sequence length after compression')
    parser.add_argument('--compressor_heads', type=int, default=8,
                        help='Number of attention heads in compressor')
    
    # 优化器参数
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_extractor', type=float, default=1e-5,
                        help='Learning rate for extractor')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_step', type=int, default=10,
                        help='Step size for step learning rate scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Gamma for step learning rate scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine scheduler')
    
    # 数据加载参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--i2v', action='store_true', default=False,)
    parser.add_argument('--debug', type=int, default=0)
    
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = GanTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()