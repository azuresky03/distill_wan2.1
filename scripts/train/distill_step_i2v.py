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

from fastvideo.distill.solver import EulerSolver, extract_into_tensor
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.models.mochi_hf.pipeline_mochi import linear_quadratic_schedule
from fastvideo.utils.checkpoint import (resume_lora_optimizer, save_checkpoint, save_lora_checkpoint)
from fastvideo.utils.communications import broadcast
from scripts.train.util.hidden_communication_data_wrapper import sp_parallel_dataloader_wrapper
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.utils.fsdp_util import (apply_fsdp_checkpointing, get_dit_fsdp_kwargs)
from fastvideo.utils.load import load_transformer
from fastvideo.utils.parallel_states import (destroy_sequence_parallel_group, get_sequence_parallel_state,
                                             initialize_sequence_parallel_state)
from fastvideo.utils.validation import log_validation


from scripts.dataset.hidden_datasets import (LatentDataset, latent_collate_function)
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

from scripts.train.util.util import load_weights, load_wan
from wan.utils.utils import cache_video
from scripts.train.model.model_cfg import WanModelCFG, WanAttentionBlock

from scripts.train.util.fm_solvers_unipc import FlowUniPCMultistepScheduler
import numpy as np

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


def distill_one_step(
    transformer,
    model_type,
    teacher_transformer,
    ema_transformer,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    # solver,
    noise_random_generator,
    gradient_accumulation_steps,
    sp_size,
    max_grad_norm,
    num_euler_timesteps,
    multiphase,
    not_apply_cfg_solver,
    distill_cfg,
    ema_decay,
    pred_decay_weight,
    pred_decay_type,
    hunyuan_teacher_disable_cfg,
    guidance_cfg=0,
    null_encoded=None,
    max_seq_len=32760,
    cfg_drop=False,
    CFG_TYPE = 0,
    debug_save_dir = "",
    vae=None,
):
    total_loss = 0.0
    optimizer.zero_grad()
    model_pred_norm = {
        "fro": 0.0,  # codespell:ignore
        "largest singular value": 0.0,
        "absolute mean": 0.0,
        "absolute max": 0.0,
    }
    for _ in range(gradient_accumulation_steps):
        latents, encoder_hidden_states, attention_mask, encoder_attention_mask, y, clip_feature = next(loader)
        model_input = latents # hunyuan like ??
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        timesteps_all = noise_scheduler.timesteps
         
        index = torch.randint(0, 50 - 6, (bsz, ), device=model_input.device).long()
        # print("timesteps ==>", timesteps_all.shape, index, )
        
        if sp_size > 1:
            broadcast(index)
            broadcast(noise)
        # Add noise according to flow matching.
        # sigmas = get_sigmas(start_timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        # sigmas = extract_into_tensor(noise_scheduler.sigmas, index, model_input.shape)
        # sigmas_prev = extract_into_tensor(noise_scheduler.sigmas_prev, index, model_input.shape)

        # timesteps = (sigmas * noise_scheduler.config.num_train_timesteps).view(-1)
        # if squeeze to [], unsqueeze to [1]
        timesteps = timesteps_all[index]

        #timesteps_prev = (sigmas_prev * noise_scheduler.config.num_train_timesteps).view(-1)
        timesteps_prev = [timesteps_all[index + 1], timesteps_all[index + 2], timesteps_all[index + 3], timesteps_all[index + 4], timesteps_all[index + 5], timesteps_all[index + 6]]


        # noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        # main_print(f"timesteps {timesteps}, timesteps_prev {timesteps_prev}, index {index}")
        guidance_tensor = torch.tensor([guidance_cfg*1000],
                                            device=noisy_model_input.device,
                                            dtype=torch.bfloat16)
        # Predict the noise residual
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = [noisy_model_input[i] for i in range(noisy_model_input.size(0))]
            model_pred = transformer(x,timesteps,None,max_seq_len,batch_context=encoder_hidden_states,context_mask=encoder_attention_mask,guidance=guidance_tensor,y=y,clip_fea=clip_feature)[0] #43200

        # # if accelerator.is_main_process:
        # main_print(f"index: {index} multiphase: {multiphase}")
        # model_pred, end_index = solver.euler_style_multiphase_pred(noisy_model_input, model_pred, index, multiphase)
        model_pred_org = model_pred.clone()
        # if accelerator.is_main_process:
        noise_scheduler.model_outputs = [None] * noise_scheduler.config.solver_order
        noise_scheduler.lower_order_nums = 0
        noise_scheduler._step_index = int(index.item())
        noise_scheduler.last_sample = None
        # model_pred = noise_scheduler.convert_model_output(sample=noisy_model_input, model_output=model_pred)
        model_pred, model_pred_prev = noise_scheduler.step(sample=noisy_model_input, model_output=model_pred, timestep=timesteps,return_dict=False)


        ## save video

        '''
        guidance_tensor = torch.tensor([guide_scale*1000],
                                            device=latent_model_input[0].device,
                                            dtype=torch.bfloat16)
        noise_pred = self.model(latent_model_input, t=timestep,context=context,seq_len=seq_len,guidance=guidance_tensor)[0]
        '''

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_output = teacher_transformer(x,timesteps,None,max_seq_len,batch_context=encoder_hidden_states,context_mask=encoder_attention_mask,guidance=guidance_tensor,y=y,clip_fea=clip_feature)[0].float()
            
            # x_prev = solver.euler_step(noisy_model_input, teacher_output, index)
            noise_scheduler.lower_order_nums = 0
            noise_scheduler.model_outputs = [None] * noise_scheduler.config.solver_order
            noise_scheduler._step_index = int(index.item())
            noise_scheduler.last_sample = None
            x_prev = noise_scheduler.step(sample=noisy_model_input, model_output=teacher_output, timestep=timesteps,return_dict=False)[0]

        # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
        for s in range(6):
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x_prev_list = [x_prev.float()[i] for i in range(x_prev.size(0))]
                    # if ema_transformer is not None:
                    #     target_pred = ema_transformer(x_prev_list, timesteps_prev, None, max_seq_len, batch_context=encoder_hidden_states, context_mask=encoder_attention_mask)[0]
                    # else:
                    target_pred = teacher_transformer(x_prev_list,timesteps_prev[s],None,max_seq_len,batch_context=encoder_hidden_states,context_mask=encoder_attention_mask,guidance=guidance_tensor,y=y,clip_fea=clip_feature)[0] #43200

                noise_scheduler.lower_order_nums = 0
                noise_scheduler.model_outputs = [None] * noise_scheduler.config.solver_order
                noise_scheduler._step_index = int(index.item()) + 1 + s
                noise_scheduler.last_sample = None
                target, target_prev = noise_scheduler.step(sample=x_prev, model_output=target_pred, timestep=timesteps_prev[s],return_dict=False)
                x_prev = target
                # target, end_index = solver.euler_style_multiphase_pred(x_prev, target_pred, index, multiphase, True)
                # target = noise_scheduler.convert_model_output(sample=x_prev, model_output=target_pred)


        huber_c = 0.001
        # loss = loss.mean()
        loss = (torch.mean(torch.sqrt((model_pred.float() - target.float())**2 + huber_c**2) - huber_c) /
                gradient_accumulation_steps)
        loss += (torch.mean(torch.sqrt((model_pred_prev.float() - target_prev.float())**2 + huber_c**2) - huber_c) /
                gradient_accumulation_steps)
        
        if pred_decay_weight > 0:
            if pred_decay_type == "l1":
                pred_decay_loss = (torch.mean(torch.sqrt(model_pred.float()**2)) * pred_decay_weight /
                                   gradient_accumulation_steps)
                loss += pred_decay_loss
            elif pred_decay_type == "l2":
                # essnetially k2?
                pred_decay_loss = (torch.mean(model_pred.float()**2) * pred_decay_weight / gradient_accumulation_steps)
                loss += pred_decay_loss
            else:
                assert NotImplementedError("pred_decay_type is not implemented")

        # calculate model_pred norm and mean
        get_norm(model_pred.detach().float(), model_pred_norm, gradient_accumulation_steps)
        loss.backward()

        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        total_loss += avg_loss.item()

    # update ema
    if ema_transformer is not None:
        reshard_fsdp(ema_transformer)
        for p_averaged, p_model in zip(ema_transformer.parameters(), transformer.parameters()):
            with torch.no_grad():
                p_averaged.copy_(torch.lerp(p_averaged.detach(), p_model.detach(), 1 - ema_decay))

    grad_norm = transformer.clip_grad_norm_(max_grad_norm)
    optimizer.step()
    lr_scheduler.step()

    if debug_save_dir:
        torch.cuda.empty_cache()
        assert vae is not None
        with torch.no_grad():
            for value,name in [(latents, "latents"), (model_pred, "model_pred"), (target, "target")]:
                video = vae.decode([value.squeeze(0).float()])[0]
                print(f"rank{dist.get_rank()} value shape: {value.shape}, video shape: {video[0].shape}")

                cache_video(tensor=video[None],save_file=f"{debug_save_dir}/rank{dist.get_rank()}_{name}.mp4",fps=16,nrow=1,normalize=True,value_range=(-1, 1))
        dist.barrier()

    return total_loss, grad_norm.item(), model_pred_norm


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    null_encoded = torch.load(args.null_encoded_path, map_location="cpu",weights_only=True)
    null_encoded = null_encoded.to(device)

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.

    # Create model:
    '''
        self.step_ratio = timesteps // euler_timesteps
        self.euler_timesteps = (np.arange(1, euler_timesteps + 1) * self.step_ratio).round().astype(np.int64) - 1
        self.euler_timesteps_prev = np.asarray([0] + self.euler_timesteps[:-1].tolist())
        self.sigmas = sigmas[self.euler_timesteps]
        self.sigmas_prev = np.asarray([sigmas[0]] +
                                      sigmas[self.euler_timesteps[:-1]].tolist())  # either use sigma0 or 0

        self.euler_timesteps = torch.from_numpy(self.euler_timesteps).long()
        self.euler_timesteps_prev = torch.from_numpy(self.euler_timesteps_prev).long()
        self.sigmas = torch.from_numpy(self.sigmas)
        self.sigmas_prev = torch.from_numpy(self.sigmas_prev)
    '''

    main_print(f"--> loading model from {args.ckpt_dir}")

    cfg = WAN_CONFIGS[args.task]

    # old_transformer = load_wan(
    #     config=cfg,
    #     checkpoint_dir=args.ckpt_dir,
    #     device_id=device,
    #     rank=rank,
    # )
    # state_dict = load_weights(args.resume_from_weight)
    # model_config = dict(old_transformer.config)
    # model_config["guidance_embed"] = True
    # transformer = WanModelCFG(**model_config)
    # result = transformer.load_state_dict(state_dict,strict=False)
    # if rank <= 0:
    #     print("Missing keys:", result.missing_keys)
    #     print("Unexpected keys:", result.unexpected_keys)
    #     print(f"CFG: load student distill model success {args.resume_from_weight}")
    transformer = WanModelCFG.from_pretrained(args.resume_from_weight).train()
    transformer.requires_grad_(True)

    # teacher_transformer = WanModelCFG(**model_config)
    # result = teacher_transformer.load_state_dict(state_dict,strict=False)
    # if rank <= 0:
    #     print("Missing keys:", result.missing_keys)
    #     print("Unexpected keys:", result.unexpected_keys)
    #     print(f"CFG: load teacher distill model success {args.resume_from_weight}")
    teacher_transformer = WanModelCFG.from_pretrained(args.resume_from_weight)
    teacher_transformer.requires_grad_(False)
    if args.use_ema:
        ema_transformer = deepcopy(transformer)
    else:
        ema_transformer = None
    
    # del state_dict, old_transformer
    torch.cuda.empty_cache()
    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M")
    main_print(f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}")
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
        (WanAttentionBlock,)
    )

    transformer = FSDP(
        transformer,
        **fsdp_kwargs,
    )
    teacher_transformer = FSDP(
        teacher_transformer,
        **fsdp_kwargs,
    )
    if args.use_ema:
        ema_transformer = FSDP(
            ema_transformer,
            **fsdp_kwargs,
        )
    main_print("--> model loaded")
    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, no_split_modules, args.selective_checkpointing)
        apply_fsdp_checkpointing(teacher_transformer, no_split_modules, args.selective_checkpointing)
        if args.use_ema:
            apply_fsdp_checkpointing(ema_transformer, no_split_modules, args.selective_checkpointing)
    # Set model as trainable.
    transformer.train()
    teacher_transformer.requires_grad_(False)
    if args.use_ema:
        ema_transformer.requires_grad_(False)

    DEBUG = False
    if DEBUG:
        from wan.modules.vae import WanVAE
        vae = WanVAE(vae_pth="/vepfs-zulution/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth")
        # autocast_type = next(vae.model.parameters()).dtype
        autocast_type = torch.bfloat16
        vae.model = vae.model.to(device).to(autocast_type)
        vae.model.eval()
        vae.model.requires_grad_(False)
    else:
        vae = None

    # noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=args.shift)
    # if args.scheduler_type == "pcm_linear_quadratic":
    #     linear_steps = int(noise_scheduler.config.num_train_timesteps * args.linear_range)
    #     sigmas = linear_quadratic_schedule(
    #         noise_scheduler.config.num_train_timesteps,
    #         args.linear_quadratic_threshold,
    #         linear_steps,
    #     )
    #     sigmas = torch.tensor(sigmas).to(dtype=torch.float32)
    # else:
    #     sigmas = noise_scheduler.sigmas
    # args.shift = 
    noise_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)
    # sigmas = np.array(noise_scheduler.sigmas)
    
    euler_timesteps = 50 #args.num_euler_timesteps
    # # step_ratio = noise_scheduler.config.num_train_timesteps // euler_timesteps
    # # euler_timesteps = (np.arange(1, euler_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
    # # new_sigmas = sigmas[euler_timesteps]

    # sigmas_prev = np.asarray([sigmas[0]] + sigmas[euler_timesteps[:-1]].tolist())

    # noise_scheduler.sigmas = torch.from_numpy(new_sigmas).to(device)
    # noise_scheduler.sigmas_prev = torch.from_numpy(sigmas_prev).to(device)
    noise_scheduler.num_inference_steps = 50 #50 #-1e6
    noise_scheduler.set_timesteps(50, device=device, shift=args.shift)

    timesteps = noise_scheduler.timesteps
    print("ssss ==>> timesteps:", timesteps)


    # solver = EulerSolver(
    #     sigmas.numpy()[::-1],
    #     noise_scheduler.config.num_train_timesteps,
    #     euler_timesteps=args.num_euler_timesteps,
    # )
    # solver.to(device)

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    if args.resume_from_lora_checkpoint:
        transformer, optimizer, init_steps = resume_lora_optimizer(transformer, args.resume_from_lora_checkpoint,
                                                                   optimizer)
    main_print(f"optimizer: {optimizer}")

    # todo add lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * world_size,
        num_training_steps=args.max_train_steps * world_size,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path)
    sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps * args.sp_size / args.train_sp_batch_size)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # if rank <= 0:
    #     project = args.tracker_project_name or "fastvideo"
    #     wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (world_size * args.gradient_accumulation_steps / args.sp_size * args.train_sp_batch_size)
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    dist.barrier()

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    # todo future
    for i in range(init_steps):
        next(loader)

    # log_validation(args, transformer, device,
    #             torch.bfloat16, 0, scheduler_type=args.scheduler_type, shift=args.shift, num_euler_timesteps=args.num_euler_timesteps, linear_quadratic_threshold=args.linear_quadratic_threshold,ema=False)
    def get_num_phases(multi_phased_distill_schedule, step):
        # step-phase,step-phase
        multi_phases = multi_phased_distill_schedule.split(",")
        phase = multi_phases[-1].split("-")[-1]
        for step_phases in multi_phases:
            phase_step, phase = step_phases.split("-")
            if step <= int(phase_step):
                return int(phase)
        return phase

    for step in range(init_steps + 1, args.max_train_steps + 1):
        start_time = time.time()
        assert args.multi_phased_distill_schedule is not None
        num_phases = get_num_phases(args.multi_phased_distill_schedule, step)

        if DEBUG:
            from pathlib import Path
            debug_save_dir = f"/vepfs-zulution/wangxuekuan/code/fast-video/Wan2.1/outputs/exp14_5_debug/{step}"
            Path(debug_save_dir).mkdir(parents=True, exist_ok=True)
        else:
            debug_save_dir = ""

        if rank == 0:
            mid = int(args.cfg)
            a = random.randint(mid-2,mid+3)
            # a = random.random() * (up_cfg - low_cfg) + low_cfg
            a_tensor = torch.tensor([a], dtype=torch.float32,device=device)
        else:
            a_tensor = torch.tensor([-1], dtype=torch.float32,device=device)

        # 主进程广播 a_tensor 到所有其他进程
        dist.broadcast(a_tensor, src=0)
        assert a_tensor[0].item() >0
        guidance_cfg = a_tensor[0].item()
        # print("")

        main_print(f"guidance_cfg: {guidance_cfg}")
        loss, grad_norm, pred_norm = distill_one_step(
            transformer,
            args.model_type,
            teacher_transformer,
            ema_transformer,
            optimizer,
            lr_scheduler,
            loader,
            noise_scheduler,
            # solver,
            noise_random_generator,
            args.gradient_accumulation_steps,
            args.sp_size,
            args.max_grad_norm,
            args.num_euler_timesteps,
            num_phases,
            args.not_apply_cfg_solver,
            args.distill_cfg,
            args.ema_decay,
            args.pred_decay_weight,
            args.pred_decay_type,
            args.hunyuan_teacher_disable_cfg,
            guidance_cfg,
            null_encoded,
            args.max_seq_len,
            step%10==11,
            CFG_TYPE=args.cfg_type,
            debug_save_dir=debug_save_dir,
            vae=vae
        )

        step_time = time.time() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)

        main_print(f"Step {step}/{args.max_train_steps} - Loss: {loss:.4f} - Step time: {step_time:.2f}s")
        progress_bar.set_postfix({
            "loss": f"{loss:.4f}",
            "step_time": f"{step_time:.2f}s",
            "grad_norm": grad_norm,
            "phases": num_phases,
        })
        progress_bar.update(1)

        if step % args.checkpointing_steps == 0:
            if args.use_lora:
                # Save LoRA weights
                save_lora_checkpoint(transformer, optimizer, rank, args.output_dir, step)
            else:
                # Your existing checkpoint saving code
                print("args.use_ema +++>>>", args.use_ema)
                if args.use_ema:
                    save_checkpoint(ema_transformer, rank, args.output_dir, step)
                else:
                    save_checkpoint(transformer, rank, args.output_dir, step)
            dist.barrier()
        torch.cuda.empty_cache()


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
    parser.add_argument("--max_seq_len", type=int, default=32760) #32760 480p 75600 720p
    parser.add_argument("--cfg_type", type=int, default=0)

    ###orginal training arguments

    parser.add_argument("--model_type", type=str, default="mochi", help="The type of model to train.")

    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_height", type=int, default=480)
    parser.add_argument("--num_width", type=int, default=848)
    parser.add_argument("--num_frames", type=int, default=163)
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
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
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
    parser.add_argument(
        "--not_apply_cfg_solver",
        action="store_true",
        help="Whether to apply the cfg_solver.",
    )
    parser.add_argument("--distill_cfg", type=float, default=3.0, help="Distillation coefficient.")
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
    parser.add_argument("--use_ema", default=False, help="Whether to use EMA.")
    parser.add_argument("--multi_phased_distill_schedule", type=str, default=None)
    parser.add_argument("--pred_decay_weight", type=float, default=0.0)
    parser.add_argument("--pred_decay_type", default="l1")
    parser.add_argument("--hunyuan_teacher_disable_cfg", action="store_true")
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    main(args)
