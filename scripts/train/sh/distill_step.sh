export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# /vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/outputs/exp10_distill_cfg/checkpoint-200/diffusion_pytorch_model.safetensors
# --resume_from_weight /cv/wangxuekuan/release_model/wanx/distill_cfg_t2v/exp10_distill_cfg/checkpoint-500/ \
# --master_weight_type bf16\
torchrun --nnodes 1 --nproc_per_node 8 --master-port 29516\
    scripts/train/distill_step.py \
    --max_seq_len 75600 \
    --resume_from_weight /cv/zhangpengpeng/cv/video_generation/Wan2.1/data/outputs/exp17_distill_cfg_i2v_720/checkpoint-400 \
    --master_weight_type bf16\
    --cfg 5 \
    --ckpt_dir /vepfs-zulution/wangxuekuan/code/algo_wanx_service/models/Wan2.1-I2V-14B-720P \
    --null_encoded_path /vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/mixkit/meta/null.pt\
    --data_json_path "./data/processed/videos2caption.json"\
    --output_dir "./data/outputs/debug"\
    --checkpointing_steps 50 \
    --seed 42\
    --gradient_checkpointing\
    --train_batch_size=1\
    --num_latent_t 21 \
    --sp_size 2 \
    --train_sp_batch_size 1\
    --dataloader_num_workers 4\
    --gradient_accumulation_steps=1\
    --max_train_steps 1000\
    --learning_rate=1e-6 \
    --mixed_precision="bf16"\
    --validation_steps 1000000\
    --validation_sampling_steps "2,4,8" \
    --checkpoints_total_limit 3\
    --allow_tf32 \
    --ema_decay 0.97 \
    --shift 5 \
    --multi_phased_distill_schedule "4000-1" \
    --not_apply_cfg_solver \
    --use_cpu_offload \
    --use_ema \
    --k 1 \
    --interval_steps 10 \
    --distill_mode half \
    --i2v