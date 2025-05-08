export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# /vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/outputs/exp10_distill_cfg/checkpoint-200/diffusion_pytorch_model.safetensors
# --resume_from_weight /cv/wangxuekuan/release_model/wanx/distill_cfg_t2v/exp10_distill_cfg/checkpoint-500/ \
# --master_weight_type bf16\
torchrun --nnodes 1 --nproc_per_node 8 --master-port 29516\
    distill_exp/train/post_train.py \
    --height 720 \
    --width 1280 \
    --resume_from_weight "/cv/zhangpengpeng/cv/video_generation/DMD2_wanx/outputs/cache/time_0424_2046|41/checkpoint_model_000499/feedforward/" \
    --master_weight_type bf16\
    --cfg 5 \
    --gan_base_path "/vepfs-zulution/models/Wan2.1-T2V-14B" \
    --null_encoded_path /vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/mixkit/meta/null.pt\
    --data_json_path "/cv/zhangpengpeng/cv/video_generation/Wan2.1/data/processed/mixkit/720/videos2caption.json"\
    --output_dir "./output/post_gan/debug"\
    --checkpointing_steps 100 \
    --seed 42\
    --gradient_checkpointing\
    --train_batch_size=1\
    --num_latent_t 21 \
    --sp_size 8 \
    --train_sp_batch_size 1\
    --dataloader_num_workers 4\
    --gradient_accumulation_steps 1\
    --max_train_steps 2000\
    --learning_rate 1e-6 \
    --mixed_precision="bf16"\
    --allow_tf32 \
    --shift 5 \
    --use_cpu_offload \
    --sample_steps 1 \
    --gan \
    --gan_warmup_steps 10 \
    --r1_loss \
    --r1_noise_strength 0.05 \
    --r1_loss_weight 0.3 \
    --gan_update_per_gen 1 \
    --gan_learning_rate 1e-6 \
    # --gan_path /cv/zhangpengpeng/cv/video_generation/Wan2.1/output/post_gan/debug/gan_warmup.safetensors \