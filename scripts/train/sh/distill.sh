export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

DATA_DIR=./data
# IP=[MASTER NODE IP]

# --master_weight_type bf16\
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
torchrun --nnodes 1 --nproc_per_node 8 --master-port 29516\
    scripts/train/distill.py\
    --cfg_type 0\
    --max_seq_len 15600 \
    --master_weight_type bf16\
    --ckpt_dir /vepfs-zulution/models/Wan2.1-T2V-14B\
    --null_encoded_path /vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/mixkit/meta/null.pt\
    --data_json_path "$DATA_DIR/mixkit/processed_480/videos2caption.json"\
    --output_dir "$DATA_DIR/outputs/exp4_1node_480"\
    --checkpointing_steps 50\
    --seed 42\
    --pretrained_model_name_or_path $DATA_DIR/hunyuan\
    --dit_model_name_or_path $DATA_DIR/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt\
    --model_type "hunyuan" \
    --cache_dir "$DATA_DIR/.cache"\
    --validation_prompt_dir "$DATA_DIR/HD-Mixkit-Finetune-Hunyuan/validation"\
    --gradient_checkpointing\
    --train_batch_size=1\
    --num_latent_t 20 \
    --sp_size 2 \
    --train_sp_batch_size 1\
    --dataloader_num_workers 4\
    --gradient_accumulation_steps=1\
    --max_train_steps 1000\
    --learning_rate=1e-6\
    --mixed_precision="bf16"\
    --validation_steps 1000000\
    --validation_sampling_steps "2,4,8" \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 5\
    --tracker_project_name Hunyuan_Distill \
    --num_height 720 \
    --num_width 1280 \
    --num_frames  81 \
    --shift 5 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --not_apply_cfg_solver 