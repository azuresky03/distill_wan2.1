PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

torchrun --nnodes 1 --nproc_per_node 8 --master-port 29521 \
    RL/scripts/train_wan_lora.py \
    --num_train_steps 10000 \
    --save_steps 10 \
    --output_dir /cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/outputs/debug \
    --log_dir /cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/outputs/valid_exp1/wan_prompt \
    --sp_size 1 \
    --seed 42 \
    --prompt_path "/cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/datasets/moviegen_benchmark.txt" \
    --reward_fn="HPSReward" \
    --reward_fn_kwargs='{"version": "v2.1"}' \
    --bf16 \
    --model_path "/vepfs-zulution/models/Wan2.1-T2V-14B" \
    --transfromer_dir "/vepfs-zulution/models/Wan2.1-T2V-14B" \
    --lora_rank 128 \
    --lora_alpha 64 \
    --sample_steps 3 \
    --shift 5 \
    --train_batch_size 1 \
    --backprop_strategy "last" \
    --num_sampled_frames 1 \
    --num_decoded_latents 1 \
    --gradient_checkpointing \
    --t5_on_cpu  \
    --use_cpu_offload \
    --debug \
    --eval_prompt_path /cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/datasets/valid2.txt \
    --no_cfg_distill \
    --resume_from_checkpoint /cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/outputs/exp2.7/checkpoint-50 \
    --height 256 \
    --width 256 \
    --frame_number 81 \
    --random_decode \
    # --eval_prompt_path /vepfs-zulution/zhangpengpeng/cv/video_generation/HunyuanVideo/test_prompts.txt \