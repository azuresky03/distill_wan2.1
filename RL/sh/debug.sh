PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

torchrun --nnodes 1 --nproc_per_node 8 --master-port 29521 \
    RL/scripts/train_wan_lora.py \
    --num_train_steps 10000 \
    --save_steps 10 \
    --output_dir /cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/outputs/debug/full \
    --log_dir /cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/outputs/valid_exp1/wan_prompt \
    --sp_size 8 \
    --seed 42 \
    --prompt_path "/cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/datasets/moviegen_benchmark.txt" \
    --reward_fn="HPSReward" \
    --reward_fn_kwargs='{"version": "v2.1"}' \
    --bf16 \
    --model_path "/vepfs-zulution/models/Wan2.1-T2V-14B" \
    --transfromer_dir "/cv/wangxuekuan/exp_models/distill/wanx-t2v/s5_exp0/checkpoint-1000" \
    --lora_rank 128 \
    --lora_alpha 64 \
    --sample_steps 3 \
    --shift 15 \
    --guidance_scale 8 \
    --train_batch_size 1 \
    --backprop_strategy "tail" \
    --backprop_num_steps 3 \
    --num_sampled_frames 1 \
    --num_decoded_latents 1 \
    --gradient_checkpointing \
    --t5_on_cpu  \
    --use_cpu_offload \
    --eval_prompt_path /cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/datasets/valid2.txt \
    --height 480 \
    --width 832 \
    --frame_number 81 \
    --extra_ckpt "/cv/zhangpengpeng/cv/video_generation/DMD2_wanx/outputs/cache/time_0409_1448|26/checkpoint_model_000499/feedforward.bin" \
    # --full_param \
    # --eval_prompt_path /vepfs-zulution/zhangpengpeng/cv/video_generation/HunyuanVideo/test_prompts.txt \
    # --resume_from_checkpoint /cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/outputs/exp2.7/checkpoint-50 \