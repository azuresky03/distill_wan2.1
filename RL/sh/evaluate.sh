PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

torchrun --nnodes 1 --nproc_per_node 8 --master-port 29521 RL/scripts/evaluate_lora.py \
    --sample_steps 7 \
    --model_path /vepfs-zulution/models/Wan2.1-T2V-14B \
    --transfromer_dir /vepfs-zulution/models/Wan2.1-T2V-14B \
    --lora_checkpoint /cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/outputs/exp2.9/checkpoint-400 \
    --prompt_path /cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/datasets/valid2.txt \
    --num_noise_variations 3 \
    --output_dir /cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/exp_results/exp2.9 \
    --seed 42 \
    --bf16 \