# export WANDB_MODE="offline"
# CUDA_VISIBLE_DEVICES=5

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

GPU_NUM=1
MODEL_PATH="/vepfs-zulution/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
MODEL_TYPE="hunyuan"
DATA_MERGE_PATH="/vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/movie/merge_480p.txt"
OUTPUT_DIR="/vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/movie/processed_480"
VALIDATION_PATH="assets/prompt.txt"

# OUTPUT_DIR="/vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/mixkit/processed_480"
# DATA_MERGE_PATH="/vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/mixkit/merge_480.txt"

# OUTPUT_DIR="/vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/sample_dataset"
# DATA_MERGE_PATH="/vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/sample_dataset_src/merge.txt"
torchrun --nproc_per_node=$GPU_NUM --master-port 29512\
    scripts/data_preprocess/preprocess_vae_latents.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --num_frames 81 \
    --train_batch_size=1 \
    --max_height=480 \
    --max_width=832 \
    --dataloader_num_workers 8 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 16 \
    --video_length_tolerance_range 3 \
    --drop_short_ratio 1

# GPU_NUM=8
# MODEL_PATH="/vepfs-zulution/models/Wan2.1-T2V-14B"
# # OUTPUT_DIR="/vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/sample_dataset"
# # DATA_MERGE_PATH="/vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/sample_dataset_src/merge.txt"
# torchrun --nproc_per_node=$GPU_NUM --master-port 29513\
#     scripts/data_preprocess/preprocess_text_embeddings.py \
#     --model_type $MODEL_TYPE \
#     --model_path $MODEL_PATH \
#     --output_dir=$OUTPUT_DIR 

# torchrun --nproc_per_node=1 \
#     fastvideo/data_preprocess/preprocess_validation_text_embeddings.py \
#     --model_type $MODEL_TYPE \
#     --model_path $MODEL_PATH \
#     --output_dir=$OUTPUT_DIR \
#     --validation_prompt_txt $VALIDATION_PATH