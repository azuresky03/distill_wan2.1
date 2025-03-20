PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

GPU_NUM=8
MODEL_PATH="/vepfs-zulution/wangxuekuan/code/algo_wanx_service/models/Wan2.1-I2V-14B-720P"
MODEL_TYPE="hunyuan"
DATA_MERGE_PATH="/cv/zhangpengpeng/cv/video_generation/Wan2.1/data/mixkit/merge_480.txt"
OUTPUT_DIR="/cv/zhangpengpeng/cv/video_generation/Wan2.1/data/processed/mixkit/merge_480"
VALIDATION_PATH="assets/prompt.txt"

torchrun --nproc_per_node=$GPU_NUM --master-port 29512\
    scripts/data_preprocess/preprocess_wan_dataset.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --num_frames 81 \
    --train_batch_size 1 \
    --max_height 480 \
    --max_width 832 \
    --dataloader_num_workers 8 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 16 \
    --video_length_tolerance_range 3 \
    --drop_short_ratio 1
