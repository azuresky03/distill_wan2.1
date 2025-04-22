#!/bin/bash

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 获取可用的GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# 设置DDP相关的环境变量
export MASTER_ADDR='localhost'
export MASTER_PORT='12355'

# 训练参数
BASE_MODEL_PATH="/vepfs-zulution/models/Wan2.1-T2V-14B"
OUTPUT_DIR="./output/gan_training"
BATCH_SIZE=1
NUM_WORKERS=4
EPOCHS=50
SAVE_FREQ=1

H=720
W=1280

# 模型相关参数
NUM_LAYERS=1
LAYER_IDXS="9 19 29 39"
DESIRED_LEN=16
COMPRESSOR_HEADS=8

# 优化器参数
LR=1e-5
LR_EXTRACTOR=1e-5
WEIGHT_DECAY=0.01

# 使用torchrun启动分布式训练（local_rank会自动传入）
torchrun --nproc_per_node=$NUM_GPUS \
    distill_exp/train/train_gan.py \
    --resume "/cv/zhangpengpeng/cv/video_generation/Wan2.1/output/gan_training/checkpoints/best.pth" \
    --debug 20 \
    --h $H \
    --w $W \
    --distilled_model_path "/cv/wangxuekuan/release_model/wanx/distill_cfg_t2v/exp10_distill_cfg/checkpoint-500" \
    --json_path "/cv/zhangpengpeng/cv/video_generation/Wan2.1/data/processed/mixkit/video/wan_exp10_cfg_480/video_meta.json" \
    --base_model_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --save_freq $SAVE_FREQ \
    --num_layers $NUM_LAYERS \
    --layer_idxs $LAYER_IDXS \
    --desired_len $DESIRED_LEN \
    --compressor_heads $COMPRESSOR_HEADS \
    --lr $LR \
    --lr_extractor $LR_EXTRACTOR \
    --weight_decay $WEIGHT_DECAY