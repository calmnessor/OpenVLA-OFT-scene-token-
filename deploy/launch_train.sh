#!/bin/bash
# ============================================================
# VGGT-Omega + OpenVLA-OFT 融合训练 — 训练启动脚本
# ============================================================
set -e

# ========== 配置参数 (按需修改) ==========
GPU_COUNT=1                        # GPU 数量 (用于 torchrun --nproc-per-node)
VLA_PATH="openvla/openvla-7b"      # 或本地路径 ~/checkpoints/openvla-7b
DATA_DIR="$HOME/datasets/modified_libero_rlds"
RUN_DIR="$HOME/runs"
VGGT_CKPT="$HOME/checkpoints/vggt_omega_1b_512.pt"

# 数据集选择: libero_spatial_no_noops | libero_object_no_noops | libero_goal_no_noops | libero_10_no_noops
DATASET="libero_spatial_no_noops"

# 训练超参
BATCH_SIZE=1                       # batch size per GPU (A100 80GB: 用2-4; 小显存: 用1)
LEARNING_RATE=5e-4
MAX_STEPS=5000                     # 论文用 150005 (完整), 测试用 5000
LR_DECAY_STEPS=100000
SAVE_FREQ=1000                     # 论文用 10000
NUM_IMAGES=3                       # 3 视角: primary + left_wrist + right_wrist
USE_PROPRIO=True

# WandB (可选)
WANDB_ENTITY="your-entity"
WANDB_PROJECT="vggt-openvla-oft"

# ========== 环境激活 ==========
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vggt-openvla-oft

cd ~/openvla-oft

# ========== 启动训练 ==========
echo "========== VGGT-Omega + OpenVLA-OFT 融合训练 =========="
echo "GPU数量:     $GPU_COUNT"
echo "数据集:      $DATASET"
echo "图像视角数:  $NUM_IMAGES"
echo "VGGT权重:    $VGGT_CKPT"
echo "Batch size:  $BATCH_SIZE"
echo "Max steps:   $MAX_STEPS"
echo "========================================================"

torchrun --standalone --nnodes 1 --nproc-per-node $GPU_COUNT vla-scripts/finetune.py \
    --vla_path "$VLA_PATH" \
    --data_root_dir "$DATA_DIR" \
    --dataset_name "$DATASET" \
    --run_root_dir "$RUN_DIR" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input $NUM_IMAGES \
    --use_proprio $USE_PROPRIO \
    --use_scene_tokens True \
    --vggt_checkpoint "$VGGT_CKPT" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_steps_before_decay $LR_DECAY_STEPS \
    --max_steps $MAX_STEPS \
    --save_freq $SAVE_FREQ \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --grad_accumulation_steps 1 \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --run_id_note "vggt_omega_scene_tokens--L1_regression--continuous_acts"

echo "========== 训练结束 =========="
