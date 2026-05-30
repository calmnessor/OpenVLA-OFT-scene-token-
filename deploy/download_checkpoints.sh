#!/bin/bash
# ============================================================
# 下载所有预训练权重 (OpenVLA + VGGT-Omega) 和 LIBERO 数据集
# ============================================================
set -e

# ---------- 配置 ----------
# HuggingFace 缓存目录 (可改为你的数据盘路径)
export HF_HOME=${HF_HOME:-~/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-~/hf_cache}

echo "HF_HOME=$HF_HOME"
mkdir -p $HF_HOME

# ---------- 1. 下载 OpenVLA-7B 基础模型 ----------
echo "========== 下载 OpenVLA-7B 基础模型 =========="
# 方法1: 从 HuggingFace 自动下载 (训练脚本首次运行时会自动下载)
# 方法2: 手动下载
huggingface-cli download openvla/openvla-7b --local-dir ~/checkpoints/openvla-7b || true
# 如果上面的命令失败 (需要登录), 使用 Python:
python -c "
from huggingface_hub import snapshot_download
snapshot_download('openvla/openvla-7b', local_dir='$HOME/checkpoints/openvla-7b')
print('OpenVLA-7B downloaded.')
" || echo "OpenVLA-7B will be auto-downloaded on first training run."

# ---------- 2. 下载 VGGT-Omega-1B-512 权重 ----------
echo ""
echo "========== 下载 VGGT-Omega-1B-512 =========="
echo "注意: 需要先在 https://huggingface.co/facebook/VGGT-Omega 申请并获取访问权限!"
echo ""

VGGT_CKPT="$HOME/checkpoints/vggt_omega_1b_512.pt"
if [ -f "$VGGT_CKPT" ]; then
    echo "VGGT-Omega checkpoint 已存在: $VGGT_CKPT"
else
    echo "正在下载 VGGT-Omega-1B-512..."
    python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='facebook/VGGT-Omega',
    filename='vggt_omega_1b_512.pt',
    local_dir='$HOME/checkpoints',
)
print(f'VGGT-Omega downloaded to: {path}')
" || echo "请手动下载 VGGT-Omega: https://huggingface.co/facebook/VGGT-Omega"
fi

# ---------- 3. 下载 LIBERO RLDS 数据集 ----------
echo ""
echo "========== 下载 LIBERO RLDS 数据集 (~10 GB) =========="
LIBERO_DATA="$HOME/datasets/modified_libero_rlds"
if [ -d "$LIBERO_DATA" ]; then
    echo "LIBERO 数据集已存在: $LIBERO_DATA"
else
    echo "正在 clone LIBERO RLDS 数据集..."
    git clone git@hf.co:datasets/openvla/modified_libero_rlds "$LIBERO_DATA" || \
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('openvla/modified_libero_rlds', local_dir='$LIBERO_DATA', repo_type='dataset')
print('LIBERO dataset downloaded.')
" || echo "请手动下载: git clone git@hf.co:datasets/openvla/modified_libero_rlds"
fi

echo ""
echo "========== 下载完成 =========="
echo "检查点:"
echo "  OpenVLA-7B:    ~/checkpoints/openvla-7b/"
echo "  VGGT-Omega:    ~/checkpoints/vggt_omega_1b_512.pt"
echo "  LIBERO数据:    ~/datasets/modified_libero_rlds/"
