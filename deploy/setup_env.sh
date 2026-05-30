#!/bin/bash
# ============================================================
# VGGT-Omega + OpenVLA-OFT 融合训练 — 云服务器环境搭建脚本
# 适用: Ubuntu 22.04, NVIDIA A100/H100, CUDA 12.1+
# ============================================================
set -e

echo "========== Step 1: 创建 Conda 环境 =========="
conda create -n vggt-openvla-oft python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vggt-openvla-oft

echo "========== Step 2: 安装 PyTorch (2.3.1, CUDA 12.1) =========="
# PyTorch 2.3 兼容 openvla-oft 和 vggt-omega 两个项目
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

echo "========== Step 3: 安装 openvla-oft =========="
cd ~/openvla-oft
pip install -e .

echo "========== Step 4: 安装 Flash Attention 2 =========="
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

echo "========== Step 5: 安装 vggt-omega =========="
cd ~/vggt-omega
pip install -r requirements.txt
pip install -e .

echo "========== Step 6: 安装 LIBERO 评估环境 =========="
cd ~
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r ~/openvla-oft/experiments/robot/libero/libero_requirements.txt

echo "========== 环境搭建完成! =========="
echo "下一步: 运行 download_checkpoints.sh 下载预训练权重"
