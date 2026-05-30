# VGGT-Omega + OpenVLA-OFT 云服务器部署完整指南

## 0. 服务器要求

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | 1x A100 80GB / H100 80GB | 2-4x A100 80GB |
| VRAM | ~53 GB (batch=1) | 80 GB (batch=2-4) |
| CPU | 16 核 | 32 核 |
| RAM | 64 GB | 128 GB |
| 磁盘 | 100 GB | 200 GB+ (含数据集) |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
| CUDA | 12.1+ | 12.4 |
| Python | 3.10 | 3.10.14 |

## 1. 上传代码到服务器

### 1.1 上传 openvla-oft (含我们修改的文件)

```bash
# 在本地 Windows 机器上，打包修改后的 openvla-oft
cd "e:/jushenzhineng/论文/3d gaussian/5月22日/5.27-5.30方案"

# 打包 (排除 .git 和 cache)
tar -czf openvla-oft-modified.tar.gz \
    --exclude='openvla-oft/.git' \
    --exclude='openvla-oft/cache' \
    openvla-oft/

# 上传到服务器
scp openvla-oft-modified.tar.gz user@your-server-ip:~
```

### 1.2 上传 vggt-omega

```bash
# 在本地
cd "e:/jushenzhineng/论文/3d gaussian/5月22日/5.27-5.30方案"
tar -czf vggt-omega.tar.gz --exclude='vggt-omega/.git' vggt-omega/
scp vggt-omega.tar.gz user@your-server-ip:~
```

### 1.3 上传部署脚本

```bash
scp deploy/*.sh user@your-server-ip:~
```

### 1.4 在服务器上解压

```bash
ssh user@your-server-ip
cd ~
tar -xzf openvla-oft-modified.tar.gz
tar -xzf vggt-omega.tar.gz

# 确认我们修改的核心文件存在
ls openvla-oft/prismatic/models/scene_projector.py          # 新增
ls openvla-oft/prismatic/extern/hf/modeling_prismatic.py    # 已修改
ls openvla-oft/vla-scripts/finetune.py                       # 已修改
ls openvla-oft/prismatic/vla/datasets/datasets.py            # 已修改
ls openvla-oft/prismatic/util/data_utils.py                  # 已修改
```

## 2. 环境搭建 (一次性)

### 2.1 完整安装步骤

> **关键**: openvla-oft 依赖 `torch==2.2.0` + `transformers==4.40.1`(fork) + `timm==0.9.10`，版本必须严格匹配

```bash
# ===== Step 1: 创建 conda 环境 =====
conda create -n vggt-openvla-oft python=3.10 -y
conda activate vggt-openvla-oft

# ===== Step 2: 安装 PyTorch 2.2.0 (CUDA 12.1) =====
# 必须用 2.2.0 版本，与 openvla-oft 兼容
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ===== Step 3: 安装 openvla-oft =====
cd ~/openvla-oft
pip install -e .

# ===== Step 4: 安装 Flash Attention 2 =====
pip install packaging ninja
ninja --version  # 确认 ninja 可用
pip install "flash-attn==2.5.5" --no-build-isolation

# ===== Step 5: 安装 vggt-omega (跳过其 torch 版本依赖) =====
cd ~/vggt-omega
# 手动安装 vggt-omega 依赖 (排除 torch/torchvision，因为它们已安装)
pip install numpy'<2' Pillow einops safetensors opencv-python
pip install -e . --no-deps

# ===== Step 6: 安装 LIBERO 评估环境 =====
cd ~
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
cd ~/openvla-oft
pip install -r experiments/robot/libero/libero_requirements.txt

# ===== Step 7: 验证安装 =====
python -c "
import torch; print(f'PyTorch: {torch.__version__}');
import timm; print(f'TIMM: {timm.__version__}');
import transformers; print(f'Transformers: {transformers.__version__}');
from prismatic.models.scene_projector import SceneProjector; print('SceneProjector: OK');
from vggt_omega.models import VGGTOmega; print('VGGTOmega: OK');
print('All imports passed!')
"
```

### 2.2 预期输出版本

```
PyTorch: 2.2.0+cu121
TIMM: 0.9.10
Transformers: 4.40.1
SceneProjector: OK
VGGTOmega: OK
All imports passed!
```

## 3. 下载预训练权重和数据集

### 3.1 下载 VGGT-Omega 权重

> **注意**: 需要先在 https://huggingface.co/facebook/VGGT-Omega 申请访问权限

```bash
# 登录 HuggingFace
huggingface-cli login

# 下载权重
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='facebook/VGGT-Omega',
    filename='vggt_omega_1b_512.pt',
    local_dir='$HOME/checkpoints',
)
print(f'Downloaded to: {path}')
"
# 预期: ~/checkpoints/vggt_omega_1b_512.pt (~4 GB)
```

### 3.2 下载 LIBERO RLDS 数据集

```bash
# 克隆数据集 (约 10 GB)
git clone git@hf.co:datasets/openvla/modified_libero_rlds ~/datasets/modified_libero_rlds

# 如果 git clone 失败，用 Python 下载:
python -c "
from huggingface_hub import snapshot_download
snapshot_download('openvla/modified_libero_rlds', local_dir='$HOME/datasets/modified_libero_rlds', repo_type='dataset')
"
```

### 3.3 OpenVLA-7B

训练脚本首次运行时会从 HuggingFace 自动下载，无需手动操作。如果需要手动下载：

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('openvla/openvla-7b', local_dir='$HOME/checkpoints/openvla-7b')
"
```

## 4. 启动训练

### 4.1 快速测试 (验证一切正常)

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vggt-openvla-oft
cd ~/openvla-oft

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir ~/datasets/modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --run_root_dir ~/runs \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 3 \
    --use_proprio True \
    --use_scene_tokens True \
    --vggt_checkpoint ~/checkpoints/vggt_omega_1b_512.pt \
    --batch_size 1 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 100 \
    --save_freq 50 \
    --save_latest_checkpoint_only True \
    --image_aug True \
    --lora_rank 32 \
    --grad_accumulation_steps 1
```

如果 100 步跑通无报错，说明环境、代码、权重、数据全部就绪。

### 4.2 正式训练 (论文配置)

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vggt-openvla-oft
cd ~/openvla-oft

# 4 GPU + 每卡 batch=8 (需要 A100 80GB × 4)
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir ~/datasets/modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --run_root_dir ~/runs \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 3 \
    --use_proprio True \
    --use_scene_tokens True \
    --vggt_checkpoint ~/checkpoints/vggt_omega_1b_512.pt \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 150005 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity "YOUR_WANDB_ENTITY" \
    --wandb_project "YOUR_WANDB_PROJECT" \
    --run_id_note "vggt_omega_scene_tokens"
```

### 4.3 单卡低显存配置

```bash
# 如果只有 1 张 A100，用 batch_size=1 + gradient_accumulation
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    ... \
    --batch_size 1 \
    --grad_accumulation_steps 8   # 等效 batch_size=8
```

### 4.4 不同数据集

```bash
# LIBERO-Spatial
--dataset_name libero_spatial_no_noops --num_images_in_input 2

# LIBERO-Object
--dataset_name libero_object_no_noops --num_images_in_input 2

# LIBERO-Goal
--dataset_name libero_goal_no_noops --num_images_in_input 2

# LIBERO-Long (10 tasks)
--dataset_name libero_10_no_noops --num_images_in_input 2
```

## 5. 训练监控

### WandB
训练指标自动上报到 WandB (如果配置了 `--wandb_entity` 和 `--wandb_project`):
- `VLA Train/Loss`: L1 回归损失
- `VLA Train/Curr Action L1 Loss`: 当前动作 L1
- `VLA Train/Next Actions L1 Loss`: 未来动作 L1
- `VLA Train/Learning Rate`: 学习率

### 本地日志
```bash
# 查看训练输出
ls ~/runs/
tail -f ~/runs/<run-id>/log.txt
```

## 6. 显存估算

| 配置 | 显存 (A100 80GB) |
|------|-----------------|
| OpenVLA-7B (LoRA, bf16) | ~25 GB |
| VGGT-Omega-1B (frozen, 3帧) | ~8 GB |
| Scene Projector | ~0.01 GB |
| 训练激活值 (batch=1) | ~20 GB |
| **总计 (batch=1)** | **~53 GB** |
| batch=2 | ~60 GB |
| batch=4 | ~73 GB |

## 7. 关键注意事项

### GPU 一致性问题
OpenVLA-OFT 论文指出：**训练和评估必须用同一型号 GPU**，否则性能可能大幅下降。如果要在不同 GPU 上评估，先在目标 GPU 上 merge LoRA 权重:

```bash
python vla-scripts/merge_lora_weights_and_save.py \
    --vla_path openvla/openvla-7b \
    --lora_adapter_path ~/runs/<run-id>/lora_adapter \
    --save_path ~/merged_model
```

### Python 版本
**必须使用 Python 3.10**。3.11+ 会导致 `timm==0.9.10` 不兼容。3.8/3.9 会导致 `transformers` fork 问题。

### 版本检查清单
- [ ] `torch==2.2.0`
- [ ] `timm==0.9.10` (TIMM 版本严格检查，0.9.10/11/12/16 四种之一)
- [ ] `transformers==4.40.1` (来自 openvla-oft fork)
- [ ] `tokenizers==0.19.1`
- [ ] `peft==0.11.1`
- [ ] `flash-attn==2.5.5`

### VGGT-Omega 授权
VGGT-Omega 权重需要 HuggingFace 授权。提前在 https://huggingface.co/facebook/VGGT-Omega 提交申请。

### LIBERO 视角数
- ALOHA 数据集: `--num_images_in_input 3` (primary + left_wrist + right_wrist)
- LIBERO 数据集: `--num_images_in_input 2` (primary + wrist)
- **注意**: 如果用 `num_images_in_input=3` 但数据集只有 2 个视角，加载会失败

## 8. 文件结构总览

服务器上的文件布局:

```
~/
├── openvla-oft/                          # 我们的修改版
│   ├── prismatic/
│   │   ├── extern/hf/modeling_prismatic.py  ← 已修改
│   │   ├── models/scene_projector.py        ← 新增
│   │   ├── vla/datasets/datasets.py         ← 已修改
│   │   └── util/data_utils.py              ← 已修改
│   └── vla-scripts/finetune.py              ← 已修改
│
├── vggt-omega/                           # VGGT-Omega 原版代码
│   └── vggt_omega/
│       └── models/vggt_omega.py
│
├── checkpoints/
│   ├── vggt_omega_1b_512.pt              # VGGT-Omega 权重 (~4 GB)
│   └── openvla-7b/                        # OpenVLA-7B (自动下载)
│
├── datasets/
│   └── modified_libero_rlds/              # LIBERO RLDS 数据集 (~10 GB)
│       ├── libero_spatial_no_noops/
│       ├── libero_object_no_noops/
│       ├── libero_goal_no_noops/
│       └── libero_10_no_noops/
│
├── runs/                                  # 训练输出
│   └── <run-id>/
│       ├── lora_adapter/                  # LoRA 权重
│       ├── scene_projector--*.pt          # Scene Projector 权重
│       └── dataset_statistics.json
│
└── deploy/                                # 部署脚本
    ├── setup_env.sh
    ├── download_checkpoints.sh
    └── launch_train.sh
```

## 9. 常见问题

### Q: 训练时 OOM 怎么办?
A: 减小 `--batch_size` 到 1，增大 `--grad_accumulation_steps` 补偿。或使用 CPU offload:
```bash
--vggt_checkpoint ~/checkpoints/vggt_omega_1b_512.pt  # VGGT 始终在 GPU
# 可考虑用 float16 替代 bfloat16 (需要修改代码)
```

### Q: VGGT-Omega 加载失败?
A: 检查 `vggt-omega/` 是否在 Python path 中。VGGTSceneExtractor 会自动将 checkpoint 目录加入 sys.path。确保 vggt-omega 已安装: `pip install -e ~/vggt-omega --no-deps`

### Q: transformers 版本冲突?
A: 必须使用 openvla-oft 的 fork: `pip install "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"`

### Q: LIBERO 数据集 "camera views" 不对?
A: 检查 `prismatic/vla/datasets/datasets.py` 中 `load_camera_views` 的设置:
- aloha 数据集: `("primary", "left_wrist", "right_wrist")`
- 非 aloha 数据集: `("primary", "wrist")`
