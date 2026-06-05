# VGGT-Omega + OpenVLA-OFT 融合框架

将 VGGT-Omega 的 3D 场景理解能力与 OpenVLA-OFT 的视觉-语言-动作策略结合，提升机器人操作的空间推理能力。

## 当前进度

### 已完成

- **架构融合**：VGGT-Omega 的 register tokens（16/视角）通过 SceneProjector（Linear + LayerNorm）投影到 LLM 嵌入空间，与 OpenVLA 的 vision patch tokens 拼接后送入 Llama-7B
- **训练流程**：VGGT-Omega 冻结，仅训练 SceneProjector（~8M）和 LoRA 权重（r=32）
- **数据集**：支持 LIBERO-Spatial/Object/Goal/10 四个任务套件（RLDS 格式）
- **label masking bug 修复**：修正 action_chunk_len 从字符串长度改为 token 数量，消除 BPE 前缀偏移导致的级联自回归误差

### 训练命令

```bash
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /PATH/TO/RLDS/DATASETS/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /YOUR/RUN/DIR/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_scene_tokens True \
  --vggt_checkpoint /PATH/TO/VGGT/model.pt \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --image_aug True \
  --lora_rank 32
```

### 待完成

- [ ] 训练至 loss 收敛并评估 LIBERO 成功率
- [ ] Cross-attention 方案：vision tokens 作为 Query，VGGT patch tokens 作为 Key/Value（替代当前简单拼接）
- [ ] RLBench 平台迁移评估

## 目录结构

```
code/                           # 代码项目
├── openvla-oft/                # OpenVLA-OFT 主代码库
├── AAAI26-SemanticVLA/         # AAAI26-SemanticVLA 改进版本
├── vggt-omega/                 # VGGT-Omega 3D 场景理解
├── dlimp_openvla/              # DLimp 数据管道库
└── deploy/                     # 部署脚本（训练启动、checkpoint 下载等）

docs/                           # 文档与方案分析
├── VGGT-Omega_OpenVLA-OFT_融合框架.md
├── SD-Pruner_SH-Fuser集成VGGT-Omega场景分析.md
├── 项目架构与微调指南.md
├── RLBench数据管道使用指南.md
└── 新服务器迁移指南.md

papers/                         # 论文资料
├── 论文/
└── 论文笔记/

config/                         # 实验环境配置
└── 实验环境/

saved_from_server/              # 从云服务器保存的模型/数据（gitignore）
```

## 关键修改文件

```
code/openvla-oft/
├── vla-scripts/finetune.py              # VGGTSceneExtractor、训练配置、主循环
├── prismatic/extern/hf/modeling_prismatic.py  # _process_scene_tokens()、forward 流程
├── prismatic/models/scene_projector.py  # SceneProjector: Linear(2048→4096) + LayerNorm
└── prismatic/vla/datasets/datasets.py   # label masking bug 修复
```

## 依赖

- [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
- [VGGT-Omega](https://github.com/facebookresearch/vggt)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
