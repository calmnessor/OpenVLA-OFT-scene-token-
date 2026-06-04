# VGGT-Ω + OpenVLA-OFT 场景Token任务感知过滤改进方案

## 1. 问题定义

### 1.1 当前方案

VGGT-Ω 推理多视图图像后，输出每帧 16 个 register token（编码全局 3D 几何信息），全部通过 SceneProjector 投影后直接拼接到 OpenVLA-OFT 的视觉序列中，送入 LLM。

```
多视图RGB → VGGT-Ω → register tokens (N×16) → SceneProjector → concat to visual → LLM
```

### 1.2 存在的问题

Register token 编码了**整个场景**的 3D 几何信息，包含大量与操作任务无关的背景几何噪音：

- 远处墙壁、天花板、窗户的几何结构
- 纹理缺失区域（大面积白墙、地板）
- 操作空间之外（>3m）的物体

这些无关信息注入 LLM 后：
- 占用序列长度（N×16 个额外的 token）
- 引入与当前任务无关的 3D 空间噪音
- LLM 需要在注意力计算中自行分辨哪些几何信息有用

### 1.3 改进目标

在 scene token 注入 LLM 之前，对其进行任务感知的过滤/加权，只保留对当前任务有用的 3D 几何特征。

---

## 2. 借鉴 VGGT-Det (CVPR 2026)

### 2.1 VGGT-Det 解决的是什么问题

VGGT-Det 提出了 **Sensor-Geometry-Free** 的室内多视图 3D 目标检测方法。传统方法（如 NeRF-Det）需要精确的相机内参/外参做 3D 特征投影，VGGT-Det 用冻结的 VGGT-1B 预训练模型替代了整个几何 pipeline。

### 2.2 借鉴的核心机制

#### 机制一：Attention Map 空间门控

**在 VGGT-Det 中的作用：**

VGGT-Det 从 VGGT-1B 的 aggregator 内部提取 attention map，利用它作为空间重要性分布，指导 3D 点云采样（Attention-Guided Query Generation）。具体操作：

1. 捕获 aggregator global attention 层的注意力权重矩阵
2. 对所有 head 和 query 取平均，得到每个 patch 的"被关注度"
3. 结合 depth 掩码过滤远处无效区域
4. 归一化后作为采样概率，top-k 取高注意力区域
5. 这些区域的 3D 点成为检测 query 锚点

核心代码路径：[VGGT-Det aggregator.py:261-312] 和 [vggtdet.py:405-459]

**关键发现：VGGT 的 attention 天然聚焦于有几何结构的物体区域，而不是平坦的墙面或远处背景。**

#### 机制二：Task Query 条件化

**在 VGGT-Det 中的作用：**

VGGT-Det 引入可学习的 `task_query` embedding，作为 Transformer Decoder 的额外输入 token，与多视图 VGGT 特征做 cross-attention，实现任务感知的特征聚合。

```python
if if_task_query:
    self.task_query = nn.Parameter(torch.Tensor(1, token_dim))
    expanded_task_query = self.task_query.unsqueeze(1).expand(-1, batch_size, -1)
    tgt = torch.cat([tgt, expanded_task_query], dim=0)  # 拼入 decoder 输入
```

虽然 VGGT-Det 的 task_query 是可学习的静态 embedding（不同任务共享），但框架天然支持替换为任务相关的条件信号。

#### 机制三：Depth 掩码过滤

**在 VGGT-Det 中的作用：**

用 VGGT 预测的深度图做有效性过滤——深度超过 1000m 的像素被标记为无效，其 attention 采样概率置零。这防止了远处/无效区域的 3D 点被选为 query。

```python
depth_mask = depth_map > 1000
norm_attn_img_up[depth_mask] = 0.0  # 无效区域概率清零
```

### 2.3 三个机制的关系

```
VGGT Attention Map ─→ 空间重要性分布 (哪里可能有物体)
        ×
VGGT Depth Map ──────→ 深度有效性约束  (哪里是有效3D空间)
        ×
Task Query ──────────→ 任务条件化      (当前任务关心什么)
        =
任务感知的3D特征选择
```

VGGT-Det 用了前两个（attention + depth），第三个用了静态可学习 query。我们的方案将三个机制整合，并把 task query 升级为真正的语言条件化。

---

## 3. 迁移方案：Task-Conditioned Spatial Gating

### 3.1 设计原则

| VGGT-Det 机制 | 迁移到 VGGT-Ω + VLA 的方式 | 变更 |
|--------------|--------------------------|------|
| Attention Map → 空间重要性 | Depth + DepthConf → 空间几何质量 | 用 VGGT-Ω 已有的 depth 预测替代 attention map 提取，避免改 aggregator |
| Depth Mask → 深度约束 | depth < D_max 过滤远处 | 直接把 depth 阈值适配到操作场景（3m 而非 1000m） |
| Task Query → 任务条件化 | VLA language instruction embedding 作为条件 | 从静态 embedding 升级为语言驱动的动态条件 |

### 3.2 不直接用 Attention Map 的原因

1. VGGT-Ω 使用 `F.scaled_dot_product_attention`，不保留中间 attention 矩阵，需要改底层代码
2. VGGT-Ω 的 depth/depth_conf 是已有输出，零额外计算开销
3. depth_conf 在语义上与 attention map 高度相关：无纹理区域 depth_conf 低，同时也不会被 attention 关注

### 3.3 整体架构

```
                    多视图 RGB 图像 (N 帧)
                         │
                         ▼
              ┌─────────────────────┐
              │   VGGT-Ω (frozen)   │
              │                     │
              │  Aggregator         │
              │  DenseHead → depth  │
              │            → depth_conf │
              └─────────────────────┘
                 │              │
                 ▼              ▼
        scene_tokens       depth, depth_conf
        (B, N×16, 2048)    (B, N, H, W)
                 │              │
                 │              ├──────────────────┐
                 │              │                  │
                 │              ▼                  ▼
                 │    ┌────────────────┐  ┌──────────────┐
                 │    │ Spatial Stats  │  │  VLA LLM     │
                 │    │ 逐帧计算:       │  │  Embedding   │
                 │    │ mean_conf      │  │              │
                 │    │ std_conf       │  │  instruction │
                 │    │ valid_ratio    │  │  → lang_feat │
                 │    │ mean_depth     │  │  (B, 4096)   │
                 │    └────────┬───────┘  └──────┬───────┘
                 │             │                 │
                 │             ▼                 ▼
                 │    ┌─────────────────────────────────┐
                 │    │  gate_net (~50K params)         │
                 │    │                                 │
                 │    │  spatial_feat = encoder(stats)  │
                 │    │  lang_feat    = proj(lang)      │
                 │    │  gate         = fusion(concat)  │
                 │    │                 → (B, N) ∈[0,1] │
                 │    └────────────────┬────────────────┘
                 │                     │
                 │                     ▼
                 │            gate_weights expand
                 │            → (B, N×16, 1)
                 │                     │
                 └─────────────────────┤
                                       ▼
                            scene_tokens × gate_weights
                                       │
                                       ▼
                              SceneProjector
                              (Linear + LayerNorm)
                                       │
                                       ▼
                        gated_scene_embeddings (B, N×16, 4096)
                                       │
                                       ▼
                        concat([visual_patches, proprio,
                                gated_scene, text_tokens])
                                       │
                                       ▼
                              LLM → Action Head
                                       │
                                       ▼
                                  物理动作
```

---

## 4. 详细实现

### 4.1 新增文件：`prismatic/models/spatial_gate.py`

```python
"""
spatial_gate.py — Task-Conditioned Spatial Gating Module

借鉴 VGGT-Det (CVPR 2026) 的 Attention-Guided Query Generation 思想：
  - VGGT-Det: attention_map × depth_mask → 3D点采样概率
  - 本模块: depth_conf × depth_valid × lang_feat → scene token 帧级门控

将 VGGT-Det 中静态的 task_query 升级为 VLA language instruction 驱动的动态条件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskConditionedSpatialGate(nn.Module):
    """
    任务条件化空间门控模块。

    对 VGGT-Ω 每帧的 scene token 进行加权：
      - depth_conf 高 + depth 在操作范围内 → 高质量几何 → 高权重
      - depth_conf 低 + depth 过远       → 背景噪音   → 低权重
      - 权重同时受任务语言特征调制，不同任务关注不同的空间区域

    Args:
        scene_dim:    VGGT-Ω scene token 维度 (2048)
        lang_dim:     VLA LLM 隐藏层维度 (4096)
        gate_hidden:  门控网络隐藏层维度
        depth_max:    有效深度上限（米），超过视为背景
    """

    def __init__(
        self,
        scene_dim: int = 2048,
        lang_dim: int = 4096,
        gate_hidden: int = 256,
        depth_max: float = 3.0,
    ):
        super().__init__()
        self.depth_max = depth_max

        # 空间统计编码器：4维统计量 → 隐藏特征
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, gate_hidden),
            nn.GELU(),
        )

        # 语言特征投影：4096 → 隐藏特征
        self.lang_proj = nn.Sequential(
            nn.Linear(lang_dim, gate_hidden),
            nn.GELU(),
        )

        # 融合层：空间特征 + 语言特征 → 门控值
        self.fusion = nn.Sequential(
            nn.Linear(gate_hidden * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, gate_hidden // 2),
            nn.GELU(),
            nn.Linear(gate_hidden // 2, 1),
            nn.Sigmoid(),  # 输出 ∈ [0, 1]
        )

    def _compute_spatial_stats(
        self, depth: torch.Tensor, depth_conf: torch.Tensor
    ) -> torch.Tensor:
        """
        从 depth/depth_conf 逐帧计算空间几何质量统计量。

        Args:
            depth:      (B, N_frames, H, W)  预测深度（米）
            depth_conf: (B, N_frames, H, W)  深度置信度
        Returns:
            stats: (B, N_frames, 4)
                   [mean_conf, std_conf, valid_ratio, mean_valid_depth]
        """
        B, N, H, W = depth.shape
        depth_flat = depth.view(B, N, -1)
        conf_flat = depth_conf.view(B, N, -1)

        # 有效深度区域：在操作空间范围内且非零
        valid_mask = (depth_flat < self.depth_max) & (depth_flat > 1e-3)
        valid_count = valid_mask.float().sum(dim=-1) + 1e-8

        # 统计量1: 有效区域平均置信度
        mean_conf = (conf_flat * valid_mask.float()).sum(dim=-1) / valid_count

        # 统计量2: 置信度标准差（纹理多样性）
        # 只在有效区域计算
        conf_centered = (conf_flat - mean_conf.unsqueeze(-1)) * valid_mask.float()
        std_conf = torch.sqrt((conf_centered ** 2).sum(dim=-1) / valid_count + 1e-8)

        # 统计量3: 有效深度区域占比
        valid_ratio = valid_count / (H * W)

        # 统计量4: 有效区域平均深度（反映物体距离）
        mean_valid_depth = (depth_flat * valid_mask.float()).sum(dim=-1) / valid_count

        stats = torch.stack([mean_conf, std_conf, valid_ratio, mean_valid_depth], dim=-1)
        return stats

    def forward(
        self,
        scene_tokens: torch.Tensor,  # (B, N_frames*16, 2048)
        depth: torch.Tensor,          # (B, N_frames, H, W)
        depth_conf: torch.Tensor,     # (B, N_frames, H, W)
        lang_feat: torch.Tensor,      # (B, lang_dim)
    ) -> torch.Tensor:
        """
        Args:
            scene_tokens: VGGT-Ω register tokens，已展开为 (B, N_frames*16, 2048)
            depth:        预测深度图
            depth_conf:   深度置信度
            lang_feat:    来自 LLM embedding 层的 instruction 均值池化特征
        Returns:
            gated_tokens: (B, N_frames*16, 2048) 加权后的 scene tokens
        """
        B, total_tokens, C = scene_tokens.shape
        num_frames = depth.shape[1]
        num_registers = total_tokens // num_frames  # 16

        # Step 1: 逐帧空间统计
        spatial_stats = self._compute_spatial_stats(depth, depth_conf)  # (B, N, 4)

        # Step 2: 空间编码
        spatial_feat = self.spatial_encoder(spatial_stats)  # (B, N, gate_hidden)

        # Step 3: 语言编码
        lang_feat_proj = self.lang_proj(lang_feat)  # (B, gate_hidden)
        lang_feat_proj = lang_feat_proj.unsqueeze(1).expand(-1, num_frames, -1)

        # Step 4: 融合 → 门控权重
        fused = torch.cat([spatial_feat, lang_feat_proj], dim=-1)  # (B, N, 2*gate_hidden)
        gate_weights = self.fusion(fused).squeeze(-1)  # (B, N) ∈ [0,1]

        # Step 5: 权重扩展到每个 register token
        gate_weights = gate_weights.unsqueeze(-1).expand(-1, -1, num_registers)
        gate_weights = gate_weights.reshape(B, total_tokens).unsqueeze(-1)

        # Step 6: 软门控（乘性，非硬选择，保证梯度平滑）
        gated_tokens = scene_tokens * gate_weights

        return gated_tokens
```

### 4.2 修改：`prismatic/models/scene_projector.py`

```python
class SceneProjector(nn.Module):
    def __init__(self, scene_dim=2048, llm_dim=4096, use_spatial_gate=True):
        super().__init__()
        self.projector = nn.Sequential(OrderedDict([
            ("scene_linear", nn.Linear(scene_dim, llm_dim, bias=True)),
            ("scene_norm", nn.LayerNorm(llm_dim)),
        ]))
        if use_spatial_gate:
            self.spatial_gate = TaskConditionedSpatialGate(
                scene_dim=scene_dim,
                lang_dim=llm_dim,
            )

    def forward(
        self,
        scene_tokens: torch.Tensor,   # (B, N*16, 2048)
        depth: torch.Tensor = None,    # (B, N, H, W)
        depth_conf: torch.Tensor = None,  # (B, N, H, W)
        lang_feat: torch.Tensor = None,   # (B, llm_dim)
    ) -> torch.Tensor:
        if hasattr(self, 'spatial_gate') and depth is not None:
            scene_tokens = self.spatial_gate(
                scene_tokens, depth, depth_conf, lang_feat
            )
        return self.projector(scene_tokens)
```

### 4.3 修改：`modeling_prismatic.py` 中的 `predict_action()`

在 `_process_scene_tokens` 调用处传入 depth 和 lang_feat：

```python
# ========== 原有代码 ==========
# Step 4: 视觉编码后的语言嵌入提取
language_embeddings = input_embeddings[~all_actions_mask].reshape(
    input_embeddings.shape[0], -1, input_embeddings.shape[2]
)

# [新增] 取 language_embeddings 的均值作为任务表示
lang_feat = language_embeddings.mean(dim=1)  # (B, llm_dim)

# ========== 原有代码 ==========
# Step 6: 拼接 VGGT-Omega 场景 token
if scene_tokens is not None:
    projected_patch_embeddings = self._process_scene_tokens(
        projected_patch_embeddings, scene_tokens,
        depth=vggt_outputs.get("depth"),           # [新增]
        depth_conf=vggt_outputs.get("depth_conf"),  # [新增]
        lang_feat=lang_feat,                        # [新增]
    )
```

`_process_scene_tokens` 方法同步修改：

```python
def _process_scene_tokens(
    self, projected_patch_embeddings, scene_tokens,
    depth=None, depth_conf=None, lang_feat=None,
):
    if scene_tokens is not None:
        scene_embeddings = self.scene_projector(
            scene_tokens,
            depth=depth,
            depth_conf=depth_conf,
            lang_feat=lang_feat,
        )
        return torch.cat([projected_patch_embeddings, scene_embeddings], dim=1)
    return projected_patch_embeddings
```

### 4.4 训练配置

| 组件 | 状态 | 参数量 |
|------|------|--------|
| VGGT-Omega | frozen | ~1B |
| gate_net | **trainable** | ~50K |
| SceneProjector | **trainable** | ~8M |
| LLM (LoRA, rank=32) | **trainable** | ~40M |
| Action Head | **trainable** | ~2M |

训练数据集：与 OpenVLA-OFT fine-tuning 相同（RLBench / LIBERO 等多视图操作数据集）。

损失函数：与现有 OpenVLA-OFT 一致（L1 回归或 Diffusion loss），gate_net 无额外辅助 loss。

梯度流：

```
Action Loss (L1)
    → LLM output hidden states
        → multimodal_embeddings 中的 gated_scene_embeddings
            → SceneProjector(gate_weights × scene_tokens)
                → gate_weights = gate_net(depth_stats, lang_feat)
                    → gate_net 参数  ← 正常反向传播
```

软门控保证梯度处处可导：`∂L/∂w_gate = ∂L/∂scene_embed · scene_token_value`

---

## 5. VGGT-Det → 本方案的迁移映射表

| 概念 | VGGT-Det (CVPR 2026) | 本方案 |
|------|----------------------|--------|
| **3D 几何来源** | VGGT-1B (aggregator tokens) | VGGT-Ω (register tokens) |
| **空间重要性** | Attention map → patch 被关注度 | Depth × DepthConf → 几何质量 |
| **空间过滤** | depth_mask > 1000m | depth > 3m (操作空间) |
| **任务条件** | 可学习静态 task_query | LLM instruction embedding |
| **采样/加权方式** | top-k 硬采样 3D 点 | sigmoid 软门控 token 权重 |
| **下游任务** | 3D 目标检测 (Transformer Decoder) | 机器人动作预测 (LLM + Action Head) |
| **训练方式** | 端到端（VGGT frozen） | 端到端（VGGT-Ω frozen） |

---

## 6. 消融实验设计建议

| 变体 | gate_net | lang_cond | 说明 |
|------|----------|-----------|------|
| Baseline | 无 | 无 | 原始方案，所有 scene token 等权注入 |
| Heuristic Gate | 启发式 | 无 | `gate = valid_ratio × mean_conf` |
| Learned Gate | 可学习 | 无 | gate_net 只输入 depth_stats |
| Task-Conditioned Gate | 可学习 | 有 | **完整方案** |
| Oracle Gate | GT 物体 mask | — | 上限参考：用 GT 分割标注做门控 |
