# Q-Former 场景 Token 任务感知调制：创新点与模型架构

## 1. 问题定位

### 1.1 当前 VGGT-Ω + OpenVLA-OFT 的缺陷

```
多视图RGB → VGGT-Ω (frozen) → register tokens [N×16, 2048]
                                    ↓
                            SceneProjector (Linear)
                                    ↓
                            [B, 48, 4096] concat → LLM
```

**三个缺陷**：

1. **无差别注入**：48 个 register token 全部等权拼入 LLM，背景几何噪音（墙壁、天花板、远处物体）与任务相关 3D 信息无区分
2. **无任务条件化**：所有任务获得完全相同的 3D 先验，"开门"和"倒水"关注的 3D 空间区域不同，但模型不知道
3. **无跨帧几何合成**：registers 以分布式编码方式承载 3D 信息——单个 register 不包含完整的几何概念（"物体形状"可能分散在多个 register 中），直接 concat 无法合成分布式几何碎片

### 1.2 已有方法的覆盖范围

| 方法 | 机制 | 是否压缩 | 是否语言条件化 | 是否跨帧合成 | 局限 |
|------|------|----------|--------------|------------|------|
| Evo-0 | visual Q → CA → VGGT | 否 | 否 | 否 | 单向检索，无 query 间协作 |
| 3D-Mix CrossAttn | MLLM Q → CA → VGGT | 否 | 间接 | 否 | 同上，且 performance 差 (56%) |
| 3D-Mix GatedFusion | Post-LLM 门控融合 | 否 | 间接 | 否 | LLM 感知不到几何 |
| Compressor-VLA | 可学习 Q + FiLM → CA → **2D visual** | 是 | 是 | 部分 | 压缩对象是 2D visual，非 3D 几何 |
| 空间门控方案 | depth stats + lang → frame gate | 否 | 部分 | 否 | 帧级粒度太粗 |

**未覆盖的空白**：对 VGGT-Ω 的 **分布式 3D 几何编码** 做**语言条件化的协同选择与合成**，并**压缩为紧凑的几何-语义 token** 注入 LLM。

---

## 2. 模型架构

### 2.1 总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                        完整模型架构                                   │
│                                                                      │
│   多视图 RGB (N×512×512)             多视图 RGB (N×224×224)           │
│         │                                     │                      │
│         ▼                                     ▼                      │
│  ┌──────────────┐                  ┌────────────────────┐            │
│  │  VGGT-Ω      │                  │ SigLIP + DINOv2    │            │
│  │  (frozen)    │                  │ Vision Backbone    │            │
│  │              │                  │ (frozen/ FiLM可选) │            │
│  │  48 token    │                  │ 768 token          │            │
│  │  [B,48,2048] │                  │ [B,768,D_vis]      │            │
│  └──────┬───────┘                  └─────────┬──────────┘            │
│         │                                    │                       │
│         │                                    ▼                       │
│         │                          ┌────────────────────┐            │
│         │                          │ Visual Projector   │            │
│         │                          │ (D_vis → 4096)     │            │
│         │                          │ [B,768,4096]       │            │
│         │                          └─────────┬──────────┘            │
│         │                                    │                       │
│         ▼                                    │                       │
│  ┌──────────────────────────────────────────┐│                       │
│  │       Q-Former Scene Token Modulator     ││                       │
│  │      (~25M, trainable)                   ││                       │
│  │                                           ││                       │
│  │  ┌─────────────────────────────────────┐  ││                       │
│  │  │ Scene Projection                    │  ││                       │
│  │  │ Linear(2048→4096) + LayerNorm       │  ││                       │
│  │  │ → scene_features [B, 48, 4096]      │  ││                       │
│  │  └─────────────────────────────────────┘  ││                       │
│  │                    │                      ││                       │
│  │  ┌────────────────┴────────────────────┐  ││                       │
│  │  │ Query Initialization                │  ││                       │
│  │  │                                     │  ││                       │
│  │  │ learnable_query [K, 4096]           │  ││                       │
│  │  │    + lang_bias(lang_feat)           │  ││                       │
│  │  │ → conditioned_queries [B, K, 4096]  │  ││                       │
│  │  └────────────────┬────────────────────┘  ││                       │
│  │                   │                       ││                       │
│  │                   ▼                       ││                       │
│  │  ┌─────────────────────────────────────┐  ││                       │
│  │  │ × L 层 Q-Former Block               │  ││                       │
│  │  │                                     │  ││                       │
│  │  │  +--------+   +--------+   +-----+  │  ││                       │
│  │  │  | Self-  |   | Cross- |   | FFN |  │  ││                       │
│  │  │  | Attn   | → | Attn   | → |     |  │  ││                       │
│  │  │  +--------+   +--------+   +-----+  │  ││                       │
│  │  │  queries间    Q=queries    几何→语义  │  ││                       │
│  │  │  "协同合成"   "分布式检索"   "翻译"   │  ││                       │
│  │  │                                     │  ││                       │
│  │  │  每步: 残差连接 + LayerNorm          │  ││                       │
│  │  └────────────────┬────────────────────┘  ││                       │
│  │                   │                       ││                       │
│  │                   ▼                       ││                       │
│  │  ┌─────────────────────────────────────┐  ││                       │
│  │  │ FiLM Modulation                     │  ││                       │
│  │  │ γ, β = lang_proj(lang_feat)         │  ││                       │
│  │  │ [B, 4096] each, bottleneck: 256d    │  ││                       │
│  │  │ x = (1+γ)⊙x + β                     │  ││                       │
│  │  │ "语言条件化特征变换"                 │  ││                       │
│  │  └────────────────┬────────────────────┘  ││                       │
│  │                   │                       ││                       │
│  │                   ▼                       ││                       │
│  │  ┌─────────────────────────────────────┐  ││                       │
│  │  │ Modality Embedding                  │  ││                       │
│  │  │ vggt_modal [1, K, 4096] (learnable) │  ││                       │
│  │  │ x += vggt_modal                     │  ││                       │
│  │  │ "标记来源 = VGGT 几何-语义"          │  ││                       │
│  │  └────────────────┬────────────────────┘  ││                       │
│  └───────────────────┼───────────────────────┘│                       │
│                      │                        │                       │
│                      ▼                        ▼                       │
│              ┌──────────────────────────────────────┐                 │
│              │         Multi-Modal Sequence          │                 │
│              │                                      │                 │
│              │  [BOS] [vision:768] [prop:1]         │                 │
│              │        [VGGT_geo-sem:K] [text]       │                 │
│              │                                      │                 │
│              │  VGGT token 位置: vision 之后         │                 │
│              │  所有 text token 双向 attend 到 VGGT   │                 │
│              └──────────────────┬───────────────────┘                 │
│                                 │                                     │
│                                 ▼                                     │
│              ┌──────────────────────────────────────┐                 │
│              │     Llama-7B (LoRA, rank=32)         │                 │
│              │                                      │                 │
│              │  Geometry-Aware Causal Reasoning     │                 │
│              └──────────────────┬───────────────────┘                 │
│                                 │                                     │
│                                 ▼                                     │
│              ┌──────────────────────────────────────┐                 │
│              │     Action Head (L1 Regression)      │                 │
│              │     → 连续动作序列                     │                 │
│              └──────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Token 数量变化

| 阶段 | Token 数量 | 维度 |
|------|-----------|------|
| VGGT-Ω 原始输出 | N×16 (默认 48) | 2048 |
| Scene Projection | 48 | 4096 |
| Q-Former 压缩后 | **K (推荐 8)** | 4096 |
| FiLM 调制后 | K | 4096 |
| + Modality Embedding | K | 4096 |
| 注入 LLM | **K** | 4096 |

LLM 总 visual token: 768 + 1 (proprio) + K = **777**（相比原来的 817，减少 5%）

---

## 3. 核心创新：从分布式几何编码到协同语义合成

### 3.1 VGGT Register Token 的本质

VGGT-Ω 的 register tokens 是**分布式 3D 几何编码**——单个 register 不包含完整的 3D 概念：

```
register token 通过 24 层交替 attention 聚合 3D 信息:

  register_0  ← 主要编码参考坐标系/相机位姿
  register_3  ← 富含深度不连续区域的边界信息
  register_7  ← 倾向于编码表面法向和形状
  register_12 ← 编码遮挡关系和多视图一致性
  ...

"物体形状" = register_3(边界) + register_7(法向) + register_12(遮挡)
            ↑ 分散在多个 register 中，需要协同合成

"抓取点"   = register_5(位置) + register_8(高度) + register_3(形状边界)
            ↑ 不同任务需要不同的 register 子集和合成方式
```

### 3.2 已有方法的局限：单向检索，无法合成

```
Evo-0 / 3D-Mix CrossAttn / Compressor-VLA:
  visual query → CA → VGGT features → 每个 query 独立检索
  
  问题: 每个 query 只能"捡"到零散的几何碎片
       query_0 发现深度边界 → 不知道法向在哪
       query_1 发现表面法向 → 不知道遮挡关系在哪
       → 无法将碎片合成为完整的 3D 理解
```

### 3.3 本方案的核心贡献：协同几何合成

```
Q-Former Self-Attention:
  
  query_0 ←→ query_1 ←→ query_2 ←→ ... ←→ query_K
  
  每层 Self-Attention 中，K 个 query 互相"通报"各自发现：
  
  query_0: "我在关注 register_3，发现了深度边界"
  query_1: "我发现了 register_7 的表面法向，应该和你的边界有关"
  query_2:  "那我换个方向，去关注 register_12 的遮挡关系"
  
  经过 L 层迭代协调:
  → K 个 query 形成互补分工
  → 每个 query 覆盖一个不同的几何子概念
  → K 个 query 的集合覆盖了"完整任务相关 3D 场景"
```

### 3.4 与已有方法的本质区别

| | 已有方法 | 本方案 |
|---|---|---|
| **几何信息处理** | 单向检索（query→token） | **协同合成**（query↔query + query→token） |
| **Query 间关系** | 独立工作 | **Self-Attention 协调分工** |
| **输出关系** | 各自独立的结果 | **互补合成的完整 3D 理解** |
| **信息保真度** | 碎片化（每个 query 只有部分信息） | **结构化**（K 个 query 覆盖不同子概念） |

### 3.5 这个创新在哪个层次

```
┌─────────────────────────────────────────────────────────────┐
│  范式创新: 无                                                │
│  架构创新: 无（Q-Former 本身已有）                            │
│                                                            │
│  ★ 概念创新:                                                │
│    首次识别 VGGT register 的分布式编码性质，                   │
│    并提出 Self-Attention 协同合成来解决已有方法的碎片化检索问题 │
│                                                            │
│  ★ 场景创新:                                                │
│    首次将 Q-Former 完整架构应用于 VGGT 3D 几何 token 的任务感知 │
│    调制，填补了 3D-Mix / Evo-0 / Compressor-VLA 之间的空白    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Q-Former Block 详细结构

```
┌─────────────────────────────────────────────────────────┐
│              Q-Former Block × L (L=2)                    │
│                                                         │
│  输入: queries [B, K, 4096], scene_features [B, 48, 4096]│
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ① Self-Attention                                │   │
│  │                                                 │   │
│  │   Q = K = V = LayerNorm(queries)                │   │
│  │   MultiHeadAttention(Q, K, V)                   │   │
│  │   num_heads = 8                                │   │
│  │                                                 │   │
│  │   作用: query 间互相通报各自的几何发现，         │   │
│  │        协调分工，避免信息重复，形成互补覆盖        │   │
│  │        同时实现跨帧 3D 信息对齐融合               │   │
│  │                                                 │   │
│  │   queries = queries + SelfAttn(norm(queries))    │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ② Cross-Attention                               │   │
│  │                                                 │   │
│  │   Q = LayerNorm(queries)  [B, K, 4096]          │   │
│  │   K = LayerNorm(scene)    [B, 48, 4096]          │   │
│  │   V = LayerNorm(scene)    [B, 48, 4096]          │   │
│  │   MultiHeadAttention(Q, K, V)                   │   │
│  │                                                 │   │
│  │   作用: 按 Self-Attn 协调后的分工，               │   │
│  │        从 48 个 scene token 中各自检索不同几何信息  │   │
│  │        形成互补分布式选择                          │   │
│  │                                                 │   │
│  │   queries = queries + CrossAttn(norm(queries),   │   │
│  │                                   scene)          │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ③ Feed-Forward Network                          │   │
│  │                                                 │   │
│  │   LayerNorm → Linear(4096→16384) → GELU         │   │
│  │             → Linear(16384→4096)                 │   │
│  │                                                 │   │
│  │   作用: 将检索到的原始几何特征                    │   │
│  │         "翻译"为 LLM 更容易理解的语义化表示        │   │
│  │         (VGGT 的几何 "语言" → LLM 的语义 "语言")   │   │
│  │                                                 │   │
│  │   queries = queries + FFN(norm(queries))        │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  输出: queries [B, K, 4096]                              │
└─────────────────────────────────────────────────────────┘
```

### 4.1 Self-Attention 为什么在这个场景必要

在 BLIP-2 的原始 Q-Former 中，Self-Attention 的作用是 query 间信息交互。但这个作用在 VGGT register token 场景下变得更加关键：

| | BLIP-2 (257 patch tokens) | 本方案 (48 register tokens) |
|---|---|---|
| Token 信息类型 | 空间相邻的 patch 特征 | **分布式 3D 几何编码** |
| 单个 token 的信息完整性 | 相对完整（局部感受野） | **不完整**（几何概念分散） |
| Self-Attn 的价值 | 信息融合/去冗余 | **协同合成**——合成碎片化的几何概念 |

因为 VGGT register 的编码是分布式的（"物体形状"分散在多个 register 中），单个 query 的 cross-attention 检索只能得到碎片。Self-Attention 让 K 个 query 协同工作，各自负责不同的几何子概念，形成完整覆盖。

---

## 5. FiLM 调制与层级语言条件化

### 5.1 设计动机

Q-Former 和 FiLM 的语言注入形成**层级互补**而非冗余：

```
┌─────────────────────────────────────────────────────────────┐
│                 层级语言条件化                                │
│                                                             │
│  语言指令: "open the top drawer"                             │
│                                                             │
│  层级1: Q-Former Query Bias                                 │
│    "top" → 偏置 query 关注场景上半部分的几何                   │
│    "drawer" → 偏置 query 关注抽屉类物体的 3D 结构              │
│    作用: 空间选择 — "看哪里" (spatial/token selection)        │
│                                                             │
│  层级2: FiLM Channel Modulation                             │
│    "open" → 强化深度通道（需要判断拉出的距离）                 │
│    "drawer" → 强化法向通道（抽屉是平面结构）                   │
│    作用: 特征变换 — "怎么看" (feature/channel transformation) │
│                                                             │
│  Q-Former = 摄影师根据主题"对焦"                               │
│  FiLM     = 根据任务调整"成像参数"                             │
│  → 互补而非冗余                                               │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 初始化策略

```python
# FiLM 零初始化 — 训练起点为恒等变换
nn.init.zeros_(gamma_proj[-1].weight)  # γ = 0 → (1+γ) = 1
nn.init.zeros_(beta_proj[-1].weight)   # β = 0

# 训练初期 FiLM 无效果，Q-Former 主导学习
# 随训练进行，FiLM 逐渐学会对筛选后的特征做精细调制
```

---

## 6. Learnable Modality Embedding

**设计**：可学习向量 `[1, 1, 4096]`，加法注入所有 K 个 VGGT token。

**动机**：

```
当前 VLA 的多模态序列:
  [vision_patches | proprio | scene_tokens | text]
  所有 token 靠位置区分类型，无显式来源标记

加入 Modality Embedding:
  [vision_patches | proprio | VGGT+K+| text]
                           ↑
               LLM 的 attention 可以学习:
               "这是 VGGT 几何-语义 token，用不同的 pattern 处理"
```

**成本**：4096 参数（约 16KB），可忽略。

---

## 7. 训练策略

### 7.1 单阶段端到端

```python
optimizer = AdamW([
    {"params": scene_projector.parameters(), "lr": 5e-4},  # Q-Former + FiLM + Modal + Proj
    {"params": lora_params,                "lr": 5e-4},   # LLM LoRA
    {"params": action_head.parameters(),   "lr": 5e-4},   # Action Head
])
```

**不需要多阶段训练**。原因：
- Q-Former 仅 ~25M 参数，48 token × K query 的搜索空间极小
- Action Loss (L1) 直接监督："选什么几何特征 → 动作预测更准"
- BLIP-2 需要多阶段是因为做开放域 vision-text 对齐，我们不存在这个需求

### 7.2 组件训练状态

| 组件 | 状态 | 参数量 | 学习率 |
|------|------|--------|--------|
| VGGT-Ω | frozen | ~1B | — |
| Vision Backbone | frozen | ~300M | — |
| **Q-Former** | **trainable** | **~12M** | **5e-4** |
| **FiLM γ/β** | **trainable** | **~4M** | **5e-4** |
| **Modality Emb** | **trainable** | **~4K** | **5e-4** |
| SceneProjector | trainable | ~8M | 5e-4 |
| LLM (LoRA rank=32) | trainable | ~40M | 5e-4 |
| Action Head | trainable | ~2M | 5e-4 |
| **总计新增 trainable** | | **~25M** | |

---

## 8. 消融实验设计

| 变体 | Q-Former | FiLM | Modal Emb | K | 验证目标 |
|------|----------|------|-----------|---|---------|
| **Baseline** | 无 | 无 | 无 | 48 | 纯 concat 基线 |
| A | ✅ | 无 | 无 | 8 | Q-Former 本身的贡献 |
| B | ✅ | ✅ | 无 | 8 | FiLM 是否有额外增益 |
| C | ✅ | ✅ | ✅ | 8 | Modal Emb 是否有额外增益 |
| D | ✅ | ✅ | ✅ | 4 | 压缩比的敏感性 |
| E | ✅ | ✅ | ✅ | 16 | 压缩比的敏感性 |
| F | CA-only | 无 | 无 | 48 | Self-Attn 的价值验证 |

变体 F 是**最关键的消融**——去掉 Self-Attn，只保留 Cross-Attn 做 token 级加权（不压缩，48 token 全保留）。如果 Q-Former 的 Self-Attn 真的有"协同合成"的价值，完整 Q-Former（压缩到 K=8）应该明显优于 CA-only（48 token 不压缩）。

---

## 9. 创新总结

### 9.1 概念层面的贡献

> **首次识别 VGGT register token 的分布式 3D 编码性质，并提出 Self-Attention 协同合成来解决已有方法（Evo-0、3D-Mix、Compressor-VLA）的碎片化检索问题。**

已有方法用 cross-attention 做的是"分布式检索"（每个 query 各自从几何编码中捡碎片），本方案用 Q-Former 的 Self-Attention 做的是"协同合成"（K 个 query 协调分工，将碎片合成完整 3D 理解）。

### 9.2 技术层面的贡献

| 技术点 | 与已有方法的区别 |
|--------|----------------|
| **Q-Former × VGGT 3D geometry** | Evo-0/3D-Mix 只用 CA，无 Self-Attn；Compressor-VLA 面向 2D visual |
| **层级语言条件化** | Q-Former query bias（空间选择） + FiLM（channel 变换），互补而非冗余 |
| **模态类型标记** | 在 VLA 中首次对 3D 几何 token 做显式的来源类型嵌入 |
| **Pre-LLM 几何融合** | 3D-Mix Visual Fusion 在 Pre-LLM 融合失败 (4.69%)，本方案通过选择+压缩+独立 token 解决了这个失败 |

### 9.3 核心故事线

> 3D-Mix showed that naive pre-LLM fusion of VGGT tokens harms performance (Visual Fusion: 4.69%). We identify the root cause — VGGT's registers are a **distributed** 3D encoding where no single token carries complete geometric semantics. Simple cross-attention (as in Evo-0, 3D-Mix CrossAttn, Compressor-VLA) can only **retrieve** fragmented geometric cues. We introduce Q-Former's self-attention mechanism to **collaboratively synthesize** these distributed cues into coherent task-relevant geometry tokens, achieving the first successful pre-LLM 3D geometry fusion for VLAs.

---

## 10. 伪代码

### 10.1 SceneQFormer

```python
class SceneQFormer(nn.Module):
    def __init__(self, scene_dim=2048, llm_dim=4096, num_queries=8, num_layers=2):
        super().__init__()
        self.num_queries = num_queries
        self.scene_proj = nn.Linear(scene_dim, llm_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, llm_dim) * 0.02)
        self.lang_to_bias = nn.Sequential(
            nn.Linear(llm_dim, llm_dim), nn.GELU(),
            nn.Linear(llm_dim, llm_dim * num_queries),
        )
        self.layers = nn.ModuleList([
            QFormerLayer(llm_dim) for _ in range(num_layers)
        ])

    def forward(self, scene_tokens, lang_feat):
        B = scene_tokens.shape[0]
        K_scene = V_scene = self.scene_proj(scene_tokens)  # [B, 48, 4096]
        bias = self.lang_to_bias(lang_feat).view(B, self.num_queries, -1)
        queries = self.query_tokens + bias
        for layer in self.layers:
            queries = layer(queries, K_scene, V_scene)
        return queries  # [B, K, 4096]
```

### 10.2 SceneTokenFiLM

```python
class SceneTokenFiLM(nn.Module):
    def __init__(self, llm_dim=4096, bottleneck=256):
        super().__init__()
        self.gamma_proj = nn.Sequential(
            nn.Linear(llm_dim, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, llm_dim))
        self.beta_proj = nn.Sequential(
            nn.Linear(llm_dim, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, llm_dim))
        self.modal_emb = nn.Parameter(torch.zeros(1, 1, llm_dim))
        nn.init.zeros_(self.gamma_proj[-1].weight)
        nn.init.zeros_(self.gamma_proj[-1].bias)
        nn.init.zeros_(self.beta_proj[-1].weight)
        nn.init.zeros_(self.beta_proj[-1].bias)

    def forward(self, x, lang_feat):
        gamma = self.gamma_proj(lang_feat).unsqueeze(1)
        beta = self.beta_proj(lang_feat).unsqueeze(1)
        return (1 + gamma) * x + beta + self.modal_emb
```

### 10.3 SceneProjector（集成版）

```python
class SceneProjector(nn.Module):
    def __init__(self, scene_dim=2048, llm_dim=4096,
                 use_qformer=True, use_film=True,
                 num_queries=8, num_qformer_layers=2):
        super().__init__()
        self.use_qformer = use_qformer
        if use_qformer:
            self.qformer = SceneQFormer(
                scene_dim=scene_dim, llm_dim=llm_dim,
                num_queries=num_queries,
                num_layers=num_qformer_layers)
        if use_film:
            self.film = SceneTokenFiLM(llm_dim=llm_dim)
        # Fallback: simple projection
        self.projector = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(scene_dim, llm_dim)),
            ("norm", nn.LayerNorm(llm_dim)),
        ]))

    def forward(self, scene_tokens, lang_feat=None):
        if self.use_qformer and lang_feat is not None:
            x = self.qformer(scene_tokens, lang_feat)      # 48 → K
        else:
            x = self.projector(scene_tokens)                # 48 → 48
        if self.use_film and lang_feat is not None:
            x = self.film(x, lang_feat)
        return x
```
