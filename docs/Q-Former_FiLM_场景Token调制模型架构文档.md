# VGGT-QFormer：基于 Q-Former 的任务感知 3D 几何场景 Token 调制架构

## 1. 模型总览

```
                        ┌──────────────────────────────────────────────────────┐
                        │                    VGGT-QFormer                          │
                        │                                                        │
  N×512×512 multi-view   │      N×224×224 multi-view        Language Instruction │
       │                 │            │                              │           │
       ▼                 │            ▼                              │           │
┌──────────────┐         │  ┌────────────────────┐                   │           │
│   VGGT-Ω     │         │  │ SigLIP + DINOv2    │                  │           │
│   (frozen)   │         │  │ Vision Backbone    │                   │           │
│              │         │  │ (frozen / FiLM)    │                   │           │
│ register tok │         │  │ visual patches     │                   │           │
│ [B,32,2048]  │         │  │ [B,768,D_vis]      │                   │           │
└──────┬───────┘         │  └────────┬───────────┘                   │           │
       │                 │           │                               │           │
       │                 │           ▼                               │           │
       │                 │  ┌────────────────────┐                   │           │
       │                 │  │ Visual Projector   │                   │           │
       │                 │  │ (D_vis → 4096)     │                   │           │
       │                 │  └────────┬───────────┘                   │           │
       │                 │           │                               │           │
       ▼                 │           │                               ▼           │
┌──────────────────────────────────────────────────────────────────────────────┐ │
│                     SceneProjector (trainable, ~734M)                          │ │
│                                                                                │ │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │ │
│  │ ① SceneQFormer                                                          │  │ │
│  │                                                                          │  │ │
│  │  scene_tokens [B, 32, 2048]          language_embedding [B, 4096]        │  │ │
│  │       │                                       │                          │  │ │
│  │       ▼                                       ▼                          │  │ │
│  │  SceneProj(2048→4096)            lang_to_bias: 4096 → 4096×K             │  │ │
│  │       │                                       │                          │  │ │
│  │       ▼                                       ▼                          │  │ │
│  │  [B, 32, 4096]                query_bias [B, K, 4096]                    │  │ │
│  │       │                                       │                          │  │ │
│  │       │                    ┌──────────────────┘                          │  │ │
│  │       │                    │                                             │  │ │
│  │       │    learnable_query [K, 4096] + bias = conditioned_queries        │  │ │
│  │       │                    │                                             │  │ │
│  │       ▼                    ▼                                             │  │ │
│  │  ┌─────────────────────────────────────────┐                            │  │ │
│  │  │     Q-Former Block × 2                  │                            │  │ │
│  │  │                                         │                            │  │ │
│  │  │  Self-Attn ──→ Cross-Attn ──→ FFN       │                            │  │ │
│  │  │  (K queries  (queries attend  (几何→语义) │                            │  │ │
│  │  │   协同合成)    to 32 scene tok)           │                            │  │ │
│  │  │                                         │                            │  │ │
│  │  │  Pre-LN + 每步残差连接                   │                            │  │ │
│  │  └──────────────────┬──────────────────────┘                            │  │ │
│  │                     │                                                   │  │ │
│  │                     ▼                                                   │  │ │
│  │              Output Proj + Residual                                     │  │ │
│  └─────────────────────┬───────────────────────────────────────────────────┘  │ │
│                        │                                                      │ │
│                        ▼                                                      │ │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │ │
│  │ ② SceneTokenFiLM                                                        │  │ │
│  │                                                                          │  │ │
│  │  compressed_tokens [B, K, 4096]     language_embedding [B, 4096]         │  │ │
│  │       │                                    │                             │  │ │
│  │       │              ┌─────────────────────┤                             │  │ │
│  │       │              │                     │                             │  │ │
│  │       │              ▼                     ▼                             │  │ │
│  │       │   lang → bottleneck(256) → γ   lang → bottleneck(256) → β       │  │ │
│  │       │              │                     │                             │  │ │
│  │       │              └─────────┬───────────┘                             │  │ │
│  │       │                        ▼                                         │  │ │
│  │       │          x = (1 + γ) ⊙ x + β    (per-channel modulation)        │  │ │
│  │       ▼                                                                  │  │ │
│  └─────────────────────┬───────────────────────────────────────────────────┘  │ │
│                        │                                                      │ │
│                        ▼                                                      │ │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │ │
│  │ ③ Modality Embedding                                                     │  │ │
│  │                                                                          │  │ │
│  │  vggt_modal [1, 1, 4096]  (learnable parameter, ~4K params)             │  │ │
│  │  x = x + vggt_modal                                                      │  │ │
│  │  → 标记 "这是 VGGT 几何-语义 token"                                       │  │ │
│  └─────────────────────┬───────────────────────────────────────────────────┘  │ │
│                        │                                                      │
└────────────────────────┼──────────────────────────────────────────────────────┘
                         │
                         ▼  [B, K, 4096]
                         │
                         │
┌────────────────────────┼──────────────────────────────────────────────────────┐
│                        ▼                                                      │
│         ┌──────────────────────────────────────────────┐                      │
│         │           Multi-Modal Input Sequence          │                      │
│         │                                              │                      │
│         │  [BOS] [vision:768] [prop:1] [VGGT:K] [text] │                      │
│         │                                              │                      │
│         │  VGGT token 位置: vision 与 text 之间          │                      │
│         │  所有 text token 双向 attend 到 VGGT token     │                      │
│         └──────────────────────┬───────────────────────┘                      │
│                                │                                              │
│                                ▼                                              │
│         ┌──────────────────────────────────────────────┐                      │
│         │          Llama-7B (LoRA, rank=32)             │                      │
│         │                                              │                      │
│         │       Geometry-Aware Causal Reasoning         │                      │
│         └──────────────────────┬───────────────────────┘                      │
│                                │                                              │
│                                ▼                                              │
│         ┌──────────────────────────────────────────────┐                      │
│         │       Action Head (L1 Regression)             │                      │
│         │       → 连续动作序列 [NUM_CHUNK, ACT_DIM]      │                      │
│         └──────────────────────────────────────────────┘                      │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Token 数量变化

| 阶段 | Token 数 | 维度 | 说明 |
|------|----------|------|------|
| VGGT-Ω 输出 (2帧) | 32 | 2048 | 每帧 16 个 register token |
| SceneProjection 后 | 32 | 4096 | 线性投影到 LLM 空间 |
| Q-Former 压缩后 | **8** | 4096 | 压缩比 4:1 |
| FiLM + Modal Emb | **8** | 4096 | 调制不改变数量 |
| 注入 LLM | **8** | 4096 | 作为独立 token segment |

LLM 总 visual token: 768 + 1 (proprio) + 8 (VGGT) = **777**（对比原始方案的 817）

---

## 2. 核心创新：从分布式几何编码到协同语义合成

### 2.1 VGGT Register Token 的本质

VGGT-Ω 通过 24 层交替 attention（Frame Attention → Global Attention → Register Attention）将多视图 3D 信息聚合到 register token 中。关键性质：**register token 是分布式 3D 几何编码**——单个 register 不包含完整的 3D 概念。

```
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

### 2.2 已有方法的局限：单向检索，无法合成

Evo-0 / 3D-Mix CrossAttn / Compressor-VLA 均使用纯 Cross-Attention：每个 query 独立从 VGGT token 中检索信息。

```
    已有方法: visual/MLLM query → Cross-Attn → VGGT features

    问题: 每个 query 独自"捡"零散的几何碎片
         query_0 发现了深度边界 → 不知道法向在哪
         query_1 发现了表面法向 → 不知道遮挡关系在哪
         → 无法将碎片合成为完整的 3D 理解
```

### 2.3 本方案的核心贡献：协同几何合成

**Q-Former 的 Self-Attention 让 K 个 query 协同工作**，互相"通报"各自发现，形成互补分工：

```
    本方案 Q-Former:

    Self-Attention 中 query 间的信息交互:
    
    query_0: "我在关注 register_3，发现了深度边界"
    query_1: "我发现了 register_7 的表面法向，应该和你的边界有关"
    query_2: "那我换个方向，去关注 register_12 的遮挡关系"

    → K 个 query 形成互补分工
    → 每个 query 覆盖一个不同的几何子概念
    → K 个 query 的集合覆盖了"完整任务相关 3D 场景"
```

### 2.4 与已有方法的本质区别

| | Evo-0 / 3D-Mix CA / Compressor-VLA | 本方案 VGGT-QFormer |
|---|---|---|
| **几何信息处理** | 单向检索（query → token） | **协同合成**（query ↔ query + query → token） |
| **Query 间关系** | 独立工作，各自检索 | **Self-Attention 协调分工** |
| **输出关系** | 各自独立的结果 | **互补合成的完整 3D 理解** |
| **压缩能力** | 无（全量 token 保留） | **有**（48 → 8，压缩比 6:1） |
| **语言条件化** | 间接或无 | **直接**（query bias + FiLM 双层注入） |

### 2.5 创新层次定位

```
┌─────────────────────────────────────────────────────────────┐
│  范式创新: 无                                                │
│  架构创新: 无（Q-Former 组件本身已有）                        │
│                                                            │
│  ★ 概念创新:                                                │
│    首次识别 VGGT register 的分布式编码性质，                   │
│    并提出 Self-Attention 协同合成解决已有方法的碎片化检索问题   │
│                                                            │
│  ★ 场景创新:                                                │
│    首次将 Q-Former 完整架构应用于 VGGT 3D 几何 token 的任务感知 │
│    调制，填补了 3D-Mix / Evo-0 / Compressor-VLA 之间的空白    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 各模块详细设计

### 3.1 SceneQFormer — 场景 Token 选择与压缩（核心）

**模块输入输出**：
- 输入: VGGT-Ω register tokens [B, N×16, 2048] + language embedding [B, 4096]
- 输出: 压缩的 task-conditioned 几何 tokens [B, K, 4096]

**关键设计决策**：

| 设计点 | 选择 | 理由 |
|--------|------|------|
| Query 数量 K | 8 | 平衡信息保持与压缩效率 |
| Q-Former 层数 L | 2 | Evo-0 实验支持，过深可能过拟合 |
| 注意力头数 | 8 | Llama-7B 兼容，计算效率适中 |
| FFN 倍数 | 4× | 标准 Transformer 配置 |
| 语言注入方式 | Query Bias | 语言直接偏置 query 初始化，而非间接通过 CA K/V |
| 初始化 | 零偏置 + 小 query | 训练初期 query 通用，逐步语言条件化 |
| Pre-LN | 是 | 每子层前 LayerNorm，训练稳定性更好 |

**Q-Former Block 结构**：

```
┌──────────────────────────────────────────────────────────┐
│              Q-Former Block × 2                           │
│                                                          │
│  输入: queries [B, K, 4096], scene_features [B, 32, 4096]│
│                                                          │
│  ① Self-Attention (Multihead, 8 heads)                   │
│     Q = K = V = LayerNorm(queries)                       │
│     queries = queries + Dropout(SA(queries))              │
│     作用: query 间协同合成分布式 3D 编码                  │
│                                                          │
│  ② Cross-Attention (Multihead, 8 heads)                  │
│     Q = LayerNorm(queries)  [B, K, 4096]                 │
│     K = V = scene_features  [B, 32, 4096]                │
│     queries = queries + Dropout(CA(queries, scene))       │
│     作用: 根据 SA 协调的分工从 scene token 检索           │
│                                                          │
│  ③ Feed-Forward Network                                  │
│     x = LayerNorm(queries)                                │
│     x = Linear(4096 → 16384) → GELU → Dropout            │
│     x = Linear(16384 → 4096) → Dropout                   │
│     queries = queries + x                                 │
│     作用: 将原始几何特征翻译为 LLM 可理解的语义化表示     │
│                                                          │
│  输出: queries [B, K, 4096]                               │
└──────────────────────────────────────────────────────────┘
```

### 3.2 SceneTokenFiLM — 任务感知特征调制

**模块输入输出**：
- 输入: Q-Former 输出 [B, K, 4096] + language embedding [B, 4096]
- 输出: 调制后的 tokens [B, K, 4096]

**调制公式**：

```
x = (1 + γ) ⊙ x + β

γ = bottleneck_scale(lang_feat)  → [B, 4096]
β = bottleneck_shift(lang_feat)  → [B, 4096]

bottleneck: Linear(4096→256) → GELU → Linear(256→4096)
→ 参数效率: 4.2M (vs 33.6M 无 bottleneck)
```

**初始化策略**：

```python
# γ 和 β 的末层投影零初始化 → 训练初期 FiLM = 恒等变换
nn.init.zeros_(scale_fc2.weight); nn.init.zeros_(scale_fc2.bias)
nn.init.zeros_(shift_fc2.weight); nn.init.zeros_(shift_fc2.bias)
```

**与 Q-Former 语言条件化的层级互补关系**：

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

**FiLM 放在 Q-Former 之后的原因**：
1. 调制的是已语义化的"几何-语义混合特征"，不破坏 VGGT 预训练结构
2. 计算效率：K × 4096（K=8） vs 32 × 4096（放在之前）
3. Q-Former 先做"选择"，FiLM 再做"调制"——逻辑顺序清晰

### 3.3 Modality Embedding — Token 来源标记

```python
self.modality_embedding = nn.Parameter(torch.zeros(1, 1, 4096))

# Forward:
x = x + self.modality_embedding  # broadcast 到所有 B 和 K
```

**设计动机**：

```
当前 VLA 多模态序列中，所有 token 仅靠位置区分类型:

  [BOS] [vision_patches] [proprio] [scene_tokens] [text]
    ↑       ↑              ↑         ↑             ↑
   无显式来源标记，LLM 需要从位置隐式学习 token 类型

加入 Modality Embedding 后:
  [BOS] [vision_patches] [proprio] [VGGT+K+] [text]
                                         ↑
                            LLM attention 学习:
                            "这是 VGGT 几何-语义 token，用不同 pattern 处理"
```

- 成本：4096 参数（约 16KB），可忽略
- 零初始化：训练初期无效果，逐步学习
- 消融实验可验证实际贡献

---

## 4. 与已有方法的系统性对比

### 4.1 对比总览表

| | Evo-0 | 3D-Mix CA | 3D-Mix Gated | Compressor-VLA | **VGGT-QFormer** |
|---|---|---|---|---|---|
| **解决的问题** | 2D视觉融合3D | 语义+几何融合 | **语义/几何混合比例** | 2D视觉压缩 | **几何token任务感知筛选** |
| **核心操作** | CA检索 | CA检索 | **门控混合** | CA压缩 | **CA选择 + SA协同合成** |
| **注入位置** | Pre-MLLM | Post-MLLM | Post-MLLM | Pre-LLM | **Pre-LLM** |
| **Query 来源** | 2D ViT | MLLM输出 | N/A(门控) | Learnable | **Learnable + Lang** |
| **Self-Attn** | 无 | 无 | 无 | 无 | **有（协同合成）** |
| **压缩** | 无 | 无 | 无 | 有(2D visual) | **有(3D geometry)** |
| **语言条件化** | 无 | 间接 | 间接(语义池化) | 有(FiLM) | **双层(query bias + FiLM)** |
| **与GatedFusion的关系** | 竞争 | 竞争 | — | 互补 | **正交（可组合）** |

### 4.2 3D-Mix GatedFusion 与本方案的根本区别：不同的问题

**3D-Mix GatedFusion 解决的是另一个问题**：

```
3D-Mix GatedFusion 的语义条件自适应门控:

  语义特征 S_j ──→ 门控 g_j = σ(W · [S_j ; F_geo_j]) ──→ f_j = g ⊙ S_j + (1-g) ⊙ F_geo_j
  几何特征 F_j ──→                                          ↑
                                                    混合比例：语义×g + 几何×(1-g)
  
  核心操作: 在语义信息和几何信息之间做加权混合
  解决的问题: "在输出融合阶段，应该用多少语义信息 vs 多少几何信息？"
```

**本方案 Q-Former 解决的是筛选问题**：

```
本方案 Q-Former 的语言条件化 token 筛选:

  32 个 VGGT register token ──→ Q-Former (lang-conditioned cross-attention) ──→ 8 个精选几何 token
  语言指令 ──→ query bias ──→                                          ↑
                                                         只选出任务相关的几何 token
  
  核心操作: 从几何 token 集合中筛选出任务相关的子集
  解决的问题: "在注入 LLM 之前，哪些几何 token 对当前任务有用？哪些是背景噪音？"
```

这是两个**正交的问题**——3D-Mix 关注的是"几何 vs 语义的权重分配"，本方案关注的是"几何信息内部的筛选去噪"。两者不是竞争关系，甚至可以组合使用（先筛选后混合）。

### 4.3 3D-Mix 9 种方案覆盖情况

3D-Mix 系统评估了 9 种 VGGT→VLA 融合方案，但**未覆盖**本方案的设计：

| 3D-Mix 已覆盖 | 本方案的不同之处 |
|-------------|----------------|
| Visual Fusion (#9): 2D ViT + VGGT 做 CA 混合 → **失败 (4.69%)** | 本方案：独立 token segment，不混合 → 避免特征污染 |
| CrossAttn Fusion (#4): MLLM→CA→VGGT → **56.25%** | 本方案：Learnable+lang query，Pre-LLM 注入，直接语言条件化 |
| **GatedFusion (#5)**: 语义条件化自适应门控 → **68.23% (最优)** | 本方案解决的是**不同的问题**：GatedFusion 调节语义/几何的**混合比例**，本方案做的是几何 token 的**预先筛选**——两个正交方向，可以互补 |

### 4.4 本方案与 GatedFusion 的互补关系

```
                     VGGT register tokens [32, 2048]
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
          ┌─────────────────────┐   ┌─────────────────────┐
          │ 本方案 (Pre-LLM)    │   │ 3D-Mix (Post-LLM)   │
          │                     │   │                     │
          │ Q-Former 筛选       │   │ GatedFusion 混合     │
          │ "哪些几何 token     │   │ "几何 vs 语义        │
          │  对任务有用？"      │   │  各占多少比例？"    │
          │                     │   │                     │
          │ 32 → 8 压缩筛选     │   │ g ⊙ semantic +      │
          │                     │   │ (1-g) ⊙ geometric   │
          └─────────┬───────────┘   └──────────┬──────────┘
                    │                          │
                    ▼                          ▼
           注入 LLM 输入              注入 Action Expert
        (LLM 做 geometry-aware      (动作解码时使用几何引导)
         reasoning)
```

**两个方向解决不同的问题，可以同时使用**：Q-Former 先筛选出任务相关的几何 token 注入 LLM（让 LLM 做几何感知推理），GatedFusion 再在 Action Expert 层面做语义/几何的自适应混合（让动作解码利用几何引导）。

### 4.5 Early vs Late Fusion 的哲学分歧

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Early Fusion (本方案):                                         │
│     "几何信息应该参与 LLM 推理过程，帮助模型从空间层面理解任务"     │
│                                                                 │
│     ✓ LLM 可做 geometry-aware reasoning                          │
│     ✓ 语言直接条件化几何选择（更精准）                              │
│     ✓ 压缩后 token 少，注意力开销可控                             │
│                                                                 │
│   Late Fusion (3D-Mix GatedFusion):                             │
│     "几何信息只需在动作解码时使用，不应干扰 LLM 语义推理"           │
│                                                                 │
│     ✓ LLM 注意力完全用于语义推理                                  │
│     ✓ 已验证有效 (68.23%)                                        │
│                                                                 │
│   核心研究问题：用 Q-Former 做语言条件化 Early fusion，             │
│   能否与/超越 Late fusion 的 GatedFusion？                       │
│                                                                 │
│   注意：两者解决的是正交问题（筛选 vs 混合），并非互斥              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 训练策略

### 5.1 单阶段端到端训练

**不需要 BLIP-2 风格的多阶段训练**。原因：
- Q-Former ~730M 参数，32 token × 8 query 的搜索空间极小
- Action L1 Loss 直接监督："选什么几何特征 → 动作预测更准"
- 没有 vision-language alignment 的预训练需求
- 数据规模（RLDS 数万 trajectories）足够

### 5.2 组件训练状态

| 组件 | 状态 | 参数量 | 学习率 |
|------|------|--------|--------|
| VGGT-Ω | frozen | ~1.0B | — |
| SigLIP + DINOv2 | frozen | ~0.3B | — |
| Visual Projector | frozen | ~17M | — |
| **SceneQFormer** | **trainable** | **730M** | 5e-4 |
| **SceneTokenFiLM** | **trainable** | **4.2M** | 5e-4 |
| **Modality Embedding** | **trainable** | **4K** | 5e-4 |
| LLM (LoRA, rank=32) | trainable | ~40M | 5e-4 |
| Action Head | trainable | ~2M | 5e-4 |
| **总计新增 trainable** | | **~734M** | |

### 5.3 训练配置

```python
# 优化器
optimizer = AdamW(
    [
        {"params": scene_projector.parameters(), "lr": 5e-4},
        {"params": lora_params,                 "lr": 5e-4},
        {"params": action_head.parameters(),    "lr": 5e-4},
    ]
)

# 学习率调度
scheduler = MultiStepLR(
    optimizer,
    milestones=[100_000],  # 100K steps 后衰减 10x
    gamma=0.1,
)

# 训练步数
max_steps = 150_000

# 批次大小
batch_size = 8 (per GPU) × grad_accumulation_steps
```

### 5.4 训练稳定性措施

1. **FiLM 零初始化**：γ, β 从 0 开始 → 训练初期 FiLM = 恒等变换
2. **Q-Former query 小初始化**：std=0.02 → 避免极端初始 attention 分布
3. **Language bias 零初始化**：末层 weight/bias = 0 → query 初始无语言偏置
4. **梯度裁剪 + warmup**：Q-Former cross-attention 早期可能产生大梯度

---

## 6. 消融实验设计

### 6.1 核心消融

| 变体 | Q-Former | FiLM | Modal Emb | K | 验证目标 |
|------|----------|------|-----------|---|---------|
| **Baseline** | 无 | 无 | 无 | 32 | 纯 concat 基线 |
| A | ✅ | 无 | 无 | 8 | Q-Former 本身贡献 |
| B | ✅ | ✅ | 无 | 8 | FiLM 额外增益 |
| C | ✅ | ✅ | ✅ | 8 | Modal Emb 额外增益 (Full pipeline) |
| **F** | **CA-only** | 无 | 无 | 32 | **Self-Attn 价值验证（关键消融）** |

变体 F 是**最关键的消融**——去掉 Self-Attn 仅保留 Cross-Attn（不压缩）。如果完整 Q-Former（K=8）显著优于 CA-only（32 token 不压缩），则直接证明 Self-Attention 的"协同合成"价值。

### 6.2 压缩比敏感性

| 变体 | K | 压缩比 | 验证目标 |
|------|---|--------|---------|
| D | 4 | 8:1 | 激进压缩的信息瓶颈 |
| C | 8 | 4:1 | 推荐配置 |
| E | 16 | 2:1 | 保守压缩 |

### 6.3 Q-Former 深度敏感性

| 变体 | L | 验证目标 |
|------|---|---------|
| G | 1 | 单层是否足够 |
| C | 2 | 推荐配置 |
| H | 4 | 更深是否带来额外收益 |

### 6.4 推荐实验顺序

1. **Baseline vs A**：验证 Q-Former 本身价值（最大不确定性）
2. **A vs B**：验证 FiLM 额外贡献
3. **B vs C (Full)**：验证 Modality Embedding
4. **CA-only (F) vs A**：验证 Self-Attention 的协同合成价值（**关键实验**）
5. **调优 K 和 L**：压缩比和深度的敏感性

---

## 7. 参数量分析

| 组件 | 子模块 | 参数量 |
|------|--------|--------|
| SceneQFormer | scene_proj (2048→4096) | 8.4M |
| | query_tokens (8 × 4096) | 33K |
| | lang_to_bias (4096→4096×8) | 151.0M |
| | ×2 QFormerLayer: | |
| | - Self-Attention (×2) | 134.2M |
| | - Cross-Attention (×2) | 134.2M |
| | - FFN (×2) | 268.4M |
| | output_proj + norms | 33.6M |
| | **SceneQFormer 小计** | **730.0M** |
| SceneTokenFiLM | γ bottleneck (4096→256→4096) | 2.1M |
| | β bottleneck (4096→256→4096) | 2.1M |
| | **FiLM 小计** | **4.2M** |
| Modality Embedding | 1 × 4096 | 4K |
| **新增总计** | | **734.3M** |

对比：LLM LoRA (rank=32) 约 40M，Visual Projector 约 17M。

---

## 8. 项目文件结构

```
code/openvla-oft/prismatic/
├── models/
│   ├── scene_qformer.py         # QFormerLayer + SceneQFormer
│   ├── scene_projector.py       # SceneTokenFiLM + SceneProjector（集成版）
│   ├── film_vit_wrapper.py      # FiLMedVisionTransformer (已有，vision backbone FiLM)
│   └── action_heads.py          # L1RegressionActionHead / DiffusionActionHead
├── extern/hf/
│   └── modeling_prismatic.py    # PrismaticForConditionalGeneration + OpenVLAForActionPrediction
│                                 #   - forward(): 训练前向（lang_feat → scene_projector）
│                                 #   - predict_action(): 推理动作预测
│                                 #   - NUM_PATCHES: 适配 Q-Former 压缩 token 数
└── vla-scripts/
    └── finetune.py               # 训练主脚本
                                   #   - VGGTSceneExtractor: VGGT-Ω 特征提取
                                   #   - SceneProjector 初始化 + checkpoint 保存
                                   #   - NUM_PATCHES 计算适配
```

---

## 9. 论文定位

### 9.1 核心故事线

> 3D-Mix showed that naive pre-LLM fusion of VGGT tokens harms performance (Visual Fusion: 4.69%). We identify the root cause — VGGT's registers are a **distributed** 3D encoding where no single token carries complete geometric semantics. Simple cross-attention (as in Evo-0, 3D-Mix CrossAttn, Compressor-VLA) can only **retrieve** fragmented geometric cues. We introduce Q-Former's self-attention mechanism to **collaboratively synthesize** these distributed cues into coherent task-relevant geometry tokens, achieving the first successful pre-LLM 3D geometry fusion for VLAs.

### 9.2 差异化方向

1. **可解释性分析**：通过 Q-Former cross-attention 权重可视化，证明模型学会了选择任务相关的 3D 区域
2. **层叠语言条件化的机制性消融**：证明 Query Bias（空间选择）+ FiLM（channel 变换）的层级互补性
3. **压缩效率系统性分析**：压缩比-性能曲线，证明精选几何信息优于全量注入
4. **Pre-LLM vs Post-LLM 融合的理论分析**：提供 Early Fusion 优于 Late Fusion 的场景条件

### 9.3 目标会议建议

| 会议 | 可行性 | 策略 |
|------|--------|------|
| **CoRL / RSS** | 有机会 | 机器人实验深度 + 真实世界验证 |
| **AAAI** | 可考虑 | 与 3D-Mix / Evo-0 直接对比 + 可解释性分析 |
| **ICRA** | 可行 | 系统性消融 + 工程贡献定位 |
