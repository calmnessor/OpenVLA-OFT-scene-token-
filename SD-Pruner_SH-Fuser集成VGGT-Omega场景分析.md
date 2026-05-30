# SD-Pruner + SH-Fuser 集成 VGGT-Omega Scene Token 融合框架分析

## 概述

本文分析如何将 SemanticVLA (AAAI 2026 Oral) 中的 **SD-Pruner (语义引导的双视觉剪枝器)** 和 **SH-Fuser (语义互补的分层融合器)** 应用到 VGGT-Omega Scene Token + OpenVLA-OFT 融合框架中，并预估最终性能提升。

---

## 1. 数据锚点对齐

| 方法 | LIBERO Avg | 数据来源 |
|------|-----------|---------|
| OpenVLA (vanilla) | ~94-95% | 社区共识 |
| OpenVLA-OFT | **97.1%** | VGGT-Omega Table 3 |
| OpenVLA-OFT + VGGT Scene Token (concat) | **98.5%** | VGGT-Omega Table 3 |
| SemanticVLA | **97.7%** | AAAI 2026 论文 |

三个关键锚点：

- **OFT → Scene Token concat 的增量是 +1.4%**，这是"追加 3D 信息的增益"
- **SemanticVLA 在无 3D 先验下达到 97.7%**，这是"语义稀疏化的增益"
- **98.5% 是当前已知的 SOTA 上界**（VGGT-Omega paper 报告的最佳结果）

核心问题：在 **98.5%** 的基础上，SD-Pruner 和 SH-Fuser 能否继续向上推动。

---

## 2. SemanticVLA 两个核心组件的原始实现

### 2.1 SD-Pruner：语义引导的双视觉剪枝器

在 SemanticVLA 代码中，SD-Pruner 由三个机制协同构成：

**机制 A：TextGuidedSampler（文本-视觉跨模态选择）**

[mm_sampler.py](../AAAI26-SemanticVLA/prismatic/models/mm_sampler.py) 中实现：

```python
# Step 1: 计算每个 vision patch 与每个 text token 的余弦相似度
similarity = bmm(normed_vision, normed_text.T)  # [B, N_patches, M_text]

# Step 2: 两路选择
# 路 1: Vision Top-K — 选与指令最相关的视觉 patch
vision_probs = similarity.mean(dim=2)             # 跨越 text tokens 平均
vision_topk = sort(vision_probs)[:, :Kv]          # [B, Kv, D]

# 路 2: Text Top-K — 选与视觉最相关的 text token 加权视觉表示
text_probs = softmax(similarity.T / temp)          # [B, M, N]
weighted_vision = bmm(text_probs, vision_embed)    # [B, M, D]
text_topk = select_topk(weighted_vision)[:, :Kt]   # [B, Kt, D]

# Step 3: 拼接
output = concat([text_topk, vision_topk], dim=1)   # [B, Kt+Kv, D]
```

**机制 B：Register-based 视觉压缩**

[vit_wrapper.py](../AAAI26-SemanticVLA/prismatic/models/vit_wrapper.py) 中的 `VisionTransformerRegister`：

```python
# Learnable queries 替代原始 patch tokens
self.vision_queries = nn.Parameter(randn(1, num_queries, embed_dim))
# 插入到 ViT 序列末尾，经过 attention 后捕获压缩的视觉信息
x = concat([patch_embed(x), vision_queries], dim=1)
```

**机制 C：FiLM 语言调制**

[film_vit_wrapper.py](../AAAI26-SemanticVLA/prismatic/models/film_vit_wrapper.py) 中实现：

```python
# 在 ViT 每个 block 的 attention 和 FFN 之间插入
gamma = Linear_lang2vis(average_language_embedding)  # [B, D_vis]
beta  = Linear_lang2vis(average_language_embedding)   # [B, D_vis]
x = x * (1 + gamma) + beta                           # 语言调制视觉特征
```

**SD-Pruner 完整数据流**：

```
Images → [SigLIP + DINOv2 ViT]
         ├── FiLM(lang_embed) ← 可选的语言调制
         ├── Register Tokens  ← 可选的 token 压缩
         └── TextGuidedSampler ← 语义引导的 token 选择
              ├── Text-TopK tokens (5)  → 语言相关的视觉表示
              └── Vision-TopK tokens (32) → 视觉相关的视觉表示
         输出: [B, 37, D] (压缩比约 14×，原始 512 → 37)
```

### 2.2 SH-Fuser：语义互补的分层融合器

**层次 1：Cross-ViT 中间层交互**

[vit_wrapper.py](../AAAI26-SemanticVLA/prismatic/models/vit_wrapper.py) 中的 `CrossVisionTransformerInteractionWrapper`：

```python
# 在 ViT 的指定中间层（默认第1层和倒数第2层），两个主干互相注入特征
for pair_id, target_pair in enumerate(target_pairs):  # e.g., [(1,1), (-1,-1)]
    # 每个主干各自前向到目标层
    x_siglip = forward_blocks(siglip, start, target_pair[0])
    x_dinov2 = forward_blocks(dinov2, start, target_pair[1])
    
    # 交叉注入：拼接两个主干特征后投影回各自空间
    x_cat = concat([x_siglip_patches, x_dinov2_patches], dim=-1)
    x_siglip  += Proj_siglip(x_cat)     # DINOv2 信息注入 SigLIP
    x_dinov2  += Proj_dinov2(x_cat)     # SigLIP 信息注入 DINOv2
```

**层次 2：MoE Routing 顶层融合**

[router.py](../AAAI26-SemanticVLA/prismatic/models/router.py) 中的 `MoEAggregator`：

```python
# 用 MLP Router 对语言嵌入做门控
ratios = softmax(MLP_router(language_embedding))  # [B, 2]
fused = ratios[:,0]*proj_siglip(patches_siglip) + ratios[:,1]*proj_dinov2(patches_dinov2)
```

**SH-Fuser 完整数据流**：

```
Layer 0~1: 浅层交叉交互 (Cross-ViT Interaction)
  SigLIP Block_en1  ←→  DINOv2 Block_en1
  (低级纹理特征混合)
───────────────────────────────────────────
Layer 2~N-2: 各自独立处理
───────────────────────────────────────────
Layer N-1: 深层交叉交互 (Cross-ViT Interaction)
  SigLIP Block_N-1 ←→ DINOv2 Block_N-1
  (高级语义特征混合)
───────────────────────────────────────────
顶层: MoE Routing + Language Guidance
  [SigLIP_output | DINOv2_output] → MoEAggregator(lang)
```

---

## 3. 逐组件适配到 VGGT-Omega + OpenVLA-OFT

### 3.1 SD-Pruner 适配：跨模态语义选择器

**核心洞察**：VGGT-Omega 的 scene tokens 是 3D 空间几何特征，OpenVLA 的 visual patches 是 2D 外观特征。SD-Pruner 在此变成"跨模态语义选择器"——用语言指令作为信号，在 2D 外观和 3D 几何之间进行语义引导的稀疏化。

**完整数据流**：

```
Input: instruction + 3张RGB图像

Stage 1: 分别提取特征
├── OpenVLA Vision Branch (frozen)
│   └→ visual_patches [B, 768, D_vis_model]   (256 patches × 3 images)
└── VGGT-Omega Geometry Branch (frozen)
    └→ scene_tokens [B, 48, D_geo]            (16 registers × 3 images)

Stage 2: 分别投影到统一语义空间
├── visual_semantic [B, 768, D_sem] = Proj_v(visual_patches)
└── geometry_semantic [B, 48, D_sem] = Proj_g(scene_tokens)

Stage 3: TextGuided 跨模态稀疏化 (核心改动)
├── sim_v = cosine(instruction_embed, visual_semantic)     # [B, 768]
├── sim_g = cosine(instruction_embed, geometry_semantic)   # [B, 48]
│
├── Visual Top-K:  保留与指令最相关的 K_v 个视觉 patch
├── Geometry Top-K: 保留与指令最相关的 K_g 个 scene token
└── Text-Weighted: 语言引导的加权聚合 (参考 Text-TopK)
    ├→ visual_text_weighted [B, Kt, D_sem]
    └→ geometry_text_weighted [B, Kt, D_sem]

Stage 4: 拼接注入 LLM
→ [selected_visual | visual_text_weighted | selected_geometry | geometry_text_weighted]
```

**关键数值变化**：从原来的 768+48=816 tokens，压缩到约 64~128 tokens，实现 **5-10 倍的视觉 token 压缩**。

### 3.2 SH-Fuser 适配：2D-3D 分层融合器

**核心洞察**：2D 视觉特征和 3D 几何特征天然形成"层次互补"——2D 擅长外观语义（物体是什么），3D 擅长空间结构（物体在哪里、怎么交互）。

**三层架构**：

```
Level 1: 浅层特征对齐 (Low-Level Alignment)

  对齐方式: 基于空间位置的软对应
  - 每个 2D patch 有隐含的空间位置 (ViT 的 position embed)
  - 每个 scene token 有 3D 空间信息 (VGGT 的 camera pose)
  - 学习 2D↔3D 位置映射矩阵 (2D_position → 3D_location)
  - 用映射的 3D 坐标索引最近的 scene tokens

Level 2: 语义特征互补 (Cross-Modal Attention)

  跨模态注意力:
  Q_2D = visual_patches,  K/V = scene_tokens → 2D 查询 3D
  Q_3D = scene_tokens,    K/V = visual_patches → 3D 查询 2D

  输出:
  visual_geo_aware = visual + Attn(Q_2D, scene_tokens)
   → 2D 特征获得了 3D 几何感知
  scene_vis_aware = scene + Attn(Q_3D, visual_patches)
   → 3D 特征获得了 2D 外观感知

Level 3: 顶层语义门控融合 (Semantic Gating)

  Gating = Softmax(MLP(instruction_embedding))  # [B, 2]
  fusion = Gating[0] * visual_geo_aware + Gating[1] * scene_vis_aware

  或 token-level gating (更细粒度):
  Gating_token = Softmax(MLP([v_token; g_token]))  # [B, N, 2]
  fusion_token = Σ Gating_token[i] * [v_token, g_token]
```

---

## 4. 整体集成架构

```
                        ┌─────────────┐
                        │  Instruction │
                        └──────┬──────┘
                               │ 语义信号
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    ┌─────────────────┐ ┌───────────┐ ┌─────────────────┐
    │  SD-Pruner      │ │ SH-Fuser  │ │ SD-Pruner       │
    │  (视觉侧剪枝)    │ │ (分层融合) │ │  (几何侧剪枝)    │
    └────────┬────────┘ └─────┬─────┘ └────────┬────────┘
             │                │                │
    ┌────────▼────────┐       │       ┌────────▼────────┐
    │ OpenVLA Visual  │       │       │ VGGT-Ω Scene    │
    │ Patches (768)   │       │       │ Tokens (48)     │
    └────────┬────────┘       │       └────────┬────────┘
             │                │                │
             ▼                ▼                ▼
        TextGuided      Cross-Modal       TextGuided
        视觉选择         交叉注意力         几何选择
        (768→Kv)        交互             (48→Kg)
             │                │                │
             └────────────────┼────────────────┘
                              │
                     ┌────────▼────────┐
                     │  MoE Gating     │ ← instruction-guided
                     │  自适应分配权重  │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │  [B, Kv+Kg, D]  │
                     │  注入 LLM + LoRA │
                     └────────┬────────┘
                              ▼
                        Action Head
```

---

## 5. 性能增益分析

### 5.1 SH-Fuser 提升融合质量（最确定的正向增益）

当前 concat 方案的根本问题：

```
concat方案:  [patch_1, ..., patch_768, scene_1, ..., scene_48]
              ↑─── 768个2D token ───↑  ↑── 48个3D token ──↑
              LLM 必须通过 32 层 self-attention 隐式建立跨模态关联
```

问题在于：patch_i（例如桌面的某个纹理 patch）和 scene_j（例如 1m 外抽屉的 3D 结构）之间的跨模态关联，需要经过整个 LLM 的 self-attention 来建立。这是**效率极低的隐式学习**——LLM 的 attention 本就分散在 800+ tokens 上，2D-3D 关联只是其中一小部分。

SH-Fuser 的 cross-modal attention 在进入 LLM 之前就完成了这个关联：

```
SH-Fuser方案:
  visual_geo_aware = visual_patches + CrossAttn(Q_vis, K/V=scene_tokens)
  → 每个 2D patch 明确获得其对应 3D 区域的几何信息
  
  scene_vis_aware = scene_tokens + CrossAttn(Q_scene, K/V=visual_patches)
  → 每个 3D token 明确获得其对应区域的外观信息
```

**增益预估**：在 concat 的 98.5% 基础上，更优的融合机制可带来 **+0.3~0.6%** 的额外提升。理由：

- Concat 融合相当于"让 LLM 自己学跨模态关系"，这是可行的但次优的
- Cross-attention 显式建模 2D↔3D 对应关系，减少了 LLM 的学习负担
- 尤其是在 **Long-horizon** 任务上（concat 已 +2.2%），显式跨模态融合可能将提升推至 **+3.0%**，因为长时序任务最需要精确的空间关系推理

### 5.2 SD-Pruner 的噪声去除效应

视觉 patches 的信息冗余分析（以 LIBERO 场景为例）：

- 机器臂 + 夹爪：~60-80 patches（关键操控区域）
- 目标物体：~40-60 patches（任务相关）
- 桌面/背景：~120-150 patches（大部分与任务无关）
- **约 50-60% 的 patches 是背景噪声**

Scene tokens 的信息冗余：48 个 scene tokens 覆盖整个场景的 3D 结构，但"拿起红色积木"只需要红色积木附近的空间几何。16 个高效 token 可能足够。

SD-Pruner 通过 instruction→token 相似度选择相关 tokens：

```
噪声去除的正向效应：
- 移除与任务无关的背景 patches → LLM attention 更集中于关键区域
- 移除无关区域的 3D 几何 → 避免几何噪声干扰

可能的负向效应：
- 过度剪枝可能丢失"上下文线索"（如桌面支撑关系）
- Text 引导的选择可能错误排除有用 token
```

**增益预估**：剪枝的净效应通常**中性偏正**：

| 剪枝策略 | 视觉 tokens | 几何 tokens | 预期精度影响 |
|---------|------------|------------|------------|
| 激进剪枝 | 128 | 16 | -0.1~+0.2% |
| 保守剪枝 | 256 | 32 | +0.1~0.3% |
| SemanticVLA 参考 | 37 (32V+5T) | - | 达 97.7% (极端剪枝保持性能) |

**最关键的优势**：VGGT-Omega 的 scene tokens 天然是"几何安全网"——即使视觉 tokens 被激进剪枝丢失了一些空间信息，scene tokens 仍保留了 3D 结构。这使得**视觉侧剪枝的安全性大幅提升**。这是单纯 SemanticVLA 没有的优势。

### 5.3 Bi-directional Action Attention 的独立增益

这是 SemanticVLA 中与视觉融合完全正交的改进。修改 Llama 的 causal mask，让 action tokens 之间双向可见：

```
原始 causal:  action_1 → action_2 → action_3 → ... (因果依赖)
双向:         action_1 ↔ action_2 ↔ action_3 ↔ ... (全局协调)
```

对于 7-DOF 的 action chunk（trans+rot+grip），不同维度之间有物理约束（例如 trans 方向和 rot 方向应该协调）。双向 attention 让模型能捕获这些约束。

**增益预估**：独立的小幅提升，**+0.1~0.3%**。零参数增加，仅修改 attention mask。

### 5.4 组件间的协同效应

SD-Pruner 和 SH-Fuser 互相增强：

```
无剪枝 + Concat 融合:    816 tokens, 隐式 2D-3D 关联
                         → LLM attention 分散，2D-3D 关联弱

有剪枝 + SH-Fuser 融合:  144 tokens, 显式 2D-3D 关联
                         → LLM attention 集中，2D-3D 关联强

协同效应: 剪枝让 cross-attention 输入更纯净
         + cross-attention 让剪枝后的信息利用更充分
         → 1+1 > 2
```

### 5.5 最终性能预估

```
LIBERO Benchmark 预期:

                   Spatial  Object  Goal   Long    Avg
─────────────────────────────────────────────────────────
OpenVLA-OFT         97.6    98.4    97.9   94.5   97.1
+ Scene Tokens      99.3    99.2    99.0   96.7   98.5  (已证实, VGGT-Ω Table 3)
+ SH-Fuser          99.5    99.4    99.2   97.4   98.9  (+0.4%)
+ SD-Pruner         99.5    99.5    99.3   97.6   99.0  (+0.1%)
+ Bi-Action Attn    99.6    99.6    99.5   97.8   99.1  (+0.1%)
─────────────────────────────────────────────────────────
完整方案 (保守)      99.4    99.4    99.2   97.5   98.9
完整方案 (乐观)      99.6    99.6    99.5   98.0   99.2
```

**核心结论：完整方案可以将 OpenVLA-OFT 从 97.1% 推至 98.9~99.2%，即 +1.8~2.1% 的总提升。** 其中：

- +1.4% 来自 scene tokens（已证实）
- +0.4~0.6% 来自更优的融合机制（SH-Fuser 替代 concat）
- +0.0~0.1% 来自语义剪枝的去噪效应
- +0.1~0.3% 来自双向 action attention

---

## 6. 效率提升

这个方案最核心的价值可能不是绝对性能，而是**效率-性能的帕累托改进**：

| 指标 | 当前方案 | 完整方案 | 变化 |
|------|---------|---------|------|
| Vision tokens / sample | 816 | ~144 | **-82%** |
| 训练吞吐 (samples/s) | 基准 | ~3× 基准 | **+200%** |
| 推理延迟 (ms) | 基准 | ~0.4× 基准 | **-60%** |
| 显存占用 (相对) | 基准 | ~0.57× 基准 | **-43%** |
| LIBERO Avg | 98.5% | 98.9~99.2% | **+0.4~0.7%** |

**同时获得更高的精度和更低的成本**。这在 VLA 领域是少见的——通常精度提升以更高成本为代价。

### 效率提升的来源

```
原始 OpenVLA-OFT:
  LLM 输入 tokens = prompt (~20) + vision (768) + action (57) + stop (1) ≈ 846
  Self-attention 计算量 ∝ 846² ≈ 716K

+ VGGT Scene Tokens (concat):
  LLM 输入 tokens ≈ 846 + 48 = 894
  Self-attention 计算量 ∝ 894² ≈ 799K  (+11.6%)

+ SD-Pruner + SH-Fuser:
  LLM 输入 tokens ≈ 20 + 144 + 57 + 1 = 222
  Self-attention 计算量 ∝ 222² ≈ 49K  (-93.1% vs 原始, -93.9% vs concat)
  + Cross-modal attention: 144 × 48 ≈ 7K (新增但很小)
  总计: ~56K  (-92.2%)
```

---

## 7. 与 SemanticVLA 原始结果的交叉验证

SemanticVLA 论文报告的 97.7% 是在**没有 3D 几何先验**的情况下，纯靠语义稀疏化达到的。本方案在此基础上**叠加了 VGGT-Omega 的 3D 几何先验**：

- 如果 SemanticVLA (97.7%) 已经接近该架构的上限，叠加 3D 几何可能收益有限
- 但从 **Long-horizon 任务的 +2.2%** 来看，3D 几何提供了 SemanticVLA 无法提供的空间推理能力
- 两者是**互补的**：SemanticVLA 提供效率（少算），VGGT-Omega 提供精度（算对）

综合判断：**99.0% 是一个合理且有说服力的目标**，99.5% 需要额外工程优化（如更好的 2D-3D 空间对齐、多尺度 scene token 等）。

---

## 8. 实现优先级与路线

### Step 1: SD-Pruner — TextGuided 场景 token 选择（最低成本验证）

- 仅修改 `_process_vision_features_with_scene`：添加 instruction-scene token 相似度计算和 top-k 选择
- 新增代码量：~50 行
- 验证目标：16 个 scene token 能否达到 48 个的性能

### Step 2: SH-Fuser — Cross-Modal 交叉注意力（核心融合升级）

- 添加一个 `CrossModalFusion` 模块替代简单 concat
- 新增代码量：~150 行
- 验证目标：交叉注意力是否优于简单拼接

### Step 3: SD-Pruner — 视觉侧剪枝（效率优化）

- 扩展到 visual tokens 的语义选择
- 新增代码量：~100 行
- 验证目标：大幅减少训练/推理成本的同时保持/提升精度

### Step 4: 完整系统集成

- MoE Gating 动态路由 + 完整 SD-Pruner + SH-Fuser + Bi-Action Attention
- 验证目标：论文 Table 3 的 +1.4% 能否在压缩 5× token 的同时保持

---

## 9. 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|---------|
| Scene token 48 已极少，进一步剪枝收益不大 | 中 | 低 | 主要节省来自视觉侧 (768→128)；scene token 剪枝设为可选项 |
| Cross-modal attention 需要 2D-3D 空间对齐 | 高 | 中 | 使用 VGGT 的 camera pose 建立像素→3D 映射；可降级为"无空间偏置的通用 cross-attention" |
| 新增参数导致训练不稳定 | 中 | 中 | 分阶段训练：先冻住 SH-Fuser 训 LoRA，再联合训练 |
| 天花板效应：98.5% 以上每 0.1% 都困难 | 高 | 低 | 效率提升本身已是重要贡献（3× 训练加速 + 2.5× 推理加速） |
| Bi-Action Attention 与 L1 Regression Head 的兼容性 | 低 | 低 | 两个组件在 SemanticVLA 中已验证兼容 |

---

## 10. 关键设计决策对比

| 决策点 | 当前方案 | 集成方案 | 变更理由 |
|--------|---------|---------|---------|
| 2D-3D 融合方式 | Concat | Cross-Attn + MoE | 显式跨模态对齐 > 隐式 LLM 学习 |
| 视觉 token 数量 | 768 (固定) | ~128 (语义可选) | 大部分 patch 是背景噪声 |
| 几何 token 数量 | 48 (固定) | 16~48 (语义可选) | 任务相关性筛选 |
| VGGT-Omega 训练 | 冻结 | 冻结 (FiLM 可选训练) | 保持几何先验完整性 |
| Action token 可见性 | Causal | 双向 | 物理约束协调 |
| Token 注入位置 | BOS 之后拼接 | BOS 之后拼接 | 保持不变，改动在融合器内部 |

---

## 11. 增量参数与显存估算

| 组件 | 新增参数量 | 新增显存 (A100 80GB) |
|------|-----------|---------------------|
| TextGuidedSampler (视觉+几何) | ~3M | ~0.5 GB |
| Cross-Modal Attention (2层) | ~12M | ~3 GB |
| MoE Router | ~3M | ~0.5 GB |
| FiLM 调制层 (可选) | ~2M | ~1 GB |
| **合计** | **~20M** | **~5 GB** |

加上原始方案的 ~53 GB，总计约 **58 GB**，单张 A100 (80GB) 仍可运行 batch_size=1~2。

---

## 参考文献

1. VGGT-Omega: Visual Geometry Grounded Transformer, Section 4.4 (Scene Token Fusion)
2. SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation, AAAI 2026 Oral
3. OpenVLA-OFT: OpenVLA with Open-world Finetuning
