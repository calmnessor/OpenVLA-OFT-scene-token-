# VGGT-Ω Scene Token + OpenVLA-OFT 融合复现框架

## 1. 论文依据

VGGT-Ω Section 4.4 已给出融合方案和实验结果：

> "Given the input images, we extract registers (scene tokens) from VGGT-Ω and concatenate them with the original OpenVLA-OFT input tokens."

| Method | Spatial | Object | Goal | Long | **Avg** |
|--------|---------|--------|------|------|---------|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| **+ Frozen Scene Tokens** | **99.3** | **99.2** | **99.0** | **96.7** | **98.5** |

---

## 2. VGGT-Ω Scene Token 提取

### 2.1 数据流

```
输入图像 I₁, I₂, I₃  (3张RGB, 各 H×W×3)
        │
        ▼
DINOv3 ViT (frozen) → Patch Embedding
        │
        ▼
每帧拼接: [Camera_Token(1) | Register_Tokens(16) | Patch_Tokens(H'W')]
  总计: [N, 1+16+H'W', 1024]   (N=帧数)
        │
        ▼
交替Attention (L层):
  ├── Frame Attention: 帧内自注意力 (所有token)
  ├── Global Attention (75%层): 跨帧全token交互
  └── Register Attention (25%层): 仅16个register跨帧交互
        │
        ▼
输出 tokens [N, 1+16+H'W', 1024]
        │
        ▼
前17个token: camera_and_register_tokens [N, 17, 1024]
  ├── [:, :1, :]  → Camera Token  (用于预测相机参数)
  └── [:, 1:, :]  → Scene Tokens  (Register, 16个) ← 我们要用这个
```

### 2.2 关键Shape

```python
# VGGT-Ω forward 输出
predictions = vggt_omega(images)  # images: [B, N, 3, H, W]

# 提取 registers (排除camera token)
camera_and_reg = predictions["camera_and_register_tokens"]  # [B, N, 17, 1024]
registers = camera_and_reg[:, :, 1:, :]                      # [B, N, 16, 1024]

# 展平为统一序列
scene_tokens = registers.reshape(B, N * 16, 1024)  # [B, 48, 1024]  (N=3时)
```

### 2.3 VGGT-Ω 配置

- 使用预训练的 `VGGT-Omega-1B-512` checkpoint
- **完全冻结**（`torch.no_grad()` + `model.eval()`）
- 图像分辨率: 512 (与2D重建任务一致)
- 不使用camera head和depth head（推理时可禁用节省显存）

---

## 3. 融合架构设计

### 3.1 总体数据流

```
┌─────────────────────────────────────────────────────────┐
│                  VGGT-Ω (Frozen)                         │
│                                                         │
│  Images [B, 3, 3, H, W]                                 │
│      │                                                  │
│      ▼                                                  │
│  DINOv3 → Alt-Attn → registers [B, 3, 16, 1024]        │
│                                                         │
│  flatten → scene_tokens [B, 48, 1024]                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   Scene Projector   │   ← 可训练
            │  Linear(1024→4096)  │     (llm_dim=4096 for Llama-7B)
            └─────────┬───────────┘
                      │
                      ▼
            scene_embeddings [B, 48, 4096]
                       │
                       │
┌──────────────────────┼──────────────────────────────────┐
│  OpenVLA-OFT (LoRA Trainable)                           │
│                                                         │
│  Images [B, 3, 3, 224, 224]  ← 原始VLA输入分辨率        │
│      │                                                  │
│      ▼                                                  │
│  SigLIP + DINOv2 → patch_tokens [B, 256*3, D_vis]      │
│      │                                                  │
│      ▼                                                  │
│  Projector → projected_patches [B, 768, 4096]           │
│      │                                                  │
│      │  ← concat scene_embeddings                       │
│      ▼                                                  │
│  [projected_patches | scene_embeddings]                 │
│   [B, 768, 4096]     [B, 48, 4096]                     │
│      → [B, 816, 4096]                                   │
│      │                                                  │
│      │  ← concat proprio token                          │
│      ▼                                                  │
│  [B, 817, 4096]  ← 插入到 [BOS] 之后                    │
│      │                                                  │
│      ▼                                                  │
│  Llama-7B (LoRA rank=32) → hidden_states               │
│      │                                                  │
│      ▼                                                  │
│  Action Head (L1 Regression) → 连续动作序列              │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Token维度对比

| 组件 | Token数量 | 维度 |
|------|----------|------|
| OpenVLA-OFT patch tokens | 768 (256×3) | 4096 (llm_dim) |
| VGGT-Ω scene tokens | 48 (16×3) | 1024 → 4096 (投影后) |
| **融合后 vision tokens** | **816** | 4096 |
| Proprio token | 1 | 4096 |
| **注入LLM的总vision tokens** | **817** | 4096 |

增量开销: +48 tokens (仅增加约6%的视觉token，但携带全局3D几何信息)

---

## 4. 代码改动清单

### 4.1 新增文件

```
openvla-oft/
└── prismatic/
    └── models/
        └── scene_projector.py          # NEW: Scene Projector 模块
```

```python
# prismatic/models/scene_projector.py
import torch.nn as nn

class SceneProjector(nn.Module):
    """Project VGGT-Ω scene tokens (register) into LLM embedding space."""
    
    def __init__(self, scene_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(scene_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )
    
    def forward(self, scene_tokens):
        """
        Args:
            scene_tokens: [B, N*16, 1024]  flattened registers from VGGT-Ω
        Returns:
            [B, N*16, llm_dim]
        """
        return self.projector(scene_tokens)
```

### 4.2 修改文件

#### 文件1: `prismatic/extern/hf/modeling_prismatic.py`

**修改点1**: `OpenVLAForActionPrediction.__init__` — 添加 scene_projector

```python
# 在 __init__ 中添加 (~line 80附近)
from prismatic.models.scene_projector import SceneProjector

class OpenVLAForActionPrediction(PrismaticPreTrainedModel):
    def __init__(self, config: OpenVLAConfig):
        # ... 现有初始化代码 ...
        
        # NEW: Scene projector for VGGT-Ω scene tokens
        self.scene_projector = SceneProjector(
            scene_dim=1024,
            llm_dim=self.llm_dim  # 4096 for Llama-7B
        )
```

**修改点2**: `_process_vision_features` → `_process_vision_features_with_scene`

```python
# 新增方法 (~line 438之后)
def _process_vision_features_with_scene(
    self, pixel_values, scene_tokens, language_embeddings=None, use_film=False
):
    """Process vision features with optional FiLM + VGGT-Ω scene tokens"""
    # 原始vision处理
    if use_film:
        patch_features = self.vision_backbone(pixel_values, language_embeddings)
    else:
        patch_features = self.vision_backbone(pixel_values)
    
    projected_patch_embeddings = self.projector(patch_features)
    # [B, N*256, llm_dim]
    
    # 注入VGGT-Ω scene tokens
    if scene_tokens is not None:
        scene_embeddings = self.scene_projector(scene_tokens)
        # [B, N*16, llm_dim]
        projected_patch_embeddings = torch.cat(
            [projected_patch_embeddings, scene_embeddings], dim=1
        )
        # [B, N*272, llm_dim]
    
    return projected_patch_embeddings
```

**修改点3**: `forward()` — 添加 scene_tokens 参数

```python
# 修改 forward 签名 (~line 499)
def forward(
    self,
    input_ids=None,
    attention_mask=None,
    pixel_values=None,
    labels=None,
    inputs_embeds=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    output_projector_features=None,
    return_dict=None,
    proprio=None,
    proprio_projector=None,
    noisy_actions=None,
    noisy_action_projector=None,
    diffusion_timestep_embeddings=None,
    use_film=False,
    scene_tokens=None,          # NEW: [B, N*16, 1024]
):
    # ... 现有代码 ...
    
    # 修改 line 586: 替换 _process_vision_features 调用
    # 原始:
    # projected_patch_embeddings = self._process_vision_features(
    #     pixel_values, language_embeddings, use_film
    # )
    
    # 改为:
    projected_patch_embeddings = self._process_vision_features_with_scene(
        pixel_values, scene_tokens, language_embeddings, use_film
    )
    
    # ... 后续代码不变 (proprio/action/label处理自动适配新token数) ...
```

#### 文件2: `vla-scripts/finetune.py`

**修改点1**: 添加 VGGT-Ω 模型加载

```python
# 在配置类和训练函数之间 (~line 90后)
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images

class VGGTSceneExtractor:
    """封装VGGT-Ω的scene token提取"""
    def __init__(self, checkpoint_path, device="cuda"):
        self.model = VGGTOmega(enable_camera=False, enable_depth=False).to(device).eval()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.device = device
        
        # 冻结所有参数
        for p in self.model.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def extract_scene_tokens(self, images_512):
        """
        Args:
            images_512: [B, N, 3, 512, 512]  已经resize到512的图片
        Returns:
            scene_tokens: [B, N*16, 1024]
        """
        predictions = self.model(images_512.to(self.device))
        camera_and_reg = predictions["camera_and_register_tokens"]
        registers = camera_and_reg[:, :, 1:, :]  # [B, N, 16, 1024]
        return registers.reshape(registers.shape[0], -1, registers.shape[-1])
```

**修改点2**: BatchTransform — 额外输出512分辨率图片

```python
# 在 RLDSBatchTransform.__call__ 中
# 除了224的pixel_values外，额外输出512分辨率的图片用于VGGT-Ω
def __call__(self, rlds_batch):
    # ... 现有代码 ...
    
    # 为VGGT-Ω准备512分辨率图片 (使用不同的image_transform)
    vggt_images = []
    # primary image @ 512
    vggt_primary = self.vggt_image_transform(
        Image.fromarray(rlds_batch["observation"]["image_primary"][0])
    )
    vggt_images.append(vggt_primary)
    
    # wrist images @ 512
    if self.use_wrist_image:
        for k in rlds_batch["observation"].keys():
            if "wrist" in k:
                vggt_wrist = self.vggt_image_transform(
                    Image.fromarray(rlds_batch["observation"][k][0])
                )
                vggt_images.append(vggt_wrist)
    
    return_dict["vggt_images"] = torch.stack(vggt_images)  # [N, 3, 512, 512]
    
    return return_dict
```

**修改点3**: 训练循环 — 提取scene tokens并传入模型

```python
# 在训练step中 (~line 393)
for batch in train_dataloader:
    # 提取VGGT-Ω scene tokens (离线或在线)
    vggt_images = batch["vggt_images"].to(device_id)
    scene_tokens = vggt_extractor.extract_scene_tokens(vggt_images)
    
    output = vla(
        input_ids=batch["input_ids"].to(device_id),
        attention_mask=batch["attention_mask"].to(device_id),
        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
        labels=batch["labels"],
        output_hidden_states=True,
        proprio=batch["proprio"] if use_proprio else None,
        proprio_projector=proprio_projector,
        scene_tokens=scene_tokens,  # NEW
    )
```

#### 文件3: `prismatic/vla/datasets/datasets.py`

在 `RLDSBatchTransform.__init__` 中添加第二个image_transform（用于512分辨率）:

```python
@dataclass
class RLDSBatchTransform:
    # ... 现有字段 ...
    vggt_image_transform: ImageTransform = None  # NEW: 512分辨率用
```

---

## 5. 训练配置

```bash
# 与OpenVLA-OFT标准训练一致，额外参数：
--scene_token_dim 1024          # VGGT-Ω register的embedding维度
--vggt_checkpoint /path/to/vggt_omega_1b_512.pt
--vggt_image_resolution 512     # VGGT-Ω输入分辨率
```

训练超参保持与OpenVLA-OFT一致：
- LoRA rank=32
- Learning rate=5e-4
- Batch size=8 per GPU
- Max steps=150K, decay at 100K
- L1 regression action head

---

## 6. 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| VGGT-Ω是否训练 | **冻结** | 论文实验支持; 避免灾难性遗忘; 节省显存 |
| Scene Token注入位置 | **patch token之后拼接** | 论文方案; 实现简单; 不破坏LLM的position encoding |
| VGGT图像分辨率 | **512** (与预训练一致) | VGGT-Ω在512分辨率训练, 改变分辨率可能影响token质量 |
| Scene Projector结构 | **Linear + LayerNorm** | 最简设计; 论文暗示简单拼接即可, projector只是维度对齐 |
| 训练时VGGT-Ω推理 | **在线** (每个batch实时提取) | 避免存储大量中间特征; GPU消耗可控(单batch推理) |
| 优化: 离线预提取 | 可选 | 如果显存紧张, 可预先对全量数据集提取scene tokens存盘 |

---

## 7. 显存估算

| 组件 | 显存 (A100 80GB) |
|------|------------------|
| VGGT-Ω-1B (frozen, 3 frames) | ~8 GB |
| OpenVLA-7B (LoRA, bf16) | ~25 GB |
| Scene Projector (可忽略) | ~0.01 GB |
| 训练激活值 (batch=1) | ~20 GB |
| **总计** | ~53 GB |

单张A100 (80GB) 可以跑 batch_size=2~4。

---

## 8. 预期效果

基于论文Table 3，在LIBERO上预期提升：

- Spatial: 97.6% → ~99.3% (+1.7%)
- Object:  98.4% → ~99.2% (+0.8%)
- Goal:    97.9% → ~99.0% (+1.1%)
- Long:    94.5% → ~96.7% (+2.2%)
- **Avg:    97.1% → ~98.5% (+1.4%)**

Long任务提升最大（+2.2%），说明3D几何先验对长时序空间推理帮助最大。

---

## 9. 潜在扩展方向

1. **多尺度Scene Token**: 提取不同层的register token (浅层几何/深层语义)
2. **Cross-Attention注入**: 不用简单拼接，而是让LLM通过cross-attention query scene tokens
3. **Scene-Aware Action Head**: Scene tokens不只注入LLM，也条件化Action Head
4. **端到端微调VGGT-Ω**: 用VLA的L1 loss回传梯度到VGGT (显存代价大)
5. **与SemanticVLA结合**: Scene tokens作为额外的"几何锚点"辅助SD-Pruner的剪枝决策
