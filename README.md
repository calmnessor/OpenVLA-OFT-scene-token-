# VGGT-Ω + OpenVLA-OFT: 3D Geometry-Aware Vision-Language-Action Policy

Reproduction of VGGT-Ω Section 4.4: integrating frozen VGGT-Ω scene tokens (register tokens) as a 3D geometry prior into OpenVLA-OFT for robot manipulation.

## Motivation

VLA models struggle with **spatial reasoning** — they see 2D pixels but need to reason in 3D. VGGT-Ω learns rich 3D geometric representations from multi-view reconstruction. We inject VGGT-Ω's 3D knowledge into OpenVLA-OFT by concatenating scene register tokens with vision patch embeddings via a lightweight linear projector.

## Architecture

```
VGGT-Ω 1B (Frozen)                  OpenVLA-7B (LoRA rank=32)
     │                                       │
Images @ 512px                        Images @ 224px
     │                                       │
  DINOv3 ViT                         SigLIP + DINOv2
     │                                       │
Register Tokens                       Patch Tokens
 [B, N×16, 2048]                      [B, 768, 4096]
     │                                       │
SceneProjector                             concat ← Proprio [1, 4096]
 Linear(2048→4096)                              │
 (pure Linear, no norm)                    concat ← Scene Tokens [B, N×16, 4096]
     │                                       │
 [B, N×16, 4096] ────────────────────────────┘
     │
     ▼
 Llama-7B (LoRA) → L1RegressionActionHead → 7-DOF × 8-step chunk
```

**Key design choice**: SceneProjector uses a pure `nn.Linear(2048, 4096, bias=True)` without LayerNorm. VGGT-Ω tokens are already well-normalized, and the LLM's internal RMSNorm handles any remaining distribution mismatch. LayerNorm in bfloat16 cannot receive training updates at value 1.0 (precision trap).

> Per VGGT-Ω (Section 4.4): *"Given the input images, we extract registers (scene tokens) from VGGT-Ω and concatenate them with the original OpenVLA-OFT input tokens."*

## Results

### LIBERO-Spatial (10 tasks × 50 trials = 500 episodes)

| | Success Rate |
|---|---|
| OpenVLA-OFT (paper baseline) | 97.6% |
| VGGT-Ω + OpenVLA-OFT (paper) | **99.3%** |
| **This reproduction (10K steps)** | **92.8%** |

Per-task breakdown:

| Task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|------|---|---|---|---|---|---|---|---|---|----|
| Rate | 100% | 96% | 100% | 94% | 98% | 48% | 100% | 96% | 96% | 100% |

- 9/10 tasks ≥ 94%, 4 tasks at perfect 100%
- Task 6 is the only bottleneck (48%); overall average still 92.8%
- Training only 10K optimizer steps (~5% of paper's 200K); further training should close the gap

## Key Implementation Details

| Component | Detail |
|-----------|--------|
| Base VLA | OpenVLA-7B (Prismatic VLM + Llama-7B), OFT pre-trained checkpoint |
| Geometry Encoder | VGGT-Ω 1B @ 512px, 16 register tokens/frame, **frozen** |
| Scene Projector | `Linear(2048→4096, bias=True)`, ~8.4M params, trainable |
| LLM Fine-tuning | LoRA rank=32, target: q/k/v/o/gate/up/down_proj |
| Action Head | L1RegressionActionHead (2-block MLP ResNet), loaded from OFT 150K checkpoint |
| Input | 2 camera views @ 224px (VLA) + 2 views @ 512px (VGGT) + proprio + instruction |
| Precision | bfloat16 |
| Effective Batch | 8 (batch_size=1 × grad_accum=8) |
| Learning Rate | 5e-4, no warmup, MultiStepLR (milestone=100K, gamma=0.1) |
| Optimizer | AdamW |

## Training

```bash
cd openvla-oft
python vla-scripts/train_vggt_launch.py
```

Config is defined in `train_vggt_launch.py`:

```python
vla_path = "/root/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
vggt_checkpoint = "/root/checkpoints/vggt_omega_1b_512/vggt_omega_1b_512.pt"
dataset_name = "libero_spatial_no_noops"
use_scene_tokens = True
use_l1_regression = True
use_proprio = True
num_images_in_input = 2
lora_rank = 32
lora_target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
merge_lora_during_training = False  # Must be False to avoid OOM
learning_rate = 5e-4
lr_warmup_steps = 0
max_steps = 10000
grad_accumulation_steps = 8
batch_size = 1
resume = True  # Load pre-trained action head from OFT checkpoint
resume_step = 150000
torch_dtype = "bfloat16"
```

**Hardware**: RTX 4090 48GB GPU + 15GB RAM. With `merge_lora_during_training=False` and `shuffle_buffer_size=20K`, peak RAM usage stays within limits.

## Evaluation

```bash
cd openvla-oft
python vla-scripts/run_eval_vggt.py
```

Evaluation config:
- LIBERO-Spatial, 10 tasks × 50 trials
- scene tokens enabled, 2 views @ 256px
- 8-step open-loop, center crop
- Uses merged model at `runs/vggt-eval-checkpoint/`

## Merging LoRA for Evaluation

```bash
python vla-scripts/merge_for_eval.py
```

**Important**: After merging, verify that `scene_projector.projector.bias` is non-zero. The merge script must restore trained SceneProjector weights from `scene_projector--latest_checkpoint.pt` — if the base model checkpoint lacks SceneProjector weights, they are initialized to zero and the merge will silently lose trained scene projection parameters, causing severe performance degradation (observed drop: 92.8% → 32.2%).

## Core Files

```
openvla-oft/
├── vla-scripts/
│   ├── train_vggt_launch.py          # Training entry point
│   ├── finetune.py                   # Training loop (modified for scene tokens)
│   ├── run_eval_vggt.py              # LIBERO-Spatial evaluation
│   └── merge_for_eval.py             # LoRA merge + SceneProjector restore
├── prismatic/
│   ├── extern/hf/modeling_prismatic.py  # VLA forward with scene token concat
│   └── models/scene_projector.py        # Linear(2048→4096) scene projector
└── experiments/robot/
    ├── robot_utils.py                   # VGGT-Ω scene token extraction
    └── openvla_utils.py                 # VLA inference utilities
```

## Dependencies

- PyTorch 2.5.0, CUDA 12.4, bfloat16
- OpenVLA-OFT: [github.com/moojink/openvla-oft](https://github.com/moojink/openvla-oft)
- VGGT-Ω: [github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- LIBERO: [github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- transformers 4.44.2, peft, robosuite, mujoco

## Known Issues

1. **bfloat16 LayerNorm precision trap**: At value 1.0, bfloat16 resolution is ~0.0078. Gradients for LayerNorm weights fall below this threshold, preventing training. Solution: remove LayerNorm from SceneProjector (pure Linear).

2. **Merge corruption**: `merge_for_eval.py` must restore trained SceneProjector weights from the training checkpoint. If skipped, the merged model uses randomly initialized SceneProjector weights and success rate drops to ~30%.

3. **OOM during merge**: `merge_lora_during_training=True` loads the full 7B model on CPU during checkpoint save, doubling memory. Keep it `False` on 15GB RAM machines.

## Acknowledgments

Based on [VGGT-Ω](https://vggt-omega.github.io/) (Meta FAIR) and [OpenVLA-OFT](https://openvla-oft.github.io/) (Stanford/MIT).

## License

This is a multi-license repository:

- Original integration work is available under the MIT terms described in
  [`LICENSE`](LICENSE).
- OpenVLA-OFT-derived portions retain the upstream MIT license and copyright
  notice in [`LICENSES/OpenVLA-OFT-MIT.txt`](LICENSES/OpenVLA-OFT-MIT.txt).
- VGGT-derived materials are governed by the VGGT License and Acceptable Use
  Policy in [`LICENSES/VGGT-LICENSE.txt`](LICENSES/VGGT-LICENSE.txt); they are
  not relicensed under MIT.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and
license scope details.
