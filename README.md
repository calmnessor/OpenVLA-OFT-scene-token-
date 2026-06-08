# VGGT-Ω + OpenVLA-OFT: 3D Geometry-Aware Vision-Language-Action Policy

Reproduction of the VGGT-Ω Section 4.4 experiment: integrating frozen VGGT-Ω scene tokens as a 3D geometry prior into OpenVLA-OFT for robot manipulation.

## Motivation

Vision-Language-Action (VLA) models struggle with **spatial reasoning** — they see 2D pixels but need to reason in 3D. VGGT-Ω learns rich 3D geometric representations from multi-view reconstruction. The idea: inject VGGT-Ω's learned 3D knowledge into OpenVLA-OFT's action prediction pipeline via a lightweight projector, without fine-tuning the geometry model.

## Architecture

```
VGGT-Ω 1B (Frozen)                  OpenVLA-7B (LoRA rank=32)
     │                                       │
Images @ 512px                        Images @ 224px
     │                                       │
  DINOv3 ViT                         SigLIP + DINOv2
     │                                       │
Register Tokens                       Patch Tokens
 [B, N×16, 1024]                      [B, 768, 4096]
     │                                       │
SceneProjector                             concat ← Proprio [1, 4096]
 Linear(1024→4096)                              │
 LayerNorm                                 concat ← Scene Tokens [B, N×16, 4096]
     │                                       │
 [B, N×16, 4096] ────────────────────────────┘
     │
     ▼
 Llama-7B (LoRA) → L1 Action Head → 7-DOF Continuous Actions × 8-step Chunk
```

**Per VGGT-Ω paper (Section 4.4):**
> "Given the input images, we extract registers (scene tokens) from VGGT-Ω and concatenate them with the original OpenVLA-OFT input tokens."

## Key Implementation Details

| Component | Detail |
|-----------|--------|
| Base VLA | OpenVLA-7B (Prismatic VLM + Llama-7B) |
| Geometry Encoder | VGGT-Ω 1B, 16 register tokens/frame, **frozen** |
| Scene Projector | `Linear(1024→4096) + LayerNorm`, ~8M params, trainable |
| LLM Fine-tuning | LoRA rank=32 (~111M trainable params) |
| Action Head | L1 Regression (continuous 7-DOF actions) |
| Input | 3 camera views @ 224px (VLA) + 3 views @ 512px (VGGT) + language instruction |
| Precision | bfloat16 |
| Dataset | LIBERO (Spatial / Object / Goal / 10) in RLDS format |

## Core Files

```
openvla-oft/
├── vla-scripts/finetune.py                         # Training entry point
├── prismatic/
│   ├── extern/hf/modeling_prismatic.py             # VLA forward with scene token integration
│   └── models/scene_projector.py                   # Linear + LayerNorm scene projector
└── experiments/robot/libero/run_libero_eval.py    # LIBERO evaluation

vggt-omega/
└── vggt_omega/models/                             # VGGT-Ω model (frozen, inference-only)
```

## Expected Results (from VGGT-Ω Table 3)

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| OpenVLA-OFT (baseline) | 97.6% | 98.4% | 97.9% | 94.5% | 97.1% |
| **+ VGGT-Ω Frozen Scene Tokens** | **99.3%** | **99.2%** | **99.0%** | **96.7%** | **98.5%** |

Largest gain on **Long tasks (+2.2%)**, indicating that 3D geometry priors most benefit long-horizon spatial reasoning.

## Training

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir ~/datasets/modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --use_l1_regression True \
    --use_scene_tokens True \
    --vggt_checkpoint ~/checkpoints/vggt_omega_1b_512/model.pt \
    --num_images_in_input 3 \
    --use_proprio True \
    --batch_size 2 \
    --grad_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --lora_rank 32 \
    --max_steps 150005
```

**Requirements:** 1× A100/A800 80GB or A100 40GB with `--load_in_8bit True`.

## Dependencies

- PyTorch 2.2.0, CUDA 12.1+, bfloat16 support
- OpenAI VLA: [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
- Geometry: [VGGT-Ω](https://github.com/facebookresearch/vggt) (checkpoint access required)
- Benchmark: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- Flash Attention 2

## Acknowledgments

Based on [VGGT-Ω](https://vggt-omega.github.io/) (Meta FAIR) and [OpenVLA-OFT](https://openvla-oft.github.io/) (Stanford/MIT).
