"""
Train OpenVLA-OFT with VGGT-Omega scene tokens on LIBERO-Spatial.
Starts from moojink pretrained weights (97.6%) and adds frozen VGGT scene tokens.
"""
import os
import sys
from pathlib import Path

# Add openvla-oft to path
sys.path.insert(0, "/root/openvla-oft")
sys.path.insert(0, "/root/openvla-oft/vla-scripts")

from dataclasses import dataclass
from typing import Optional

import finetune as ft


@dataclass
class LiberoSpatialVGGTConfig(ft.FinetuneConfig):
    """LIBERO-Spatial + VGGT-Omega scene tokens training config."""

    # Model: start from moojink pretrained weights (97.6% on LIBERO-Spatial)
    vla_path: str = "/root/checkpoints/openvla-7b-oft-finetuned-libero-spatial"

    # Dataset
    data_root_dir: Path = Path("/root/datasets/rlds")
    dataset_name: str = "libero_spatial_no_noops"
    run_root_dir: Path = Path("runs")
    shuffle_buffer_size: int = 100_000

    # VGGT-Omega scene tokens
    use_scene_tokens: bool = True
    vggt_checkpoint: str = "/root/checkpoints/vggt_omega_1b_512/vggt_omega_1b_512.pt"

    # OFT architecture (matching moojink training)
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    use_proprio: bool = True
    num_images_in_input: int = 2  # third-person + wrist

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = False  # Faster saving

    # Training
    batch_size: int = 8
    grad_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 1000
    num_steps_before_decay: int = 100_000
    max_steps: int = 10000
    
    # Checkpointing
    save_freq: int = 2000
    save_latest_checkpoint_only: bool = True
    
    # Precision
    torch_dtype: str = "bfloat16"
    load_in_8bit: bool = False
    
    # Resume from moojink action_head and proprio_projector
    resume: bool = True
    resume_step: int = 150000
    
    # Image augmentation
    image_aug: bool = True
    
    # Logging
    wandb_entity: str = ""
    wandb_project: str = ""
    run_id_note: Optional[str] = "vggt-scene-tokens-from-moojink"
    
    use_val_set: bool = False


if __name__ == "__main__":
    ft.finetune.__wrapped__(LiberoSpatialVGGTConfig())
