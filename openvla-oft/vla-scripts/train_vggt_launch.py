import sys, os
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/root/openvla-oft")
sys.path.insert(0, "/root/openvla-oft/vla-scripts")

import finetune as ft
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Cfg(ft.FinetuneConfig):
    vla_path: str = "/root/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
    data_root_dir: Path = Path("/root/datasets/rlds")
    dataset_name: str = "libero_spatial_no_noops"
    run_root_dir: Path = Path("runs")
    shuffle_buffer_size: int = 20_000
    use_scene_tokens: bool = True
    vggt_checkpoint: str = "/root/checkpoints/vggt_omega_1b_512/vggt_omega_1b_512.pt"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    use_proprio: bool = True
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"  # Explicit: avoid PrismaticProjector
    merge_lora_during_training: bool = False
    batch_size: int = 1           # Reduced from 8 due to OOM
    grad_accumulation_steps: int = 8  # Effective batch = 8
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100_000
    max_steps: int = 10000
    save_freq: int = 2000
    save_latest_checkpoint_only: bool = True
    torch_dtype: str = "bfloat16"
    load_in_8bit: bool = False
    resume: bool = True
    resume_step: int = 150000
    image_aug: bool = True
    wandb_entity: str = "suziyang63-zzy"
    wandb_project: str = "openvla-oft-vggt"
    run_id_note: Optional[str] = "vggt-scene-tokens-bs1-ga8"
    use_val_set: bool = False

ft.finetune.__wrapped__(Cfg())
