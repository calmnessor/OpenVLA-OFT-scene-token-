"""Apply all VGGT+OpenVLA-OFT changes to finetune.py"""
from pathlib import Path

path = Path("Afford+VLA/scene token+OpenVLA-OFT/openvla-oft/vla-scripts/finetune.py")
content = path.read_text()

# 1. Replace all bfloat16 -> float16
content = content.replace("torch.bfloat16", "torch.float16")

# 2. low_cpu_mem_usage=True -> False
content = content.replace("low_cpu_mem_usage=True", "low_cpu_mem_usage=False")

# 3. Add SceneProjector import
content = content.replace(
    "from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead",
    "from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead\nfrom prismatic.models.scene_projector import SceneProjector"
)

# 4. Add VGGTSceneExtractor class before @dataclass
vggt_class = '''
# === VGGT-Omega Scene Token Extractor ===
class VGGTSceneExtractor:
    """Extract scene tokens (registers) from frozen VGGT-Omega for VLA training.

    VGGT-Omega outputs 16 register tokens per input frame that encode global
    3D geometry through alternating attention across views. These serve as a
    spatial prior for the VLA policy without any finetuning of VGGT-Omega.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        import sys
        vggt_path = Path(checkpoint_path).parent
        if str(vggt_path) not in sys.path:
            sys.path.insert(0, str(vggt_path))

        from vggt_omega.models import VGGTOmega

        self.device = device
        self.model = VGGTOmega(
            enable_camera=False,
            enable_depth=False,
            enable_alignment=False,
        ).to(device).eval()

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model_keys = set(self.model.state_dict().keys())
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        missing = model_keys - set(filtered_dict.keys())
        skipped = len(state_dict) - len(filtered_dict)
        if missing:
            print(f"[VGGTSceneExtractor] Missing keys: {missing}")
        if skipped:
            print(f"[VGGTSceneExtractor] Skipped {skipped} keys not in model (e.g. camera_head, dense_head)")
        self.model.load_state_dict(filtered_dict, strict=True)

        for p in self.model.parameters():
            p.requires_grad = False

        print(f"[VGGTSceneExtractor] Loaded from {checkpoint_path}, "
              f"params frozen: {sum(p.numel() for p in self.model.parameters())}")

    @torch.no_grad()
    def extract_scene_tokens(self, images_512: torch.Tensor) -> torch.Tensor:
        predictions = self.model(images_512.to(self.device, dtype=torch.float16))
        camera_and_reg = predictions["camera_and_register_tokens"]
        registers = camera_and_reg[:, :, 1:, :]
        B, N = registers.shape[0], registers.shape[1]
        return registers.reshape(B, N * 16, registers.shape[-1])

'''
content = content.replace("\n@dataclass\nclass FinetuneConfig:", vggt_class + "@dataclass\nclass FinetuneConfig:")

# 5. Add use_scene_tokens + vggt_checkpoint config
content = content.replace(
    "use_proprio: bool = False                        # If True, includes robot proprioceptive state in input",
    "use_proprio: bool = False                        # If True, includes robot proprioceptive state in input\n"
    "    use_scene_tokens: bool = False                   # If True, uses VGGT-Omega scene tokens as 3D geometry prior\n"
    "    vggt_checkpoint: Optional[str] = None             # Path to VGGT-Omega checkpoint file (.pt)"
)

# 6. Add scene_tokens param to run_forward_pass
content = content.replace(
    "    num_diffusion_steps_train=None,\n) -> Tuple[torch.Tensor, Dict[str, float]]:",
    "    num_diffusion_steps_train=None,\n    scene_tokens=None,\n) -> Tuple[torch.Tensor, Dict[str, float]]:"
)

# 7. Add scene_tokens to vla() call in run_forward_pass
content = content.replace(
    "            use_film=use_film,\n        )\n\n    # Get action masks needed for logging",
    "            use_film=use_film,\n            scene_tokens=scene_tokens,\n        )\n\n    # Get action masks needed for logging"
)

# 8. run_diffusion_sampling: add scene_tokens param
content = content.replace(
    "    use_film,\n) -> torch.Tensor:\n    \"\"\"\n    Run diffusion sampling",
    "    use_film,\n    scene_tokens=None,\n) -> torch.Tensor:\n    \"\"\"\n    Run diffusion sampling"
)

# 9. run_diffusion_sampling: add scene_tokens to vla() call
content = content.replace(
    "                use_film=use_film,\n            )\n            # Get last layer hidden states",
    "                use_film=use_film,\n                scene_tokens=scene_tokens,\n            )\n            # Get last layer hidden states"
)

# 10. save_training_checkpoint: add scene_projector save
content = content.replace(
    "        # Wait for model components to be saved\n    dist.barrier()",
    "        if cfg.use_scene_tokens:\n"
    "            torch.save(\n"
    "                vla.module.scene_projector.state_dict(),\n"
    "                checkpoint_dir / f\"scene_projector--{checkpoint_name_suffix}\"\n"
    "            )\n\n"
    "        # Wait for model components to be saved\n    dist.barrier()"
)

# 11. run_validation: add vggt_extractor param
content = content.replace(
    "    val_time_limit,\n) -> None:\n    \"\"\"\n    Compute validation",
    "    val_time_limit,\n    vggt_extractor=None,\n) -> None:\n    \"\"\"\n    Compute validation"
)

# 12. run_validation: add scene token extraction
content = content.replace(
    "    with torch.no_grad():\n        for batch in val_dataloader:\n            # Always compute L1 loss",
    "    with torch.no_grad():\n        for batch in val_dataloader:\n"
    "            # [VGGT-Omega] Extract scene tokens for validation\n"
    "            if cfg.use_scene_tokens and vggt_extractor is not None:\n"
    "                vggt_images = batch[\"vggt_images\"].to(device_id)\n"
    "                scene_tokens = vggt_extractor.extract_scene_tokens(vggt_images)\n"
    "            else:\n"
    "                scene_tokens = None\n\n"
    "            # Always compute L1 loss"
)

# 13. run_validation: add scene_tokens to run_forward_pass call
content = content.replace(
    "                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,\n"
    "            )\n\n            # Add the loss value",
    "                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,\n"
    "                scene_tokens=scene_tokens,\n"
    "            )\n\n            # Add the loss value"
)

# 14. finetune(): replace scene_projector after model loading
old_load = """    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    ).to(device_id)

    # Set number of images in VLA input"""
new_load = """    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    ).to(device_id)

    # Replace scene_projector with correct dimensions (HF cache may have wrong dims from remote)
    vla.scene_projector = SceneProjector(scene_dim=2048, llm_dim=vla.llm_dim).to(device_id, dtype=torch.float16)

    # Set number of images in VLA input"""
content = content.replace(old_load, new_load)

# 15. Add action_head = None initialization
content = content.replace(
    "\n    # If applicable, instantiate continuous action head for L1 regression",
    "\n    action_head = None\n    noisy_action_projector = None\n\n    # If applicable, instantiate continuous action head for L1 regression"
)

# 16. Add NUM_SCENE_TOKENS
old_patches = """    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1"""
new_patches = """    # For VGGT-Omega scene tokens, add N*16 tokens (16 register tokens per input frame)
    if cfg.use_scene_tokens:
        NUM_SCENE_TOKENS = cfg.num_images_in_input * 16
        NUM_PATCHES += NUM_SCENE_TOKENS
    else:
        NUM_SCENE_TOKENS = 0
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1"""
content = content.replace(old_patches, new_patches)

# 17. Add VGGT extractor init
content = content.replace(
    "\n    # Instantiate optimizer",
    "\n    # [VGGT-Omega] Initialize scene token extractor (frozen, no gradient)\n"
    "    if cfg.use_scene_tokens:\n"
    '        assert cfg.vggt_checkpoint is not None, "Must provide --vggt_checkpoint when --use_scene_tokens=True!"\n'
    "        vggt_extractor = VGGTSceneExtractor(cfg.vggt_checkpoint, device=f\"cuda:{device_id}\")\n"
    "    else:\n"
    "        vggt_extractor = None\n"
    "\n    # Instantiate optimizer"
)

# 18. Add VGGT image transform
old_wrist = """    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # Create training and optional validation datasets"""
new_wrist = """    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # [VGGT-Omega] Create 512-resolution image transform for scene token extraction
    if cfg.use_scene_tokens:
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode
        vggt_image_transform = Compose([
            Resize(512, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(512),
            ToTensor(),
        ])
    else:
        vggt_image_transform = None

    # Create training and optional validation datasets"""
content = content.replace(old_wrist, new_wrist)

# 19. Add vggt_image_transform to batch_transform
content = content.replace(
    "        use_proprio=cfg.use_proprio,\n    )\n    train_dataset",
    "        use_proprio=cfg.use_proprio,\n        vggt_image_transform=vggt_image_transform,\n    )\n    train_dataset"
)

# 20. Add scene token extraction in training loop
old_loop = """        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute training metrics and loss"""
new_loop = """        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # [VGGT-Omega] Extract scene tokens from 512-resolution images
            if cfg.use_scene_tokens and vggt_extractor is not None:
                vggt_images = batch["vggt_images"].to(device_id)
                scene_tokens = vggt_extractor.extract_scene_tokens(vggt_images)
            else:
                scene_tokens = None

            # Compute training metrics and loss"""
content = content.replace(old_loop, new_loop)

# 21. Add scene_tokens to training run_forward_pass call
content = content.replace(
    "                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,\n"
    "            )\n\n            # Normalize loss to account for gradient accumulation",
    "                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,\n"
    "                scene_tokens=scene_tokens,\n"
    "            )\n\n            # Normalize loss to account for gradient accumulation"
)

# 22. Add CSV logging
content = content.replace(
    "            # Push Metrics to W&B (every wandb_log_freq gradient steps)",
    "            # Save metrics to local CSV log\n"
    "            csv_log_path = run_dir / \"training_metrics.csv\"\n"
    "            if gradient_step_idx == 1 and not csv_log_path.exists():\n"
    "                with open(csv_log_path, \"w\") as f:\n"
    '                    f.write(",".join(sorted(smoothened_metrics.keys())) + "\\n")\n'
    "            with open(csv_log_path, \"a\") as f:\n"
    '                f.write(",".join(str(smoothened_metrics[k]) for k in sorted(smoothened_metrics.keys())) + "\\n")\n'
    "\n            # Push Metrics to W&B (every wandb_log_freq gradient steps)"
)

# 23. Add vggt_extractor to validation call
content = content.replace(
    "                    val_time_limit=cfg.val_time_limit,\n                )\n                # Set model back to training mode",
    "                    val_time_limit=cfg.val_time_limit,\n                    vggt_extractor=vggt_extractor,\n                )\n                # Set model back to training mode"
)

# 24. Fix docstring
content = content.replace("torch.bfloat16 data type", "torch.float16 data type")

path.write_text(content)
print("All patches applied successfully!")
print(f"File size: {len(content)} chars")
