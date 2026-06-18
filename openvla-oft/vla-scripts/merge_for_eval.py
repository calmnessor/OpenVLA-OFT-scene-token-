"""
Merge LoRA adapter into base model for evaluation, preserving SceneProjector weights.

Key: The base model checkpoint (moojink) does NOT contain SceneProjector weights.
We must save scene_projector from the trained model BEFORE merging, then restore it
into the merged model. Otherwise trained scene_norm is lost.
"""
import os, sys, json, shutil
import torch
from transformers import AutoModelForVision2Seq
from peft import PeftModel

SRC_CHECKPOINT = "/root/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
LORA_ADAPTER = "/root/openvla-oft/runs/openvla-7b-oft-finetuned-libero-spatial/lora_adapter"
RUN_DIR = "/root/openvla-oft/runs/openvla-7b-oft-finetuned-libero-spatial"
EVAL_DIR = "/root/openvla-oft/runs/vggt-eval-checkpoint"


def check_and_fix_scene_norm(model):
    """Check scene_norm for NaN and fix if needed. Returns True if fix was applied."""
    sp = model.scene_projector.projector
    wn = sp.scene_norm.weight.data
    bn = sp.scene_norm.bias.data
    has_nan = torch.isnan(wn).any() or torch.isnan(bn).any()
    print(f"  scene_norm.weight: NaN={torch.isnan(wn).any().item()}, "
          f"min={wn.min():.6f}, max={wn.max():.6f}, mean={wn.mean():.6f}")
    print(f"  scene_norm.bias:   NaN={torch.isnan(bn).any().item()}, "
          f"min={bn.min():.6f}, max={bn.max():.6f}, mean={bn.mean():.6f}")
    if has_nan:
        print("  WARNING: scene_norm has NaN! Resetting to ones/zeros.")
        wn = torch.ones_like(wn)
        bn = torch.zeros_like(bn)
        sp.scene_norm.weight.data = wn
        sp.scene_norm.bias.data = bn
        return True
    return False


def main():
    print("Loading base model...")
    base_vla = AutoModelForVision2Seq.from_pretrained(
        SRC_CHECKPOINT,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    print(f"Base model loaded. Scene projector: {type(base_vla.scene_projector)}")

    print("Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_vla, LORA_ADAPTER)

    # Save scene_projector from PeftModel BEFORE merging
    # This captures trained scene_norm (and scene_linear base+LoRA for reference)
    scene_projector_state = None
    has_saved_scene = False
    scene_ckpt_path = os.path.join(RUN_DIR, "scene_projector--latest_checkpoint.pt")
    if os.path.exists(scene_ckpt_path):
        print(f"Loading saved scene_projector from {scene_ckpt_path}")
        scene_projector_state = torch.load(scene_ckpt_path, map_location="cpu")
        has_saved_scene = True
    else:
        print("No saved scene_projector checkpoint found — extracting from current model state...")
        # Extract from current PeftModel (scene_norm is a base param, NOT in LoRA adapter)
        scene_projector_state = {
            k: v.cpu().clone()
            for k, v in peft_model.scene_projector.state_dict().items()
        }

    print("Merging LoRA...")
    merged_vla = peft_model.merge_and_unload()

    print("Restoring scene_projector...")
    # Filter state to only restore non-LoRA params (scene_norm)
    # scene_linear base params in merged model are correct (base + lora_B @ lora_A)
    restore_keys = [k for k in scene_projector_state if "lora" not in k.lower()]
    restore_state = {k: scene_projector_state[k] for k in restore_keys}
    print(f"  Restoring keys: {list(restore_state.keys())}")
    merged_vla.scene_projector.load_state_dict(restore_state, strict=False)

    # Verify scene_norm health
    print("Checking scene_norm after restore...")
    needs_fix = check_and_fix_scene_norm(merged_vla)
    if needs_fix and not has_saved_scene:
        print("  NOTE: scene_norm was never saved from training (training script bug).")
        print("  Using default initialization. Re-train with fixed finetune.py to properly save scene_norm.")

    # Quick forward check
    print("Running SceneProjector forward test...")
    x = torch.randn(1, 32, 2048, dtype=torch.bfloat16)
    out = merged_vla.scene_projector(x)
    ok = not torch.isnan(out).any().item()
    print(f"  Output NaN: {not ok}, min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
    if not ok:
        print("  FATAL: SceneProjector still produces NaN after fix!")
        print("  Re-initializing entire SceneProjector...")
        from prismatic.models.scene_projector import SceneProjector
        merged_vla.scene_projector = SceneProjector(scene_dim=2048, llm_dim=merged_vla.llm_dim).to(torch.bfloat16)
        print("  SceneProjector replaced with fresh initialization.")

    print(f"Saving merged model to {EVAL_DIR}...")
    os.makedirs(EVAL_DIR, exist_ok=True)
    merged_vla.save_pretrained(EVAL_DIR)

    # Copy config files
    print("Copying config files...")
    config_files = [
        "added_tokens.json", "special_tokens_map.json", "tokenizer_config.json",
        "tokenizer.json", "tokenizer.model", "processor_config.json",
        "preprocessor_config.json", "processing_prismatic.py",
    ]
    for fn in config_files:
        src = os.path.join(SRC_CHECKPOINT, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(EVAL_DIR, fn))

    # Copy model component checkpoints from run dir
    component_files = [
        ("action_head--latest_checkpoint.pt", "action_head--13188_checkpoint.pt"),
        ("proprio_projector--latest_checkpoint.pt", "proprio_projector--13188_checkpoint.pt"),
        ("dataset_statistics.json", "dataset_statistics.json"),
    ]
    for src_name, dst_name in component_files:
        src = os.path.join(RUN_DIR, src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(EVAL_DIR, dst_name))
            print(f"  {src_name} -> {dst_name}")

    # Copy modeling files from run dir (if they exist, they include scene_projector support)
    for fn in ["configuration_prismatic.py", "modeling_prismatic.py"]:
        src = os.path.join(RUN_DIR, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(EVAL_DIR, fn))
            print(f"  Copied {fn}")

    print(f"\nEval checkpoint ready at {EVAL_DIR}")
    print("Files:")
    for f in sorted(os.listdir(EVAL_DIR)):
        sz = os.path.getsize(os.path.join(EVAL_DIR, f))
        print(f"  {f} ({sz/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
