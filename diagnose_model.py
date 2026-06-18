"""Diagnose the merged model to find root causes of low success rate."""
import os, sys
import torch
import numpy as np

EVAL_DIR = "/root/openvla-oft/runs/vggt-eval-checkpoint"
RUN_DIR = "/root/openvla-oft/runs"

def check_scene_projector_bias(model):
    """Check if SceneProjector bias is all zeros."""
    sp = model.scene_projector
    print(f"\n=== SceneProjector Diagnosis ===")
    print(f"Type: {type(sp).__name__}")

    for name, param in sp.named_parameters():
        print(f"  {name}: shape={param.shape}, mean={param.data.float().mean():.6f}, "
              f"std={param.data.float().std():.6f}, "
              f"min={param.data.float().min():.6f}, max={param.data.float().max():.6f}")
        if 'bias' in name.lower():
            if param.data.abs().max() < 1e-6:
                print(f"    *** WARNING: bias is ALL ZEROS! This means it was never trained or was reset. ***")

    # Forward test
    x = torch.randn(1, 16, 2048, dtype=torch.bfloat16)
    with torch.no_grad():
        out = sp(x)
    print(f"  Forward test: NaN={torch.isnan(out).any().item()}, "
          f"range=[{out.min():.4f}, {out.max():.4f}], mean={out.mean():.4f}")

def check_action_head(model):
    """Check if action head was properly loaded from pre-trained checkpoint."""
    ah = model.action_head
    print(f"\n=== ActionHead Diagnosis ===")

    for name, param in ah.named_parameters():
        print(f"  {name}: shape={param.shape}, mean={param.data.float().mean():.6f}, "
              f"std={param.data.float().std():.6f}")
        if 'layer_norm' in name.lower():
            if abs(param.data.float().mean() - 1.0) < 0.001 and param.data.float().std() < 0.001:
                print(f"    ** LayerNorm at default init (all 1.0) - may indicate weights not loaded **")

    # Check total params
    total = sum(p.numel() for p in ah.parameters())
    print(f"  Total parameters: {total:,} (~{total/1e6:.0f}M)")

def find_checkpoints():
    """List all training checkpoints."""
    print(f"\n=== Training Checkpoints ===")
    for d in sorted(os.listdir(RUN_DIR)):
        if d.startswith("openvla"):
            run_path = os.path.join(RUN_DIR, d)
            print(f"\n  {d}/")
            for f in sorted(os.listdir(run_path)):
                fpath = os.path.join(run_path, f)
                if os.path.isfile(fpath):
                    sz = os.path.getsize(fpath)
                    print(f"    {f} ({sz/1e6:.1f} MB)")
                elif os.path.isdir(fpath):
                    n_files = len(os.listdir(fpath))
                    print(f"    {f}/ ({n_files} files)")

def compare_action_head():
    """Compare merged action_head with pre-trained checkpoint."""
    print(f"\n=== ActionHead Comparison ===")
    # Check original OFT checkpoint
    oft_ckpt = "/root/checkpoints/openvla-7b-oft-finetuned-libero-spatial/action_head--150000_checkpoint.pt"
    merged_ckpt = os.path.join(EVAL_DIR, "action_head--13188_checkpoint.pt")

    if os.path.exists(oft_ckpt):
        oft_state = torch.load(oft_ckpt, map_location="cpu")
        print(f"Original OFT checkpoint (150K):")
        for k, v in oft_state.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, mean={v.float().mean():.6f}, std={v.float().std():.6f}")
    else:
        print(f"Original OFT checkpoint NOT FOUND: {oft_ckpt}")

    if os.path.exists(merged_ckpt):
        merged_state = torch.load(merged_ckpt, map_location="cpu")
        print(f"\nMerged checkpoint:")
        for k, v in merged_state.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, mean={v.float().mean():.6f}, std={v.float().std():.6f}")

        # Check if they differ significantly
        if os.path.exists(oft_ckpt):
            print(f"\nDifference check:")
            for k in merged_state:
                if k in oft_state and isinstance(merged_state[k], torch.Tensor):
                    diff = (merged_state[k].float() - oft_state[k].float()).abs()
                    print(f"  {k}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
    else:
        print(f"Merged checkpoint NOT FOUND: {merged_ckpt}")

def check_scene_checkpoint():
    """Check if scene_projector checkpoints exist in training directory."""
    print(f"\n=== Scene Checkpoint Files ===")
    for d in sorted(os.listdir(RUN_DIR)):
        if d.startswith("openvla"):
            run_path = os.path.join(RUN_DIR, d)
            scene_files = [f for f in os.listdir(run_path) if "scene" in f.lower()]
            if scene_files:
                print(f"\n  {d}/")
                for f in sorted(scene_files):
                    fpath = os.path.join(run_path, f)
                    state = torch.load(fpath, map_location="cpu")
                    print(f"    {f} ({os.path.getsize(fpath)/1e6:.1f} MB)")
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            print(f"      {k}: shape={v.shape}, mean={v.float().mean():.6f}, std={v.float().std():.6f}")
            else:
                print(f"\n  {d}/: NO scene checkpoint files! (TRAINING DID NOT SAVE SCENE WEIGHTS)")

    # Also check lora_adapter
    for d in sorted(os.listdir(RUN_DIR)):
        if d.startswith("openvla"):
            lora_path = os.path.join(RUN_DIR, d, "lora_adapter")
            if os.path.isdir(lora_path):
                adapter = torch.load(os.path.join(lora_path, "adapter_model.safetensors"), map_location="cpu")
                scene_keys = [k for k in adapter.keys() if "scene" in k.lower()]
                if scene_keys:
                    print(f"\n  {d}/lora_adapter: HAS scene keys!")
                    for k in scene_keys:
                        print(f"    {k}: shape={adapter[k].shape}")
                else:
                    print(f"\n  {d}/lora_adapter: NO scene keys (scene_projector NOT trained via LoRA)")

def main():
    print("=" * 60)
    print("MODEL DIAGNOSTIC REPORT")
    print("=" * 60)

    find_checkpoints()
    check_scene_checkpoint()

    if os.path.exists(EVAL_DIR):
        print(f"\n=== Loading merged model from {EVAL_DIR} ===")
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            EVAL_DIR,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu",
        )
        check_scene_projector_bias(model)
        if hasattr(model, 'action_head'):
            check_action_head(model)
    else:
        print(f"Eval directory NOT FOUND: {EVAL_DIR}")

    compare_action_head()

if __name__ == "__main__":
    main()
