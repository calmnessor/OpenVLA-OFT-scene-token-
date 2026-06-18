import torch, os, sys, json, shutil
sys.path.insert(0, "/root/openvla-oft")
from transformers import AutoModelForVision2Seq
from peft import PeftModel

SRC = "/root/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
LORA = "/root/openvla-oft/runs/openvla-7b-oft-finetuned-libero-spatial/lora_adapter"
RUN = "/root/openvla-oft/runs/openvla-7b-oft-finetuned-libero-spatial"
EVAL = "/root/openvla-oft/runs/vggt-eval-checkpoint"

print("Loading base model...")
base = AutoModelForVision2Seq.from_pretrained(
    SRC, torch_dtype=torch.bfloat16, trust_remote_code=True,
    low_cpu_mem_usage=True, device_map="cpu"
)
print(f"Loaded. SceneProjector: {type(base.scene_projector)}")

print("Loading LoRA adapter...")
peft = PeftModel.from_pretrained(base, LORA)
print("Merging LoRA...")
merged = peft.merge_and_unload()

print("Checking scene_norm before fix...")
wn = merged.scene_projector.projector.scene_norm.weight
bn = merged.scene_projector.projector.scene_norm.bias
print(f"  weight: has_nan={torch.isnan(wn).any()}, sample={wn[:3]}")
print(f"  bias: has_nan={torch.isnan(bn).any()}, sample={bn[:3]}")

print("Fixing scene_norm...")
merged.scene_projector.projector.scene_norm.weight.data = torch.ones(4096, dtype=torch.bfloat16)
merged.scene_projector.projector.scene_norm.bias.data = torch.zeros(4096, dtype=torch.bfloat16)

# Verify
x = torch.randn(1, 32, 2048, dtype=torch.bfloat16)
out = merged.scene_projector(x)
print(f"SceneProjector test: NaN={torch.isnan(out).any().item()}")

print("Saving merged model...")
os.makedirs(EVAL, exist_ok=True)
merged.save_pretrained(EVAL)
print("Saved.")

# Copy tokenizer/config files from SRC
print("Copying config files...")
for fn in [
    "added_tokens.json", "special_tokens_map.json", "tokenizer_config.json",
    "tokenizer.json", "tokenizer.model", "processor_config.json",
    "preprocessor_config.json", "processing_prismatic.py",
]:
    src = os.path.join(SRC, fn)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(EVAL, fn))

# Copy action_head, proprio_projector, dataset_statistics from RUN (rename checkpoints)
for fn in [
    "action_head--latest_checkpoint.pt",
    "proprio_projector--latest_checkpoint.pt",
    "dataset_statistics.json",
]:
    src = os.path.join(RUN, fn)
    if os.path.exists(src):
        if "latest_checkpoint" in fn:
            dst = fn.replace("latest_checkpoint", "13188_checkpoint")
        else:
            dst = fn
        shutil.copy2(src, os.path.join(EVAL, dst))
        print(f"  {fn} -> {dst}")

print("Done!")
