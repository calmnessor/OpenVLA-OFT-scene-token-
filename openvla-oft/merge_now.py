import os, sys, json, shutil
import torch
os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, "/root/openvla-oft")
from transformers import AutoModelForVision2Seq
from peft import PeftModel

SRC = "/root/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
LORA = "/root/openvla-oft/runs/openvla-7b-oft-finetuned-libero-spatial/lora_adapter"
RUN = "/root/openvla-oft/runs/openvla-7b-oft-finetuned-libero-spatial"
EVAL = "/root/openvla-oft/runs/vggt-eval-checkpoint"

print("Step 1/4: Loading base model...")
base_vla = AutoModelForVision2Seq.from_pretrained(
    SRC, torch_dtype=torch.bfloat16, trust_remote_code=True,
    low_cpu_mem_usage=True, device_map="cpu",
)
print(f"Done. SceneProjector: {type(base_vla.scene_projector)}")

print("Step 2/4: Loading LoRA + merging...")
peft_model = PeftModel.from_pretrained(base_vla, LORA)
sp_state = {k: v.cpu().clone() for k, v in peft_model.scene_projector.state_dict().items()}
merged_vla = peft_model.merge_and_unload()
merged_vla.scene_projector.load_state_dict(sp_state, strict=True)
print("LoRA merged, SceneProjector restored.")

x = torch.randn(1, 32, 2048, dtype=torch.bfloat16)
out = merged_vla.scene_projector(x)
print(f"Forward test: NaN={torch.isnan(out).any().item()}")

print("Step 3/4: Saving merged model...")
os.makedirs(EVAL, exist_ok=True)
merged_vla.save_pretrained(EVAL)
for fn in ["added_tokens.json", "special_tokens_map.json", "tokenizer_config.json",
            "tokenizer.json", "tokenizer.model", "processor_config.json",
            "preprocessor_config.json", "processing_prismatic.py"]:
    src = os.path.join(SRC, fn)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(EVAL, fn))

print("Step 4/4: Copying checkpoints...")
for src_name, dst_name in [
    ("action_head--latest_checkpoint.pt", "action_head--13188_checkpoint.pt"),
    ("proprio_projector--latest_checkpoint.pt", "proprio_projector--13188_checkpoint.pt"),
    ("scene_projector--latest_checkpoint.pt", "scene_projector--latest_checkpoint.pt"),
    ("dataset_statistics.json", "dataset_statistics.json"),
]:
    src = os.path.join(RUN, src_name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(EVAL, dst_name))

print(f"Done! Checkpoint at {EVAL}")
for f in sorted(os.listdir(EVAL)):
    print(f"  {f} ({os.path.getsize(os.path.join(EVAL, f))/1e6:.1f} MB)")
