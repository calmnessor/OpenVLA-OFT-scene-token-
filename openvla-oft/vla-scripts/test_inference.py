import torch, os, sys, json
sys.path.insert(0, "/root/openvla-oft")

from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from experiments.robot.openvla_utils import find_checkpoint_file, load_component_state_dict

device = "cuda:0"
CKPT = "/root/openvla-oft/runs/vggt-eval-checkpoint"

print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(CKPT, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
model = model.to(device)
model.eval()
print(f"Model loaded. llm_dim={model.llm_dim}")

# Load action_head
ah_path = find_checkpoint_file(CKPT, "action_head")
print(f"Action head: {ah_path}")
action_head = L1RegressionActionHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=7)
action_head.load_state_dict(load_component_state_dict(ah_path))
action_head = action_head.to(torch.bfloat16).to(device)
action_head.eval()

# Load proprio_projector
pp_path = find_checkpoint_file(CKPT, "proprio_projector")
print(f"Proprio projector: {pp_path}")
proprio_projector = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8)
proprio_projector.load_state_dict(load_component_state_dict(pp_path))
proprio_projector = proprio_projector.to(torch.bfloat16).to(device)
proprio_projector.eval()

# Load norm stats
with open(os.path.join(CKPT, "dataset_statistics.json")) as f:
    model.norm_stats = json.load(f)

# Test with dummy inputs
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)

from PIL import Image
import numpy as np
dummy_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
dummy_img = Image.fromarray(dummy_arr)
inputs = processor("In: What action should the robot take to test?\nOut:", dummy_img).to(device, dtype=torch.bfloat16)

print(f"Input pixel_values dtype: {inputs['pixel_values'].dtype}")
print(f"Input input_ids dtype: {inputs['input_ids'].dtype}")

# Test scene tokens
scene_tokens = torch.randn(1, 32, 2048, dtype=torch.bfloat16, device=device)

# Test 1: Without scene tokens
print("\n=== Test 1: Without scene tokens ===")
with torch.no_grad():
    action1, _ = model.predict_action(
        **inputs,
        unnorm_key="libero_spatial_no_noops",
        do_sample=False,
        proprio=torch.randn(8, dtype=torch.bfloat16, device=device),
        proprio_projector=proprio_projector,
        action_head=action_head,
        scene_tokens=None,
        use_film=False,
    )

import numpy as np
print(f"Action: {action1 if isinstance(action1, np.ndarray) else action1[0]}")
has_nan1 = np.isnan(action1).any() if isinstance(action1, np.ndarray) else torch.isnan(action1).any().item()
print(f"Has NaN: {has_nan1}")
print(f"Result: {'PASS' if not has_nan1 else 'FAIL'}")

# Test 2: Without scene tokens, without proprio
print("\n=== Test 2: Without scene tokens & proprio ===")
with torch.no_grad():
    action2, _ = model.predict_action(
        **inputs,
        unnorm_key="libero_spatial_no_noops",
        do_sample=False,
        action_head=action_head,
        scene_tokens=None,
        use_film=False,
    )

has_nan2 = np.isnan(action2).any() if isinstance(action2, np.ndarray) else torch.isnan(action2).any().item()
print(f"Has NaN: {has_nan2}")
print(f"Result: {'PASS' if not has_nan2 else 'FAIL'}")

# Test 3: With scene tokens (random)
print("\n=== Test 3: With scene tokens (random) ===")
with torch.no_grad():
    action3, _ = model.predict_action(
        **inputs,
        unnorm_key="libero_spatial_no_noops",
        do_sample=False,
        proprio=torch.randn(8, dtype=torch.bfloat16, device=device),
        proprio_projector=proprio_projector,
        action_head=action_head,
        scene_tokens=scene_tokens,
        use_film=False,
    )

has_nan3 = np.isnan(action3).any() if isinstance(action3, np.ndarray) else torch.isnan(action3).any().item()
print(f"Has NaN: {has_nan3}")
print(f"Result: {'PASS' if not has_nan3 else 'FAIL'}")

# Test 4: Check intermediate model outputs
print("\n=== Test 4: Forward pass inspection ===")
from prismatic.extern.hf.modeling_prismatic import _process_scene_tokens
# Check if scene_projector produces NaN
sp_out = model.scene_projector(scene_tokens)
print(f"SceneProjector output shape: {sp_out.shape}")
print(f"SceneProjector output has NaN: {torch.isnan(sp_out).any().item()}")
print(f"SceneProjector output stats: min={sp_out.min():.4f}, max={sp_out.max():.4f}, mean={sp_out.mean():.4f}")

# Check vision backbone output
with torch.no_grad():
    vision_out = model.vision_backbone(inputs['pixel_values'])
if isinstance(vision_out, tuple):
    vision_out = vision_out[0]
print(f"Vision backbone output has NaN: {torch.isnan(vision_out).any().item()}")
print(f"Vision backbone output stats: min={vision_out.min():.4f}, max={vision_out.max():.4f}")

# Test 5: Without scene_projector (replace with identity-like)
print("\n=== Test 5: Zero scene tokens ===")
zero_scene = torch.zeros(1, 32, 2048, dtype=torch.bfloat16, device=device)
with torch.no_grad():
    action5, _ = model.predict_action(
        **inputs,
        unnorm_key="libero_spatial_no_noops",
        do_sample=False,
        proprio=torch.randn(8, dtype=torch.bfloat16, device=device),
        proprio_projector=proprio_projector,
        action_head=action_head,
        scene_tokens=zero_scene,
        use_film=False,
    )

has_nan5 = np.isnan(action5).any() if isinstance(action5, np.ndarray) else torch.isnan(action5).any().item()
print(f"Has NaN: {has_nan5}")
print(f"Result: {'PASS' if not has_nan5 else 'FAIL'}")
