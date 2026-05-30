"""Check LIBERO dataset action value ranges for clipping issues."""
import sys
sys.path.insert(0, "/root/Afford+VLA/scene token+OpenVLA-OFT/openvla-oft")

import numpy as np
import tensorflow_datasets as tfds

# Load the spatial dataset directly
ds = tfds.builder_from_directory("/root/datasets/modified_libero_rlds/libero_spatial_no_noops")

# Sample actions
all_actions = []
count = 0
for episode in ds.as_dataset(split="train"):
    for step in episode:
        all_actions.append(step["action"].numpy())
        count += 1
        if count >= 5000:
            break
    if count >= 5000:
        break

actions = np.array(all_actions)  # [N, T, 7]
print(f"Sampled {actions.shape[0]} steps from {actions.shape[0]//8} trajectories")
print(f"Action shape per step: {actions.shape[1:]}")
print(f"\nAction statistics across all dimensions:")
print(f"  Min per dim:  {actions.min(axis=(0,1))}")
print(f"  Max per dim:  {actions.max(axis=(0,1))}")
print(f"  Mean per dim: {actions.mean(axis=(0,1))}")
print(f"  Std per dim:  {actions.std(axis=(0,1))}")
print(f"  % outside [-1, 1]: {((actions < -1) | (actions > 1)).mean(axis=(0,1)) * 100}")

# Check if any values exceed [-1, 1]
outside = (actions < -1) | (actions > 1)
if outside.any():
    print(f"\n*** WARNING: {outside.sum()} action values outside [-1, 1]!")
    print(f"  Range: [{actions.min():.4f}, {actions.max():.4f}]")
    # Show which dims
    for d in range(actions.shape[-1]):
        outs = (actions[:, :, d] < -1) | (actions[:, :, d] > 1)
        if outs.any():
            print(f"  Dim {d}: {outs.sum()} values outside [-1,1], "
                  f"min={actions[:,:,d].min():.4f}, max={actions[:,:,d].max():.4f}")
else:
    print(f"\nAll action values within [-1, 1] range - clipping not an issue")
