"""Test script: collect a single RLBench demo and save as HDF5."""
import os, sys
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import CloseJar

obs_config = ObservationConfig()
obs_config.set_all(True)
obs_config.front_camera.image_size = (256, 256)
obs_config.wrist_camera.image_size = (256, 256)
obs_config.left_shoulder_camera.image_size = (256, 256)
obs_config.right_shoulder_camera.image_size = (256, 256)

action_mode = ActionMode(EndEffectorPoseViaPlanning(), Discrete())

print("Creating env...", flush=True)
env = Environment(action_mode=action_mode, obs_config=obs_config, headless=True)
env.launch()

print("Getting task...", flush=True)
task = env.get_task(CloseJar)

print("Collecting demo via keyframe policy...", flush=True)
demos = task.get_demos(amount=1, live_demos=True)
print(f"SUCCESS: {len(demos)} demo(s)", flush=True)

demo = demos[0]
n = len(demo)
print(f"Steps: {n}", flush=True)

# Save as HDF5
out_path = "/tmp/rlbench_test_data/close_jar.hdf5"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with h5py.File(out_path, "w") as f:
    grp = f.create_group("data")
    dg = grp.create_group("demo_0")
    obs_grp = dg.create_group("obs")

    obs_grp.create_dataset("front_rgb", data=np.stack([d.front_rgb.astype(np.uint8) for d in demo]))
    obs_grp.create_dataset("wrist_rgb", data=np.stack([d.wrist_rgb.astype(np.uint8) for d in demo]))
    obs_grp.create_dataset("joint_positions", data=np.stack([d.joint_positions.astype(np.float32) for d in demo]))
    obs_grp.create_dataset("gripper_open", data=np.stack([np.array([d.gripper_open], dtype=np.float32) for d in demo]))
    ee_poses = np.stack([d.gripper_pose.astype(np.float32) for d in demo])
    obs_grp.create_dataset("ee_pos", data=ee_poses[:, :3])
    obs_grp.create_dataset("ee_quat", data=ee_poses[:, 3:])

    actions = []
    for t in range(n - 1):
        dp = demo[t+1].gripper_pose[:3].astype(np.float32) - demo[t].gripper_pose[:3].astype(np.float32)
        r1 = R.from_quat(demo[t].gripper_pose[3:].astype(np.float32))
        r2 = R.from_quat(demo[t+1].gripper_pose[3:].astype(np.float32))
        dr = (r2 * r1.inv()).as_rotvec()
        dg = np.array([demo[t+1].gripper_open - demo[t].gripper_open], dtype=np.float32)
        actions.append(np.concatenate([dp, dr, dg]))
    actions.append(np.zeros(7, dtype=np.float32))
    dg.create_dataset("actions", data=np.stack(actions))

    dg.attrs["num_steps"] = n
    dg.attrs["language_instruction"] = "close the jar"
    dg.attrs["success"] = True
    dg.create_dataset("dones", data=np.array([0]*(n-1) + [1], dtype=np.uint8))
    dg.create_dataset("rewards", data=np.array([0.0]*(n-1) + [1.0], dtype=np.float32))

print(f"Saved: {out_path}", flush=True)
env.shutdown()
print("ALL DONE!", flush=True)
