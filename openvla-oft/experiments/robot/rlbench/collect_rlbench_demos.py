"""
collect_rlbench_demos.py

Collect demonstrations for RLBench tasks using the built-in keyframe extractor.

RLBench's scripted policies extract keyframes (waypoints) from the task definition
and use a motion planner to generate smooth trajectories. This script replays
those demonstrations and saves them as HDF5 files (one per task).

Usage:
    python experiments/robot/rlbench/collect_rlbench_demos.py \
        --tasks close_jar peg_in_hole reach_and_drag \
        --demos_per_task 100 \
        --output_dir ./datasets/rlbench_raw \
        --headless True \
        --image_size 256 256
"""

import argparse
import logging
import os
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.rlbench.rlbench_utils import (
    RLBENCH_TASK_CLASSES,
    RLBENCH_TASK_INSTRUCTIONS,
    get_rlbench_env,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


HDF5_SCHEMA_VERSION = 2


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _extract_expert_joint_action(obs, expected_dim):
    """Return the RLBench command that produced this observation, if available."""
    misc = getattr(obs, "misc", None) or {}
    action = misc.get("joint_position_action")
    if action is None:
        return np.full(expected_dim, np.nan, dtype=np.float32), False

    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (expected_dim,) or not np.isfinite(action).all():
        raise ValueError(f"Invalid RLBench joint_position_action shape/value: {action}")
    if not 0.0 <= float(action[-1]) <= 1.0:
        raise ValueError(f"Expert gripper command must be in [0, 1], got {action[-1]}")
    return action, True


def _compute_transition_actions(
    ee_positions,
    ee_quaternions,
    expert_joint_actions,
    expert_action_valid,
):
    """Build [delta xyz, delta rotvec, absolute gripper command] actions."""
    num_steps = len(ee_positions)
    if num_steps < 2:
        raise ValueError(f"A demonstration must contain at least 2 observations, got {num_steps}")
    if not np.all(expert_action_valid[1:]):
        missing = (np.flatnonzero(~expert_action_valid[1:]) + 1).tolist()
        raise ValueError(f"Missing expert action for observation indices {missing[:10]}")

    actions = np.empty((num_steps - 1, 7), dtype=np.float32)
    for step_idx in range(num_steps - 1):
        delta_pos = ee_positions[step_idx + 1] - ee_positions[step_idx]
        current_rot = R.from_quat(ee_quaternions[step_idx])
        next_rot = R.from_quat(ee_quaternions[step_idx + 1])
        delta_rot = (next_rot * current_rot.inv()).as_rotvec()
        gripper_command = expert_joint_actions[step_idx + 1, -1]
        actions[step_idx] = np.concatenate([delta_pos, delta_rot, [gripper_command]])
    return actions


def collect_demos_for_task(task_name, task_class, num_demos, output_dir, headless=True, image_size=(256, 256), overwrite=False):
    """
    Collect demonstrations for a single RLBench task using the keyframe extractor.

    RLBench tasks define keyframes in their Task Description Language (TDL).
    The scripted policy extracts these keyframes, uses a motion planner to move
    between them, and we record all observations along the way.
    """
    # Use absolute path to guard against CWD changes by RLBench/CoppeliaSim
    output_path = os.path.abspath(os.path.join(output_dir, f"{task_name}.hdf5"))
    if os.path.exists(output_path) and not overwrite:
        logger.info(f"  Output already exists: {output_path} (use --overwrite to replace)")
        return output_path
    elif os.path.exists(output_path):
        logger.info(f"  Removing existing: {output_path}")
        os.remove(output_path)

    logger.info(f"  Collecting {num_demos} demos for '{task_name}'...")

    # Create environment
    env, task, _ = get_rlbench_env(task_class, headless=headless, image_size=image_size)

    # Create HDF5 file
    with h5py.File(output_path, "w") as h5f:
        grp = h5f.create_group("data")
        h5f.attrs["schema_version"] = HDF5_SCHEMA_VERSION
        h5f.attrs["task_name"] = task_name
        h5f.attrs["action_encoding"] = "delta_xyz_delta_rotvec_absolute_gripper"
        h5f.attrs["num_requested_demos"] = num_demos
        h5f.attrs["image_size"] = image_size

        num_collected = 0
        for demo_idx in range(num_demos):
            # Get a demonstration from the scripted policy (keyframe-based)
            try:
                demo = task.get_demos(
                    amount=1,
                    live_demos=True,   # Generate demos from keyframe policy in real-time
                    image_paths=False,
                    max_attempts=10,
                )[0]
            except (IndexError, RuntimeError) as e:
                logger.warning(f"    Skipping demo {demo_idx}: {e}")
                continue

            # Extract observations and actions from demo
            num_steps = len(demo)
            front_rgbs = []
            wrist_rgbs = []
            left_shoulder_rgbs = []
            right_shoulder_rgbs = []
            joint_positions = []
            gripper_opens = []
            gripper_poses = []
            gripper_joints = []
            ee_positions = []
            ee_quaternions = []
            expert_joint_actions = []
            expert_action_valid = []

            for step_idx in range(num_steps):
                obs = demo[step_idx]
                front_rgbs.append(obs.front_rgb.astype(np.uint8))
                wrist_rgbs.append(obs.wrist_rgb.astype(np.uint8))
                left_shoulder_rgbs.append(obs.left_shoulder_rgb.astype(np.uint8))
                right_shoulder_rgbs.append(obs.right_shoulder_rgb.astype(np.uint8))
                joint_positions.append(obs.joint_positions.astype(np.float32))
                gripper_opens.append(np.array([obs.gripper_open], dtype=np.float32))
                gripper_poses.append(obs.gripper_pose.astype(np.float32))
                gripper_joints.append(obs.gripper_joint_positions.astype(np.float32))

                # End-effector state: position + quaternion
                ee_pos = obs.gripper_pose[:3].astype(np.float32)
                ee_quat = obs.gripper_pose[3:].astype(np.float32)
                ee_positions.append(ee_pos)
                ee_quaternions.append(ee_quat)

                expert_action, is_valid = _extract_expert_joint_action(
                    obs, expected_dim=joint_positions[-1].shape[0] + 1
                )
                expert_joint_actions.append(expert_action)
                expert_action_valid.append(is_valid)

            ee_positions = np.stack(ee_positions, axis=0)
            ee_quaternions = np.stack(ee_quaternions, axis=0)
            expert_joint_actions = np.stack(expert_joint_actions, axis=0)
            expert_action_valid = np.asarray(expert_action_valid, dtype=np.bool_)
            actions = _compute_transition_actions(
                ee_positions,
                ee_quaternions,
                expert_joint_actions,
                expert_action_valid,
            )

            # Save demo data
            demo_grp = grp.create_group(f"demo_{num_collected}")
            obs_grp = demo_grp.create_group("obs")

            # Stack observations
            obs_grp.create_dataset("front_rgb", data=np.stack(front_rgbs, axis=0))
            obs_grp.create_dataset("wrist_rgb", data=np.stack(wrist_rgbs, axis=0))
            obs_grp.create_dataset("left_shoulder_rgb", data=np.stack(left_shoulder_rgbs, axis=0))
            obs_grp.create_dataset("right_shoulder_rgb", data=np.stack(right_shoulder_rgbs, axis=0))
            obs_grp.create_dataset("joint_positions", data=np.stack(joint_positions, axis=0))
            obs_grp.create_dataset("gripper_open", data=np.stack(gripper_opens, axis=0))
            obs_grp.create_dataset("gripper_pose", data=np.stack(gripper_poses, axis=0))
            obs_grp.create_dataset("gripper_joint_positions", data=np.stack(gripper_joints, axis=0))
            obs_grp.create_dataset("ee_pos", data=ee_positions)
            obs_grp.create_dataset("ee_quat", data=ee_quaternions)

            expert_grp = demo_grp.create_group("expert")
            expert_grp.create_dataset("joint_position_action", data=expert_joint_actions)
            expert_grp.create_dataset("action_valid", data=expert_action_valid.astype(np.uint8))
            expert_gripper_command = expert_joint_actions[:, -1:].copy()
            observed_gripper_state = np.stack(gripper_opens, axis=0)
            expert_gripper_command[~expert_action_valid] = observed_gripper_state[
                ~expert_action_valid
            ]
            expert_grp.create_dataset("gripper_command", data=expert_gripper_command)

            # One action per transition: observation[t] -> observation[t + 1].
            demo_grp.create_dataset("actions", data=actions)
            demo_grp.attrs["num_steps"] = num_steps
            demo_grp.attrs["num_transitions"] = num_steps - 1
            demo_grp.attrs["success"] = True
            demo_grp.attrs["action_encoding"] = "delta_xyz_delta_rotvec_absolute_gripper"
            demo_grp.attrs["variation_index"] = int(demo[0].misc.get("variation_index", -1))
            demo_grp.attrs["num_reset_attempts"] = int(getattr(demo, "num_reset_attempts", 1))
            demo_grp.attrs["language_instruction"] = RLBENCH_TASK_INSTRUCTIONS.get(
                task_name, task_name.replace("_", " ")
            )

            dones = np.zeros(num_steps - 1, dtype=np.uint8)
            dones[-1] = 1
            rewards = np.zeros(num_steps - 1, dtype=np.float32)
            rewards[-1] = 1.0
            demo_grp.create_dataset("dones", data=dones)
            demo_grp.create_dataset("rewards", data=rewards)

            num_collected += 1
            h5f.flush()

        h5f.attrs["num_collected_demos"] = num_collected
    # Clean up
    env.shutdown()
    logger.info(f"  Collected {num_collected} demos -> {output_path}")
    return output_path


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    tasks_to_collect = args.tasks if args.tasks else list(RLBENCH_TASK_CLASSES.keys())
    logger.info(f"Tasks to collect: {tasks_to_collect}")
    logger.info(f"Demos per task: {args.demos_per_task}")
    logger.info(f"Output directory: {args.output_dir}")

    for task_name in tasks_to_collect:
        if task_name not in RLBENCH_TASK_CLASSES:
            logger.warning(f"Unknown task '{task_name}', skipping. Available: {list(RLBENCH_TASK_CLASSES.keys())}")
            continue

        task_class = RLBENCH_TASK_CLASSES[task_name]
        collect_demos_for_task(
            task_name=task_name,
            task_class=task_class,
            num_demos=args.demos_per_task,
            output_dir=args.output_dir,
            headless=args.headless,
            image_size=tuple(args.image_size),
            overwrite=args.overwrite,
        )

    logger.info("Demo collection complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect RLBench demonstrations via keyframe extractor")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Task names to collect. If not specified, collects all 9 registered tasks.")
    parser.add_argument("--demos_per_task", type=int, default=100,
                        help="Number of demonstrations per task (default: 100)")
    parser.add_argument("--output_dir", type=str, default="./datasets/rlbench_raw",
                        help="Output directory for HDF5 files")
    parser.add_argument("--headless", type=_parse_bool, default=True,
                        help="Run headless (no GUI)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing HDF5 files")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256],
                        help="Camera image resolution (height width)")
    args = parser.parse_args()
    main(args)
