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
import tqdm

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.rlbench.rlbench_utils import (
    RLBENCH_TASK_CLASSES,
    RLBENCH_TASK_INSTRUCTIONS,
    get_rlbench_env,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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
    env, task, task_description = get_rlbench_env(task_class, headless=headless, image_size=image_size)

    # Create HDF5 file
    with h5py.File(output_path, "w") as h5f:
        grp = h5f.create_group("data")

        num_collected = 0
        for demo_idx in range(num_demos):
            # Get a demonstration from the scripted policy (keyframe-based)
            try:
                demo = task.get_demos(
                    amount=1,
                    live_demos=True,   # Generate demos from keyframe policy in real-time
                    image_paths=False,
                    max_attempts=1,    # Do not retry on failure (for speed)
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
            actions = []

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

                # Action: delta EE pose + gripper open (from keyframe)
                if step_idx < num_steps - 1:
                    next_ee_pos = demo[step_idx + 1].gripper_pose[:3].astype(np.float32)
                    next_ee_quat = demo[step_idx + 1].gripper_pose[3:].astype(np.float32)
                    next_gripper = np.array([demo[step_idx + 1].gripper_open], dtype=np.float32)
                    delta_pos = next_ee_pos - ee_pos
                    # NOTE: Quaternion delta is approximated as difference
                    delta_quat = next_ee_quat - ee_quat
                    delta_gripper = next_gripper - np.array([obs.gripper_open], dtype=np.float32)
                    action = np.concatenate([delta_pos, delta_quat, delta_gripper])
                else:
                    # Last step: zero action (terminal state)
                    action = np.zeros(8, dtype=np.float32)

                actions.append(action.astype(np.float32))

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
            obs_grp.create_dataset("ee_pos", data=np.stack(ee_positions, axis=0))
            obs_grp.create_dataset("ee_quat", data=np.stack(ee_quaternions, axis=0))

            # Store actions and metadata
            demo_grp.create_dataset("actions", data=np.stack(actions, axis=0))
            demo_grp.attrs["num_steps"] = num_steps
            demo_grp.attrs["success"] = True  # Scripted demos always succeed
            demo_grp.attrs["language_instruction"] = RLBENCH_TASK_INSTRUCTIONS.get(
                task_name, task_name.replace("_", " ")
            )

            # Dones and rewards
            dones = np.zeros(num_steps, dtype=np.uint8)
            dones[-1] = 1
            rewards = np.zeros(num_steps, dtype=np.float32)
            rewards[-1] = 1.0
            demo_grp.create_dataset("dones", data=dones)
            demo_grp.create_dataset("rewards", data=rewards)

            num_collected += 1

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
    parser.add_argument("--headless", type=bool, default=True,
                        help="Run headless (no GUI)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing HDF5 files")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256],
                        help="Camera image resolution (height width)")
    args = parser.parse_args()
    main(args)
