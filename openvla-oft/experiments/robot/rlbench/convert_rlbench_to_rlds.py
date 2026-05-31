"""
convert_rlbench_to_rlds.py

Convert collected RLBench HDF5 demonstrations to RLDS (TFRecord) format
compatible with the OpenVLA-OFT training pipeline.

The output RLDS dataset follows the Open X-Embodiment convention:
    {
        "observation": {
            "image_primary": np.ndarray (H, W, 3) uint8,
            "wrist": np.ndarray (H, W, 3) uint8,
            "proprio": np.ndarray (8,) float32,
        },
        "task": {
            "language_instruction": bytes,
        },
        "action": np.ndarray (7,) float32,   # delta pos(3) + delta rot(3) + gripper(1)
        "dataset_name": str,
    }

Usage:
    python experiments/robot/rlbench/convert_rlbench_to_rlds.py \
        --input_dir ./datasets/rlbench_raw \
        --output_dir ./datasets/rlbench_rlds \
        --task_name close_jar
"""

import argparse
import logging
import os

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Feature specification matching OpenVLA-OFT RLDS format ────────────────────
RLDS_FEATURES = tfds.features.FeaturesDict({
    "steps": tfds.features.Dataset({
        "observation": tfds.features.FeaturesDict({
            "image_primary": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8),
            "image_wrist": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8),
            "proprio": tfds.features.Tensor(shape=(8,), dtype=np.float32),
        }),
        "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
        "discount": tfds.features.Tensor(shape=(), dtype=np.float32),
        "reward": tfds.features.Tensor(shape=(), dtype=np.float32),
        "is_first": tfds.features.Tensor(shape=(), dtype=np.bool_),
        "is_last": tfds.features.Tensor(shape=(), dtype=np.bool_),
        "is_terminal": tfds.features.Tensor(shape=(), dtype=np.bool_),
        "language_instruction": tfds.features.Text(),
    }),
    "episode_metadata": tfds.features.FeaturesDict({
        "file_path": tfds.features.Text(),
    }),
})


def compute_delta_action(ee_pos, ee_quat, gripper_open, next_ee_pos, next_ee_quat, next_gripper_open):
    """Compute delta action in the format expected by OpenVLA."""
    # Position delta
    delta_pos = next_ee_pos - ee_pos

    # Rotation delta (as axis-angle difference)
    r1 = R.from_quat(ee_quat)
    r2 = R.from_quat(next_ee_quat)
    r_diff = r2 * r1.inv()
    delta_rot = r_diff.as_rotvec()

    # Gripper delta
    delta_gripper = next_gripper_open - gripper_open

    action = np.concatenate([delta_pos, delta_rot, [delta_gripper]])
    return action.astype(np.float32)


def load_hdf5_demo(hdf5_path):
    """Load a single HDF5 demo file and return list of episodes."""
    episodes = []
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        for demo_key in sorted(data.keys(), key=lambda k: int(k.split("_")[1])):
            demo = data[demo_key]
            num_steps = demo.attrs.get("num_steps", demo["actions"].shape[0])
            lang = demo.attrs.get("language_instruction", "complete the task")

            front_rgbs = demo["obs/front_rgb"][:]
            wrist_rgbs = demo["obs/wrist_rgb"][:]
            joint_positions = demo["obs/joint_positions"][:]
            gripper_opens = demo["obs/gripper_open"][:]
            ee_poses = demo["obs/ee_pos"][:]
            ee_quats = demo["obs/ee_quat"][:]

            steps = []
            for t in range(num_steps - 1):
                # Proprio: joint positions(7) + gripper_open(1)
                proprio = np.concatenate([joint_positions[t], gripper_opens[t]])

                # Action: delta from current to next
                action = compute_delta_action(
                    ee_poses[t], ee_quats[t], gripper_opens[t][0],
                    ee_poses[t + 1], ee_quats[t + 1], gripper_opens[t + 1][0],
                )

                steps.append({
                    "observation": {
                        "image_primary": front_rgbs[t],
                        "image_wrist": wrist_rgbs[t],
                        "proprio": proprio.astype(np.float32),
                    },
                    "action": action,
                    "discount": np.float32(1.0),
                    "reward": np.float32(1.0 if t == num_steps - 2 else 0.0),
                    "is_first": np.bool_(t == 0),
                    "is_last": np.bool_(t == num_steps - 2),
                    "is_terminal": np.bool_(t == num_steps - 2),
                    "language_instruction": lang,
                })

            if steps:
                episodes.append({
                    "steps": steps,
                    "episode_metadata": {"file_path": hdf5_path},
                })

    return episodes


def generate_examples(episodes):
    """Yield episodes as TFDS example format."""
    for ep in episodes:
        yield ep["steps"], ep["episode_metadata"]


def convert_task(input_dir, output_dir, task_name):
    """Convert a single task's HDF5 demos to RLDS format."""
    input_path = os.path.join(input_dir, f"{task_name}.hdf5")
    output_path = os.path.join(output_dir, task_name)

    if not os.path.exists(input_path):
        logger.warning(f"Input file not found: {input_path}")
        return

    if os.path.exists(output_path):
        logger.info(f"Output already exists: {output_path}")
        return

    logger.info(f"Loading demos from {input_path}...")
    episodes = load_hdf5_demo(input_path)
    logger.info(f"  Loaded {len(episodes)} episodes")

    if not episodes:
        logger.warning(f"No episodes found in {input_path}")
        return

    # Write as TFRecord (RLDS format)
    logger.info(f"Writing RLDS to {output_path}...")
    builder = tfds.core.builder_from_directory(output_path)
    if not hasattr(builder, "info") or builder.info is None:
        # Create new dataset
        writer = tfds.core.DatasetBuilder(
            name=task_name,
            data_dir=output_path,
            version="0.1.0",
            features=RLDS_FEATURES,
            file_format="tfrecord",
        )
    else:
        writer = builder

    writer.download_and_prepare()  # Not needed for manual writing
    logger.info(f"  Done: {output_path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = args.tasks if args.tasks else [
        f.replace(".hdf5", "") for f in os.listdir(args.input_dir) if f.endswith(".hdf5")
    ]
    logger.info(f"Tasks to convert: {tasks}")

    for task_name in tasks:
        convert_task(args.input_dir, args.output_dir, task_name)

    logger.info("Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RLBench HDF5 demos to RLDS format")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing HDF5 demo files from collect_rlbench_demos.py")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for RLDS datasets")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Task names to convert (default: all HDF5 files in input_dir)")
    args = parser.parse_args()
    main(args)
