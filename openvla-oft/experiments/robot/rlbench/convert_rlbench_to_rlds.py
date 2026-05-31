"""
convert_rlbench_to_rlds.py

Convert collected RLBench HDF5 demonstrations to RLDS (TFRecord) format
compatible with the OpenVLA-OFT training pipeline.

The output follows the standard RLDS episode structure:
    - steps: Sequence of per-timestep observations and actions
    - language_instruction: Task description (episode-level)
    - episode_metadata: Auxiliary info

Each episode stored in the HDF5 is converted to a TFRecord Example with:
    steps/observation/front_rgb    : [T, 256, 256, 3] uint8
    steps/observation/wrist_rgb    : [T, 256, 256, 3] uint8
    steps/observation/joint_positions : [T, 7] float32
    steps/observation/gripper_open : [T, 1] float32
    steps/action                   : [T, 7] float32 (delta EE pose + gripper)

The OXE pipeline maps these to the standardized format via image_obs_keys
and state_obs_keys configuration (see materialize.py rlbench config).

Usage:
    python experiments/robot/rlbench/convert_rlbench_to_rlds.py \
        --input_dir ./datasets/rlbench_raw \
        --output_dir ./datasets/rlbench_rlds \
        --tasks close_jar
"""

import argparse
import json
import logging
import os
import sys

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── RLDS Feature Specification ──────────────────────────────────────────────────
# Raw features as stored in TFRecords. The OXE pipeline's restructure() function
# will map these to image_primary / image_wrist / proprio via image_obs_keys
# and state_obs_keys config.

RLDS_FEATURES = tfds.features.FeaturesDict({
    "steps": tfds.features.Sequence({
        "observation": tfds.features.FeaturesDict({
            "front_rgb": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8, encoding_format="png"),
            "wrist_rgb": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8, encoding_format="png"),
            "joint_positions": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            "gripper_open": tfds.features.Tensor(shape=(1,), dtype=tf.float32),
        }),
        "action": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
        "language_instruction": tfds.features.Text(),
    }),
    "episode_metadata": tfds.features.FeaturesDict({
        "file_path": tfds.features.Text(),
    }),
})


def compute_delta_action(ee_pos, ee_quat, gripper_open, next_ee_pos, next_ee_quat, next_gripper_open):
    """
    Compute 7-dim action: [dx,dy,dz, drx,dry,drz, target_gripper].

    Position and rotation are DELTA (relative to current step).
    Gripper is ABSOLUTE target state (0=close, 1=open) — matches OXE convention.
    """
    delta_pos = next_ee_pos - ee_pos

    r1 = R.from_quat(ee_quat)
    r2 = R.from_quat(next_ee_quat)
    r_diff = r2 * r1.inv()
    delta_rot = r_diff.as_rotvec()

    # Absolute target gripper state (not delta)
    target_gripper = float(next_gripper_open)

    return np.concatenate([delta_pos, delta_rot, [target_gripper]]).astype(np.float32)


def load_episodes_from_hdf5(hdf5_path):
    """Load all demos from an HDF5 file, returning episodes in RLDS-ready format.

    Handles both production format (collect_rlbench_demos.py) and test format
    (deploy/test_collect_demo.py).
    """
    episodes = []
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))

        for demo_key in demo_keys:
            demo = data[demo_key]
            obs = demo["obs"]

            # Determine num_steps from available data
            if "num_steps" in demo.attrs:
                num_steps = demo.attrs["num_steps"]
            elif "actions" in demo:
                num_steps = demo["actions"].shape[0]
            else:
                num_steps = obs["ee_pos"].shape[0]

            lang = demo.attrs.get("language_instruction", "complete the task")

            front_rgbs = obs["front_rgb"][:].astype(np.uint8)
            wrist_rgbs = obs["wrist_rgb"][:].astype(np.uint8)
            joint_positions = obs["joint_positions"][:].astype(np.float32)
            gripper_opens = obs["gripper_open"][:].astype(np.float32)
            ee_poses = obs["ee_pos"][:].astype(np.float32)
            ee_quats = obs["ee_quat"][:].astype(np.float32)

            # Compute actions: delta from step t to t+1 (last step excluded)
            T = num_steps - 1
            actions = np.zeros((T, 7), dtype=np.float32)
            for t in range(T):
                actions[t] = compute_delta_action(
                    ee_poses[t], ee_quats[t], gripper_opens[t][0],
                    ee_poses[t + 1], ee_quats[t + 1], gripper_opens[t + 1][0],
                )

            # language_instruction must be a list (one per step) since it's inside a Sequence
            lang_list = [lang] * T

            episodes.append({
                "steps": {
                    "observation": {
                        "front_rgb": front_rgbs[:T],
                        "wrist_rgb": wrist_rgbs[:T],
                        "joint_positions": joint_positions[:T],
                        "gripper_open": gripper_opens[:T],
                    },
                    "action": actions,
                    "language_instruction": lang_list,
                },
                "episode_metadata": {"file_path": hdf5_path},
            })

    return episodes


def _features_to_json(feature):
    """Convert a TFDS FeatureConnector tree to JSON-serializable dict.

    Generates the features.json format that tfds.builder() expects.
    Uses only the public API (feature.keys() and feature[key]) to traverse.
    """
    import tensorflow_datasets as tfds

    if isinstance(feature, tfds.features.FeaturesDict):
        return {
            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
            "featuresDict": {
                "features": {
                    k: _features_to_json(feature[k]) for k in feature.keys()
                }
            },
        }
    elif isinstance(feature, tfds.features.Sequence):
        return {
            "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
            "sequence": {
                "feature": _features_to_json(feature.feature),
                "length": "-1",
            },
        }
    elif isinstance(feature, tfds.features.Image):
        shape = [str(s) for s in feature.shape]
        np_dtype = feature.np_dtype if hasattr(feature, "np_dtype") else feature.dtype
        dtype_str = np_dtype.name if hasattr(np_dtype, "name") else str(np_dtype)
        return {
            "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
            "image": {
                "shape": {"dimensions": shape},
                "dtype": dtype_str,
                "encodingFormat": feature._encoding_format or "png",
            },
        }
    elif isinstance(feature, tfds.features.Tensor):
        shape_list = [str(s) for s in feature.shape]
        # Use np_dtype to avoid deprecated wrapper repr (e.g. "<dtype: 'float32'>")
        np_dtype = feature.np_dtype if hasattr(feature, "np_dtype") else feature.dtype
        dtype_str = np_dtype.name if hasattr(np_dtype, "name") else str(np_dtype)
        return {
            "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
            "tensor": {
                "shape": {"dimensions": shape_list} if shape_list else {},
                "dtype": dtype_str,
                "encoding": "none",
            },
        }
    elif isinstance(feature, tfds.features.Text):
        return {
            "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
            "text": {},
        }
    else:
        logger.warning(f"Unknown feature type: {type(feature)}, serializing as-is")
        return {"pythonClassName": str(type(feature))}


def convert_task(input_dir, output_dir, task_name, prefix="rlbench_"):
    """Convert a single task's HDF5 demos to RLDS TFRecord format.

    The pipeline uses 'rlbench_{task_name}' as the dataset identifier (needed for
    auto-detection in make_oxe_dataset_kwargs). Output is written to:

        output_dir/{prefix}{task_name}/0.1.0/
            dataset_info.json
            {prefix}{task_name}-train.tfrecord-XXXXX-of-YYYYY
    """
    dataset_name = f"{prefix}{task_name}"
    hdf5_path = os.path.join(input_dir, f"{task_name}.hdf5")

    if not os.path.exists(hdf5_path):
        logger.warning(f"Input file not found: {hdf5_path}")
        return None

    # Check if already converted
    version_dir = os.path.join(output_dir, dataset_name, "0.1.0")
    if os.path.exists(os.path.join(version_dir, "dataset_info.json")):
        logger.info(f"Output already exists: {version_dir}")
        return version_dir

    logger.info(f"Loading demos from {hdf5_path}...")
    episodes = load_episodes_from_hdf5(hdf5_path)
    logger.info(f"  Loaded {len(episodes)} episodes")

    if not episodes:
        logger.warning(f"No valid episodes found in {hdf5_path}")
        return None

    os.makedirs(version_dir, exist_ok=True)

    # Serialize episodes to TFRecord
    tfrecord_path = os.path.join(version_dir, f"{dataset_name}-train.tfrecord-00000-of-00001")
    logger.info(f"Writing TFRecord to {tfrecord_path}...")
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for ep in episodes:
            example = RLDS_FEATURES.serialize_example(ep)
            writer.write(example)

    # Write dataset_info.json
    info = {
        "name": dataset_name,
        "description": f"RLBench {task_name} demonstrations",
        "version": "0.1.0",
        "fileFormat": "tfrecord",
        "moduleName": "rlbench.convert",
        "splits": [
            {
                "name": "train",
                "shardLengths": [str(len(episodes))],
                "numBytes": str(os.path.getsize(tfrecord_path)),
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
            }
        ],
    }
    info_path = os.path.join(version_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Also write features.json for TFDS compatibility
    features_path = os.path.join(version_dir, "features.json")
    feature_config = _features_to_json(RLDS_FEATURES)
    with open(features_path, "w") as f:
        json.dump(feature_config, f, indent=2)

    logger.info(f"  Done: {version_dir} ({len(episodes)} episodes)")
    return version_dir


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine tasks to convert
    if args.tasks:
        tasks = args.tasks
    else:
        tasks = [
            f.replace(".hdf5", "")
            for f in os.listdir(args.input_dir)
            if f.endswith(".hdf5")
        ]
    logger.info(f"Tasks to convert: {tasks}")

    prefix = getattr(args, "prefix", "rlbench_")
    for task_name in tasks:
        convert_task(args.input_dir, args.output_dir, task_name, prefix=prefix)

    logger.info("Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RLBench HDF5 demos to RLDS format")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing HDF5 demo files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for RLDS datasets")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Task names to convert (default: all .hdf5 files in input_dir)")
    args = parser.parse_args()
    main(args)
