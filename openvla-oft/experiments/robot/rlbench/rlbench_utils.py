"""
rlbench_utils.py

Utility functions for RLBench environment and task management.
"""

import logging
import os
from typing import Tuple

import numpy as np
from rlbench import Environment, ObservationConfig
from rlbench import ActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.observation import Observation
from rlbench.tasks import (
    CloseJar,
    MoveHanger,
    PlaceHangerOnRack,
    PlayJenga,
    PutKnifeOnChoppingBoard,
    TakeUmbrellaOutOfUmbrellaStand,
)

logger = logging.getLogger(__name__)

# ── Task Registry ──────────────────────────────────────────────────────────────
# Map task names to RLBench task classes.
# These 9 tasks require fine spatial understanding and are compatible with
# the Evo-0 paper's design philosophy.

# Five RLBench tasks from Evo-0 paper (Section IV.A), covering three categories:
#   1. Precise grasping and transport: PlayJenga, TakeUmbrellaOutOfUmbrellaStand
#   2. Precise grasping and placement: PutKnifeOnChoppingBoard
#   3. Precise motion under height/translation variation: PlaceHangerOnRack, MoveHanger
RLBENCH_TASK_CLASSES = {
    "close_jar": CloseJar,
    "play_jenga": PlayJenga,
    "put_knife_on_chopping_board": PutKnifeOnChoppingBoard,
    "take_umbrella_out_of_umbrella_stand": TakeUmbrellaOutOfUmbrellaStand,
    "place_hanger_on_rack": PlaceHangerOnRack,
    "move_hanger": MoveHanger,
}

# Task language instructions (used in VLA prompt)
RLBENCH_TASK_INSTRUCTIONS = {
    "close_jar": "close the jar",
    "play_jenga": "play jenga",
    "put_knife_on_chopping_board": "put the knife on the chopping board",
    "take_umbrella_out_of_umbrella_stand": "take the umbrella out of the umbrella stand",
    "place_hanger_on_rack": "place the hanger on the rack",
    "move_hanger": "move the hanger",
}

# Maximum episode steps per task (generous upper bound)
RLBENCH_TASK_MAX_STEPS = {
    "close_jar": 200,
    "play_jenga": 250,
    "put_knife_on_chopping_board": 200,
    "take_umbrella_out_of_umbrella_stand": 200,
    "place_hanger_on_rack": 200,
    "move_hanger": 200,
}


def get_rlbench_env(
    task_class,
    headless: bool = True,
    image_size: Tuple[int, int] = (256, 256),
) -> Tuple[Environment, str]:
    """
    Create an RLBench environment for a given task class.

    Args:
        task_class: RLBench task class (e.g., PegInHole)
        headless: Whether to run headless (no GUI)
        image_size: Resolution for camera observations

    Returns:
        (env, task_description) tuple
    """
    # Ensure headless rendering env vars are set
    if headless:
        os.environ.setdefault("COPPELIASIM_HEADLESS", "1")
        os.environ.setdefault("DISPLAY", ":99")

    # Configure camera observations
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.front_camera.set_all(True)
    obs_config.front_camera.image_size = image_size
    obs_config.wrist_camera.set_all(True)
    obs_config.wrist_camera.image_size = image_size
    obs_config.left_shoulder_camera.set_all(True)
    obs_config.left_shoulder_camera.image_size = image_size
    obs_config.right_shoulder_camera.set_all(True)
    obs_config.right_shoulder_camera.image_size = image_size

    # Configure action mode: delta EE pose via motion planning + discrete gripper
    action_mode = ActionMode(EndEffectorPoseViaPlanning(), Discrete())

    # Create environment
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=headless,
    )
    env.launch()

    # Get task
    task = env.get_task(task_class)
    task_description = task_class.__name__

    return env, task, task_description


def get_rlbench_image(obs: Observation) -> np.ndarray:
    """Get front camera image as uint8 numpy array."""
    return obs.front_rgb.astype(np.uint8)


def get_rlbench_wrist_image(obs: Observation) -> np.ndarray:
    """Get wrist camera image as uint8 numpy array."""
    return obs.wrist_rgb.astype(np.uint8)


def get_rlbench_proprio(obs: Observation) -> np.ndarray:
    """
    Extract proprioceptive state from RLBench observation.

    Returns [8]-dim array: [joint_positions(7), gripper_open(1)]
    """
    joint_pos = obs.joint_positions.astype(np.float32)
    gripper_open = np.array([obs.gripper_open], dtype=np.float32)
    return np.concatenate([joint_pos, gripper_open])


def quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    w = np.clip(w, -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if np.abs(den) < 1e-6:
        return np.zeros(3)
    return (np.array([x, y, z]) * 2.0 * np.arccos(w) / den).astype(np.float32)
