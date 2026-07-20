"""
run_rlbench_eval.py

Evaluates a trained VGGT-Omega + OpenVLA-OFT policy on RLBench tasks.

Usage:
    python experiments/robot/rlbench/run_rlbench_eval.py \
        --pretrained_checkpoint /path/to/checkpoint \
        --tasks close_jar peg_in_hole \
        --headless True \
        --num_episodes_per_task 25
"""

import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import draccus
import numpy as np
import tqdm

# Add repo root so that interpreter can find experiments.robot regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from experiments.robot.rlbench.rlbench_utils import (
    RLBENCH_TASK_CLASSES,
    RLBENCH_TASK_INSTRUCTIONS,
    RLBENCH_TASK_MAX_STEPS,
    get_rlbench_env,
    get_rlbench_image,
    get_rlbench_proprio,
    get_rlbench_wrist_image,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_geo_register_predictor,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class GenerateConfig:
    # Model-specific parameters
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    use_geo_registers: bool = False
    num_geo_registers: int = 16
    geo_register_num_layers: int = 2
    geo_register_num_heads: int = 8
    # Backward-compatible alias for older eval commands; inference uses the distilled predictor, not online VGGT.
    use_scene_tokens: bool = False
    vggt_checkpoint: Optional[str] = None
    vggt_omega_repo: Optional[str] = None
    center_crop: bool = True
    num_open_loop_steps: int = 8  # RLBench chunk=8 continuous trajectory (same as LIBERO)
    lora_rank: int = 32
    unnorm_key: str = "rlbench"

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # RLBench environment parameters
    tasks: List[str] = draccus.field(default_factory=lambda: ["close_jar"])
    headless: bool = True
    image_size: int = 256
    num_episodes_per_task: int = 25
    num_steps_wait: int = 0

    # Utils
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    seed: int = 7


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be None!"
    for task in cfg.tasks:
        assert task in RLBENCH_TASK_CLASSES, f"Unknown task '{task}'. Available: {list(RLBENCH_TASK_CLASSES.keys())}"


def run_episode(
    cfg, env, task, task_name, model, resize_size, processor=None,
    action_head=None, proprio_projector=None, geo_register_predictor=None, noisy_action_projector=None,
):
    """Run a single evaluation episode.

    The model predicts 7-dim delta actions: [dx,dy,dz, drx,dry,drz, target_gripper].
    RLBench's EndEffectorPoseViaPlanning expects 8-dim absolute actions:
    [x,y,z, qx,qy,qz,qw, gripper_open]. This function converts between the two.
    """
    from scipy.spatial.transform import Rotation as Rot

    obs = None
    for reset_attempt in range(3):
        try:
            descriptions, obs = task.reset()
            break
        except Exception as e:
            logger.warning(
                "Episode reset failed for %s (attempt %s/3): %s",
                task_name,
                reset_attempt + 1,
                e,
            )
            time.sleep(1.0)
    if obs is None:
        return False

    lang = RLBENCH_TASK_INSTRUCTIONS.get(task_name, task_name.replace("_", " "))

    # Hold the current absolute pose while objects stabilize.
    for _ in range(cfg.num_steps_wait):
        hold_action = np.concatenate(
            [np.asarray(obs.gripper_pose, dtype=np.float32), [float(obs.gripper_open)]]
        )
        obs, reward, done = task.step(hold_action.tolist())

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        logger.warning(
            "num_open_loop_steps (%s) != NUM_ACTIONS_CHUNK (%s); "
            "the evaluator will execute the first requested actions from each predicted chunk.",
            cfg.num_open_loop_steps,
            NUM_ACTIONS_CHUNK,
        )
    action_queue = deque()
    max_steps = RLBENCH_TASK_MAX_STEPS.get(task_name, 200)
    t = 0
    success = False

    try:
        while t < max_steps:
            # Prepare observation
            img = get_rlbench_image(obs)
            wrist_img = get_rlbench_wrist_image(obs)
            img_resized = resize_image_for_policy(img, resize_size)
            wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
            proprio = get_rlbench_proprio(obs)

            observation = {
                "full_image": img_resized,
                "wrist_image": wrist_img_resized,
                "state": proprio,
            }

            # Query model if action queue is empty
            if len(action_queue) == 0:
                actions = get_action(
                    cfg, model, observation, lang,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    geo_register_predictor=geo_register_predictor,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions[: cfg.num_open_loop_steps])

            delta_action = action_queue.popleft()  # [dx,dy,dz, drx,dry,drz, target_gripper]

            # RLBench training targets use an absolute gripper state in [0, 1],
            # where 0=closed and 1=open. Do not apply the LIBERO/OpenVLA
            # gripper sign flip here, or open/close will be swapped.
            delta_action = delta_action.copy()
            delta_action[-1] = float(np.round(np.clip(delta_action[-1], 0.0, 1.0)))

            # Convert 7-dim delta to 8-dim absolute action for RLBench
            current_pose = obs.gripper_pose  # [x, y, z, qx, qy, qz, qw]
            rlbench_action = np.zeros(8, dtype=np.float32)

            # Position: absolute = current + delta
            rlbench_action[:3] = current_pose[:3] + delta_action[:3]

            # Orientation: absolute = delta_rotation @ current_rotation
            r_current = Rot.from_quat(current_pose[3:])
            r_delta = Rot.from_rotvec(delta_action[3:6])
            r_target = r_delta * r_current
            rlbench_action[3:7] = r_target.as_quat()

            # Gripper: absolute target (already binarized to 0 or 1)
            rlbench_action[7] = float(np.clip(delta_action[6], 0.0, 1.0))

            # Execute
            obs, reward, done = task.step(rlbench_action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        logger.warning(f"Episode error: {e}")

    return success


def eval_rlbench(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    # Initialize model
    model = get_model(cfg)
    resize_size = get_image_resize_size(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    use_geo_registers = cfg.use_geo_registers or cfg.use_scene_tokens
    geo_register_predictor = None
    if use_geo_registers:
        geo_register_predictor = get_geo_register_predictor(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    if cfg.use_scene_tokens and not cfg.use_geo_registers:
        logger.warning("use_scene_tokens is deprecated; using the distilled geo_register_predictor at inference.")

    # Setup logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    run_id = f"EVAL-RLBENCH-{cfg.model_family}-{DATE_TIME}"
    log_path = os.path.join(cfg.local_log_dir, run_id + ".json")
    results = {}

    logger.info(f"Evaluating on tasks: {cfg.tasks}")

    for task_name in cfg.tasks:
        logger.info(f"\n{'='*50}\nTask: {task_name}\n{'='*50}")
        task_class = RLBENCH_TASK_CLASSES[task_name]
        env, task, task_desc = get_rlbench_env(task_class, headless=cfg.headless, image_size=(cfg.image_size, cfg.image_size))

        successes = 0
        for ep in tqdm.tqdm(range(cfg.num_episodes_per_task)):
            success = run_episode(
                cfg, env, task, task_name, model, resize_size,
                processor, action_head, proprio_projector, geo_register_predictor, noisy_action_projector,
            )
            if success:
                successes += 1
            if (ep + 1) % 5 == 0:
                logger.info(f"  [{ep+1}/{cfg.num_episodes_per_task}] Success: {successes}/{ep+1} = {successes/(ep+1)*100:.1f}%")

        env.shutdown()
        rate = successes / cfg.num_episodes_per_task
        results[task_name] = {
            "successes": successes,
            "total": cfg.num_episodes_per_task,
            "success_rate": rate,
        }
        logger.info(f"Task '{task_name}': {rate*100:.1f}% ({successes}/{cfg.num_episodes_per_task})")

    # Compute overall
    total_success = sum(r["successes"] for r in results.values())
    total_eps = sum(r["total"] for r in results.values())
    overall = total_success / total_eps if total_eps > 0 else 0
    results["overall"] = {"successes": total_success, "total": total_eps, "success_rate": overall}

    logger.info(f"\n{'='*50}")
    logger.info(f"Overall: {overall*100:.1f}% ({total_success}/{total_eps})")
    logger.info(f"Results saved to: {log_path}")

    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    return overall


if __name__ == "__main__":
    draccus.wrap()(eval_rlbench)()
