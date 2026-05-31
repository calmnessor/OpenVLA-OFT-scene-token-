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
from typing import List, Optional

import draccus
import numpy as np
import tqdm

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
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
    center_crop: bool = True
    num_open_loop_steps: int = 1  # RLBench uses keyframe prediction (single step)
    lora_rank: int = 32
    unnorm_key: str = "rlbench"

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # RLBench environment parameters
    tasks: List[str] = draccus.field(default_factory=lambda: ["close_jar"])
    headless: bool = True
    image_size: int = 256
    num_episodes_per_task: int = 25
    num_steps_wait: int = 10

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
    action_head=None, proprio_projector=None, noisy_action_projector=None,
):
    """Run a single evaluation episode."""
    descriptions, obs = task.reset()
    # Use the first variation's description
    lang = task_name.replace("_", " ")

    # Wait for objects to stabilize
    for _ in range(cfg.num_steps_wait):
        obs, reward, done = task.step(np.zeros(7 if cfg.num_open_loop_steps == 1 else 7))

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
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
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()

            # Process action for RLBench
            action = normalize_gripper_action(action, binarize=True)
            if cfg.model_family == "openvla":
                action = invert_gripper_action(action)

            # Execute
            obs, reward, done = task.step(action.tolist())
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

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

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
                processor, action_head, proprio_projector, noisy_action_projector,
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
