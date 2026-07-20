"""
visualize_eval.py — Run 1 eval episode and save video + action log.
Usage:
    python experiments/robot/rlbench/visualize_eval.py \
        --checkpoint runs/...--5000_chkpt \
        --task play_jenga
"""

import argparse, os, sys, json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from rlbench.backend.exceptions import InvalidActionError

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "robot"))

from experiments.robot.rlbench.rlbench_utils import (
    RLBENCH_TASK_CLASSES, RLBENCH_TASK_INSTRUCTIONS, RLBENCH_TASK_MAX_STEPS, get_rlbench_env,
    get_rlbench_image, get_rlbench_wrist_image, get_rlbench_proprio,
)
from experiments.robot.openvla_utils import get_action_head, get_geo_register_predictor, get_noisy_action_projector, get_processor, get_proprio_projector, get_vla
from experiments.robot.robot_utils import get_action, get_image_resize_size
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from scipy.spatial.transform import Rotation as Rot


def _rgb(obs, name):
    arr = getattr(obs, f"{name}_rgb", None)
    if arr is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    return arr.astype(np.uint8)


def _labeled_tile(rgb, label, size=(320, 240)):
    tile = cv2.resize(rgb, size)
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    cv2.rectangle(tile, (0, 0), (size[0], 24), (0, 0, 0), -1)
    cv2.putText(tile, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1, cv2.LINE_AA)
    return tile


def build_multiview_frame(obs, overlay_lines):
    front = _labeled_tile(_rgb(obs, "front"), "front")
    wrist = _labeled_tile(_rgb(obs, "wrist"), "wrist")
    left = _labeled_tile(_rgb(obs, "left_shoulder"), "left shoulder")
    right = _labeled_tile(_rgb(obs, "right_shoulder"), "right shoulder")
    top = np.concatenate([front, wrist], axis=1)
    bottom = np.concatenate([left, right], axis=1)
    frame = np.concatenate([top, bottom], axis=0)

    y0 = 28
    line_h = 20
    box_h = 10 + line_h * len(overlay_lines)
    cv2.rectangle(frame, (0, frame.shape[0] - box_h), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    for i, line in enumerate(overlay_lines):
        y = frame.shape[0] - box_h + 22 + i * line_h
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="play_jenga")
    parser.add_argument("--use_geo_registers", action="store_true")
    parser.add_argument("--use_scene_tokens", action="store_true", help="Deprecated alias for --use_geo_registers")
    parser.add_argument("--num_geo_registers", type=int, default=16)
    parser.add_argument("--geo_register_num_layers", type=int, default=2)
    parser.add_argument("--geo_register_num_heads", type=int, default=8)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--num_steps_wait", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--reset_retries", type=int, default=10)
    args = parser.parse_args()

    task_name = args.task
    task_instruction = RLBENCH_TASK_INSTRUCTIONS.get(task_name, task_name.replace("_", " "))
    checkpoint = os.path.abspath(args.checkpoint)

    # Load norm stats
    stats_path = os.path.join(checkpoint, "dataset_statistics.json")
    with open(stats_path) as f:
        stats = json.load(f)
    norm_key = list(stats.keys())[0]
    print(f"norm key: {norm_key}")

    # Model config
    cfg_dict = {
        "pretrained_checkpoint": checkpoint,
        "model_family": "openvla",
        "use_l1_regression": True,
        "use_proprio": True,
        "num_images_in_input": 2,
        "center_crop": True,
        "lora_rank": 32,
        "unnorm_key": norm_key,
        "load_in_8bit": False,
        "load_in_4bit": False,
        "num_open_loop_steps": 8,
        "use_film": False,
        "use_geo_registers": args.use_geo_registers or args.use_scene_tokens,
        "num_geo_registers": args.num_geo_registers,
        "geo_register_num_layers": args.geo_register_num_layers,
        "geo_register_num_heads": args.geo_register_num_heads,
        "use_scene_tokens": args.use_scene_tokens,
    }
    from experiments.robot.rlbench.run_rlbench_eval import GenerateConfig
    cfg = GenerateConfig(**cfg_dict)

    print("Loading model...")
    model = get_vla(cfg)
    resize_size = get_image_resize_size(cfg)
    processor = get_processor(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
    geo_register_predictor = None
    if cfg.use_geo_registers or cfg.use_scene_tokens:
        geo_register_predictor = get_geo_register_predictor(cfg, model.llm_dim)
    print("Model loaded.")

    # RLBench env. RLBench/CoppeliaSim occasionally fails waypoint validation
    # during randomized reset, so retry with a fresh environment before giving up.
    task_class = RLBENCH_TASK_CLASSES[task_name]
    env = None
    task = None
    obs = None
    last_reset_error = None
    for reset_attempt in range(args.reset_retries):
        try:
            if env is not None:
                env.shutdown()
            env, task, _ = get_rlbench_env(task_class, headless=args.headless, image_size=(256, 256))
            desc, obs = task.reset()
            if reset_attempt > 0:
                print(f"RLBench reset succeeded on retry {reset_attempt + 1}/{args.reset_retries}.")
            break
        except Exception as exc:
            last_reset_error = exc
            print(f"RLBench reset failed on attempt {reset_attempt + 1}/{args.reset_retries}: {exc}")
            if env is not None:
                try:
                    env.shutdown()
                except Exception:
                    pass
                env = None
    if obs is None:
        raise RuntimeError(f"RLBench reset failed after {args.reset_retries} attempts") from last_reset_error
    for _ in range(args.num_steps_wait):
        hold_action = np.concatenate(
            [np.asarray(obs.gripper_pose, dtype=np.float32), [float(obs.gripper_open)]]
        )
        obs, _, _ = task.step(hold_action.tolist())

    frames = []
    action_queue = deque()
    max_steps = args.max_steps or RLBENCH_TASK_MAX_STEPS.get(task_name, 250)
    log_lines = []
    frames.append(build_multiview_frame(obs, ["initial observation"]))

    for t in range(max_steps):
        # Query the policy from the pre-action observation.
        front_rgb = obs.front_rgb.astype(np.uint8)
        wrist_rgb = obs.wrist_rgb.astype(np.uint8)
        gripper_open = obs.gripper_open
        new_chunk = False

        if len(action_queue) == 0:
            from PIL import Image
            img_arr = np.array(Image.fromarray(front_rgb).resize((224, 224)))
            wrist_arr = np.array(Image.fromarray(wrist_rgb).resize((224, 224)))
            proprio = np.concatenate([obs.joint_positions, [gripper_open]]).astype(np.float32)

            observation = {
                "full_image": img_arr,
                "wrist_image": wrist_arr,
                "state": proprio,
            }

            raw_actions = get_action(
                cfg, model, observation, task_instruction,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                geo_register_predictor=geo_register_predictor,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
            )
            action_queue.extend(raw_actions[:cfg.num_open_loop_steps])
            new_chunk = True

        delta_action = action_queue.popleft().copy()
        pre_pose = np.asarray(obs.gripper_pose, dtype=np.float32).copy()

        # Gripper
        target_gripper = float(np.round(np.clip(delta_action[-1], 0.0, 1.0)))

        # Build RLBench absolute action
        current_pose = obs.gripper_pose
        rlbench_action = np.zeros(8, dtype=np.float32)
        rlbench_action[:3] = current_pose[:3] + delta_action[:3]
        r_current = Rot.from_quat(current_pose[3:])
        r_delta = Rot.from_rotvec(delta_action[3:6])
        rlbench_action[3:7] = (r_delta * r_current).as_quat()
        rlbench_action[7] = target_gripper

        delta_mm = np.linalg.norm(delta_action[:3]) * 1000
        delta_rot = np.linalg.norm(delta_action[3:6])
        try:
            # Execute and render the post-action observation, so motion is visible in this frame.
            obs, reward, done = task.step(rlbench_action.tolist())
            actual_move_mm = np.linalg.norm(np.asarray(obs.gripper_pose[:3], dtype=np.float32) - pre_pose[:3]) * 1000
        except InvalidActionError as exc:
            actual_move_mm = 0.0
            target_pos = rlbench_action[:3]
            line = (
                f"Step {t:3d} | INVALID_ACTION | cmd_delta_pos={delta_mm:5.1f}mm | "
                f"delta_rot={delta_rot:.3f} | raw_gripper={float(delta_action[-1]):+.3f} "
                f"| gripper={target_gripper:.1f} | current_pos={np.round(pre_pose[:3], 4)} "
                f"| target_pos={np.round(target_pos, 4)} | error={exc}"
            )
            log_lines.append(line)
            overlay = [
                f"step={t} INVALID_ACTION",
                f"delta_xyz(mm)=({delta_action[0]*1000:+.1f}, {delta_action[1]*1000:+.1f}, {delta_action[2]*1000:+.1f}) | norm={delta_mm:.1f}mm",
                f"target_pos={np.round(target_pos, 4)}",
                str(exc),
            ]
            frames.append(build_multiview_frame(obs, overlay))
            print(line)
            break
        line = (
            f"Step {t:3d} | cmd_delta_pos={delta_mm:5.1f}mm | actual_move={actual_move_mm:5.1f}mm "
            f"| delta_rot={delta_rot:.3f} | raw_gripper={float(delta_action[-1]):+.3f} "
            f"| gripper={target_gripper:.1f} | obs_gripper={float(obs.gripper_open):.2f} | done={done}"
        )
        log_lines.append(line)

        overlay = [
            f"step={t} new_chunk={int(new_chunk)} reward={reward:.2f} done={done}",
            f"delta_xyz(mm)=({delta_action[0]*1000:+.1f}, {delta_action[1]*1000:+.1f}, {delta_action[2]*1000:+.1f}) | norm={delta_mm:.1f}mm",
            f"delta_rot={delta_rot:.3f} | raw_gripper={float(delta_action[-1]):+.3f} | target_gripper={target_gripper:.1f}",
            f"actual_move={actual_move_mm:.1f}mm | obs_gripper={obs.gripper_open:.2f}",
        ]
        frames.append(build_multiview_frame(obs, overlay))

        if done:
            print(f"✅ SUCCESS at step {t}!")
            break

    env.shutdown()

    # Save
    out_dir = Path("videos_eval")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{Path(checkpoint).name}_{task_name}.mp4"

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    log_path = str(out_path).replace(".mp4", ".log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"Video: {out_path}  ({len(frames)} frames)")
    print(f"Log:   {log_path}")
    print(f"\nAction summary:")
    print(f"  Final done: {done}")
    print(f"  Final gripper: {obs.gripper_open:.2f}")
    print(f"  Final EE pos: {obs.gripper_pose[:3]}")


if __name__ == "__main__":
    main()
