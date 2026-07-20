"""
Run multiple RLBench visual evaluation episodes with one model load.

Example:
    python experiments/robot/rlbench/visualize_eval_multi.py \
        --checkpoint runs/...--4000_chkpt \
        --task close_jar \
        --num_episodes 20 \
        --stop_on_success
"""

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rlbench.backend.exceptions import InvalidActionError
from scipy.spatial.transform import Rotation as Rot

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "robot"))

from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_geo_register_predictor,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    get_vla,
)
from experiments.robot.rlbench.rlbench_utils import (  # noqa: E402
    RLBENCH_TASK_CLASSES,
    RLBENCH_TASK_INSTRUCTIONS,
    RLBENCH_TASK_MAX_STEPS,
    get_rlbench_env,
)
from experiments.robot.rlbench.visualize_eval import build_multiview_frame  # noqa: E402
from experiments.robot.robot_utils import get_action, get_image_resize_size  # noqa: E402
from experiments.robot.rlbench.run_rlbench_eval import GenerateConfig  # noqa: E402


def save_video(frames, out_path, fps=12):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def reset_episode(task_class, headless, reset_retries):
    env = None
    task = None
    obs = None
    last_error = None
    for attempt in range(reset_retries):
        try:
            if env is not None:
                env.shutdown()
            env, task, _ = get_rlbench_env(task_class, headless=headless, image_size=(256, 256))
            desc, obs = task.reset()
            if attempt > 0:
                print(f"RLBench reset succeeded on retry {attempt + 1}/{reset_retries}.")
            return env, task, obs
        except Exception as exc:
            last_error = exc
            print(f"RLBench reset failed on attempt {attempt + 1}/{reset_retries}: {exc}")
            if env is not None:
                try:
                    env.shutdown()
                except Exception:
                    pass
                env = None
    raise RuntimeError(f"RLBench reset failed after {reset_retries} attempts") from last_error


def run_episode(
    episode_idx,
    cfg,
    model,
    processor,
    action_head,
    proprio_projector,
    geo_register_predictor,
    noisy_action_projector,
    task_class,
    task_instruction,
    args,
    out_dir,
):
    env = None
    frames = []
    log_lines = []
    done = False
    invalid_action = False
    reward = 0.0
    obs = None

    try:
        env, task, obs = reset_episode(task_class, args.headless, args.reset_retries)

        for _ in range(args.num_steps_wait):
            hold_action = np.concatenate([np.asarray(obs.gripper_pose, dtype=np.float32), [float(obs.gripper_open)]])
            obs, reward, done = task.step(hold_action.tolist())

        frames.append(build_multiview_frame(obs, [f"episode={episode_idx}", "initial observation"]))
        action_queue = deque()
        max_steps = args.max_steps or RLBENCH_TASK_MAX_STEPS.get(args.task, 250)

        for t in range(max_steps):
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
                    cfg,
                    model,
                    observation,
                    task_instruction,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    geo_register_predictor=geo_register_predictor,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(raw_actions[: cfg.num_open_loop_steps])
                new_chunk = True

            delta_action = action_queue.popleft().copy()
            pre_pose = np.asarray(obs.gripper_pose, dtype=np.float32).copy()
            target_gripper = float(np.round(np.clip(delta_action[-1], 0.0, 1.0)))

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
                obs, reward, done = task.step(rlbench_action.tolist())
                actual_move_mm = np.linalg.norm(np.asarray(obs.gripper_pose[:3], dtype=np.float32) - pre_pose[:3]) * 1000
            except InvalidActionError as exc:
                invalid_action = True
                target_pos = rlbench_action[:3]
                line = (
                    f"Step {t:3d} | INVALID_ACTION | cmd_delta_pos={delta_mm:5.1f}mm | "
                    f"delta_rot={delta_rot:.3f} | raw_gripper={float(delta_action[-1]):+.3f} "
                    f"| gripper={target_gripper:.1f} | current_pos={np.round(pre_pose[:3], 4)} "
                    f"| target_pos={np.round(target_pos, 4)} | error={exc}"
                )
                log_lines.append(line)
                frames.append(
                    build_multiview_frame(
                        obs,
                        [
                            f"episode={episode_idx} step={t} INVALID_ACTION",
                            f"delta_xyz(mm)=({delta_action[0]*1000:+.1f}, {delta_action[1]*1000:+.1f}, {delta_action[2]*1000:+.1f}) | norm={delta_mm:.1f}mm",
                            f"target_pos={np.round(target_pos, 4)}",
                            str(exc),
                        ],
                    )
                )
                print(line)
                break

            line = (
                f"Step {t:3d} | cmd_delta_pos={delta_mm:5.1f}mm | actual_move={actual_move_mm:5.1f}mm "
                f"| delta_rot={delta_rot:.3f} | raw_gripper={float(delta_action[-1]):+.3f} "
                f"| gripper={target_gripper:.1f} | obs_gripper={float(obs.gripper_open):.2f} | done={done}"
            )
            log_lines.append(line)
            frames.append(
                build_multiview_frame(
                    obs,
                    [
                        f"episode={episode_idx} step={t} new_chunk={int(new_chunk)} reward={reward:.2f} done={done}",
                        f"delta_xyz(mm)=({delta_action[0]*1000:+.1f}, {delta_action[1]*1000:+.1f}, {delta_action[2]*1000:+.1f}) | norm={delta_mm:.1f}mm",
                        f"delta_rot={delta_rot:.3f} | raw_gripper={float(delta_action[-1]):+.3f} | target_gripper={target_gripper:.1f}",
                        f"actual_move={actual_move_mm:.1f}mm | obs_gripper={obs.gripper_open:.2f}",
                    ],
                )
            )
            if done:
                print(f"SUCCESS episode={episode_idx} step={t}")
                break
    finally:
        if env is not None:
            env.shutdown()

    status = "success" if done else "invalid" if invalid_action else "fail"
    stem = f"episode_{episode_idx:03d}_{status}"
    if done:
        stem += f"_step_{len(log_lines) - 1:03d}"
    out_mp4 = out_dir / f"{stem}.mp4"
    out_log = out_dir / f"{stem}.log"
    save_video(frames, out_mp4, fps=args.fps)
    with open(out_log, "w") as f:
        f.write("\n".join(log_lines))

    final_gripper = float(obs.gripper_open) if obs is not None else None
    final_pos = np.asarray(obs.gripper_pose[:3]).tolist() if obs is not None else None
    summary = {
        "episode": episode_idx,
        "status": status,
        "success": bool(done),
        "invalid_action": bool(invalid_action),
        "num_frames": len(frames),
        "num_log_lines": len(log_lines),
        "final_gripper": final_gripper,
        "final_ee_pos": final_pos,
        "video": str(out_mp4),
        "log": str(out_log),
    }
    print(f"Episode {episode_idx}: {status} | video={out_mp4}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="close_jar")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--stop_on_success", action="store_true")
    parser.add_argument("--use_geo_registers", action="store_true")
    parser.add_argument("--use_scene_tokens", action="store_true", help="Deprecated alias for --use_geo_registers")
    parser.add_argument("--num_geo_registers", type=int, default=16)
    parser.add_argument("--geo_register_num_layers", type=int, default=2)
    parser.add_argument("--geo_register_num_heads", type=int, default=8)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--num_steps_wait", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--reset_retries", type=int, default=10)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--output_dir", default="videos_eval_multi")
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    task_name = args.task
    task_instruction = RLBENCH_TASK_INSTRUCTIONS.get(task_name, task_name.replace("_", " "))

    stats_path = os.path.join(checkpoint, "dataset_statistics.json")
    with open(stats_path) as f:
        stats = json.load(f)
    norm_key = list(stats.keys())[0]
    print(f"norm key: {norm_key}")

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
    cfg = GenerateConfig(**cfg_dict)

    print("Loading model once for all episodes...")
    model = get_vla(cfg)
    _ = get_image_resize_size(cfg)
    processor = get_processor(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
    geo_register_predictor = None
    if cfg.use_geo_registers or cfg.use_scene_tokens:
        geo_register_predictor = get_geo_register_predictor(cfg, model.llm_dim)
    print("Model loaded.")

    task_class = RLBENCH_TASK_CLASSES[task_name]
    run_name = f"{Path(checkpoint).name}_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    summaries = []
    for episode_idx in range(args.num_episodes):
        summary = run_episode(
            episode_idx,
            cfg,
            model,
            processor,
            action_head,
            proprio_projector,
            geo_register_predictor,
            noisy_action_projector,
            task_class,
            task_instruction,
            args,
            out_dir,
        )
        summaries.append(summary)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summaries, f, indent=2)
        successes = sum(item["success"] for item in summaries)
        print(f"Progress: {episode_idx + 1}/{args.num_episodes} | successes={successes}")
        if summary["success"] and args.stop_on_success:
            print("Stopping early after first success.")
            break

    successes = sum(item["success"] for item in summaries)
    print(f"Final success rate: {successes}/{len(summaries)} = {successes / max(len(summaries), 1):.3f}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
