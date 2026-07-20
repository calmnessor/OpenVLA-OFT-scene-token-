"""
Replay and validate RLBench HDF5 expert actions.

This script is a sanity check for the data/action interface used by RLBench
OpenVLA evaluation:

1. Offline alignment: verifies that HDF5 actions exactly reconstruct the next
   recorded end-effector pose in the collected demonstration.
2. Optional online replay: executes the same 7D delta-action convention through
   RLBench's current action mode, using the same delta-to-absolute conversion as
   visualize_eval.py / run_rlbench_eval.py.

Important: HDF5 demos were collected under randomized object placements. Online
replay starts from a new RLBench reset, so task success is not a strict replay
guarantee unless the same scene randomization is reproduced. The online replay is
mainly for checking action scale, gripper semantics, planner compatibility, and
whether expert-scale deltas are executable.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "robot"))

def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")




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
    frame = np.concatenate(
        [np.concatenate([front, wrist], axis=1), np.concatenate([left, right], axis=1)],
        axis=0,
    )

    line_h = 20
    box_h = 10 + line_h * len(overlay_lines)
    cv2.rectangle(frame, (0, frame.shape[0] - box_h), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    for i, line in enumerate(overlay_lines):
        y = frame.shape[0] - box_h + 22 + i * line_h
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1, cv2.LINE_AA)
    return frame


def load_demo(hdf5_path, demo_key):
    with h5py.File(hdf5_path, "r") as f:
        root_attrs = dict(f.attrs)
        demo = f["data"][demo_key]
        attrs = dict(demo.attrs)
        obs = demo["obs"]
        payload = {
            "root_attrs": root_attrs,
            "attrs": attrs,
            "front_rgb": np.asarray(obs["front_rgb"], dtype=np.uint8),
            "wrist_rgb": np.asarray(obs["wrist_rgb"], dtype=np.uint8),
            "left_shoulder_rgb": np.asarray(obs["left_shoulder_rgb"], dtype=np.uint8)
            if "left_shoulder_rgb" in obs
            else None,
            "right_shoulder_rgb": np.asarray(obs["right_shoulder_rgb"], dtype=np.uint8)
            if "right_shoulder_rgb" in obs
            else None,
            "ee_pos": np.asarray(obs["ee_pos"], dtype=np.float32),
            "ee_quat": np.asarray(obs["ee_quat"], dtype=np.float32),
            "gripper_open": np.asarray(obs["gripper_open"], dtype=np.float32).reshape(-1),
            "actions": np.asarray(demo["actions"], dtype=np.float32),
            "rewards": np.asarray(demo["rewards"], dtype=np.float32) if "rewards" in demo else None,
            "dones": np.asarray(demo["dones"], dtype=np.uint8) if "dones" in demo else None,
        }
        if "expert" in demo:
            expert = demo["expert"]
            payload["expert_gripper_command"] = (
                np.asarray(expert["gripper_command"], dtype=np.float32).reshape(-1)
                if "gripper_command" in expert
                else None
            )
            payload["expert_action_valid"] = (
                np.asarray(expert["action_valid"], dtype=np.bool_).reshape(-1)
                if "action_valid" in expert
                else None
            )
        else:
            payload["expert_gripper_command"] = None
            payload["expert_action_valid"] = None
    return payload


def validate_offline_alignment(demo):
    actions = demo["actions"]
    ee_pos = demo["ee_pos"]
    ee_quat = demo["ee_quat"]
    gripper_open = demo["gripper_open"]
    commands = demo["expert_gripper_command"]
    valid = demo["expert_action_valid"]

    num_steps = int(ee_pos.shape[0])
    expected_shape = (num_steps - 1, 7)
    result = {
        "num_steps": num_steps,
        "num_transitions": int(actions.shape[0]),
        "action_shape": list(actions.shape),
        "expected_action_shape": list(expected_shape),
        "finite_actions": bool(np.isfinite(actions).all()),
        "shape_ok": bool(actions.shape == expected_shape),
        "action_encoding": demo["attrs"].get("action_encoding", ""),
    }
    if actions.shape != expected_shape:
        result["all_checks_passed"] = False
        return result

    expected_delta_pos = ee_pos[1:] - ee_pos[:-1]
    expected_delta_rot = (R.from_quat(ee_quat[1:]) * R.from_quat(ee_quat[:-1]).inv()).as_rotvec()
    max_pos_error = float(np.max(np.abs(actions[:, :3] - expected_delta_pos)))
    max_rot_error = float(np.max(np.abs(actions[:, 3:6] - expected_delta_rot)))

    if commands is not None and commands.shape[0] == num_steps:
        gripper_target = commands[1:]
        gripper_source = "expert/gripper_command[1:]"
    else:
        gripper_target = gripper_open[1:]
        gripper_source = "obs/gripper_open[1:]"
    max_gripper_error = float(np.max(np.abs(actions[:, 6] - gripper_target)))

    delta_pos_norm_mm = np.linalg.norm(actions[:, :3], axis=1) * 1000.0
    delta_rot_norm = np.linalg.norm(actions[:, 3:6], axis=1)
    result.update(
        {
            "max_position_alignment_error_m": max_pos_error,
            "max_rotation_alignment_error_rad": max_rot_error,
            "max_gripper_alignment_error": max_gripper_error,
            "gripper_alignment_source": gripper_source,
            "delta_pos_norm_mm": {
                "min": float(np.min(delta_pos_norm_mm)),
                "mean": float(np.mean(delta_pos_norm_mm)),
                "p95": float(np.percentile(delta_pos_norm_mm, 95)),
                "max": float(np.max(delta_pos_norm_mm)),
            },
            "delta_rot_norm_rad": {
                "min": float(np.min(delta_rot_norm)),
                "mean": float(np.mean(delta_rot_norm)),
                "p95": float(np.percentile(delta_rot_norm, 95)),
                "max": float(np.max(delta_rot_norm)),
            },
            "gripper_command_counts": {
                "close_0": int((actions[:, 6] < 0.5).sum()),
                "open_1": int((actions[:, 6] >= 0.5).sum()),
            },
            "expert_action_valid_ok": bool(valid is None or (not valid[0] and valid[1:].all())),
        }
    )
    result["all_checks_passed"] = bool(
        result["finite_actions"]
        and result["shape_ok"]
        and max_pos_error < 1e-5
        and max_rot_error < 1e-5
        and max_gripper_error < 1e-6
        and result["expert_action_valid_ok"]
    )
    return result


def make_video(frames, output_path, fps):
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def reset_task(task_class, args):
    from experiments.robot.rlbench.rlbench_utils import get_rlbench_env

    env = None
    last_error = None
    for attempt in range(args.reset_retries):
        try:
            if env is not None:
                env.shutdown()
            env, task, _ = get_rlbench_env(task_class, headless=args.headless, image_size=tuple(args.image_size))
            _, obs = task.reset()
            return env, task, obs, attempt + 1
        except Exception as exc:
            last_error = exc
            print(f"reset failed {attempt + 1}/{args.reset_retries}: {exc}")
            if env is not None:
                try:
                    env.shutdown()
                except Exception:
                    pass
                env = None
    raise RuntimeError(f"RLBench reset failed after {args.reset_retries} attempts") from last_error


def delta_to_rlbench_action(obs, delta_action, gripper_mode):
    delta_action = np.asarray(delta_action, dtype=np.float32).copy()
    if gripper_mode == "round":
        gripper = float(np.round(np.clip(delta_action[6], 0.0, 1.0)))
    elif gripper_mode == "raw":
        gripper = float(np.clip(delta_action[6], 0.0, 1.0))
    else:
        raise ValueError(f"Unknown gripper mode: {gripper_mode}")

    current_pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    action = np.zeros(8, dtype=np.float32)
    action[:3] = current_pose[:3] + delta_action[:3]
    action[3:7] = (R.from_rotvec(delta_action[3:6]) * R.from_quat(current_pose[3:])).as_quat()
    action[7] = gripper
    return action, gripper


def replay_online(demo, args, out_dir):
    from rlbench.backend.exceptions import InvalidActionError
    from experiments.robot.rlbench.rlbench_utils import RLBENCH_TASK_CLASSES

    if args.task not in RLBENCH_TASK_CLASSES:
        raise ValueError(f"Unknown RLBench task {args.task!r}; available: {sorted(RLBENCH_TASK_CLASSES)}")
    task_class = RLBENCH_TASK_CLASSES[args.task]
    env, task, obs, reset_attempt = reset_task(task_class, args)
    frames = [build_multiview_frame(obs, ["initial reset for expert-action online replay"])]
    log_lines = []
    success = False
    invalid = False
    error = False
    error_message = None
    final_reward = 0.0
    final_done = False
    final_step = -1

    actions = demo["actions"]
    max_steps = min(args.max_steps or actions.shape[0], actions.shape[0])

    try:
        for step_idx in range(max_steps):
            pre_pose = np.asarray(obs.gripper_pose, dtype=np.float32).copy()
            delta_action = actions[step_idx]
            delta_mm = float(np.linalg.norm(delta_action[:3]) * 1000.0)
            delta_rot = float(np.linalg.norm(delta_action[3:6]))
            try:
                rlbench_action, gripper = delta_to_rlbench_action(obs, delta_action, args.gripper_mode)
                obs, reward, done = task.step(rlbench_action.tolist())
                actual_move_mm = float(
                    np.linalg.norm(np.asarray(obs.gripper_pose[:3], dtype=np.float32) - pre_pose[:3]) * 1000.0
                )
            except InvalidActionError as exc:
                invalid = True
                final_step = step_idx
                line = (
                    f"Step {step_idx:3d} | INVALID_ACTION | cmd_delta_pos={delta_mm:5.1f}mm "
                    f"| delta_rot={delta_rot:.3f} | gripper={gripper:.1f} "
                    f"| current_pos={np.round(pre_pose[:3], 4)} "
                    f"| target_pos={np.round(rlbench_action[:3], 4)} | error={exc}"
                )
                print(line)
                log_lines.append(line)
                frames.append(
                    build_multiview_frame(
                        obs,
                        [
                            f"step={step_idx} INVALID_ACTION",
                            f"expert delta norm={delta_mm:.1f}mm | delta_rot={delta_rot:.3f}",
                            f"target_pos={np.round(rlbench_action[:3], 4)}",
                            str(exc),
                        ],
                    )
                )
                break
            except Exception as exc:
                error = True
                error_message = repr(exc)
                final_step = step_idx
                line = (
                    f"Step {step_idx:3d} | ERROR | cmd_delta_pos={delta_mm:5.1f}mm "
                    f"| delta_rot={delta_rot:.3f} | current_pos={np.round(pre_pose[:3], 4)} "
                    f"| current_quat={np.round(pre_pose[3:], 4)} | error={exc}"
                )
                print(line)
                log_lines.append(line)
                frames.append(
                    build_multiview_frame(
                        obs,
                        [
                            f"step={step_idx} ERROR",
                            f"expert delta norm={delta_mm:.1f}mm | delta_rot={delta_rot:.3f}",
                            f"current_quat={np.round(pre_pose[3:], 4)}",
                            str(exc),
                        ],
                    )
                )
                break

            final_reward = float(reward)
            final_done = bool(done)
            final_step = step_idx
            line = (
                f"Step {step_idx:3d} | cmd_delta_pos={delta_mm:5.1f}mm | actual_move={actual_move_mm:5.1f}mm "
                f"| delta_rot={delta_rot:.3f} | gripper={gripper:.1f} "
                f"| obs_gripper={float(obs.gripper_open):.2f} | reward={float(reward):.2f} | done={done}"
            )
            log_lines.append(line)
            frames.append(
                build_multiview_frame(
                    obs,
                    [
                        f"step={step_idx} expert_delta_replay reward={float(reward):.2f} done={done}",
                        f"delta_xyz(mm)=({delta_action[0]*1000:+.1f}, {delta_action[1]*1000:+.1f}, {delta_action[2]*1000:+.1f})",
                        f"norm={delta_mm:.1f}mm | delta_rot={delta_rot:.3f} | gripper={gripper:.1f}",
                        f"actual_move={actual_move_mm:.1f}mm | obs_gripper={float(obs.gripper_open):.2f}",
                    ],
                )
            )
            if done:
                success = True
                break
    finally:
        try:
            env.shutdown()
        except Exception:
            pass

    status = "success" if success else "invalid" if invalid else "error" if error else "fail"
    video_path = out_dir / f"{args.demo}_{status}_online_replay.mp4"
    log_path = out_dir / f"{args.demo}_{status}_online_replay.log"
    make_video(frames, video_path, args.fps)
    log_path.write_text("\n".join(log_lines) + "\n")

    return {
        "executed": True,
        "status": status,
        "success": success,
        "invalid_action": invalid,
        "error": error,
        "error_message": error_message,
        "reset_attempt": reset_attempt,
        "steps_executed": int(final_step + 1),
        "final_reward": final_reward,
        "final_done": final_done,
        "video": str(video_path),
        "log": str(log_path),
        "caveat": (
            "Online replay uses a fresh RLBench reset. The stored HDF5 scene randomization is not reproduced, "
            "so task success is informative but not a strict requirement for action-format correctness."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate and optionally replay RLBench HDF5 expert actions")
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--task", default="close_jar")
    parser.add_argument("--demo", default="demo_0", help="Demo key, e.g. demo_0")
    parser.add_argument("--output_dir", default="expert_replay_checks")
    parser.add_argument("--execute", action="store_true", help="Also execute expert deltas in RLBench")
    parser.add_argument("--headless", type=parse_bool, default=True)
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--reset_retries", type=int, default=20)
    parser.add_argument("--gripper_mode", choices=["round", "raw"], default="round")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    demo_key = args.demo if args.demo.startswith("demo_") else f"demo_{args.demo}"
    demo = load_demo(args.hdf5, demo_key)

    out_dir = Path(args.output_dir) / f"{Path(args.hdf5).stem}_{demo_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "hdf5": os.path.abspath(args.hdf5),
        "task": args.task,
        "demo": demo_key,
        "root_attrs": {k: str(v) for k, v in demo["root_attrs"].items()},
        "demo_attrs": {k: str(v) for k, v in demo["attrs"].items()},
        "offline_alignment": validate_offline_alignment(demo),
        "online_replay": {"executed": False},
    }
    if args.execute:
        summary["online_replay"] = replay_online(demo, args, out_dir)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
