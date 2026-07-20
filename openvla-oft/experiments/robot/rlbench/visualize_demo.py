"""
visualize_demo.py

将 RLBench 演示轨迹（HDF5 格式）可视化为 MP4 视频。

用法:
    # 单条轨迹，仅前置摄像头
    python visualize_demo.py --hdf5 datasets/rlbench_raw/close_jar.hdf5 --demo 0

    # 全部轨迹，全部摄像头
    python visualize_demo.py --hdf5 datasets/rlbench_raw/close_jar.hdf5 --demo all --cameras all

    # 指定轨迹，前置+腕部
    python visualize_demo.py --hdf5 datasets/rlbench_raw/close_jar.hdf5 --demo 5 --cameras front wrist
"""

import argparse
import json
import logging
import os
from datetime import datetime

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 摄像头名称 → HDF5 数据集键名
CAMERA_KEYS = {
    "front": "front_rgb",
    "wrist": "wrist_rgb",
    "left_shoulder": "left_shoulder_rgb",
    "right_shoulder": "right_shoulder_rgb",
}

OVERLAY_KEYS = ["gripper_open", "joint_positions"]


def make_video(frames, output_path, fps=10):
    """将 BGR 帧列表写入 MP4 文件"""
    if not frames:
        logger.warning("无帧可写入")
        return
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    logger.info(f"  已保存: {output_path} ({len(frames)} 帧, {fps} fps)")


def add_overlay_text(frame, step_idx, num_steps, gripper_open):
    """在帧上叠加步数计数器和夹爪状态"""
    text_lines = [
        f"Step: {step_idx + 1}/{num_steps}",
        f"Gripper: {gripper_open:.2f}",
    ]
    y0 = 20
    for i, line in enumerate(text_lines):
        y = y0 + i * 22
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


def get_camera_frames(demo, camera_key):
    """从 demo 组中提取一个摄像头的所有帧，叠加状态信息"""
    obs = demo["obs"]
    rgbs = obs[camera_key][:].astype(np.uint8)
    num_steps = rgbs.shape[0]

    # 夹爪状态用于叠加显示
    if "gripper_open" in obs:
        gripper_opens = obs["gripper_open"][:].flatten()
    else:
        gripper_opens = [0.0] * num_steps

    frames = []
    for t in range(num_steps):
        frame = cv2.cvtColor(rgbs[t], cv2.COLOR_RGB2BGR)
        frame = add_overlay_text(frame, t, num_steps, float(gripper_opens[t]))
        frames.append(frame)

    return frames

def validate_schema_v2_demo(demo, schema_version=None):
    """Validate structure, transition alignment, and expert-command semantics."""
    checks = {}
    report = {
        "schema_version": int(schema_version) if schema_version is not None else None,
        "checks": checks,
    }

    required_obs = {
        "front_rgb",
        "wrist_rgb",
        "joint_positions",
        "gripper_open",
        "gripper_pose",
        "gripper_joint_positions",
        "ee_pos",
        "ee_quat",
    }
    obs = demo["obs"]
    missing_obs = sorted(required_obs - set(obs.keys()))
    checks["required_observation_fields"] = not missing_obs
    report["missing_observation_fields"] = missing_obs

    required_demo = {"actions", "dones", "rewards", "expert"}
    missing_demo = sorted(required_demo - set(demo.keys()))
    checks["required_demo_fields"] = not missing_demo
    report["missing_demo_fields"] = missing_demo
    if missing_obs or missing_demo:
        report["all_checks_passed"] = False
        return report

    expert = demo["expert"]
    required_expert = {"joint_position_action", "gripper_command", "action_valid"}
    missing_expert = sorted(required_expert - set(expert.keys()))
    checks["required_expert_fields"] = not missing_expert
    report["missing_expert_fields"] = missing_expert
    if missing_expert:
        report["all_checks_passed"] = False
        return report

    num_steps = obs["front_rgb"].shape[0]
    num_transitions = num_steps - 1
    obs_lengths = {key: int(obs[key].shape[0]) for key in required_obs}
    checks["observation_lengths"] = all(length == num_steps for length in obs_lengths.values())

    actions = np.asarray(demo["actions"], dtype=np.float32)
    commands = np.asarray(expert["gripper_command"], dtype=np.float32).reshape(-1)
    observed = np.asarray(obs["gripper_open"], dtype=np.float32).reshape(-1)
    valid = np.asarray(expert["action_valid"], dtype=np.bool_).reshape(-1)
    joint_actions = np.asarray(expert["joint_position_action"], dtype=np.float32)

    checks["action_shape"] = actions.shape == (num_transitions, 7)
    checks["transition_metadata"] = (
        int(demo.attrs.get("num_steps", -1)) == num_steps
        and int(demo.attrs.get("num_transitions", -1)) == num_transitions
    )
    checks["transition_array_lengths"] = (
        demo["dones"].shape == (num_transitions,)
        and demo["rewards"].shape == (num_transitions,)
    )
    checks["expert_shapes"] = (
        commands.shape == (num_steps,)
        and valid.shape == (num_steps,)
        and joint_actions.shape == (num_steps, 8)
    )

    max_position_error = None
    max_rotation_error = None
    max_gripper_error = None
    if checks["action_shape"] and checks["expert_shapes"]:
        expected_delta_pos = np.asarray(obs["ee_pos"], dtype=np.float32)[1:] - np.asarray(
            obs["ee_pos"], dtype=np.float32
        )[:-1]
        quaternions = np.asarray(obs["ee_quat"], dtype=np.float32)
        current_rot = R.from_quat(quaternions[:-1])
        next_rot = R.from_quat(quaternions[1:])
        expected_delta_rot = (next_rot * current_rot.inv()).as_rotvec()

        max_position_error = float(np.max(np.abs(actions[:, :3] - expected_delta_pos)))
        max_rotation_error = float(np.max(np.abs(actions[:, 3:6] - expected_delta_rot)))
        max_gripper_error = float(np.max(np.abs(actions[:, 6] - commands[1:])))
        checks["delta_position_alignment"] = max_position_error < 1e-5
        checks["delta_rotation_alignment"] = max_rotation_error < 1e-5
        checks["expert_gripper_alignment"] = max_gripper_error < 1e-6
        checks["finite_actions"] = bool(np.isfinite(actions).all())
    else:
        checks["delta_position_alignment"] = False
        checks["delta_rotation_alignment"] = False
        checks["expert_gripper_alignment"] = False
        checks["finite_actions"] = False

    checks["expert_action_validity"] = (
        valid.shape == (num_steps,) and not bool(valid[0]) and bool(valid[1:].all())
    )
    checks["gripper_command_range"] = bool(
        np.isfinite(commands).all() and ((commands >= 0.0) & (commands <= 1.0)).all()
    )

    report.update(
        {
            "num_steps": int(num_steps),
            "num_transitions": int(num_transitions),
            "observation_lengths": obs_lengths,
            "expert_close_frames": int((commands < 0.5).sum()),
            "expert_open_frames": int((commands >= 0.5).sum()),
            "observed_closed_frames": int((observed < 0.5).sum()),
            "observed_open_frames": int((observed >= 0.5).sum()),
            "command_state_mismatch_frames": int((commands != observed).sum()),
            "max_position_alignment_error": max_position_error,
            "max_rotation_alignment_error": max_rotation_error,
            "max_gripper_alignment_error": max_gripper_error,
        }
    )
    report["all_checks_passed"] = bool(all(checks.values()))
    return report


def get_validation_frames(demo):
    """Create a front/wrist replay with command and action diagnostics."""
    obs = demo["obs"]
    front = np.asarray(obs["front_rgb"], dtype=np.uint8)
    wrist = np.asarray(obs["wrist_rgb"], dtype=np.uint8)
    observed = np.asarray(obs["gripper_open"], dtype=np.float32).reshape(-1)
    commands = np.asarray(demo["expert"]["gripper_command"], dtype=np.float32).reshape(-1)
    valid = np.asarray(demo["expert"]["action_valid"], dtype=np.bool_).reshape(-1)
    actions = np.asarray(demo["actions"], dtype=np.float32)

    frames = []
    num_steps = front.shape[0]
    for t in range(num_steps):
        front_bgr = cv2.cvtColor(front[t], cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist[t], cv2.COLOR_RGB2BGR)
        if front_bgr.shape[:2] != wrist_bgr.shape[:2]:
            wrist_bgr = cv2.resize(wrist_bgr, (front_bgr.shape[1], front_bgr.shape[0]))

        image = np.concatenate([front_bgr, wrist_bgr], axis=1)
        cv2.putText(image, "FRONT", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            image,
            "WRIST",
            (front_bgr.shape[1] + 8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        panel = np.zeros((96, image.shape[1], 3), dtype=np.uint8)
        observed_name = "OPEN" if observed[t] >= 0.5 else "CLOSE"
        command_name = "OPEN" if commands[t] >= 0.5 else "CLOSE"
        mismatch = abs(float(observed[t] - commands[t])) > 0.5
        command_color = (0, 180, 255) if mismatch else (0, 220, 0)
        cv2.putText(
            panel,
            f"step {t + 1}/{num_steps} | observed={observed_name} | expert={command_name} | valid={int(valid[t])}",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            command_color,
            1,
            cv2.LINE_AA,
        )

        if t < actions.shape[0]:
            delta_mm = float(np.linalg.norm(actions[t, :3]) * 1000.0)
            delta_rot = float(np.linalg.norm(actions[t, 3:6]))
            next_command = "OPEN" if actions[t, 6] >= 0.5 else "CLOSE"
            action_text = (
                f"next action: dpos={delta_mm:.1f} mm | drot={delta_rot:.3f} rad | gripper={next_command}"
            )
        else:
            action_text = "terminal observation"
        cv2.putText(
            panel,
            action_text,
            (8, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "orange = physical state differs from expert command",
            (8, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        frames.append(np.concatenate([image, panel], axis=0))
    return frames


def write_validation_artifacts(demo, demo_key, output_dir, schema_version, fps):
    report = validate_schema_v2_demo(demo, schema_version=schema_version)
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, f"{demo_key}_validation.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  validation report: {report_path}")

    if not report["all_checks_passed"]:
        logger.error(f"  validation failed for {demo_key}: {report['checks']}")
        return report

    video_path = os.path.join(output_dir, f"{demo_key}_validation.mp4")
    make_video(get_validation_frames(demo), video_path, fps=fps)
    return report


def visualize_demo(hdf5_path, demo_key, cameras, output_dir, fps=10, validate=False):
    """为单条演示轨迹生成各摄像头视角的视频"""
    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())
        if demo_key not in demos:
            logger.warning(f"轨迹 '{demo_key}' 在 {hdf5_path} 中不存在。可用: {demos}")
            return

        demo = f["data"][demo_key]
        num_steps = demo["obs"]["front_rgb"].shape[0]
        lang = demo.attrs.get("language_instruction", "未知任务")

        logger.info(f"  {demo_key}: {num_steps} 步, 指令: '{lang}'")

        if validate:
            write_validation_artifacts(
                demo,
                demo_key,
                output_dir,
                schema_version=f.attrs.get("schema_version"),
                fps=fps,
            )

        for cam_name in cameras:
            cam_key = CAMERA_KEYS[cam_name]
            if cam_key not in demo["obs"]:
                logger.warning(f"  摄像头 '{cam_name}' 不在轨迹中，跳过")
                continue

            frames = get_camera_frames(demo, cam_key)
            out_name = f"{demo_key}_{cam_name}.mp4"
            out_path = os.path.join(output_dir, out_name)
            make_video(frames, out_path, fps=fps)


def main(args):
    hdf5_path = args.hdf5
    if not os.path.exists(hdf5_path):
        logger.error(f"HDF5 文件不存在: {hdf5_path}")
        return

    # 生成输出目录名
    task_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"./videos_{task_name}_{ts}"

    cameras = args.cameras
    if cameras == ["all"]:
        cameras = list(CAMERA_KEYS.keys())

    with h5py.File(hdf5_path, "r") as f:
        all_demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))

    if args.demo == "all":
        demo_keys = all_demos
    else:
        demo_keys = [f"demo_{args.demo}"]

    logger.info(f"HDF5: {hdf5_path}")
    logger.info(f"任务: {task_name}")
    logger.info(f"待可视化轨迹数: {len(demo_keys)}")
    logger.info(f"摄像头: {cameras}")
    logger.info(f"输出目录: {output_dir}")

    for dk in demo_keys:
        visualize_demo(
            hdf5_path,
            dk,
            cameras,
            output_dir,
            fps=args.fps,
            validate=args.validate,
        )

    logger.info(f"完成。视频已保存到 {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 RLBench 演示轨迹可视化为 MP4 视频")
    parser.add_argument("--hdf5", type=str, required=True,
                        help="HDF5 演示文件路径")
    parser.add_argument("--demo", type=str, default="0",
                        help="轨迹序号，或 'all' 表示全部（默认: 0）")
    parser.add_argument("--cameras", type=str, nargs="+", default=["front"],
                        choices=["front", "wrist", "left_shoulder", "right_shoulder", "all"],
                        help="要渲染的摄像头（默认: front）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认: ./videos_<task>_<timestamp>/）")
    parser.add_argument("--fps", type=int, default=10,
                        help="视频帧率（默认: 10）")
    parser.add_argument("--validate", action="store_true",
                        help="验证 schema-v2 action 对齐并输出双视角诊断视频和 JSON")
    args = parser.parse_args()
    main(args)
