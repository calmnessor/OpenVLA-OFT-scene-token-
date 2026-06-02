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
import logging
import os
from datetime import datetime

import cv2
import h5py
import numpy as np

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


def visualize_demo(hdf5_path, demo_key, cameras, output_dir, fps=10):
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
        visualize_demo(hdf5_path, dk, cameras, output_dir, fps=args.fps)

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
    args = parser.parse_args()
    main(args)
