#!/usr/bin/env python
"""Extract VGGT-Omega teacher caches for RLBench HDF5 demonstrations.

The converter consumes the resulting files with --teacher_cache_dir. Each output
file contains:
  - teacher_scene_registers: [T, V, 17, 2048]
  - teacher_depth: [T, V, H, W]

Default views are front_rgb and wrist_rgb to match OpenVLA's two-image RLBench
policy input.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image


def load_model(checkpoint: Path, vggt_repo: Path):
    if str(vggt_repo) not in sys.path:
        sys.path.insert(0, str(vggt_repo))
    from vggt_omega.models import VGGTOmega

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run VGGT-Omega teacher extraction.")
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to("cuda")


def preprocess_arrays(arrays: list[np.ndarray], image_resolution: int, vggt_repo: Path) -> torch.Tensor:
    if str(vggt_repo) not in sys.path:
        sys.path.insert(0, str(vggt_repo))
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for idx, arr in enumerate(arrays):
            path = Path(tmpdir) / f"view_{idx}.png"
            Image.fromarray(arr.astype(np.uint8)).save(path)
            image_paths.append(str(path))
        return load_and_preprocess_images(image_paths, image_resolution=image_resolution).to("cuda")


def extract_demo(model, obs_group, views: list[str], image_resolution: int, vggt_repo: Path):
    num_steps = obs_group[views[0]].shape[0]
    register_chunks = []
    depth_chunks = []

    for t in range(num_steps - 1):
        arrays = [obs_group[view][t] for view in views]
        images = preprocess_arrays(arrays, image_resolution, vggt_repo)
        with torch.inference_mode():
            pred = model(images)

        registers = pred["camera_and_register_tokens"].detach().float().cpu().numpy()[0]
        depth = pred["depth"].detach().float().cpu().numpy()[0]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        register_chunks.append(registers.astype(np.float32))
        depth_chunks.append(depth.astype(np.float32))

    return np.stack(register_chunks, axis=0), np.stack(depth_chunks, axis=0)


def extract_task(args, model, task_name: str):
    input_path = Path(args.input_dir) / f"{task_name}.hdf5"
    if not input_path.exists():
        print(f"Skip missing task HDF5: {input_path}")
        return

    out_task_dir = Path(args.output_dir) / task_name
    out_task_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for demo_key in demo_keys:
            out_path = out_task_dir / f"{demo_key}.npz"
            if out_path.exists() and not args.overwrite:
                print(f"Skip existing cache: {out_path}")
                continue

            obs = f["data"][demo_key]["obs"]
            teacher_scene_registers, teacher_depth = extract_demo(
                model,
                obs,
                args.views,
                args.image_resolution,
                Path(args.vggt_repo),
            )
            np.savez_compressed(
                out_path,
                teacher_scene_registers=teacher_scene_registers,
                teacher_depth=teacher_depth,
            )
            print(
                f"Wrote {out_path}: registers={teacher_scene_registers.shape}, depth={teacher_depth.shape}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing RLBench HDF5 files")
    parser.add_argument("--output_dir", required=True, help="Directory for per-demo teacher .npz files")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--vggt_repo", default="/home/jovyan/home/suziyang/code/vggt-omega")
    parser.add_argument("--checkpoint", required=True, help="VGGT-Omega checkpoint .pt")
    parser.add_argument("--image_resolution", type=int, default=256)
    parser.add_argument("--views", nargs="+", default=["front_rgb", "wrist_rgb"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    tasks = args.tasks or sorted(path.stem for path in input_dir.glob("*.hdf5"))
    model = load_model(Path(args.checkpoint), Path(args.vggt_repo))

    os.makedirs(args.output_dir, exist_ok=True)
    for task_name in tasks:
        extract_task(args, model, task_name)


if __name__ == "__main__":
    main()
