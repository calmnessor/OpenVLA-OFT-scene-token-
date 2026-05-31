# RLBench 数据管道使用指南

## 概述

RLBench 数据管道实现以下流程：

```
RLBench 环境 → collect_rlbench_demos.py → HDF5 原始数据
                                              ↓
                                   convert_rlbench_to_rlds.py
                                              ↓
                                   RLDS TFRecord 格式数据集
                                              ↓
                                   OpenVLA-OFT 训练/评估管道
```

数据格式转换的核心约定：

- **收集时**：记录原始观测（RGB、关节角、末端位姿），供后续灵活复算
- **转换时**：从原始位姿重新计算 7 维 delta action，写入 RLDS
- **训练时**：模型预测 7 维 delta action，前 6 维归一化，最后一维（夹爪）不做归一化
- **评估时**：将 7 维 delta action 转为 RLBench 需要的 8 维绝对位姿

---

## 1. 支持的 5 个任务（Evo-0 论文对齐）

Evo-0 论文 (Section IV.A) 选取了 5 个 RLBench 任务，覆盖三类精确操作：

| 任务名 | RLBench 类 | 语言指令 | 类别 | 最大步数 |
|--------|-----------|---------|------|---------|
| `play_jenga` | PlayJenga | play jenga | 精确抓取 + 搬运 | 250 |
| `put_knife_on_chopping_board` | PutKnifeOnChoppingBoard | put the knife on the chopping board | 精确抓取 + 放置 | 200 |
| `take_umbrella_out_of_umbrella_stand` | TakeUmbrellaOutOfUmbrellaStand | take the umbrella out of the umbrella stand | 精确抓取 + 搬运 | 200 |
| `place_hanger_on_rack` | PlaceHangerOnRack | place the hanger on the rack | 高度/平移变化操作 | 200 |
| `move_hanger` | MoveHanger | move the hanger | 高度/平移变化操作 | 200 |

任务注册、语言指令和最大步数定义在 `experiments/robot/rlbench/rlbench_utils.py`。

---

## 2. 环境准备（云服务器）

### 2.1 启动 Xvfb 虚拟显示

RLBench 需要显示器，headless 模式下使用 Xvfb：

```bash
Xvfb :99 -screen 0 1280x1024x24 +extension GLX -ac +render -noreset &
export DISPLAY=:99
export COPPELIASIM_HEADLESS=1
```

### 2.2 激活 conda 环境

```bash
conda activate vggt-openvla-oft
cd /opt/openvla-oft
```

---

## 3. 数据收集

### 3.1 收集命令

```bash
python experiments/robot/rlbench/collect_rlbench_demos.py \
    --tasks play_jenga put_knife_on_chopping_board \
        take_umbrella_out_of_umbrella_stand place_hanger_on_rack move_hanger \
    --demos_per_task 100 \
    --output_dir ./datasets/rlbench_raw \
    --overwrite
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tasks` | 全部 9 个 | 要收集的任务名列表 |
| `--demos_per_task` | 100 | 每个任务收集的演示数 |
| `--output_dir` | `./datasets/rlbench_raw` | HDF5 文件输出目录 |
| `--headless` | True | 无 GUI 模式 |
| `--image_size` | 256 256 | 相机分辨率 (H W) |
| `--overwrite` | False | 覆盖已有的 HDF5 文件 |

**注意事项：**

- 必须在 `/opt/openvla-oft` 目录下运行（或等效的 `~/Afford+VLA/scene token+OpenVLA-OFT/openvla-oft/`）
- 不加 `--overwrite` 时已有数据会被跳过；加 `--overwrite` 会删除旧文件重新收集
- RLBench 的脚本策略并非每次都能成功，失败的 demo 会自动跳过 — 例如收集 5 条可能只成功 4 条，属正常现象
- 脚本内部使用 `os.path.abspath()` 将输出路径转为绝对路径，防止 CoppeliaSim 运行时改变当前工作目录导致文件写入错误位置

### 3.2 收集原理

RLBench 内置了基于 Task Description Language (TDL) 的脚本策略（keyframe extractor）。收集脚本调用 `task.get_demos()` 获取由运动规划器生成的演示轨迹，记录每一步的：

- 4 个相机 RGB：front / wrist / left_shoulder / right_shoulder
- 关节位置 (7,)：`joint_positions`
- 夹爪开合度 (1,)：`gripper_open`
- 末端位姿 (7,)：`gripper_pose` = [x, y, z, qx, qy, qz, qw]
- 原始 action (8,)：脚本内部使用，**训练时不会被使用**

### 3.3 HDF5 文件结构

```
{task_name}.hdf5
└── data/
    ├── demo_0/
    │   ├── obs/
    │   │   ├── front_rgb        [T, 256, 256, 3] uint8
    │   │   ├── wrist_rgb        [T, 256, 256, 3] uint8
    │   │   ├── left_shoulder_rgb  [T, 256, 256, 3] uint8
    │   │   ├── right_shoulder_rgb [T, 256, 256, 3] uint8
    │   │   ├── joint_positions  [T, 7] float32
    │   │   ├── gripper_open     [T, 1] float32
    │   │   ├── ee_pos           [T, 3] float32
    │   │   └── ee_quat          [T, 4] float32
    │   ├── actions              [T, 8] float32 (参考用)
    │   ├── dones                [T,]  uint8
    │   ├── rewards              [T,]  float32
    │   └── attrs: num_steps, success, language_instruction
    ├── demo_1/
    └── ...
```

---

## 4. 数据转换：HDF5 → RLDS

### 4.1 转换命令

```bash
python experiments/robot/rlbench/convert_rlbench_to_rlds.py \
    --input_dir ./datasets/rlbench_raw \
    --output_dir ./datasets/rlbench_rlds \
    --tasks play_jenga put_knife_on_chopping_board
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--input_dir` | 包含 HDF5 文件的目录 |
| `--output_dir` | RLDS 数据集输出目录 |
| `--tasks` | 要转换的任务名（默认：input_dir 下所有 .hdf5 文件） |

### 4.2 输出的 RLDS 目录结构

```
rlbench_rlds/
└── rlbench_play_jenga/
    └── 0.1.0/
        ├── dataset_info.json       # 数据集元信息（名称、版本、分片信息）
        ├── features.json           # TFDS Feature 树结构
        └── rlbench_play_jenga-train.tfrecord-00000-of-00001
```

数据集命名规则为 `rlbench_{task_name}`，这一前缀在 OXE pipeline 中被 `materialize.py` 自动识别。

### 4.3 RLDS 数据格式

每个 TFRecord 的 `tf.train.Example` 包含：

```
steps: Sequence[
    observation: {
        front_rgb         Image(256, 256, 3) uint8  PNG 编码
        wrist_rgb         Image(256, 256, 3) uint8  PNG 编码
        joint_positions   Tensor(7,) float32
        gripper_open      Tensor(1,) float32
    }
    action:               Tensor(7,) float32
    language_instruction: Text
]
episode_metadata: {
    file_path:            Text
}
```

### 4.4 7 维 action 定义

```
action = [dx, dy, dz,  drx, dry, drz,  target_gripper]
          └─ delta ─┘  └─ delta rotvec ─┘  └─ absolute ─┘
```

从 HDF5 中的原始位姿数据计算得出：

```python
def compute_delta_action(ee_pos, ee_quat, gripper_open, next_ee_pos, next_ee_quat, next_gripper_open):
    # 位置：delta EE position
    delta_pos = next_ee_pos - ee_pos

    # 旋转：delta rotation vector (rotvec)
    r1 = R.from_quat(ee_quat)
    r2 = R.from_quat(next_ee_quat)
    r_diff = r2 * r1.inv()
    delta_rot = r_diff.as_rotvec()

    # 夹爪：absolute target（非 delta!）
    target_gripper = float(next_gripper_open)

    return [dx, dy, dz, drx, dry, drz, target_gripper]
```

**关键约定**（与 OXE 标准一致）：
- 位置 + 旋转：**相对位移**（delta），需要归一化
- 夹爪：**绝对目标状态**（0 = 闭合, 1 = 张开），不归一化

对应的归一化掩码：

```python
action_normalization_mask = [True, True, True, True, True, True, False]  # 不归一化夹爪
absolute_action_mask      = [False, False, False, False, False, False, True]  # 夹爪是绝对值
```

---

## 5. OXE Pipeline 集成

### 5.1 自动检测

当 `dataset_name` 以 `rlbench_` 开头时，`materialize.py` 自动路由到 RLBench 配置，无需在 `OXE_DATASET_CONFIGS` 中注册：

```python
# materialize.py: make_oxe_dataset_kwargs()
if dataset_name.startswith("rlbench_"):
    return _make_rlbench_dataset_kwargs(...)
```

### 5.2 RLBench 专用配置

`_make_rlbench_dataset_kwargs()` 自动生成以下配置：

```python
{
    "image_obs_keys": {"primary": "front_rgb", "wrist": "wrist_rgb"},
    "state_obs_keys": ["joint_positions", "gripper_open"],  # 拼接为 proprio (8,)
    "absolute_action_mask": [False]*6 + [True],
    "action_normalization_mask": [True]*6 + [False],
    "language_key": "language_instruction",
    "standardize_fn": OXE_STANDARDIZATION_TRANSFORMS["rlbench"],
}
```

### 5.3 数据变换

`transforms.py` 中的 `rlbench_dataset_transform` 是一个轻量级 pass-through，仅做 dtype 转换：

```python
def rlbench_dataset_transform(trajectory):
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    return trajectory
```

RLBench 数据已经是标准化格式（256x256 RGB、joint_positions、delta actions），无需像其他 OXE 数据集那样进行复杂的重映射。

### 5.4 dlimp pipeline 数据流

```
make_dataset_from_rlds()
  → dl.DLataset.from_rlds(tfrecords_path)    # 读取 TFRecord，展开 Sequence
  → traj_map(rlbench_dataset_transform)       # 类型转换
  → restructure()                             # 映射观测键 + 拼接 proprio
  → 后续: chunk, frame transform, normalize
```

`restructure()` 执行的映射：
- `image_obs_keys`: `front_rgb` → `image_primary`, `wrist_rgb` → `image_wrist`
- `state_obs_keys`: `joint_positions` (7,) + `gripper_open` (1,) → 拼接为 `proprio` (8,)

---

## 6. 验证数据管道

### 6.1 快速检查 HDF5 数据

```bash
python -c "
import h5py, numpy as np
with h5py.File('./datasets/rlbench_raw/play_jenga.hdf5', 'r') as f:
    demos = sorted(f['data'].keys())
    for d in demos[:3]:
        n = f['data'][d].attrs['num_steps']
        lang = f['data'][d].attrs.get('language_instruction', 'N/A')
        print(f'{d}: {n} steps, lang=\"{lang}\"')
    print(f'Total demos: {len(demos)}')
"
```

### 6.2 验证 RLDS 可加载

```bash
python -c "
from prismatic.vla.datasets.rlds.oxe.materialize import _make_rlbench_dataset_kwargs
from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
from prismatic.vla.constants import NormalizationType

kwargs = _make_rlbench_dataset_kwargs(
    'rlbench_play_jenga', './datasets/rlbench_rlds',
    load_camera_views=('primary', 'wrist'),
    load_proprio=True, load_language=True,
    action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
)
ds, stats = make_dataset_from_rlds(train=True, standardize_fn=kwargs.pop('standardize_fn'), **kwargs)
print(f'Trajectories: {stats[\"num_trajectories\"]}, Transitions: {stats[\"num_transitions\"]}')

for batch in ds.take(1):
    print(f'Action shape: {batch[\"action\"].shape}')            # (N, 7)
    print(f'Proprio shape: {batch[\"observation\"][\"proprio\"].shape}')  # (N, 8)
    print(f'Language: {batch[\"task\"][\"language_instruction\"]}')
print('OK')
"
```

---

## 7. 训练

### 7.1 GPU 精度选择

OpenVLA-7B 训练使用混合精度。不同 GPU 支持的精度不同：

| GPU | 支持精度 | `--torch_dtype` | 备注 |
|-----|---------|-----------------|------|
| A100 / H100 | bf16, fp16, fp32 | `bfloat16`（默认） | bf16 原生支持，无需额外配置 |
| **V100 32GB** | fp16, fp32 | **`float16`** | **不支持 bf16，必须指定 float16** |
| V100 16GB | fp16 + 8-bit | `float16` + `--load_in_8bit True` | 显存不足时启用 8-bit 量化 |

### 7.2 V100 训练命令（完整）

V100 不支持 bf16，且 32GB 显存跑 fp16 + 双图 + LoRA 时 batch_size=4 会导致 OOM。经验配置：`batch_size=1 + grad_accumulation_steps=4`。

```bash
cd /opt/openvla-oft

WANDB_MODE=disabled torchrun --nproc_per_node=1 vla-scripts/finetune.py \
    --vla_path /root/checkpoints/openvla-7b \
    --dataset_name rlbench_play_jenga \
    --data_root_dir ./datasets/rlbench_rlds \
    --torch_dtype float16 \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_lora True \
    --lora_rank 32 \
    --batch_size 1 \
    --grad_accumulation_steps 4 \
    --shuffle_buffer_size 500 \
    --max_steps 500 \
    --save_freq 250 \
    --learning_rate 2e-5 \
    --run_root_dir ./runs
```

关键参数说明：

| 参数 | V100 推荐值 | 说明 |
|------|-----------|------|
| `--vla_path` | `/root/checkpoints/openvla-7b` | OpenVLA-7B 基座模型路径 |
| `--torch_dtype` | `float16` | **V100 必须用 float16，不支持 bf16** |
| `--num_images_in_input` | `2` | 使用 front + wrist 双相机 |
| `--use_proprio` | `True` | 输入关节 + 夹爪 proprio |
| `--use_lora` | `True` | LoRA 微调（仅 1.45% 参数 可训练） |
| `--batch_size` | `1` | V100 32GB fp16 只能跑 bs=1 |
| `--grad_accumulation_steps` | `4` | 梯度累积，等效 batch_size=4 |
| `--shuffle_buffer_size` | `≤数据总量` | 小数据集需调小，否则 dataloader 卡住 |
| `--load_in_8bit` | `False` | 32GB V100 不需要；16GB 设为 True |

A100/H100 上训练只需去掉 `--torch_dtype float16`（默认 bf16），`batch_size` 可加大到 8-16。

### 7.3 分布式启动

训练脚本基于 PyTorch DDP，必须用 `torchrun` 启动：

```bash
# 单 GPU
torchrun --nproc_per_node=1 vla-scripts/finetune.py ...

# 多 GPU（例如 4 卡）
torchrun --nproc_per_node=4 vla-scripts/finetune.py ...
```

直接 `python vla-scripts/finetune.py ...` 会报错 `Default process group has not been initialized`。

### 7.4 训练输出

```
runs/openvla-7b+rlbench_play_jenga+b4+lr-2e-05+lora-r32+dropout-0.0--image_aug/
├── dataset_statistics.json    # 数据统计（用于推理时 unnormalize action）
├── checkpoints/               # LoRA adapter 检查点
└── ...
```

多任务混合训练支持 OXE mixture 机制，可传入多个 `dataset_name` 及权重。

---

## 8. 评估

### 8.1 运行评估

```bash
python experiments/robot/rlbench/run_rlbench_eval.py \
    --pretrained_checkpoint /path/to/checkpoint \
    --tasks play_jenga \
    --headless True \
    --num_episodes_per_task 25
```

### 8.2 评估时的 Action 转换

模型预测 7 维 delta action，RLBench 的 `EndEffectorPoseViaPlanning` 要求 8 维绝对位姿。`run_rlbench_eval.py` 自动完成转换：

```python
# 模型输出: [dx, dy, dz, drx, dry, drz, target_gripper]

# 位置: absolute = current + delta
rlbench_action[:3] = current_pose[:3] + delta_action[:3]

# 旋转: absolute = delta_rotation ⊗ current_rotation（quaternion 乘法）
r_current = R.from_quat(current_pose[3:])
r_delta = R.from_rotvec(delta_action[3:6])
r_target = r_delta * r_current
rlbench_action[3:7] = r_target.as_quat()

# 夹爪: 二值化 + 反转（OpenVLA 夹爪语义为 1=close, 0=open）
delta_action = normalize_gripper_action(delta_action, binarize=True)
delta_action = invert_gripper_action(delta_action)
rlbench_action[7] = float(np.clip(delta_action[6], 0.0, 1.0))
```

---

## 9. 服务器信息

| 项目 | 值 |
|------|-----|
| 地址 | `223.109.239.30:31440` |
| 用户 | `root` |
| 环境 | `vggt-openvla-oft` (conda) |
| 项目路径 | `/opt/openvla-oft`（软链到 `~/Afford+VLA/scene token+OpenVLA-OFT/openvla-oft/`） |
| 基座模型 | `/root/checkpoints/openvla-7b/` |
| GPU | Tesla V100-SXM2-32GB (32 GiB) |
| SSH 脚本 | `deploy/ssh_run.py`（通过 paramiko 上传文件/执行命令） |

---

## 10. 关键文件索引

| 文件 | 作用 |
|------|------|
| `openvla-oft/experiments/robot/rlbench/rlbench_utils.py` | 环境配置、任务注册、观测提取 |
| `openvla-oft/experiments/robot/rlbench/collect_rlbench_demos.py` | 收集演示 → HDF5 |
| `openvla-oft/experiments/robot/rlbench/convert_rlbench_to_rlds.py` | HDF5 → RLDS TFRecord |
| `openvla-oft/experiments/robot/rlbench/run_rlbench_eval.py` | 模型评估 |
| `openvla-oft/prismatic/vla/datasets/rlds/oxe/materialize.py` | OXE 自动检测 + RLBench 配置 |
| `openvla-oft/prismatic/vla/datasets/rlds/oxe/transforms.py` | RLBench 数据变换 |
| `openvla-oft/prismatic/vla/constants.py` | Action dim / proprio dim / chunk 配置 |
| `deploy/ssh_run.py` | SSH 远程执行工具 |

---

## 11. 完整工作流速查（V100 32GB）

```bash
# 1. 启动环境
export DISPLAY=:99 COPPELIASIM_HEADLESS=1
conda activate vggt-openvla-oft
cd /opt/openvla-oft

# 2. 收集数据（100 条/任务 × 5 任务 = 500 条演示）
python experiments/robot/rlbench/collect_rlbench_demos.py \
    --tasks play_jenga put_knife_on_chopping_board \
        take_umbrella_out_of_umbrella_stand place_hanger_on_rack move_hanger \
    --demos_per_task 100 \
    --output_dir ./datasets/rlbench_raw \
    --overwrite

# 3. 转换为 RLDS
python experiments/robot/rlbench/convert_rlbench_to_rlds.py \
    --input_dir ./datasets/rlbench_raw \
    --output_dir ./datasets/rlbench_rlds

# 4. 训练（V100 fp16 + LoRA）
WANDB_MODE=disabled torchrun --nproc_per_node=1 vla-scripts/finetune.py \
    --vla_path /root/checkpoints/openvla-7b \
    --dataset_name rlbench_play_jenga \
    --data_root_dir ./datasets/rlbench_rlds \
    --torch_dtype float16 \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_lora True \
    --batch_size 1 \
    --grad_accumulation_steps 4 \
    --max_steps 50000 \
    --save_freq 5000 \
    --learning_rate 2e-5 \
    --run_root_dir ./runs

# 5. 评估
python experiments/robot/rlbench/run_rlbench_eval.py \
    --pretrained_checkpoint ./runs/<run_dir>/checkpoints/step-XXXXX \
    --tasks play_jenga --headless True
```

A100/H100 上训练只需去掉 `--torch_dtype float16`，`batch_size` 可调大到 8-16。

---

## 12. 常见问题排查

### 12.1 收集后文件不在预期位置

**现象**：日志显示保存到 `./datasets/rlbench_raw/play_jenga.hdf5`，但目录是空的。

**原因**：RLBench/CoppeliaSim 运行时会改变当前工作目录，导致相对路径解析到其他位置。

**解决**：脚本已使用 `os.path.abspath()` 转绝对路径。升级到最新版本后重试。

### 12.2 训练 OOM

**现象**：`torch.cuda.OutOfMemoryError: CUDA out of memory`

**解决**（按优先级）：
1. `--batch_size 1 --grad_accumulation_steps N` — 减小 batch size，用梯度累积补偿
2. `--torch_dtype float16` — V100 确认使用 fp16
3. `--load_in_8bit True` — 启用 8-bit 量化（会损失少量精度）
4. `--num_images_in_input 1` — 只用 front 相机（放弃 wrist）

### 12.3 `RuntimeError: Current CUDA Device does not support bfloat16`

**原因**：V100 不支持 bf16，训练脚本默认使用 bf16。

**解决**：加 `--torch_dtype float16`。

### 12.4 `Default process group has not been initialized`

**原因**：直接 `python` 运行 DDP 脚本。

**解决**：用 `torchrun --nproc_per_node=1` 启动。

### 12.5 `Cannot copy out of meta tensor; no data!`

**原因**：`low_cpu_mem_usage=True` 与自定义模型类不兼容（特定 PyTorch 版本）。

**解决**：脚本已改用 `device_map` 方式加载，更新到最新版本。

### 12.6 训练后 config.json 被修改

**现象**：每次训练 `finetune.py` 会修改 `/root/checkpoints/openvla-7b/config.json`。

**原因**：训练脚本自动注册 OpenVLA HF Auto Classes，修改 `auto_map` 字段。会创建 `.back.YYYYMMDD_HHMMSS` 备份。

**解决**：这是正常行为。如需恢复，复制最新的 backup 文件：`cp config.json.back.XXXXXX config.json`。

### 12.7 收集时 demo 被跳过

**现象**：`WARNING: Skipping demo N: Could not collect demos.`

**原因**：RLBench 脚本策略在该次尝试中未成功完成任务。

**解决**：正常现象，无需处理。收集脚本会自动重试直到收集够 `--demos_per_task` 条成功演示。
