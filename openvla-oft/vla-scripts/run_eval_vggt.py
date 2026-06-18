import os, sys
sys.path.insert(0, "/root/openvla-oft")
sys.path.insert(0, "/root/LIBERO")
os.chdir("/root/openvla-oft/experiments/robot/libero")
sys.path.insert(0, os.getcwd())
sys.path.append("../..")
os.environ["WANDB_MODE"] = "disabled"
os.environ["MUJOCO_GL"] = "egl"

from run_libero_eval import GenerateConfig, TaskSuite, eval_libero

cfg = GenerateConfig(
    model_family="openvla",
    pretrained_checkpoint="/root/openvla-oft/runs/vggt-eval-checkpoint",
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_scene_tokens=True,
    vggt_checkpoint="/root/checkpoints/vggt_omega_1b_512/vggt_omega_1b_512.pt",
    use_proprio=True,
    center_crop=True,
    num_open_loop_steps=8,
    lora_rank=32,
    unnorm_key="libero_spatial_no_noops",
    load_in_8bit=False,
    load_in_4bit=False,
    task_suite_name=TaskSuite.LIBERO_SPATIAL.value,
    num_steps_wait=10,
    num_trials_per_task=50,
    initial_states_path="DEFAULT",
    env_img_res=256,
    run_id_note="vggt-openvla-oft-13188",
    local_log_dir="./experiments/logs",
    use_wandb=False,
    wandb_entity="",
    wandb_project="",
    seed=7,
)

print("Starting evaluation...")
print(f"Checkpoint: {cfg.pretrained_checkpoint}")
print(f"Scene tokens: {cfg.use_scene_tokens}")
print(f"VGGT checkpoint: {cfg.vggt_checkpoint}")
print(f"Task suite: {cfg.task_suite_name}")
print(f"Trials per task: {cfg.num_trials_per_task}")

final_rate = eval_libero.__wrapped__(cfg)
print("")
print(f"FINAL SUCCESS RATE: {final_rate:.4f} ({final_rate*100:.1f}%)")
