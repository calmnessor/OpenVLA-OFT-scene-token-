"""Test RLBench RLDS data loading through the OpenVLA pipeline."""
import sys
sys.path.insert(0, '.')
from prismatic.vla.datasets.rlds.oxe.materialize import _make_rlbench_dataset_kwargs
from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
from prismatic.vla.constants import NormalizationType

print("=== Test: make_dataset_from_rlds() ===")
try:
    kwargs = _make_rlbench_dataset_kwargs(
        "rlbench_close_jar",
        "/tmp/rlbench_rlds_test",
        load_camera_views=("primary", "wrist"),
        load_proprio=True,
        load_language=True,
        action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
    )
    print("kwargs keys:", list(kwargs.keys()))
    print("  name:", kwargs["name"])
    print("  data_dir:", kwargs["data_dir"])
    print("  image_obs_keys:", kwargs["image_obs_keys"])
    print("  state_obs_keys:", kwargs["state_obs_keys"])
    print("  language_key:", kwargs.get("language_key", "N/A"))
    print("  standardize_fn:", kwargs["standardize_fn"].__name__)

    dataset, stats = make_dataset_from_rlds(
        train=True,
        standardize_fn=kwargs.pop("standardize_fn"),
        **kwargs,
    )
    print("Dataset created:", dataset)
    nt = stats.get("num_trajectories", "?")
    ntr = stats.get("num_transitions", "?")
    print(f"Stats: trajectories={nt}, transitions={ntr}")

    # Try iterating
    for batch in dataset.take(1):
        print("Batch keys:", list(batch.keys()))
        if "observation" in batch:
            for k, v in batch["observation"].items():
                if hasattr(v, "shape"):
                    print(f"  observation/{k}: shape={v.shape}, dtype={v.dtype}")
        print("  action shape:", batch["action"].shape)
        if "task" in batch:
            print("  task keys:", list(batch["task"].keys()))
            li = batch["task"]["language_instruction"]
            print(f"  language_instruction: {li}")
    print("Test PASSED!")
except Exception as e:
    print(f"Test FAILED: {e}")
    import traceback
    traceback.print_exc()
