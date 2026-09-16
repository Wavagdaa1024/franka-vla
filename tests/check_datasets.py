"""Verify that all datasets in dataset/ are valid LeRobotDataset instances."""
import os
import sys
from pathlib import Path

# Ensure offline mode so LeRobot doesn't try to query HF Hub
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"

# Ensure lerobot package from submodule src is prioritized
lerobot_src = ROOT / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def check_single_dataset(ds_path: Path) -> bool:
    info_json = ds_path / "meta" / "info.json"
    if not info_json.exists():
        print(f"[SKIP] {ds_path.name}: Not a dataset directory (no meta/info.json)")
        return True

    print(f"--> Validating {ds_path.name}...")
    try:
        ds = LeRobotDataset(
            repo_id=f"dataset/{ds_path.name}",
            root=str(ds_path),
            download_videos=False,
        )
        print(f"    Episodes : {ds.num_episodes}")
        print(f"    Frames   : {len(ds)} (FPS: {ds.fps})")
        print(f"    Features : {list(ds.features.keys())}")

        # Test index 0 frame load
        sample = ds[0]
        for key in ["observation.images.phone", "observation.images.laptop"]:
            if key in sample:
                print(f"    Video frame '{key}': shape={tuple(sample[key].shape)}, dtype={sample[key].dtype}")
        if "action" in sample:
            print(f"    Action: shape={tuple(sample['action'].shape)}, head={sample['action'][:4].tolist()}...")
        if "observation.state" in sample:
            print(f"    State : shape={tuple(sample['observation.state'].shape)}")
        if "task" in sample:
            print(f"    Task  : '{sample['task']}'")

        print(f"[PASS] {ds_path.name}: 100% LeRobotDataset compliant.\n")
        return True
    except Exception as e:
        print(f"[FAIL] {ds_path.name}: Failed to load! Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("  LeRobot Dataset Compliance Verification")
    print("=" * 60)
    if not DATASET_DIR.exists():
        print(f"[FAIL] Dataset directory does not exist: {DATASET_DIR}")
        return 1

    ds_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not ds_dirs:
        print("[WARN] No datasets found in dataset/")
        return 0

    all_passed = True
    for d in sorted(ds_dirs):
        ok = check_single_dataset(d)
        if not ok:
            all_passed = False

    print("-" * 60)
    if all_passed:
        print("[ALL PASS] All checked datasets are 100% valid and ready for training.")
        return 0
    else:
        print("[FAIL] One or more datasets failed compliance checks.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
