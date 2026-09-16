import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(r"C:\Users\74727\Desktop\project\VLA_franka")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_pi05_expert import build_jointpos_dataset

dataset_dirs = [
    PROJECT_ROOT / "dataset" / "teleop_pick_cube_15hz_001",
    PROJECT_ROOT / "dataset" / "teleop_pick_cube_15hz_002",
    PROJECT_ROOT / "dataset" / "teleop_pick_vegetables_15hz_001",
]

all_samples, cached_files = build_jointpos_dataset(dataset_dirs)
print(f"Total samples indexed: {len(all_samples)}")
print(f"Total cached files: {len(cached_files)}")

# Check prompt distribution
from collections import Counter
c = Counter(s["task"] for s in all_samples)
print("Task distribution:")
for k, v in c.items():
    print(f"  {k}: {v} samples")
