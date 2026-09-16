import sys
import numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\74727\Desktop\project\VLA_franka")
sys.path.insert(0, str(ROOT / "lerobot" / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="dataset/teleop_pick_cube_15hz_001",
    root=str(ROOT / "dataset" / "teleop_pick_cube_15hz_001"),
    download_videos=False,
)

print(f"Total frames: {len(ds)}")
actions = []
states = []
for i in range(len(ds)):
    item = ds[i]
    actions.append(item["action"].numpy())
    states.append(item["observation.state"].numpy())

actions = np.array(actions)
states = np.array(states)

print("--- ACTIONS STATS ---")
print("ACT Mean:", np.round(np.mean(actions, axis=0), 5).tolist())
print("ACT Std: ", np.round(np.std(actions, axis=0), 5).tolist())
print("ACT Q01: ", np.round(np.percentile(actions, 1, axis=0), 5).tolist())
print("ACT Q99: ", np.round(np.percentile(actions, 99, axis=0), 5).tolist())
print("ACT Min: ", np.round(np.min(actions, axis=0), 5).tolist())
print("ACT Max: ", np.round(np.max(actions, axis=0), 5).tolist())

print("\n--- STATES STATS ---")
print("STATE Mean:", np.round(np.mean(states, axis=0), 5).tolist())
print("STATE Std: ", np.round(np.std(states, axis=0), 5).tolist())
print("STATE Q01: ", np.round(np.percentile(states, 1, axis=0), 5).tolist())
print("STATE Q99: ", np.round(np.percentile(states, 99, axis=0), 5).tolist())
