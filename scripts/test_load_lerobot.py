from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset_dir = Path(r"C:\Users\74727\Desktop\project\VLA_franka\dataset")

for d in sorted(dataset_dir.iterdir()):
    if not d.is_dir():
        continue
    print("=" * 60)
    print(f"Loading LeRobotDataset from: {d.name}")
    try:
        ds = LeRobotDataset(repo_id="local/test", root=d)
        print(f"  Length: {len(ds)} frames, num_episodes: {ds.num_episodes}")
        item0 = ds[0]
        print(f"  Item keys: {list(item0.keys())}")
        print(f"  Item[0] task_index: {item0.get('task_index')}")
        if hasattr(ds, "meta") and hasattr(ds.meta, "tasks"):
            print(f"  ds.meta.tasks: {ds.meta.tasks}")
            # Check mapping task_index -> task_string
            task_idx = item0.get("task_index").item() if hasattr(item0.get("task_index"), "item") else int(item0.get("task_index"))
            # In lerobot, tasks can be dictionary or DataFrame
            print(f"  Mapped task for item 0: {ds.meta.tasks}")
    except Exception as e:
        print(f"  Failed to load: {e}")
        import traceback
        traceback.print_exc()
