import json
from pathlib import Path

dataset_dir = Path(r"C:\Users\74727\Desktop\project\VLA_franka\dataset")
print(f"Scanning dataset directory: {dataset_dir}")

for d in sorted(dataset_dir.iterdir()):
    if not d.is_dir():
        continue
    print("\n" + "=" * 60)
    print(f"DATASET: {d.name}")
    info_file = d / "meta" / "info.json"
    if info_file.exists():
        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)
            print(f"  Total episodes: {info.get('total_episodes')}")
            print(f"  Total frames:   {info.get('total_frames')}")
            print(f"  FPS:            {info.get('fps')}")
            print(f"  Robot type:     {info.get('robot_type')}")
            feats = info.get("features", {})
            print("  Features:")
            for k, v in feats.items():
                print(f"    {k}: dtype={v.get('dtype')}, shape={v.get('shape')}")
    else:
        print("  [WARN] meta/info.json not found!")

    tasks_file = d / "meta" / "tasks.jsonl"
    if tasks_file.exists():
        with open(tasks_file, "r", encoding="utf-8") as f:
            tasks = [json.loads(line) for line in f]
            print(f"  Tasks ({len(tasks)}):")
            for t in tasks:
                print(f"    - {t}")
    episodes_file = d / "meta" / "episodes.jsonl"
    if episodes_file.exists():
        with open(episodes_file, "r", encoding="utf-8") as f:
            episodes = [json.loads(line) for line in f]
            print(f"  Total recorded episode entries in jsonl: {len(episodes)}")
            if len(episodes) > 0:
                print(f"    Sample episode[0]: {episodes[0]}")
                if "tasks" in episodes[0] or "task" in episodes[0]:
                    task_keys = set(ep.get("task", ep.get("tasks", "")) for ep in episodes)
                    print(f"    Distinct tasks across episodes: {task_keys}")
