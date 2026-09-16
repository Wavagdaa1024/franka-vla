import json
from pathlib import Path

dataset_dir = Path(r"C:\Users\74727\Desktop\project\VLA_franka\dataset")

for d in sorted(dataset_dir.iterdir()):
    if not d.is_dir():
        continue
    print("=" * 60)
    print(f"DATASET: {d.name}")
    meta_dir = d / "meta"
    for meta_file in sorted(meta_dir.iterdir()):
        print(f"  File: {meta_file.name} (size: {meta_file.stat().st_size} bytes)")
        if meta_file.name.endswith(".json"):
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if meta_file.name == "info.json":
                    pass
                else:
                    print(f"    Content: {data}")
        elif meta_file.name.endswith(".jsonl"):
            with open(meta_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
                print(f"    Line count: {len(lines)}")
                if len(lines) > 0:
                    print(f"    Line 0: {lines[0]}")
                    if len(lines) > 1:
                        # Print all tasks or unique tasks
                        tasks_seen = set()
                        for item in lines:
                            for k in ["task", "tasks"]:
                                if k in item:
                                    val = item[k]
                                    if isinstance(val, list):
                                        tasks_seen.update(val)
                                    else:
                                        tasks_seen.add(val)
                        if tasks_seen:
                            print(f"    All unique tasks in {meta_file.name}: {tasks_seen}")
