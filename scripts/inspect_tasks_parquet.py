import pandas as pd
from pathlib import Path

dataset_dir = Path(r"C:\Users\74727\Desktop\project\VLA_franka\dataset")

for d in sorted(dataset_dir.iterdir()):
    if not d.is_dir():
        continue
    print("=" * 60)
    print(f"DATASET: {d.name}")
    tasks_pq = d / "meta" / "tasks.parquet"
    if tasks_pq.exists():
        df_tasks = pd.read_parquet(tasks_pq)
        print("  Tasks DataFrame:")
        print(df_tasks)
    
    episodes_dir = d / "meta" / "episodes"
    if episodes_dir.exists():
        ep_files = list(episodes_dir.glob("*.parquet"))
        print(f"  Episode parquet files count: {len(ep_files)}")
        if ep_files:
            df_ep = pd.read_parquet(ep_files[0])
            print("  Episodes sample (first 3 rows):")
            print(df_ep.head(3))
            if "tasks" in df_ep.columns:
                all_tasks = set()
                for t in df_ep["tasks"]:
                    if isinstance(t, (list, tuple)):
                        all_tasks.update(t)
                    else:
                        all_tasks.add(t)
                print(f"  Tasks listed in episodes: {all_tasks}")
            if "task_index" in df_ep.columns:
                print(f"  Task index distribution:\n{df_ep['task_index'].value_counts()}")
