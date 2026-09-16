import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

DATASET_DIR = Path(r"C:\Users\74727\Desktop\project\VLA_franka\dataset\teleop_pick_cube_15hz_001")

def rot_x(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

def trans_x(a):
    T = np.eye(4)
    T[0, 3] = a
    return T

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def trans_z(d):
    T = np.eye(4)
    T[2, 3] = d
    return T

def franka_fk(q, ee_offset=0.107):
    mdh = [
        (0, 0.333, 0),
        (0, 0, -np.pi/2),
        (0, 0.316, np.pi/2),
        (0.0825, 0, np.pi/2),
        (-0.0825, 0.384, -np.pi/2),
        (0, 0, np.pi/2),
        (0.088, 0, np.pi/2),
        (0, ee_offset, 0)
    ]
    T = np.eye(4)
    for i in range(7):
        a, d, alpha = mdh[i]
        theta = q[i]
        T = T @ (rot_x(alpha) @ trans_x(a) @ rot_z(theta) @ trans_z(d))
    return T

data_dir = DATASET_DIR / "data" / "chunk-000"
parquet_files = sorted(data_dir.glob("file-*.parquet"))

all_init_q = []
all_grasp_q = []
all_grasp_xyz = []

for pf in parquet_files:
    tbl = pq.read_table(str(pf))
    states = np.array([r.as_py() for r in tbl.column("observation.state")])
    ep_indices = np.array(tbl.column("episode_index").to_pylist())
    unique_eps = np.unique(ep_indices)

    for ep in unique_eps:
        mask = (ep_indices == ep)
        ep_states = states[mask]
        init_q = ep_states[0, :7]
        all_init_q.append(init_q)

        # Grasp frame is where gripper first closes (g > 0.5)
        grippers = ep_states[:, 7]
        closed_indices = np.where(grippers > 0.4)[0]
        if len(closed_indices) > 0:
            grasp_idx = closed_indices[0]
        else:
            grasp_idx = len(ep_states) // 2

        grasp_q = ep_states[grasp_idx, :7]
        all_grasp_q.append(grasp_q)

        T = franka_fk(grasp_q)
        xyz = T[:3, 3]
        all_grasp_xyz.append(xyz)

all_init_q = np.array(all_init_q)
all_grasp_q = np.array(all_grasp_q)
all_grasp_xyz = np.array(all_grasp_xyz)

print(f"Total analyzed episodes: {len(all_init_q)}")
print("-" * 65)
print(f"Initial Joint 0 (rad):  mean={all_init_q[:, 0].mean():+.3f}, std={all_init_q[:, 0].std():.3f}, min={all_init_q[:, 0].min():+.3f}, max={all_init_q[:, 0].max():+.3f}")
print(f"Grasp Joint 0 (rad):    mean={all_grasp_q[:, 0].mean():+.3f}, std={all_grasp_q[:, 0].std():.3f}, min={all_grasp_q[:, 0].min():+.3f}, max={all_grasp_q[:, 0].max():+.3f}")
print(f"Grasp Joint 0 (deg):    mean={np.degrees(all_grasp_q[:, 0].mean()):+.1f} deg, range=[{np.degrees(all_grasp_q[:, 0].min()):+.1f}, {np.degrees(all_grasp_q[:, 0].max()):+.1f}] deg")
print("-" * 65)
print(f"Grasp Cartesian X (forward): mean={all_grasp_xyz[:, 0].mean():.3f} m, min={all_grasp_xyz[:, 0].min():.3f}, max={all_grasp_xyz[:, 0].max():.3f}")
print(f"Grasp Cartesian Y (left/right, +left/-right): mean={all_grasp_xyz[:, 1].mean():+.3f} m, min={all_grasp_xyz[:, 1].min():+.3f}, max={all_grasp_xyz[:, 1].max():+.3f}")
print(f"Grasp Cartesian Z (height):  mean={all_grasp_xyz[:, 2].mean():.3f} m, min={all_grasp_xyz[:, 2].min():.3f}, max={all_grasp_xyz[:, 2].max():.3f}")
print("-" * 65)
