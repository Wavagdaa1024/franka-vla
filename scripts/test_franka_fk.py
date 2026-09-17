import numpy as np

def rot_x(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float64)

def trans_x(a):
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = a
    return T

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

def trans_z(d):
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = d
    return T

def franka_fk(q, ee_offset=0.1034):
    mdh = [
        (0.0, 0.333, 0.0),
        (0.0, 0.0, -np.pi/2),
        (0.0, 0.316, np.pi/2),
        (0.0825, 0.0, np.pi/2),
        (-0.0825, 0.384, -np.pi/2),
        (0.0, 0.0, np.pi/2),
        (0.088, 0.0, np.pi/2),
        (0.0, ee_offset, 0.0)
    ]
    T = np.eye(4, dtype=np.float64)
    for i in range(7):
        a, d, alpha = mdh[i]
        theta = q[i]
        T = T @ (rot_x(alpha) @ trans_x(a) @ rot_z(theta) @ trans_z(d))
    T = T @ (rot_x(mdh[7][2]) @ trans_x(mdh[7][0]) @ trans_z(mdh[7][1]))
    return T

# Ground truth from actual robot state controller:
gt_q = [0.13291284567031209, 0.47590791339805755, 0.05501755740245183, -2.448578657852976, 0.02203638603289922, 2.8999886983368133, 0.9361168698153992]
gt_O_T_EE = np.array([
    [0.9995257650470911, 0.009573565932200157, -0.028937084601825702, 0.45539525733090064],
    [0.010316824195825414, -0.9996082168682501, 0.02564635398345217, 0.09130097892221659],
    [-0.02868022047941554, -0.025932730400555305, -0.999252175210093, 0.006749444042978656],
    [0.0, 0.0, 0.0, 1.0]
])

T_pred = franka_fk(gt_q, ee_offset=0.1034)
print("FK Position Pred (ee_offset=0.1034):", T_pred[:3, 3])
print("GT Position:                        ", gt_O_T_EE[:3, 3])
diff = np.linalg.norm(T_pred[:3, 3] - gt_O_T_EE[:3, 3])
print(f"Position Error: {diff*1000:.3f} mm")

# Find exact offset that matches libfranka's F_T_EE
for off in np.linspace(0.100, 0.110, 1001):
    T_test = franka_fk(gt_q, ee_offset=off)
    err = np.linalg.norm(T_test[:3, 3] - gt_O_T_EE[:3, 3])
    if err < 0.0005:
        print(f"Optimal ee_offset: {off:.5f}m, error: {err*1000:.3f} mm")
        break
