import numpy as np

# Ground truth from actual robot state controller:
gt_q = [0.13291284567031209, 0.47590791339805755, 0.05501755740245183, -2.448578657852976, 0.02203638603289922, 2.8999886983368133, 0.9361168698153992]
gt_O_T_EE = np.array([
    [0.9995257650470911, 0.009573565932200157, -0.028937084601825702, 0.45539525733090064],
    [0.010316824195825414, -0.9996082168682501, 0.02564635398345217, 0.09130097892221659],
    [-0.02868022047941554, -0.025932730400555305, -0.999252175210093, 0.006749444042978656],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float64)

# Franka Panda Official Craig DH Parameters:
# Link i: a_{i-1}, d_i, alpha_{i-1}, theta_i
# Joint 1: a0=0,      d1=0.333, alpha0=0,      theta1=q1
# Joint 2: a1=0,      d2=0,     alpha1=-pi/2,  theta2=q2
# Joint 3: a2=0,      d3=0.316, alpha2=pi/2,   theta3=q3
# Joint 4: a3=0.0825, d4=0,     alpha3=pi/2,   theta4=q4
# Joint 5: a4=-0.0825,d5=0.384, alpha4=-pi/2,  theta5=q5
# Joint 6: a5=0,      d6=0,     alpha5=pi/2,   theta6=q6
# Joint 7: a6=0.088,  d7=0,     alpha6=pi/2,   theta7=q7
# Flange:  a7=0,      d8=0.107, alpha7=0,      theta8=0
# EE:      F_T_EE (from ROS message: d_z = 0.1034, rotation around z is -pi/4 or pi/4)

def tf_mdh(a, d, alpha, theta):
    c, s = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [c, -s, 0, a],
        [s*ca, c*ca, -sa, -sa*d],
        [s*sa, c*sa, ca, ca*d],
        [0, 0, 0, 1]
    ], dtype=np.float64)

def get_franka_fk(q, F_T_EE=None):
    # Standard Franka Modified DH
    mdh = [
        (0.0, 0.333, 0.0),
        (0.0, 0.0, -np.pi/2),
        (0.0, 0.316, np.pi/2),
        (0.0825, 0.0, np.pi/2),
        (-0.0825, 0.384, -np.pi/2),
        (0.0, 0.0, np.pi/2),
        (0.088, 0.0, np.pi/2),
        (0.0, 0.107, 0.0)
    ]
    T = np.eye(4, dtype=np.float64)
    for i in range(7):
        a, d, alpha = mdh[i]
        T = T @ tf_mdh(a, d, alpha, q[i])
    # Flange
    a, d, alpha = mdh[7]
    T = T @ tf_mdh(a, d, alpha, 0.0)
    if F_T_EE is not None:
        T = T @ F_T_EE
    return T

# The F_T_EE from Franka state message:
F_T_EE = np.array([
    [0.70710678, -0.70710678, 0.0, 0.0],
    [0.70710678,  0.70710678, 0.0, 0.0],
    [0.0,         0.0,        1.0, 0.1034],
    [0.0,         0.0,        0.0, 1.0]
], dtype=np.float64)

T_pred = get_franka_fk(gt_q, F_T_EE=F_T_EE)
print("FK Position with F_T_EE:", T_pred[:3, 3])
print("GT Position:            ", gt_O_T_EE[:3, 3])
err = np.linalg.norm(T_pred[:3, 3] - gt_O_T_EE[:3, 3])
print(f"Position Error: {err*1000:.4f} mm")

rot_err = np.linalg.norm(T_pred[:3, :3] - gt_O_T_EE[:3, :3])
print(f"Rotation Error norm: {rot_err:.6f}")
