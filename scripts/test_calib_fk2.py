import numpy as np

gt_q = [0.13291284567031209, 0.47590791339805755, 0.05501755740245183, -2.448578657852976, 0.02203638603289922, 2.8999886983368133, 0.9361168698153992]
gt_O_T_EE = np.array([
    [0.9995257650470911, 0.009573565932200157, -0.028937084601825702, 0.45539525733090064],
    [0.010316824195825414, -0.9996082168682501, 0.02564635398345217, 0.09130097892221659],
    [-0.02868022047941554, -0.025932730400555305, -0.999252175210093, 0.006749444042978656],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float64)

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
    a, d, alpha = mdh[7]
    T = T @ tf_mdh(a, d, alpha, 0.0)
    if F_T_EE is not None:
        T = T @ F_T_EE
    return T

# Try both signs of pi/4
c = np.cos(np.pi/4)
s = np.sin(np.pi/4)
F_T_EE_1 = np.array([
    [ c, -s, 0, 0],
    [ s,  c, 0, 0],
    [ 0,  0, 1, 0.1034],
    [ 0,  0, 0, 1]
], dtype=np.float64)

F_T_EE_2 = np.array([
    [ c,  s, 0, 0],
    [-s,  c, 0, 0],
    [ 0,  0, 1, 0.1034],
    [ 0,  0, 0, 1]
], dtype=np.float64)

for idx, F_T_EE in enumerate([F_T_EE_1, F_T_EE_2]):
    T_pred = get_franka_fk(gt_q, F_T_EE=F_T_EE)
    rot_err = np.linalg.norm(T_pred[:3, :3] - gt_O_T_EE[:3, :3])
    print(f"F_T_EE_{idx+1} Rot Error norm: {rot_err:.6f}")
    if rot_err < 1e-4:
        print(f"MATCH FOUND for F_T_EE_{idx+1}!")
