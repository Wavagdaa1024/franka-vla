import numpy as np
from franka_teleop.kinematics import forward_kinematics, analytical_jacobian, damped_pinv, FRANKA_JOINT_LIMITS

q_ready = np.array([-0.0802, 0.0305, 0.0914, -2.4492, 0.0313, 2.4519, 0.7947])
T_ready = forward_kinematics(q_ready)
R_target = T_ready[:3, :3].copy()
p_target = np.array([0.55, 0.15, 0.08])

q_perturbed = q_ready.copy()
q_perturbed[6] += 0.35  # ~20 deg yaw rotation!
q_perturbed[4] -= 0.20  # ~11 deg tilt
q_perturbed[5] += 0.15

T_pert = forward_kinematics(q_perturbed)
R_pert = T_pert[:3, :3]
R_rel_init = R_target @ R_pert.T
ang_init = np.arccos(np.clip((np.trace(R_rel_init) - 1.0)/2.0, -1.0, 1.0)) * 180.0 / np.pi
print(f"Initial perturbed: 3D SO(3) angle error = {ang_init:.2f} deg, Pos err = {np.linalg.norm(T_pert[:3, 3] - p_target)*1000:.1f} mm")

q = q_perturbed.copy()
for it in range(35):
    T = forward_kinematics(q)
    p_curr = T[:3, 3]
    R_curr = T[:3, :3]
    pos_err = p_target - p_curr
    rot_err = 0.5 * (np.cross(R_curr[:, 0], R_target[:, 0]) + np.cross(R_curr[:, 1], R_target[:, 1]) + np.cross(R_curr[:, 2], R_target[:, 2]))
    if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-4:
        print(f"Converged in {it+1} iters!")
        break
    dx = np.concatenate([pos_err, rot_err])
    J = analytical_jacobian(q)
    J_pinv = damped_pinv(J, damping=1e-3)
    dq = J_pinv @ dx
    N = np.eye(7) - J_pinv @ J
    dq += N @ (0.5 * (q_ready - q))
    dq_norm = np.linalg.norm(dq)
    if dq_norm > 0.15:
        dq = dq * (0.15 / dq_norm)
    q += dq
    for j in range(7):
        q[j] = np.clip(q[j], FRANKA_JOINT_LIMITS[j][0], FRANKA_JOINT_LIMITS[j][1])

T_final = forward_kinematics(q)
p_final = T_final[:3, 3]
R_final = T_final[:3, :3]
R_rel_fin = R_target @ R_final.T
ang_final = np.arccos(np.clip((np.trace(R_rel_fin) - 1.0)/2.0, -1.0, 1.0)) * 180.0 / np.pi
print(f"Final: 3D SO(3) angle error = {ang_final:.4f} deg, Pos err = {np.linalg.norm(p_final - p_target)*1000:.3f} mm")
