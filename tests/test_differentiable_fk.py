import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from franka_teleop.kinematics import (
    forward_kinematics,
    MDH_PARAMS,
    DEFAULT_F_T_EE,
    FrankaDifferentiableKinematics
)


def test_differentiable_fk():
    print("=" * 80)
    print("  TESTING FRANKA DIFFERENTIABLE FORWARD KINEMATICS (DFK)")
    print("=" * 80)

    dfk = FrankaDifferentiableKinematics()

    # Test cases: standard configurations
    test_configs = [
        np.array([0.0, -np.pi/4.0, 0.0, -3*np.pi/4.0, 0.0, np.pi/2.0, np.pi/4.0], dtype=np.float32),
        np.array([0.1329, 0.4759, 0.0550, -2.4485, 0.0220, 2.8999, 0.9361], dtype=np.float32),
        np.array([0.45, -0.25, 0.10, -2.10, 0.05, 2.50, 0.785], dtype=np.float32),
    ]

    for idx, q_np in enumerate(test_configs):
        T_gt = forward_kinematics(q_np)
        p_gt = T_gt[:3, 3]
        z_gt = T_gt[:3, 2]

        q_t = torch.tensor(q_np, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
        p_pred, z_pred = dfk(q_t)

        z_p = z_pred.detach().numpy()[0]
        p_err_mm = np.linalg.norm(p_pred.detach().numpy()[0] - p_gt) * 1000.0
        z_vec_err = np.linalg.norm(z_p - z_gt)
        z_p_u = z_p / np.linalg.norm(z_p)
        z_gt_u = z_gt / np.linalg.norm(z_gt)
        z_dot = np.clip(np.dot(z_p_u, z_gt_u), -1.0, 1.0)
        tilt_err_deg = np.degrees(np.arccos(z_dot))

        print(f"Config #{idx+1}: Pos Error = {p_err_mm:.6f} mm, Z Vec L2 = {z_vec_err:.6e}, Tilt Error = {tilt_err_deg:.6f} deg")
        assert p_err_mm < 0.005, f"Pos error too large: {p_err_mm} mm"
        assert z_vec_err < 0.001, f"Z vector error too large: {z_vec_err}"
        assert tilt_err_deg < 0.05, f"Tilt error too large: {tilt_err_deg} deg"

        # Autograd verification
        loss = p_pred.sum() + z_pred.sum()
        loss.backward()
        grad = q_t.grad.numpy()[0]
        assert not np.isnan(grad).any(), "NaN in gradient"
        assert not np.isinf(grad).any(), "Inf in gradient"
        assert np.linalg.norm(grad) > 0.0, "Zero gradient"

    # Batch test: (B, 15, 7)
    B = 4
    T = 15
    q_batch = torch.randn(B, T, 7, dtype=torch.float32, requires_grad=True)
    p_b, z_b = dfk(q_batch)
    assert p_b.shape == (B, T, 3), f"Wrong shape: {p_b.shape}"
    assert z_b.shape == (B, T, 3), f"Wrong shape: {z_b.shape}"

    # Vertical downward loss test
    target_z = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    vert_loss = ((z_b - target_z) ** 2).mean()
    vert_loss.backward()
    assert q_batch.grad is not None
    # GPU CUDA test if available
    if torch.cuda.is_available():
        dfk_cuda = FrankaDifferentiableKinematics(device="cuda:0")
        q_cuda = torch.randn(B, T, 7, device="cuda:0", dtype=torch.float32, requires_grad=True)
        p_cu, z_cu = dfk_cuda(q_cuda)
        assert p_cu.device.type == "cuda"
        assert z_cu.device.type == "cuda"
        loss_cu = p_cu.sum() + ((z_cu - target_z.to("cuda:0")) ** 2).sum()
        loss_cu.backward()
        assert q_cuda.grad is not None and not torch.isnan(q_cuda.grad).any()
        print("  [*] GPU CUDA Forward & Backward Pass verified on", torch.cuda.get_device_name(0))

    print("\n[SUCCESS] Franka DFK passed all consistency and gradient checks 100%!")


if __name__ == "__main__":
    test_differentiable_fk()
