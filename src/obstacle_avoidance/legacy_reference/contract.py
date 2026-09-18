"""
State snapshot and reproducibility contract interface for AG-S4-001 / A0.
Ensures serialization contract covering:
1. Simulation state
2. Controller internal state
3. Policy and action queue state
4. RNG state
Explicitly marks `real_snapshot_restore_verified = False` and "真实快照恢复未验证" in A0.
"""

from dataclasses import dataclass, field, asdict
import json
from typing import Any, Dict, List, Optional
from src.types import ValidationError


@dataclass
class SimulationStateSnapshot:
    sim_time: float
    physics_step: int
    qpos: List[float]
    qvel: List[float]
    object_poses: Dict[str, List[float]]

    def validate(self) -> None:
        if self.physics_step < 0:
            raise ValidationError("physics_step must be non-negative")
        if not isinstance(self.qpos, list) or not isinstance(self.qvel, list):
            raise ValidationError("qpos and qvel must be lists")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ControllerStateSnapshot:
    target_q: List[float]
    target_qd: List[float]
    integral_errors: Optional[List[float]] = None

    def validate(self) -> None:
        if not isinstance(self.target_q, list) or not isinstance(self.target_qd, list):
            raise ValidationError("target_q and target_qd must be lists")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyQueueStateSnapshot:
    action_queue: List[List[float]]
    remaining_chunk_steps: int
    current_chunk_id: int
    policy_latent_cache: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if self.remaining_chunk_steps < 0:
            raise ValidationError("remaining_chunk_steps must be non-negative")
        if self.current_chunk_id < 0:
            raise ValidationError("current_chunk_id must be non-negative")
        if not isinstance(self.action_queue, list):
            raise ValidationError("action_queue must be a list")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RNGStateSnapshot:
    python_rng_state: Any
    numpy_rng_state: Optional[Any] = None
    torch_rng_state: Optional[Any] = None

    def validate(self) -> None:
        if self.python_rng_state is None:
            raise ValidationError("python_rng_state must not be None")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_rng_state": self.python_rng_state,
            "numpy_rng_state": self.numpy_rng_state,
            "torch_rng_state": self.torch_rng_state,
        }


@dataclass
class FullSnapshotContract:
    snapshot_id: str
    sim_state: SimulationStateSnapshot
    controller_state: ControllerStateSnapshot
    policy_queue_state: PolicyQueueStateSnapshot
    rng_state: RNGStateSnapshot
    # Strictly must be False in A0 per TASK.md and ACCEPTANCE.md
    real_snapshot_restore_verified: bool = False
    verification_note: str = "真实快照恢复未验证 (A0纯CPU阶段，缺少真实仿真与控制器硬件，仅测试序列化契约)"

    def validate(self) -> None:
        if not self.snapshot_id:
            raise ValidationError("snapshot_id cannot be empty")
        self.sim_state.validate()
        self.controller_state.validate()
        self.policy_queue_state.validate()
        self.rng_state.validate()

    def serialize(self) -> Dict[str, Any]:
        self.validate()
        return {
            "snapshot_id": self.snapshot_id,
            "sim_state": self.sim_state.to_dict(),
            "controller_state": self.controller_state.to_dict(),
            "policy_queue_state": self.policy_queue_state.to_dict(),
            "rng_state": self.rng_state.to_dict(),
            "real_snapshot_restore_verified": self.real_snapshot_restore_verified,
            "verification_note": self.verification_note,
        }

    @classmethod
    def deserialize(cls, d: Dict[str, Any]) -> "FullSnapshotContract":
        sim_s = SimulationStateSnapshot(**d["sim_state"])
        ctrl_s = ControllerStateSnapshot(**d["controller_state"])
        pol_s = PolicyQueueStateSnapshot(**d["policy_queue_state"])
        rng_s = RNGStateSnapshot(**d["rng_state"])
        snapshot = cls(
            snapshot_id=d["snapshot_id"],
            sim_state=sim_s,
            controller_state=ctrl_s,
            policy_queue_state=pol_s,
            rng_state=rng_s,
            real_snapshot_restore_verified=bool(d.get("real_snapshot_restore_verified", False)),
            verification_note=str(d.get("verification_note", "真实快照恢复未验证")),
        )
        snapshot.validate()
        return snapshot
