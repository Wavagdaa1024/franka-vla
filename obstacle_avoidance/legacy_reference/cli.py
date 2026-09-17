"""
CLI entry point for AG-S4-001 / A0 data protocol and synthetic verification.
Standard library only.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import List

from src.aggregator import AggregatorConfig, EventAggregator
from src.contract import (
    ControllerStateSnapshot,
    FullSnapshotContract,
    PolicyQueueStateSnapshot,
    RNGStateSnapshot,
    SimulationStateSnapshot,
)
from src.logger import EpisodeLogger, ManifestManager
from src.residual import compute_action_residual, forbid_cross_space_subtraction
from src.types import (
    ActionSemantics,
    ActionSpace,
    ActionUnit,
    ConstraintKey,
    LatencyBreakdown,
    StepRecord,
    TerminationReason,
    ValidationError,
)


def generate_synthetic_episode(
    output_dir: Path,
    episode_id: str,
    scene_seed: int,
    pattern: str,
) -> dict:
    """
    Generates a controlled synthetic episode according to specified pattern.
    Patterns:
    - 'zero_intervention': normal actions, no filter modification
    - 'continuous_single_call': multiple ticks in same policy call intervened on same constraint
    - 'cross_call_repeat': policy call 0 triggers C1, policy call 1 triggers C1 again
    - 'distinct_constraints': policy call 0 triggers C1, policy call 1 triggers C2
    - 'short_timeout': episode terminates at step 3 while window is open -> right-censored
    """
    logger = EpisodeLogger(
        output_dir=output_dir,
        episode_id=episode_id,
        scene_seed=scene_seed,
        aggregator_config=AggregatorConfig(H=3, epsilon=1e-4),
        source="synthetic_unit_test",
    )

    control_dt = 0.02
    chunk_size = 5

    if pattern == "zero_intervention":
        for step_idx in range(15):
            call_id = step_idx // chunk_size
            sim_t = step_idx * control_dt
            action = [0.1, 0.0, -0.05, 0.0, 0.0, 0.0]
            rec = StepRecord(
                episode_id=episode_id,
                scene_seed=scene_seed,
                policy_call_id=call_id,
                chunk_id=call_id,
                control_step=step_idx,
                sim_time=sim_t,
                observation_time=sim_t,
                proposed_action=action,
                commanded_safe_action=action,
                measured_state_delta=action,
                measured_joint_state=[0.0] * 6,
                measured_ee_state=[0.4, 0.0, 0.3, 0.0, 0.0, 0.0],
                action_space=ActionSpace.CARTESIAN_EE_DELTA.value,
                action_unit=ActionUnit.M_RAD.value,
                is_normalized=False,
                constraint_id=None,
                is_intervened=False,
                action_valid_mask=[True] * 6,
                latency_breakdown=LatencyBreakdown(5.0, 20.0, 1.0, 0.5),
                termination_reason=TerminationReason.RUNNING.value,
                source="synthetic_unit_test",
            )
            logger.log_step(rec)
        return logger.finalize(TerminationReason.SUCCESS.value)

    elif pattern == "continuous_single_call":
        # call_id=0 has steps 0..4 all intervened on C1
        c_id = ConstraintKey("obstacle_post", "wrist_link", "min_distance").to_string()
        for step_idx in range(10):
            call_id = step_idx // chunk_size
            sim_t = step_idx * control_dt
            prop = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
            if call_id == 0:
                # Intervene on every tick of call 0
                safe = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
                intervened = True
                curr_c = c_id
            else:
                safe = prop
                intervened = False
                curr_c = None

            rec = StepRecord(
                episode_id=episode_id,
                scene_seed=scene_seed,
                policy_call_id=call_id,
                chunk_id=call_id,
                control_step=step_idx,
                sim_time=sim_t,
                observation_time=sim_t,
                proposed_action=prop,
                commanded_safe_action=safe,
                measured_state_delta=safe,
                measured_joint_state=None,
                measured_ee_state=None,
                action_space=ActionSpace.CARTESIAN_EE_DELTA.value,
                action_unit=ActionUnit.M_RAD.value,
                is_normalized=False,
                constraint_id=curr_c,
                is_intervened=intervened,
                action_valid_mask=[True] * 6,
                latency_breakdown=LatencyBreakdown(4.0, 18.0, 2.0, 0.5),
                termination_reason=TerminationReason.RUNNING.value,
                source="synthetic_unit_test",
            )
            logger.log_step(rec)
        return logger.finalize(TerminationReason.SUCCESS.value)

    elif pattern == "cross_call_repeat":
        c_id = ConstraintKey("obstacle_box", "gripper_link", "min_distance").to_string()
        for step_idx in range(15):
            call_id = step_idx // chunk_size
            sim_t = step_idx * control_dt
            prop = [0.1, 0.1, 0.0, 0.0, 0.0, 0.0]
            # Call 0 and Call 1 both intervene on c_id
            if call_id in (0, 1) and (step_idx % chunk_size == 0):
                safe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                intervened = True
                curr_c = c_id
            else:
                safe = prop
                intervened = False
                curr_c = None

            rec = StepRecord(
                episode_id=episode_id,
                scene_seed=scene_seed,
                policy_call_id=call_id,
                chunk_id=call_id,
                control_step=step_idx,
                sim_time=sim_t,
                observation_time=sim_t,
                proposed_action=prop,
                commanded_safe_action=safe,
                measured_state_delta=safe,
                measured_joint_state=None,
                measured_ee_state=None,
                action_space=ActionSpace.CARTESIAN_EE_DELTA.value,
                action_unit=ActionUnit.M_RAD.value,
                is_normalized=False,
                constraint_id=curr_c,
                is_intervened=intervened,
                action_valid_mask=[True] * 6,
                latency_breakdown=LatencyBreakdown(4.5, 19.0, 2.5, 0.5),
                termination_reason=TerminationReason.RUNNING.value,
                source="synthetic_unit_test",
            )
            logger.log_step(rec)
        return logger.finalize(TerminationReason.SUCCESS.value)

    elif pattern == "distinct_constraints":
        c1 = ConstraintKey("obstacle_left", "wrist_link", "min_distance").to_string()
        c2 = ConstraintKey("obstacle_right", "forearm_link", "min_distance").to_string()
        for step_idx in range(15):
            call_id = step_idx // chunk_size
            sim_t = step_idx * control_dt
            prop = [0.1, -0.1, 0.0, 0.0, 0.0, 0.0]
            if call_id == 0 and (step_idx % chunk_size == 0):
                safe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                intervened = True
                curr_c = c1
            elif call_id == 1 and (step_idx % chunk_size == 0):
                safe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                intervened = True
                curr_c = c2
            else:
                safe = prop
                intervened = False
                curr_c = None

            rec = StepRecord(
                episode_id=episode_id,
                scene_seed=scene_seed,
                policy_call_id=call_id,
                chunk_id=call_id,
                control_step=step_idx,
                sim_time=sim_t,
                observation_time=sim_t,
                proposed_action=prop,
                commanded_safe_action=safe,
                measured_state_delta=safe,
                measured_joint_state=None,
                measured_ee_state=None,
                action_space=ActionSpace.CARTESIAN_EE_DELTA.value,
                action_unit=ActionUnit.M_RAD.value,
                is_normalized=False,
                constraint_id=curr_c,
                is_intervened=intervened,
                action_valid_mask=[True] * 6,
                latency_breakdown=LatencyBreakdown(4.0, 17.5, 1.8, 0.4),
                termination_reason=TerminationReason.RUNNING.value,
                source="synthetic_unit_test",
            )
            logger.log_step(rec)
        return logger.finalize(TerminationReason.SUCCESS.value)

    elif pattern == "short_timeout":
        # Episode intervenes at call 0, then aborts/timeouts after call 1 (horizon H=3 not reached -> censored)
        c_id = ConstraintKey("obstacle_wall", "elbow_link", "min_distance").to_string()
        for step_idx in range(6):
            call_id = step_idx // chunk_size
            sim_t = step_idx * control_dt
            prop = [0.05, 0.05, 0.0, 0.0, 0.0, 0.0]
            if call_id == 0 and step_idx == 0:
                safe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                intervened = True
                curr_c = c_id
            else:
                safe = prop
                intervened = False
                curr_c = None

            rec = StepRecord(
                episode_id=episode_id,
                scene_seed=scene_seed,
                policy_call_id=call_id,
                chunk_id=call_id,
                control_step=step_idx,
                sim_time=sim_t,
                observation_time=sim_t,
                proposed_action=prop,
                commanded_safe_action=safe,
                measured_state_delta=safe,
                measured_joint_state=None,
                measured_ee_state=None,
                action_space=ActionSpace.CARTESIAN_EE_DELTA.value,
                action_unit=ActionUnit.M_RAD.value,
                is_normalized=False,
                constraint_id=curr_c,
                is_intervened=intervened,
                action_valid_mask=[True] * 6,
                latency_breakdown=LatencyBreakdown(5.2, 21.0, 2.1, 0.6),
                termination_reason=TerminationReason.RUNNING.value,
                source="synthetic_unit_test",
            )
            logger.log_step(rec)
        # Finalize as TIMEOUT
        return logger.finalize(TerminationReason.TIMEOUT.value)

    else:
        raise ValueError(f"Unknown synthetic pattern: {pattern}")


def run_synthetic_suite(output_dir: str) -> None:
    out_path = Path(output_dir)
    manifest = ManifestManager(output_dir=out_path, run_id="AG-S4-001-A0-SYNTHETIC-SUITE")

    patterns = [
        ("ep_001_zero_intervention", 101, "zero_intervention"),
        ("ep_002_continuous_single_call", 102, "continuous_single_call"),
        ("ep_003_cross_call_repeat", 103, "cross_call_repeat"),
        ("ep_004_distinct_constraints", 104, "distinct_constraints"),
        ("ep_005_short_timeout", 105, "short_timeout"),
    ]

    print(f"[CLI] Generating {len(patterns)} synthetic episodes in {out_path}...")
    for ep_id, seed, pat in patterns:
        summary = generate_synthetic_episode(out_path, ep_id, seed, pat)
        manifest.add_episode_summary(summary)
        print(f"  - Generated {ep_id} (pattern={pat}): status={summary['termination_reason']}, "
              f"evaluable={summary['aggregation']['evaluable_windows']}, "
              f"censored={summary['aggregation']['censored_windows']}, "
              f"repeated={summary['aggregation']['repeated_conflicts']}")

    final_manifest = manifest.write_manifest()
    stats = final_manifest["aggregate_statistics"]
    print(f"[CLI] Finished. Aggregate stats: {json.dumps(stats, indent=2)}")


def run_verify() -> None:
    print("[CLI] Running self-verification of protocol contracts...")

    # 1. Verify residual calculation
    prop = [0.1, 0.2, -0.1]
    safe = [0.05, 0.2, 0.0]
    res, l2, linf = compute_action_residual(
        proposed_action=prop,
        commanded_safe_action=safe,
        action_space=ActionSpace.CARTESIAN_EE_DELTA.value,
        action_unit=ActionUnit.M_RAD.value,
        is_normalized=False,
    )
    assert len(res) == 3, "Residual dimension mismatch"
    assert math.isclose(res[0], -0.05), "Residual value incorrect"
    print("  [OK] Residual computation verified.")

    # 2. Verify snapshot contract
    sim_state = SimulationStateSnapshot(0.1, 5, [0.1, 0.2], [0.0, 0.0], {"box": [1.0, 0.0, 0.5]})
    ctrl_state = ControllerStateSnapshot([0.1, 0.2], [0.0, 0.0])
    pol_state = PolicyQueueStateSnapshot([[0.1, 0.2]], 1, 0)
    rng_state = RNGStateSnapshot(python_rng_state="mock_rng")
    contract = FullSnapshotContract("snap_01", sim_state, ctrl_state, pol_state, rng_state)
    d = contract.serialize()
    restored = FullSnapshotContract.deserialize(d)
    assert restored.real_snapshot_restore_verified is False
    assert "真实快照恢复未验证" in restored.verification_note
    print("  [OK] Snapshot contract serialization and unverified flag verified.")

    print("[CLI] All self-verifications passed.")


def run_local_preflight() -> None:
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "system": platform.system(),
        "machine": platform.machine(),
    }
    print(json.dumps(info, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="AG-S4-001 / A0 Protocol CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    syn_parser = subparsers.add_parser("run-synthetic", help="Generate synthetic test episodes")
    syn_parser.add_argument("--output-dir", default="synthetic_data", help="Output directory")

    subparsers.add_parser("verify", help="Run contract verification self-check")
    subparsers.add_parser("preflight-local", help="Print local environment information")

    args = parser.parse_args()

    if args.command == "run-synthetic":
        run_synthetic_suite(args.output_dir)
    elif args.command == "verify":
        run_verify()
    elif args.command == "preflight-local":
        run_local_preflight()


if __name__ == "__main__":
    main()
