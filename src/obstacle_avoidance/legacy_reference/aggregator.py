"""
Pure function / stateful event aggregator for AG-S4-002 / S4 End-to-End.
Rules:
1. Atomic updates: validates step, temporal order, and call sequence before mutating any aggregator state.
2. Sequential policy call indexing: tracks sequential call count so non-consecutive IDs (e.g. 10, 20)
   correctly detect repeats within horizon H.
3. Mask & unit-aware residual check: respects action_valid_mask and canonicalizes units to SI
   so inactive dimensions do not trigger interventions and physical distances remain consistent.
4. Right-censored windows: windows truncated before H observed calls elapse are strictly censored.
5. Full state serialization and restoration support.
"""

from dataclasses import dataclass, field, asdict
import math
from typing import Any, Dict, List, Optional, Set, Tuple
from src.types import ConstraintKey, StepRecord, ValidationError
from src.residual import compute_action_residual


@dataclass
class AggregatorConfig:
    H: int = 3                # Horizon in observed policy calls
    epsilon: float = 1e-4     # Minimum residual L2 norm (in canonical SI) to trigger intervention event

    def validate(self) -> None:
        if not isinstance(self.H, int) or isinstance(self.H, bool) or self.H < 1:
            raise ValidationError(f"H must be a positive integer, got {self.H}")
        if not isinstance(self.epsilon, (int, float)) or isinstance(self.epsilon, bool) or math.isnan(self.epsilon) or math.isinf(self.epsilon) or self.epsilon < 0.0:
            raise ValidationError(f"epsilon must be a non-negative finite float, got {self.epsilon}")

    def to_dict(self) -> Dict[str, Any]:
        return {"H": self.H, "epsilon": float(self.epsilon)}


@dataclass
class InterventionWindow:
    window_id: int
    origin_policy_call_id: int
    origin_call_seq_idx: int
    constraint_key: str
    opened_at_control_step: int
    opened_at_sim_time: float
    status: str = "OPEN"           # "OPEN", "REPEATED", "EXPIRED_CLEAN", "CENSORED"
    repeated_at_policy_call_id: Optional[int] = None
    repeated_at_control_step: Optional[int] = None
    closed_at_policy_call_id: Optional[int] = None
    is_evaluable: bool = False
    is_censored: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InterventionWindow":
        # Backward compatibility if origin_call_seq_idx is missing
        if "origin_call_seq_idx" not in d:
            d["origin_call_seq_idx"] = d.get("origin_policy_call_id", 0)
        return cls(**d)


@dataclass
class AggregationSummary:
    total_steps: int
    total_policy_calls: int
    total_interventions_raw_ticks: int
    total_windows_opened: int
    evaluable_windows: int
    censored_windows: int
    repeated_conflicts: int
    clean_expired_windows: int
    repeat_rate: Optional[float]
    denominator: int  # Defined as evaluable_windows
    h_horizon: int
    epsilon: float
    termination_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_policy_calls": self.total_policy_calls,
            "total_interventions_raw_ticks": self.total_interventions_raw_ticks,
            "total_windows_opened": self.total_windows_opened,
            "evaluable_windows": self.evaluable_windows,
            "censored_windows": self.censored_windows,
            "repeated_conflicts": self.repeated_conflicts,
            "clean_expired_windows": self.clean_expired_windows,
            "repeat_rate": self.repeat_rate,
            "denominator": self.denominator,
            "h_horizon": self.h_horizon,
            "epsilon": self.epsilon,
            "termination_reason": self.termination_reason,
        }


class EventAggregator:
    """
    Stateful event aggregator tracking interventions and repeat-H conflicts.
    """

    def __init__(self, config: Optional[AggregatorConfig] = None) -> None:
        self.config = config or AggregatorConfig()
        self.config.validate()

        self._window_counter: int = 0
        self._current_policy_call_id: Optional[int] = None
        self._current_call_seq_idx: int = -1
        self._seen_constraints_current_call: Set[str] = set()
        self._all_policy_call_ids: List[int] = []

        self._open_windows: List[InterventionWindow] = []
        self._closed_windows: List[InterventionWindow] = []

        self._total_steps: int = 0
        self._raw_intervened_ticks: int = 0
        self._last_control_step: Optional[int] = None
        self._last_sim_time: Optional[float] = None
        self._is_finalized: bool = False
        self._termination_reason: Optional[str] = None

    @property
    def config_values(self) -> AggregatorConfig:
        return self.config

    def process_step(self, step: StepRecord) -> None:
        """
        Process a single StepRecord in causal sequence.
        Atomic: validates all step properties and temporal ordering before mutating internal state.
        """
        if self._is_finalized:
            raise ValidationError("Cannot process step on finalized aggregator.")

        # 1. Complete validation of the step record itself
        step.validate()

        # 2. Temporal & order checks BEFORE modifying any aggregator state
        if self._last_control_step is not None and step.control_step < self._last_control_step:
            raise ValidationError(
                f"control_step must be non-decreasing, got {step.control_step} < previous {self._last_control_step}"
            )
        if self._last_sim_time is not None and step.sim_time < self._last_sim_time:
            raise ValidationError(
                f"sim_time must be non-decreasing, got {step.sim_time} < previous {self._last_sim_time}"
            )

        policy_call_id = step.policy_call_id
        if self._current_policy_call_id is not None and policy_call_id < self._current_policy_call_id:
            raise ValidationError(
                f"Out of order policy_call_id: step has {policy_call_id}, current is {self._current_policy_call_id}"
            )

        # 3. All validations passed! Now mutate aggregator state atomically.
        self._total_steps += 1
        self._last_control_step = step.control_step
        self._last_sim_time = step.sim_time

        # Detect transition to a new policy call
        if self._current_policy_call_id is None:
            self._current_policy_call_id = policy_call_id
            self._current_call_seq_idx = 0
            self._all_policy_call_ids.append(policy_call_id)
            self._seen_constraints_current_call = set()
        elif policy_call_id != self._current_policy_call_id:
            # Advance to new sequential policy call
            self._advance_policy_call(policy_call_id)

        # Check intervention
        if step.is_intervened:
            self._raw_intervened_ticks += 1
            constraint_key = step.constraint_id
            if not constraint_key:
                raise ValidationError("Intervened step missing constraint_id")

            # Check residual magnitude vs epsilon in canonical SI units, respecting valid mask
            _, l2_norm, _ = compute_action_residual(
                proposed_action=step.proposed_action,
                commanded_safe_action=step.commanded_safe_action,
                action_space=step.action_space,
                action_unit=step.action_unit,
                is_normalized=step.is_normalized,
                action_valid_mask=step.action_valid_mask,
                canonicalize_to_si=True,
            )

            if l2_norm >= self.config.epsilon:
                # Is this constraint already seen within the current policy call?
                if constraint_key not in self._seen_constraints_current_call:
                    # This is the FIRST time this constraint triggered in THIS policy call!
                    self._seen_constraints_current_call.add(constraint_key)

                    # 1. Check if it resolves any open windows monitoring this constraint
                    for w in self._open_windows:
                        if w.constraint_key == constraint_key:
                            # Verify that it is in a subsequent policy call (not the origin call)
                            if self._current_call_seq_idx > w.origin_call_seq_idx:
                                w.status = "REPEATED"
                                w.repeated_at_policy_call_id = policy_call_id
                                w.repeated_at_control_step = step.control_step
                                w.closed_at_policy_call_id = policy_call_id
                                w.is_evaluable = True
                                w.is_censored = False

                    # Move repeated windows to closed
                    still_open = []
                    for w in self._open_windows:
                        if w.status == "REPEATED":
                            self._closed_windows.append(w)
                        else:
                            still_open.append(w)
                    self._open_windows = still_open

                    # 2. Open a NEW window for this new proposal intervention event
                    self._window_counter += 1
                    new_window = InterventionWindow(
                        window_id=self._window_counter,
                        origin_policy_call_id=policy_call_id,
                        origin_call_seq_idx=self._current_call_seq_idx,
                        constraint_key=constraint_key,
                        opened_at_control_step=step.control_step,
                        opened_at_sim_time=step.sim_time,
                        status="OPEN",
                    )
                    self._open_windows.append(new_window)
                else:
                    # Same constraint triggered in another tick within the SAME policy call:
                    # Do not count as a new proposal, do not inflate windows.
                    pass

    def _advance_policy_call(self, new_policy_call_id: int) -> None:
        """
        Handles transition to new policy call, expiring windows that exceeded H horizon without repeat.
        Uses sequential observed call count to handle non-consecutive call IDs cleanly.
        """
        self._current_policy_call_id = new_policy_call_id
        self._current_call_seq_idx += 1
        self._all_policy_call_ids.append(new_policy_call_id)
        self._seen_constraints_current_call = set()

        still_open = []
        for w in self._open_windows:
            elapsed_calls = self._current_call_seq_idx - w.origin_call_seq_idx
            if elapsed_calls > self.config.H:
                # Window completed H calls without any repeated conflict
                w.status = "EXPIRED_CLEAN"
                w.closed_at_policy_call_id = new_policy_call_id
                w.is_evaluable = True
                w.is_censored = False
                self._closed_windows.append(w)
            else:
                still_open.append(w)
        self._open_windows = still_open

    def finalize_episode(self, termination_reason: str) -> AggregationSummary:
        """
        Finalize episode and classify remaining open windows as right-censored.
        """
        if self._is_finalized:
            raise ValidationError("Aggregator is already finalized.")

        self._is_finalized = True
        self._termination_reason = str(termination_reason)

        last_seq = self._current_call_seq_idx if self._current_call_seq_idx >= 0 else 0
        last_call_id = self._current_policy_call_id or 0

        for w in self._open_windows:
            elapsed_calls = last_seq - w.origin_call_seq_idx
            if elapsed_calls >= self.config.H:
                w.status = "EXPIRED_CLEAN"
                w.closed_at_policy_call_id = last_call_id
                w.is_evaluable = True
                w.is_censored = False
            else:
                # Truncated before H calls elapsed without repeat -> Right-censored!
                w.status = "CENSORED"
                w.closed_at_policy_call_id = last_call_id
                w.is_evaluable = False
                w.is_censored = True
            self._closed_windows.append(w)

        self._open_windows = []

        return self.get_summary()

    def get_summary(self) -> AggregationSummary:
        all_windows = self._closed_windows + self._open_windows

        total_windows = len(all_windows)
        evaluable = [w for w in all_windows if w.is_evaluable]
        censored = [w for w in all_windows if w.is_censored]
        repeated = [w for w in all_windows if w.status == "REPEATED"]
        clean = [w for w in all_windows if w.status == "EXPIRED_CLEAN"]

        num_evaluable = len(evaluable)
        num_censored = len(censored)
        num_repeated = len(repeated)
        num_clean = len(clean)

        repeat_rate = (num_repeated / num_evaluable) if num_evaluable > 0 else None

        return AggregationSummary(
            total_steps=self._total_steps,
            total_policy_calls=len(self._all_policy_call_ids),
            total_interventions_raw_ticks=self._raw_intervened_ticks,
            total_windows_opened=total_windows,
            evaluable_windows=num_evaluable,
            censored_windows=num_censored,
            repeated_conflicts=num_repeated,
            clean_expired_windows=num_clean,
            repeat_rate=repeat_rate,
            denominator=num_evaluable,
            h_horizon=self.config.H,
            epsilon=self.config.epsilon,
            termination_reason=self._termination_reason,
        )

    def get_windows(self) -> List[InterventionWindow]:
        return list(self._closed_windows + self._open_windows)

    def get_state(self) -> Dict[str, Any]:
        """
        Serialize aggregator state for saving/restoration contract test.
        """
        return {
            "config": self.config.to_dict(),
            "window_counter": self._window_counter,
            "current_policy_call_id": self._current_policy_call_id,
            "current_call_seq_idx": self._current_call_seq_idx,
            "seen_constraints_current_call": list(self._seen_constraints_current_call),
            "all_policy_call_ids": list(self._all_policy_call_ids),
            "open_windows": [w.to_dict() for w in self._open_windows],
            "closed_windows": [w.to_dict() for w in self._closed_windows],
            "total_steps": self._total_steps,
            "raw_intervened_ticks": self._raw_intervened_ticks,
            "last_control_step": self._last_control_step,
            "last_sim_time": self._last_sim_time,
            "is_finalized": self._is_finalized,
            "termination_reason": self._termination_reason,
        }

    @classmethod
    def load_state(cls, state_dict: Dict[str, Any]) -> "EventAggregator":
        """
        Deserialize and restore aggregator state.
        """
        cfg_d = state_dict["config"]
        config = AggregatorConfig(H=int(cfg_d["H"]), epsilon=float(cfg_d["epsilon"]))
        agg = cls(config=config)

        agg._window_counter = int(state_dict["window_counter"])
        agg._current_policy_call_id = state_dict["current_policy_call_id"]
        agg._current_call_seq_idx = int(state_dict.get("current_call_seq_idx", 0))
        agg._seen_constraints_current_call = set(state_dict["seen_constraints_current_call"])
        agg._all_policy_call_ids = list(state_dict["all_policy_call_ids"])
        agg._open_windows = [InterventionWindow.from_dict(w) for w in state_dict["open_windows"]]
        agg._closed_windows = [InterventionWindow.from_dict(w) for w in state_dict["closed_windows"]]
        agg._total_steps = int(state_dict["total_steps"])
        agg._raw_intervened_ticks = int(state_dict["raw_intervened_ticks"])
        agg._last_control_step = state_dict.get("last_control_step")
        agg._last_sim_time = state_dict.get("last_sim_time")
        agg._is_finalized = bool(state_dict["is_finalized"])
        agg._termination_reason = state_dict.get("termination_reason")
        return agg
