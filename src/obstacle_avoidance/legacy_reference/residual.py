"""
Residual calculation and validation module for AG-S4-002 / S4 End-to-End.
Ensures residual is only calculated:
1. In identical action space representation.
2. In identical physical units, with canonicalization to SI (m_rad) for threshold evaluation.
3. Strictly in unnormalized physical scale.
4. Strictly respecting action_valid_mask (inactive/padded dimensions do not contribute to physical residual norm).
5. Forbids subtracting measured joint displacement (q delta) directly from end-effector action.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from src.types import ActionSemantics, ActionSpace, ActionUnit, ValidationError


def get_canonical_unit_scales(action_unit: str, dim: int) -> List[float]:
    """
    Returns per-dimension scale factors to convert actions into canonical SI units
    (meters for translation, radians for rotation).
    """
    if action_unit == ActionUnit.M_RAD.value:
        return [1.0] * dim
    elif action_unit == ActionUnit.RAD.value:
        return [1.0] * dim
    elif action_unit == ActionUnit.MM_DEG.value:
        # First 3 dimensions are position (mm -> m: 1e-3)
        # Remaining dimensions are rotation (deg -> rad: pi / 180)
        scales = []
        for i in range(dim):
            if i < 3:
                scales.append(1e-3)
            else:
                scales.append(math.pi / 180.0)
        return scales
    else:
        raise ValidationError(f"Unknown or unsupported action unit '{action_unit}'")


def compute_action_residual(
    proposed_action: List[float],
    commanded_safe_action: List[float],
    action_space: str,
    action_unit: str,
    is_normalized: bool,
    action_valid_mask: Optional[List[bool]] = None,
    expected_space: Optional[str] = None,
    expected_unit: Optional[str] = None,
    canonicalize_to_si: bool = True,
) -> Tuple[List[float], float, float]:
    """
    Computes numerical residual between commanded_safe_action and proposed_action.
    residual = commanded_safe - proposed (i.e. correction vector applied by safety filter)

    If action_valid_mask is provided, masked-out dimensions (mask[i] == False)
    do NOT contribute to physical residual norms (l2_norm, linf_norm).

    If canonicalize_to_si is True, l2_norm and linf_norm are calculated after
    converting components into canonical SI units (meters and radians).

    Returns:
        residual_vector: List[float] (raw physical unnormalized difference)
        l2_norm: float (canonical SI norm on active valid dimensions)
        linf_norm: float (canonical SI max absolute diff on active valid dimensions)
    """
    if is_normalized:
        raise ValidationError(
            "Action residual must only be computed after unnormalization into physical space. "
            "Computing residuals in normalized space is strictly forbidden by protocol."
        )

    if not isinstance(proposed_action, list) or not isinstance(commanded_safe_action, list):
        raise ValidationError("Action vectors must be lists of floats.")

    if not proposed_action or not commanded_safe_action:
        raise ValidationError("Action vectors cannot be empty.")

    dim = len(proposed_action)
    if len(commanded_safe_action) != dim:
        raise ValidationError(
            f"Action dimension mismatch: proposed ({dim}) vs commanded_safe ({len(commanded_safe_action)})."
        )

    if action_valid_mask is not None:
        if len(action_valid_mask) != dim:
            raise ValidationError(
                f"action_valid_mask length ({len(action_valid_mask)}) must match action dimension ({dim})"
            )
        for idx, b in enumerate(action_valid_mask):
            if not isinstance(b, bool):
                raise ValidationError(f"action_valid_mask element at index {idx} must be bool, got {type(b).__name__}")

    if expected_space is not None and action_space != expected_space:
        raise ValidationError(
            f"Action space mismatch: expected {expected_space}, got {action_space}"
        )

    if expected_unit is not None and action_unit != expected_unit:
        raise ValidationError(
            f"Action unit mismatch: expected {expected_unit}, got {action_unit}"
        )

    scales = get_canonical_unit_scales(action_unit, dim) if canonicalize_to_si else [1.0] * dim

    residual: List[float] = []
    sum_sq = 0.0
    max_abs = 0.0

    for i in range(dim):
        safe_val = commanded_safe_action[i]
        prop_val = proposed_action[i]

        if not isinstance(safe_val, (int, float)) or isinstance(safe_val, bool) or math.isnan(safe_val) or math.isinf(safe_val):
            raise ValidationError(f"Invalid non-finite value in commanded_safe_action at index {i}: {safe_val}")
        if not isinstance(prop_val, (int, float)) or isinstance(prop_val, bool) or math.isnan(prop_val) or math.isinf(prop_val):
            raise ValidationError(f"Invalid non-finite value in proposed_action at index {i}: {prop_val}")

        diff_raw = float(safe_val - prop_val)
        residual.append(diff_raw)

        # Only valid (unmasked) dimensions contribute to the physical norm
        is_active = True if action_valid_mask is None else action_valid_mask[i]
        if is_active:
            diff_scaled = diff_raw * scales[i]
            sum_sq += diff_scaled * diff_scaled
            if abs(diff_scaled) > max_abs:
                max_abs = abs(diff_scaled)

    l2_norm = math.sqrt(sum_sq)
    linf_norm = max_abs

    return residual, l2_norm, linf_norm


def forbid_cross_space_subtraction(
    measured_state_delta: List[float],
    measured_space: str,
    action: List[float],
    action_space: str,
) -> None:
    """
    Enforces protocol requirement:
    '不能将实测q差直接减末端动作。'
    """
    if ("joint" in measured_space.lower() and "cartesian" in action_space.lower()) or \
       ("cartesian" in measured_space.lower() and "joint" in action_space.lower()):
        raise ValidationError(
            f"Cross-representation subtraction forbidden: measured state is in '{measured_space}' "
            f"while action is in '{action_space}'. Cannot subtract joint angles from end-effector Cartesian coordinates."
        )
