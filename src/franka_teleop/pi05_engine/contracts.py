"""Checkpoint contracts shared by training and deployment."""
from pathlib import Path, WindowsPath, PosixPath
import warnings
import math
import torch

ACTION_SEMANTICS = "v2_gripper_action_command"
PRECISION = "fp32_lora_and_projections"
LEGACY_CROPS = {"pi05_lora_pure_flow_50k": "none", "pi05_lora_crop169_20k": "16_9"}

def read_checkpoint(path):
    with torch.serialization.safe_globals([WindowsPath, PosixPath]):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a mapping")
    return checkpoint

def checkpoint_state(checkpoint):
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    if state is None and checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state = checkpoint
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Checkpoint has no valid tensor state_dict/model_state_dict")
    if checkpoint.get("precision") == PRECISION and any(v.dtype != torch.float32 for v in state.values()):
        raise ValueError("Checkpoint claims FP32 trainables but contains other dtypes")
    return state

def resolve_crop(checkpoint, path, requested="auto"):
    metadata = checkpoint.get("image_crop", checkpoint.get("args", {}).get("image_crop"))
    if metadata is not None and metadata not in ("none", "16_9"):
        raise ValueError(f"Invalid checkpoint image_crop: {metadata!r}")
    if requested not in (None, "auto", "none", "16_9"):
        raise ValueError(f"Invalid requested image_crop: {requested!r}")
    if requested not in (None, "auto"):
        if metadata is not None and metadata != requested:
            raise ValueError(f"Crop conflicts with checkpoint: {requested} != {metadata}")
        return requested
    if metadata is not None:
        return metadata
    legacy = LEGACY_CROPS.get(Path(path).parent.name)
    if legacy is None:
        raise ValueError("Legacy checkpoint has no crop metadata; supply --image-crop none or 16_9")
    warnings.warn(f"Legacy crop compatibility: {Path(path).parent.name} -> {legacy}", stacklevel=2)
    return legacy

def lora_spec(checkpoint):
    lang, expert = int(checkpoint.get("lang_rank", 16)), int(checkpoint.get("expert_rank", 32))
    if lang <= 0 or expert <= 0:
        raise ValueError("LoRA ranks must be positive")
    spec = dict(lang_rank=lang, expert_rank=expert,
                lang_alpha=float(checkpoint.get("lang_alpha", 2 * lang)),
                expert_alpha=float(checkpoint.get("expert_alpha", 2 * expert)))
    if any(not math.isfinite(spec[k]) or spec[k] <= 0 for k in ("lang_alpha", "expert_alpha")):
        raise ValueError("LoRA alpha must be positive and finite")
    return spec

def validate_resume(checkpoint, args, val_episodes=None):
    """Contract continuation; migration explicitly transfers weights only."""
    state = checkpoint_state(checkpoint)
    spec = lora_spec(checkpoint)
    if spec != dict(lang_rank=args.lang_rank, expert_rank=args.expert_rank,
                    lang_alpha=2.0 * args.lang_rank, expert_alpha=2.0 * args.expert_rank):
        raise ValueError("Checkpoint LoRA ranks/scaling differ from requested training architecture")
    if args.resume_mode == "migrate":
        return
    required = ("image_crop", "action_semantics", "precision", "val_episodes",
                "optimizer_state_dict", "scheduler_state_dict", "step", "args")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Cannot resume, missing {missing}; use --resume-mode migrate for weights-only transfer")
    expected = {"image_crop": args.image_crop, "action_semantics": ACTION_SEMANTICS, "precision": PRECISION}
    for key, value in expected.items():
        if checkpoint[key] != value:
            raise ValueError(f"Resume {key} mismatch; use explicit migration")
    if any(t.dtype != torch.float32 for t in state.values()):
        raise ValueError("FP32 resume contract contains non-FP32 trainable tensors")
    old = checkpoint["args"]
    for key in ("lang_rank", "expert_rank", "lora_dropout", "dataset", "task_filter",
                "val_seed", "num_val_episodes", "val_ratio", "lr", "steps", "warmup_steps",
                "batch_size", "grad_accum"):
        a, b = old.get(key), getattr(args, key)
        if key == "dataset":
            a, b = [str(Path(p).resolve()) for p in a or []], [str(Path(p).resolve()) for p in b]
        if a != b:
            raise ValueError(f"Resume argument mismatch for {key}: {a!r} != {b!r}; use --resume-mode migrate")
    if val_episodes is not None and set(checkpoint["val_episodes"]) != set(val_episodes):
        raise ValueError("Resume validation episodes differ from saved split")
