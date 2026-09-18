# Adapted from LeRobot v0.6.1 processor_pi05.py and normalize_processor.py.
# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Apache-2.0. Modified: explicit tensors/stats, no framework pipelines or dataset.
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .config import PI05Config
from .utils import resize_with_pad_torch


def transform(tensor, stats, mode, *, inverse=False, eps=1e-8):
    """Upstream numeric formulas, but missing/malformed stats fail explicitly."""
    if not torch.isfinite(tensor).all():
        raise ValueError("non-finite input")
    if mode == "IDENTITY":
        return tensor
    names = {"MEAN_STD": ("mean", "std"), "MIN_MAX": ("min", "max"),
             "QUANTILES": ("q01", "q99"), "QUANTILE10": ("q10", "q90")}
    if mode not in names:
        raise ValueError(f"unsupported normalization: {mode}")
    values = []
    for name in names[mode]:
        if name not in stats:
            raise ValueError(f"missing normalization statistic: {name}")
        value = torch.as_tensor(stats[name], device=tensor.device, dtype=tensor.dtype)
        if value.shape != (tensor.shape[-1],) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite and match the actual feature dimension")
        values.append(value)
    a, b = values
    if mode == "MEAN_STD":
        if (b < 0).any():
            raise ValueError("std cannot be negative")
        return tensor * b + a if inverse else (tensor - a) / (b + eps)
    span = b - a
    if (span < 0).any():
        raise ValueError("normalization upper bounds cannot be below lower bounds")
    span = torch.where(span == 0, torch.tensor(eps, device=tensor.device, dtype=tensor.dtype), span)
    return (tensor + 1) * span / 2 + a if inverse else 2 * (tensor - a) / span - 1


def make_prompts(state, tasks):
    """Preserve the pinned pi05 256-bin state-to-text convention (no extra pad)."""
    if state.ndim != 2 or not torch.isfinite(state).all():
        raise ValueError("state must be a finite [batch, state_dim] tensor")
    if isinstance(tasks, str):
        tasks = [tasks] * len(state)
    if len(tasks) != len(state) or any(not isinstance(t, str) or not t.strip() for t in tasks):
        raise ValueError("provide one non-empty task for each batch item")
    bins = np.digitize(state.detach().cpu().numpy(), bins=np.linspace(-1, 1, 257)[:-1]) - 1
    result = []
    for task, row in zip(tasks, bins, strict=True):
        cleaned = task.strip().replace("_", " ").replace("\n", " ")
        result.append(f"Task: {cleaned}, State: {' '.join(map(str, row))};\nAction: ")
    return result


@dataclass
class PreparedInput:
    images: list[torch.Tensor]
    image_masks: list[torch.Tensor]
    tokens: torch.Tensor
    token_mask: torch.Tensor


class PI05Processor:
    def __init__(self, config: PI05Config, stats: dict, tokenizer, *, profile="checkpoint"):
        self.config, self.stats, self.tokenizer = config, stats, tokenizer
        if profile not in {"checkpoint", "droid", "droid_jointpos"}:
            raise ValueError("profile must be checkpoint, droid, or droid_jointpos")
        self.profile = profile
        if profile == "droid" and (config.max_state_dim != 32 or config.max_action_dim != 32):
            raise ValueError("DROID profile requires the 32-dimensional pi05 core")
        if profile == "droid_jointpos" and (config.max_state_dim != 8 or config.max_action_dim != 32):
            raise ValueError("DROID jointpos profile requires max_state_dim=8 and max_action_dim=32")
        for key, kind in (("observation.state", "STATE"), ("action", "ACTION")):
            self._transform(torch.zeros(1, self._dim(key)), key, kind)

    def _dim(self, key):
        return 8 if self.profile in {"droid", "droid_jointpos"} else self.config.feature_dim(key)

    def _transform(self, tensor, key, kind, *, inverse=False):
        mode = self.config.normalization_mapping.get(kind, "QUANTILES")
        stats = self.stats.get(key, {})
        if self.profile == "checkpoint":
            return transform(tensor, stats, mode, inverse=inverse)
        if mode != "QUANTILES":
            raise ValueError("DROID profile requires its original quantile statistics")
        sliced = {name: value[:8] for name, value in stats.items() if isinstance(value, list)}
        transform(tensor, sliced, mode)  # Validate finite dimensions/bounds before the OpenPI formula.
        lower = torch.as_tensor(sliced["q01"], device=tensor.device, dtype=tensor.dtype)
        upper = torch.as_tensor(sliced["q99"], device=tensor.device, dtype=tensor.dtype)
        span = upper - lower + 1e-6  # Exact OpenPI DROID convention, distinct from LeRobot generic.
        return (tensor + 1) / 2 * span + lower if inverse else (tensor - lower) / span * 2 - 1

    def to_absolute_jointpos(self, delta_chunk, current_q):
        """
        Convert relative joint position deltas in action chunk to absolute target positions.
        delta_chunk: [..., chunk_size, 8] or [chunk_size, 8] (unnormalized actions from postprocess)
        current_q: [7] current robot joint positions in radians
        Returns:
            q_targets: [..., chunk_size, 7] absolute joint positions in radians
        """
        q_curr = torch.as_tensor(current_q, dtype=torch.float32)
        if q_curr.shape[-1] != 7:
            raise ValueError(f"current_q must have 7 joint dimensions, got shape {q_curr.shape}")
        deltas = delta_chunk[..., :7]
        return q_curr + deltas

    @classmethod
    def from_local(cls, config, stats_path, tokenizer_path, *, profile="checkpoint"):
        stats_file = Path(stats_path)
        tokenizer_dir = Path(tokenizer_path)
        if not stats_file.is_file():
            raise FileNotFoundError("provide an existing normalization JSON")
        if tokenizer_dir.is_file() and tokenizer_dir.suffix == ".model":
            from .tokenizer import SentencePiecePromptTokenizer
            tokenizer = SentencePiecePromptTokenizer(tokenizer_dir)
        elif tokenizer_dir.is_dir():
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, padding_side="right")
        else:
            raise FileNotFoundError("provide a local tokenizer directory or official .model file")
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        if "norm_stats" in stats:
            stats = {"observation.state": stats["norm_stats"]["state"], "action": stats["norm_stats"]["actions"]}
        return cls(config, stats, tokenizer, profile=profile)

    def prepare_state(self, raw_state):
        state = torch.as_tensor(raw_state, dtype=torch.float32, device="cpu")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[-1] != self._dim("observation.state"):
            raise ValueError("state shape must match checkpoint; no implicit feature remapping")
        normalized = self._transform(state, "observation.state", "STATE")
        # OpenPI TokenizePrompt precedes PadStatesAndActions: text uses 8 DROID
        # state values. The later padded state is unused by the pi05 action expert.
        return normalized

    def prepare(self, observations: dict, task, *, device="cpu"):
        normalized = self.prepare_state(observations["observation.state"])
        state = normalized
        prompts = make_prompts(normalized, task)
        tokenized = self.tokenizer(prompts, max_length=self.config.tokenizer_max_length, truncation=True,
                                   padding="max_length", padding_side="right", return_tensors="pt")
        present = [k for k in self.config.image_keys if k in observations]
        missing = [k for k in self.config.image_keys if k not in observations]
        if not present:
            raise ValueError("at least one checkpoint image feature is required")
        images, masks = [], []
        for key in present:
            image = torch.as_tensor(observations[key], device=device)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            # Deliberately accept one unambiguous convention, unlike heuristic upstream input.
            if image.ndim != 4 or image.shape[1] != 3 or image.shape[0] != len(state):
                raise ValueError(f"{key} must be RGB [B,3,H,W] or [3,H,W]")
            if image.dtype == torch.uint8:
                image = image.float() / 255
            elif image.dtype.is_floating_point:
                image = image.float()
            else:
                raise ValueError("images must be uint8 or floating RGB in [0,1]")
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("floating RGB must be finite and in [0,1]")
            image = image.permute(0, 2, 3, 1)
            if tuple(image.shape[1:3]) != self.config.image_resolution:
                image = resize_with_pad_torch(image, *self.config.image_resolution)
            images.append((image * 2 - 1).permute(0, 3, 1, 2).contiguous())
            masks.append(torch.ones(len(state), dtype=torch.bool, device=device))
        # Preserve upstream order: present camera slots followed by missing masked slots.
        for _ in missing:
            images.append(torch.full_like(images[0], -1))
            masks.append(torch.zeros_like(masks[0]))
        return PreparedInput(images, masks, tokenized["input_ids"].to(device),
                             tokenized["attention_mask"].to(device=device, dtype=torch.bool))

    def postprocess(self, padded_actions):
        dim = self._dim("action")
        if padded_actions.ndim != 3 or padded_actions.shape[1:] != (self.config.chunk_size, self.config.max_action_dim):
            raise ValueError("unexpected network action chunk shape")
        actions = padded_actions[..., :dim].float()
        actions = self._transform(actions, "action", "ACTION", inverse=True)
        if not torch.isfinite(actions).all():
            raise ValueError("non-finite action output")
        return actions.detach().cpu()
