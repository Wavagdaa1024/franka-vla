# Adapted from LeRobot v0.6.1 configuration_pi05.py; Apache-2.0.
# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Modified: inference-only dataclass, explicit features, no registry/optimizer/RTC.
from dataclasses import dataclass, field, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class PI05Config:
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10
    min_period: float = 0.004
    max_period: float = 4.0
    image_resolution: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 200
    input_features: dict = field(default_factory=dict)
    output_features: dict = field(default_factory=dict)
    normalization_mapping: dict = field(default_factory=lambda: {
        "VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"})

    def __post_init__(self):
        for name in ("chunk_size", "n_action_steps", "max_state_dim", "max_action_dim",
                     "num_inference_steps", "tokenizer_max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        if any(v not in {"gemma_2b", "gemma_300m"} for v in (self.paligemma_variant, self.action_expert_variant)):
            raise ValueError("unsupported Gemma variant")
        if len(self.image_resolution) != 2 or self.image_resolution[0] != self.image_resolution[1]:
            raise ValueError("image_resolution must be square")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 14 or v % 14 for v in self.image_resolution):
            raise ValueError("image_resolution must be a positive multiple of patch size 14")
        if not 0 < self.min_period < self.max_period:
            raise ValueError("invalid time embedding periods")
        if self.normalization_mapping.get("VISUAL", "IDENTITY") != "IDENTITY":
            raise ValueError("only IDENTITY visual normalization is supported")
        modes = {"IDENTITY", "MEAN_STD", "MIN_MAX", "QUANTILES", "QUANTILE10"}
        if any(self.normalization_mapping.get(k, "QUANTILES") not in modes for k in ("STATE", "ACTION")):
            raise ValueError("unsupported normalization mode")

    @property
    def image_keys(self):
        return tuple(k for k, v in self.input_features.items() if v.get("type") == "VISUAL")

    def feature_dim(self, key):
        feature = (self.input_features if key == "observation.state" else self.output_features).get(key)
        if not feature or len(feature.get("shape", [])) != 1:
            raise ValueError(f"checkpoint must declare a one-dimensional {key} feature")
        value = feature["shape"][0]
        maximum = self.max_state_dim if key == "observation.state" else self.max_action_dim
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
            raise ValueError(f"invalid {key} dimension: {value}")
        return value

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "pi05":
            raise ValueError("checkpoint config type must be pi05")
        if data.get("n_obs_steps", 1) != 1:
            raise ValueError("only n_obs_steps=1 is supported")
        if data.get("use_relative_actions") or data.get("use_peft"):
            raise ValueError("relative-action and PEFT checkpoints require separate adapters")
        if data.get("rtc_config") is not None:
            raise ValueError("RTC is outside this inference-only build")
        args = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        if "image_resolution" in args:
            args["image_resolution"] = tuple(args["image_resolution"])
        result = cls(**args)
        if not result.image_keys:
            raise ValueError("checkpoint must declare at least one VISUAL feature")
        result.feature_dim("observation.state")
        result.feature_dim("action")
        if data.get("empty_cameras", 0):
            raise ValueError("declare empty camera slots explicitly in input_features")
        return result

    @classmethod
    def from_file(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
