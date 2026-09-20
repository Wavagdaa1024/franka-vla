# -*- coding: utf-8 -*-
"""
Pi0.5 Native LoRA (Low-Rank Adaptation) Module.
Designed for robotic VLA fine-tuning following Northwestern IDEAS Lab specification.

Key Features:
  - Zero external dependencies (pure PyTorch).
  - Exact zero-initialization (B=0, delta=0 at step 0).
  - Preserves base model weights 100% frozen.
  - Lightweight adapter checkpoint saving/loading (~35MB total).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper around a frozen nn.Linear layer.
    h = W_0 * x + (alpha / r) * (B @ A) * x
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank if rank > 0 else 1.0

        # Freeze base linear layer
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        if rank > 0:
            # LoRA weights in float32 for high numerical precision & stability
            self.lora_A = nn.Parameter(
                torch.empty(rank, self.in_features, dtype=torch.float32, device=base_linear.weight.device)
            )
            # B initialized to strictly zero (ensures zero perturbation at step 0)
            self.lora_B = nn.Parameter(
                torch.zeros(self.out_features, rank, dtype=torch.float32, device=base_linear.weight.device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.dropout = nn.Identity()

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        if self.rank > 0:
            # x: (..., in_features)
            # A: (rank, in_features) -> x @ A.T: (..., rank)
            # B: (out_features, rank) -> (x @ A.T) @ B.T: (..., out_features)
            in_dtype = x.dtype
            x_dropped = self.dropout(x)
            if x_dropped.dtype != self.lora_A.dtype:
                x_dropped = x_dropped.to(self.lora_A.dtype)
            lora_act = torch.matmul(x_dropped, self.lora_A.t())
            lora_out = torch.matmul(lora_act, self.lora_B.t()) * self.scaling
            if lora_out.dtype != in_dtype:
                lora_out = lora_out.to(in_dtype)
            return base_out + lora_out
        return base_out

    def merge_weights(self):
        """In-place merges LoRA weights into base layer for fast zero-overhead inference."""
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            delta_w = (torch.matmul(self.lora_B, self.lora_A) * self.scaling).to(self.base_linear.weight.dtype)
            self.base_linear.weight.data += delta_w
            self.rank = 0
            self.lora_A = None
            self.lora_B = None


def inject_lora_into_submodule(
    module: nn.Module,
    target_attr_names: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    prefix: str = "",
) -> Dict[str, LoRALinear]:
    """
    Recursively scans module and replaces matching nn.Linear attributes with LoRALinear.
    """
    injected = {}
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and name in target_attr_names:
            lora_layer = LoRALinear(
                child,
                rank=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            setattr(module, name, lora_layer)
            injected[full_name] = lora_layer
        else:
            injected.update(
                inject_lora_into_submodule(
                    child,
                    target_attr_names=target_attr_names,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    prefix=full_name,
                )
            )
    return injected


def inject_pi05_lora(
    net: nn.Module,
    lang_rank: int = 16,
    lang_alpha: float = 32.0,
    expert_rank: int = 32,
    expert_alpha: float = 64.0,
    lora_dropout: float = 0.0,
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> Tuple[Dict[str, LoRALinear], Dict[str, LoRALinear]]:
    """
    Injects LoRA adapters into:
      1. PaliGemma-2B language model attention projections (rank 16)
      2. Gemma-300M Action Expert attention projections (rank 32)
    Freezes all base parameters in the entire network.
    Unfreezes Action In/Out projections and Time MLP.
    """
    # 1. Freeze everything first
    for p in net.parameters():
        p.requires_grad = False

    # 2. Inject LoRA into PaliGemma language model
    lang_model = net.paligemma_with_expert.paligemma.model.language_model
    lang_loras = inject_lora_into_submodule(
        lang_model,
        target_attr_names=target_modules,
        rank=lang_rank,
        lora_alpha=lang_alpha,
        lora_dropout=lora_dropout,
        prefix="paligemma.model.language_model",
    )

    # 3. Inject LoRA into Gemma Action Expert
    expert_model = net.paligemma_with_expert.gemma_expert
    expert_loras = inject_lora_into_submodule(
        expert_model,
        target_attr_names=target_modules,
        rank=expert_rank,
        lora_alpha=expert_alpha,
        lora_dropout=lora_dropout,
        prefix="gemma_expert",
    )

    # 4. Make LoRA parameters trainable
    for lora in list(lang_loras.values()) + list(expert_loras.values()):
        lora.lora_A.requires_grad = True
        lora.lora_B.requires_grad = True

    # 5. Make Action Projections and Time MLP trainable
    for m in [net.action_in_proj, net.action_out_proj, net.time_mlp_in, net.time_mlp_out]:
        for p in m.parameters():
            p.requires_grad = True

    return lang_loras, expert_loras


def extract_lora_state_dict(net: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extracts only trainable parameters:
      - All LoRA parameters (lora_A, lora_B)
      - action_in_proj, action_out_proj
      - time_mlp_in, time_mlp_out
    """
    state_dict = {}
    for name, param in net.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.data.clone().cpu()
    return state_dict


def load_lora_state_dict(net: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = False):
    """
    Loads LoRA parameters and action projection weights into net.
    Strict mode validates both unexpected keys AND missing trainable keys.
    """
    model_params = dict(net.named_parameters())
    trainable_keys = {name for name, p in net.named_parameters() if p.requires_grad}
    loaded_count = 0
    unexpected_keys = []

    for name, param in state_dict.items():
        if name in model_params:
            target_param = model_params[name]
            target_param.data.copy_(param.to(device=target_param.device, dtype=target_param.dtype))
            loaded_count += 1
        else:
            unexpected_keys.append(name)

    if strict:
        if unexpected_keys:
            raise KeyError(f"[LoRA Loader] Strict check failed: {len(unexpected_keys)} unexpected keys found in state_dict! E.g.: {unexpected_keys[:5]}")
        missing_keys = [k for k in trainable_keys if k not in state_dict]
        if missing_keys:
            raise KeyError(f"[LoRA Loader] Strict check failed: {len(missing_keys)} required trainable keys missing from checkpoint! E.g.: {missing_keys[:5]}")

    print(f"[LoRA Loader] Successfully loaded {loaded_count}/{len(state_dict)} tensors (strict={strict}).")
