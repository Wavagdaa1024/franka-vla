#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Modified: minimal inference extraction from LeRobot v0.6.1
# 7e241bd630a3719a56157a497ce5d08f244784f1; no full LeRobot imports.
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


def crop_to_16_9_torch(images: torch.Tensor) -> torch.Tensor:
    """
    Center crops tensor of shape [*B, H, W, C] to 16:9 aspect ratio.
    Adheres strictly to Northwestern IDEAS Lab and OpenPI DROID convention.
    For 640x480 (4:3), crops vertical height to 360 (offset 60px top and bottom) yielding 640x360 (16:9).
    """
    H, W = images.shape[-3], images.shape[-2]
    target_H = int(round(W * 9.0 / 16.0))
    if H > target_H:
        offset = (H - target_H) // 2
        return images[..., offset : offset + target_H, :, :]
    elif W > int(round(H * 16.0 / 9.0)):
        target_W = int(round(H * 16.0 / 9.0))
        offset = (W - target_W) // 2
        return images[..., :, offset : offset + target_W, :]
    return images


def crop_to_16_9_np(image: np.ndarray) -> np.ndarray:
    """
    Center crops numpy array of shape [H, W, C] to 16:9 aspect ratio.
    """
    h, w = image.shape[:2]
    target_h = int(round(w * 9.0 / 16.0))
    if h > target_h:
        offset = (h - target_h) // 2
        return np.ascontiguousarray(image[offset : offset + target_h, :, :])
    elif w > int(round(h * 16.0 / 9.0)):
        target_w = int(round(h * 16.0 / 9.0))
        offset = (w - target_w) // 2
        return np.ascontiguousarray(image[:, offset : offset + target_w, :])
    return image


def get_safe_dtype(dtype, device):
    if device not in {"cpu", "cuda"}:
        raise ValueError("This minimal build supports CPU and CUDA only")
    return dtype

def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

def prepare_attention_masks_4d(att_2d_masks: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """Expand boolean 2D attention masks to the additive 4D layout expected by transformers.

    Valid positions become 0.0 and masked positions the large negative openpi constant.
    """
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    result = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result

def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Padding is centered (openpi convention). For the top-left-padding variant used by
    smolvla/xvla, see :func:`resize_with_pad`.

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images

def clone_past_key_values(past_key_values):
    from transformers.cache_utils import DynamicCache
    return DynamicCache(tuple((keys.clone(), values.clone(), sliding_window)
                              for keys, values, sliding_window in past_key_values))
