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
import torch
from torch import Tensor

def sample_noise(shape, device) -> Tensor:
    """Standard-normal float32 noise, the flow-matching x_1 sample."""
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )

def euler_integrate(denoise_fn, noise: Tensor, num_steps: int) -> Tensor:
    """Upstream Euler loop with RTC hooks removed; t=1 to t=0."""
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time_tensor = torch.tensor(1.0 + step * dt, dtype=noise.dtype, device=noise.device).expand(noise.shape[0])
        x_t = x_t + dt * denoise_fn(x_t, time_tensor)
    return x_t

