"""Local pi05 checkpoint -> observations -> action chunk. No robot or server I/O."""
from collections import deque
import os
from pathlib import Path
import torch

from .config import PI05Config
from .checkpoint import load_weights
from .processing import PI05Processor


def validate_device(device):
    if device == "cpu":
        return torch.device("cpu")
    if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"}:
        raise ValueError("use cpu, or select physical GPU 0/1 via CUDA_VISIBLE_DEVICES and use cuda:0")
    return torch.device(device)


class PI05Inference:
    def __init__(self, network, processor, *, device="cpu"):
        self.device = validate_device(device)
        self.network = network.eval()
        self.processor = processor
        self.config = processor.config
        self._queue = deque()

    @classmethod
    def from_checkpoint(cls, checkpoint, *, stats_path, tokenizer_path, device="cpu", profile="checkpoint"):
        root = Path(checkpoint).expanduser().resolve()
        if not root.is_dir() or not (root / "model.safetensors").is_file():
            raise FileNotFoundError("local checkpoint must contain config.json and model.safetensors")
        target = validate_device(device)
        config = PI05Config.from_file(root / "config.json")
        # Fail on missing preprocessing assets before allocating a full network.
        processor = PI05Processor.from_local(config, stats_path, tokenizer_path, profile=profile)
        from .network import PI05Pytorch
        from accelerate import init_empty_weights
        # Keep nonpersistent rotary/position buffers on CPU; a blanket meta context
        # would leave buffers absent from safetensors impossible to materialize.
        with init_empty_weights(include_buffers=False):
            model = PI05Pytorch(config)
        load_weights(model, root / "model.safetensors")
        loaded_dtype = next(model.parameters()).dtype
        precision = "bfloat16" if (loaded_dtype == torch.bfloat16 or config.dtype == "bfloat16") else "float32"
        model.paligemma_with_expert.to_bfloat16_for_selected_params(precision)
        if precision == "bfloat16":
            model.action_in_proj.to(dtype=torch.bfloat16)
            model.action_out_proj.to(dtype=torch.bfloat16)
            model.time_mlp_in.to(dtype=torch.bfloat16)
            model.time_mlp_out.to(dtype=torch.bfloat16)
        model.to(target)
        return cls(model, processor, device=device)

    @torch.inference_mode()
    def predict_action_chunk(self, observations, task, *, noise=None):
        inputs = self.processor.prepare(observations, task, device=self.device)
        if noise is not None:
            expected = (inputs.tokens.shape[0], self.config.chunk_size, self.config.max_action_dim)
            if tuple(noise.shape) != expected or not torch.isfinite(noise).all():
                raise ValueError(f"noise must be finite with shape {expected}")
            noise = noise.to(device=self.device, dtype=torch.float32)
        actions = self.network.sample_actions(inputs.images, inputs.image_masks, inputs.tokens,
                                              inputs.token_mask, noise=noise)
        return self.processor.postprocess(actions)

    def select_action(self, observations, task):
        """Consume a short cached chunk; caller must reset on task/scene changes."""
        if not self._queue:
            chunk = self.predict_action_chunk(observations, task)[:, :self.config.n_action_steps]
            self._queue.extend(chunk.transpose(0, 1))
        return self._queue.popleft()

    def reset(self):
        self._queue.clear()
