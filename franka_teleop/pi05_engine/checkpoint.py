"""Strict, local-only safetensors loading. Never falls back to random weights."""
from pathlib import Path
import torch


def remap_keys(state):
    """Handle explicit openpi/LeRobot naming revisions without dropping weights."""
    result = {}
    for key, tensor in state.items():
        name = key.removeprefix("model.")
        for old, new in (("action_time_mlp_in.", "time_mlp_in."), ("action_time_mlp_out.", "time_mlp_out.")):
            if name.startswith(old):
                name = new + name[len(old):]
        # Transformers v4 -> v5 PaliGemma nesting (the action expert is unaffected).
        for section in ("language_model", "vision_tower", "multi_modal_projector"):
            old = f"paligemma_with_expert.paligemma.{section}."
            if name.startswith(old):
                name = f"paligemma_with_expert.paligemma.model.{section}." + name[len(old):]
        if name in result:
            raise ValueError(f"duplicate key after remapping: {name}")
        result[name] = tensor
    embedding = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    head = "paligemma_with_expert.paligemma.lm_head.weight"
    # PaliGemma's tied embedding may be saved once by safetensors.
    if embedding not in result and head in result:
        result[embedding] = result[head]
    if head not in result and embedding in result:
        result[head] = result[embedding]
    return result


def load_weights(model, file: str | Path):
    file = Path(file)
    if not file.is_file() or file.suffix != ".safetensors":
        raise FileNotFoundError("expected an existing model.safetensors file")
    from safetensors.torch import load_file
    state = remap_keys(load_file(str(file), device="cpu"))
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    wrong = [k for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape]
    if missing or extra or wrong:
        raise ValueError(f"checkpoint incompatible: missing={missing[:5]}, extra={extra[:5]}, shape_mismatch={wrong[:5]}")
    model.load_state_dict(state, strict=True, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("unmaterialized parameters remain after loading")
    model.requires_grad_(False)
    model.eval()
    return model
