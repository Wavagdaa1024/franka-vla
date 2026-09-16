"""Explicit offline entry point. Never streams commands to hardware."""
import argparse
import json
from pathlib import Path
import numpy as np
from .config import PI05Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--inspect", action="store_true", help="inspect config/files without loading model")
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--input", type=Path, help="NPZ containing observation.state and configured RGB keys")
    parser.add_argument("--task")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda:0"])
    parser.add_argument("--profile", default="checkpoint", choices=["checkpoint", "droid"])
    args = parser.parse_args()
    config = PI05Config.from_file(args.checkpoint / "config.json")
    if args.inspect:
        print(json.dumps({"images": config.image_keys, "state_dim": config.feature_dim("observation.state"),
                          "action_dim": config.feature_dim("action"), "chunk_size": config.chunk_size,
                          "weights_present": (args.checkpoint / "model.safetensors").is_file()}, indent=2))
        return
    if any(x is None for x in (args.stats, args.tokenizer, args.input, args.task, args.output)):
        parser.error("inference requires --stats, --tokenizer, --input, --task and --output")
    if args.output.exists():
        parser.error("output already exists; choose a new filename")
    from .runtime import PI05Inference
    model = PI05Inference.from_checkpoint(args.checkpoint, stats_path=args.stats,
                                          tokenizer_path=args.tokenizer, device=args.device, profile=args.profile)
    with np.load(args.input, allow_pickle=False) as data:
        batch = dict(data)
    actions = model.predict_action_chunk(batch, args.task).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.save(handle, actions)
    print(f"Saved offline actions {actions.shape} to {args.output}; no robot commands were sent.")


if __name__ == "__main__":
    main()
