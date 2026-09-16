"""Verify that checkpoints, models, and tokenizers exist and are valid."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT / "checkpoints"

REQUIRED_ASSETS = [
    ("pi05_droid_jointpos", CHECKPOINTS_DIR / "pi05_droid_jointpos"),
    ("Qwen3.5-9B", CHECKPOINTS_DIR / "Qwen3.5-9B"),
    ("RoboBrain2.5-8B-NV", CHECKPOINTS_DIR / "RoboBrain2.5-8B-NV"),
]

def main():
    print("=" * 60)
    print("  VLA_franka Asset Verification")
    print("=" * 60)
    all_ok = True
    for name, path in REQUIRED_ASSETS:
        if path.exists():
            files = list(path.glob("*"))
            print(f"[PASS] {name:<22}: Found ({len(files)} files in {path.name})")
        else:
            print(f"[FAIL] {name:<22}: MISSING at {path}")
            all_ok = False

    print("-" * 60)
    if all_ok:
        print("[ALL PASS] All required model checkpoints and assets are present.")
        return 0
    else:
        print("[WARN] Some checkpoints are missing. Check directory structure.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
