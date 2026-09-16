import os
import sys
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(r"C:\Users\74727\Desktop\project\VLA_franka")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from franka_teleop.pi05_engine.runtime import PI05Inference
from franka_teleop.pi05_engine.lora import inject_pi05_lora, extract_lora_state_dict, load_lora_state_dict

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "pi05_droid_jointpos"
STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_jointpos_norm_stats.json"
TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"

print("[Test] Loading base PI0.5 model on GPU 1...")
inference = PI05Inference.from_checkpoint(
    CHECKPOINT_DIR,
    stats_path=STATS_PATH,
    tokenizer_path=TOKENIZER_PATH,
    device="cuda:0",
    profile="droid_jointpos"
)
net = inference.network

print("\n[Test] Auditing baseline parameters before LoRA...")
total_before = sum(p.numel() for p in net.parameters())
print(f"  Total baseline params: {total_before:,} ({total_before/1e6:,.2f} M)")

print("\n[Test] Injecting LoRA adapters (PaliGemma r=16, Action Expert r=32)...")
lang_loras, expert_loras = inject_pi05_lora(
    net,
    lang_rank=16,
    lang_alpha=32.0,
    expert_rank=32,
    expert_alpha=64.0,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
)

print(f"  Injected PaliGemma LoRAs:     {len(lang_loras)} linear layers")
print(f"  Injected Action Expert LoRAs: {len(expert_loras)} linear layers")

trainable_params = [p for p in net.parameters() if p.requires_grad]
trainable_count = sum(p.numel() for p in trainable_params)
total_after = sum(p.numel() for p in net.parameters())
frozen_count = total_after - trainable_count

print("\n[Test] Parameter Isolation Audit:")
print(f"  Trainable params: {trainable_count:,} ({trainable_count/1e6:,.2f} M, {trainable_count/total_after*100:.2f}%)")
print(f"  Frozen params:    {frozen_count:,} ({frozen_count/1e6:,.2f} M, {frozen_count/total_after*100:.2f}%)")
print(f"  Total params:     {total_after:,} ({total_after/1e6:,.2f} M)")

# Verify language model base is frozen
lang_model = net.paligemma_with_expert.paligemma.model.language_model
for name, p in lang_model.named_parameters():
    if "lora_" in name:
        assert p.requires_grad, f"LoRA param {name} should be trainable!"
    else:
        assert not p.requires_grad, f"Base param {name} should be frozen!"
print("  [Pass] PaliGemma base parameters are 100% frozen, only LoRA adapters trainable.")

# Verify expert model base is frozen
expert_model = net.paligemma_with_expert.gemma_expert
for name, p in expert_model.named_parameters():
    if "lora_" in name:
        assert p.requires_grad, f"LoRA param {name} should be trainable!"
    else:
        assert not p.requires_grad, f"Base param {name} should be frozen!"
print("  [Pass] Action Expert base parameters are 100% frozen, only LoRA adapters trainable.")

# Verify action projections are trainable
for m in [net.action_in_proj, net.action_out_proj, net.time_mlp_in, net.time_mlp_out]:
    assert all(p.requires_grad for p in m.parameters()), "Action Projections and Time MLP must be trainable!"
print("  [Pass] Action Projections & Time MLP are trainable.")

# Test state dict extract & reload
print("\n[Test] Testing LoRA state dict extraction and loading...")
state_dict = extract_lora_state_dict(net)
print(f"  Extracted {len(state_dict)} trainable tensor entries.")
# Calculate state dict file size in MB
total_bytes = sum(t.nelement() * t.element_size() for t in state_dict.values())
print(f"  LoRA adapter size in memory: {total_bytes / (1024*1024):.2f} MB")

load_lora_state_dict(net, state_dict, strict=True)
print("  [Pass] State dict successfully extracted and reloaded.")

print("\n" + "=" * 60)
print("ALL LoRA UNIT TESTS PASSED EMPIRICALLY!")
print("=" * 60)
