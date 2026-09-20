"""
Comprehensive Automated Test Suite for Codex Audit Fixes (Branch: fix/pi05-codex-audit)
Verifies:
  1. Gripper Action Command vs Physical State separation.
  2. FP32 parameters, FP32 AdamW moments, and non-zero parameter updates.
  3. Preprocessing 4:3 vs 16:9 decoupling and backward compatibility.
  4. Strict checkpoint key validation (missing & unexpected keys).
  5. Camera TTL stale frame protection.
"""
import os
import sys
import time
from pathlib import Path
import unittest
import numpy as np
import torch

# Ensure repository root is in python path
REPO_ROOT = Path("C:/Users/74727/Desktop/project/VLA_franka")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "python"))

from franka_teleop.pi05_engine.config import PI05Config
from franka_teleop.pi05_engine.processing import PI05Processor
from franka_teleop.pi05_engine.lora import inject_pi05_lora, load_lora_state_dict, extract_lora_state_dict
from franka_teleop.pi05_engine.runtime import PI05Inference


class TestCodexAuditFixes(unittest.TestCase):

    def test_01_preprocessing_decoupling(self):
        """Verify 4:3 native resize-pad vs 16:9 center crop decoupling."""
        print("\n--- [Test 1] Preprocessing Decoupling ---")
        config_path = REPO_ROOT / "checkpoints/pi05_droid_jointpos/config.json"
        config = PI05Config.from_file(config_path)
        stats_path = REPO_ROOT / "checkpoints/pi05_droid_jointpos/auxiliary/openpi_droid_jointpos_norm_stats.json"
        tokenizer_path = REPO_ROOT / "checkpoints/pi05_droid_jointpos/auxiliary/paligemma_tokenizer.model"
        
        proc_none = PI05Processor.from_local(config, stats_path, tokenizer_path, profile="droid_jointpos", crop_mode="none")
        proc_169 = PI05Processor.from_local(config, stats_path, tokenizer_path, profile="droid_jointpos", crop_mode="16_9")
        
        dummy_img = torch.ones(3, 480, 640, dtype=torch.uint8) * 128
        obs = {
            "observation.images.base_0_rgb": dummy_img,
            "observation.images.left_wrist_0_rgb": dummy_img,
            "observation.state": np.zeros(8, dtype=np.float32)
        }
        
        prep_none = proc_none.prepare(obs, "test task", device="cpu")
        prep_169 = proc_169.prepare(obs, "test task", device="cpu")
        
        # In [-1, 1] range: padding is -1.0
        # For 640x480 (4:3) -> 224x168: top/bottom padding is 28 pixels: rows 0..27 and 196..223 should be -1.0
        # Row 28 should be non-padding
        img_none = prep_none.images[0][0]  # [3, 224, 224]
        self.assertTrue(torch.all(img_none[:, 0:27, :] == -1.0), "4:3 top padding should be ~28px")
        self.assertFalse(torch.all(img_none[:, 30, :] == -1.0), "4:3 content should exist at row 30")

        # For 16:9 -> 224x126: top/bottom padding is 49 pixels: row 30 is padding in 16:9!
        img_169 = prep_169.images[0][0]
        self.assertTrue(torch.all(img_169[:, 30, :] == -1.0), "16:9 top padding should cover row 30")
        self.assertFalse(torch.all(img_169[:, 55, :] == -1.0), "16:9 content should exist at row 55")
        print("  [PASS] 4:3 native preserved (no bottom cutoff) & 16:9 mode functional.")

    def test_02_gripper_action_command_semantics(self):
        """Verify that training and eval load true action gripper commands (0/1) instead of delayed state."""
        print("\n--- [Test 2] Gripper Action Command Semantics ---")
        from train_pi05_lora import build_multitask_dataset
        
        dataset_dir = REPO_ROOT / "dataset/teleop_pick_cube_15hz_002"
        if not dataset_dir.exists():
            self.skipTest("Dataset not present on this machine")
            
        train_samples, val_samples, cached_files, _, _ = build_multitask_dataset(
            [dataset_dir],
            val_ratio=0.15,
            num_val_episodes=2,
            crop_16_9=False
        )
        
        first_cf = next(iter(cached_files.values()))
        self.assertIn("actions", first_cf, "cached_files must contain 'actions' table column")
        actions_raw = first_cf["actions"]
        states_raw = first_cf["states"]
        
        # Check that action[:, 7] has discrete 0/1 command values
        grip_actions = actions_raw[:, 7]
        unique_vals = np.unique(grip_actions)
        print(f"  Actions column gripper values: {unique_vals}")
        self.assertTrue(all(v in {0.0, 1.0} for v in unique_vals), "Action gripper must be 0 or 1 commands")
        
        # Check that state[:, 7] has continuous physical measurement values
        grip_states = states_raw[:, 7]
        print(f"  States column gripper range: min={grip_states.min():.3f}, max={grip_states.max():.3f}")
        self.assertGreater(len(np.unique(grip_states)), 10, "State gripper must be continuous sensor readings")
        print("  [PASS] Action command (0/1) strictly separated from physical state.")

    def test_03_strict_checkpoint_loading(self):
        """Verify strict checkpoint loading rejects missing and unexpected keys."""
        print("\n--- [Test 3] Strict Checkpoint Loading ---")
        import torch.nn as nn
        
        class DummyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.action_in_proj = nn.Linear(32, 1024)
                self.action_out_proj = nn.Linear(1024, 32)
                self.base = nn.Linear(10, 10)
                self.base.requires_grad_(False)
                
        net = DummyNet()
        # Normal extraction
        sd = extract_lora_state_dict(net)
        self.assertEqual(len(sd), 4, "Should extract 4 tensors for 2 linear layers (weights + biases)")
        
        # Strict load succeeds on exact keys
        load_lora_state_dict(net, sd, strict=True)
        
        # Strict load fails on missing key
        incomplete_sd = {k: v for k, v in sd.items() if "out_proj" not in k}
        with self.assertRaises(KeyError):
            load_lora_state_dict(net, incomplete_sd, strict=True)
            
        # Strict load fails on unexpected key
        unexpected_sd = dict(sd)
        unexpected_sd["non_existent_layer.weight"] = torch.zeros(5)
        with self.assertRaises(KeyError):
            load_lora_state_dict(net, unexpected_sd, strict=True)
            
        print("  [PASS] Strict verification properly rejects missing and unexpected keys.")

    def test_04_fp32_precision_and_optimizer_updates(self):
        """Verify that LoRA and projection layers are FP32, AdamW moments are FP32, and receive fine updates."""
        print("\n--- [Test 4] FP32 Precision & Optimizer Updates ---")
        import torch.nn as nn
        
        # Test LoRALinear initialization dtype
        base_bf16 = nn.Linear(1024, 1024).to(dtype=torch.bfloat16)
        from franka_teleop.pi05_engine.lora import LoRALinear
        lora = LoRALinear(base_bf16, rank=16, lora_alpha=32.0)
        
        self.assertEqual(lora.lora_A.dtype, torch.float32, "lora_A must be float32")
        self.assertEqual(lora.lora_B.dtype, torch.float32, "lora_B must be float32")
        
        # Forward pass with bfloat16 input
        x_bf16 = torch.randn(2, 4, 1024, dtype=torch.bfloat16)
        out = lora(x_bf16)
        self.assertEqual(out.dtype, torch.bfloat16, "Output of LoRALinear should match base activation dtype")
        
        # Optimizer with small learning rate
        opt = torch.optim.AdamW([lora.lora_A, lora.lora_B], lr=1e-6)
        loss = out.float().mean()
        loss.backward()
        
        opt.step()
        
        # Verify optimizer moments are float32
        state_A = opt.state[lora.lora_A]
        self.assertEqual(state_A["exp_avg"].dtype, torch.float32, "AdamW exp_avg must be float32")
        self.assertEqual(state_A["exp_avg_sq"].dtype, torch.float32, "AdamW exp_avg_sq must be float32")
        
        # Verify lora_B received non-zero gradient and update (previously initialized to 0.0)
        self.assertTrue(torch.any(lora.lora_B.data != 0.0), "lora_B should have updated from exact zero")
        print("  [PASS] LoRA is FP32, AdamW moments are FP32, and fine updates are preserved.")

    def test_05_camera_ttl_guard(self):
        """Verify that get_frames returns None, None when cameras are stale."""
        print("\n--- [Test 5] Camera TTL Stale Protection ---")
        from sync_vla_agent import DualRealSenseStreamer
        
        streamer = DualRealSenseStreamer("MOCK_F", "MOCK_W", fps=15)
        now = time.time()
        # Mock valid fresh frames
        streamer.frames["front"] = np.zeros((480, 640, 3), dtype=np.uint8)
        streamer.frames["wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
        streamer.timestamps["front"] = now
        streamer.timestamps["wrist"] = now
        
        f, w = streamer.get_frames(max_age_s=0.5)
        self.assertIsNotNone(f, "Fresh frames should be returned")
        self.assertIsNotNone(w, "Fresh frames should be returned")
        
        # Mock stale front camera (>0.5s ago)
        streamer.timestamps["front"] = now - 1.0
        f_stale, w_stale = streamer.get_frames(max_age_s=0.5)
        self.assertIsNone(f_stale, "Stale frames must return None to trigger safe hold")
        self.assertIsNone(w_stale, "Stale frames must return None to trigger safe hold")
        print("  [PASS] Stale camera frames safely intercepted with None, None.")


if __name__ == "__main__":
    unittest.main()
