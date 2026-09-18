import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Ensure GPU 1 isolation before torch import
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

DEFAULT_CHECKPOINT_DIR = Path(r"C:\Users\74727\Desktop\project\embodied_midterm\checkpoints\RoboBrain2.5-8B-NV")

class VisualGroundingAgent:
    """
    Open-vocabulary 2D visual grounding agent using RoboBrain2.5-8B-NV.
    Detects semantic targets (e.g. 'the red cube', 'the brown woven basket')
    and returns precise bounding boxes and centroid coordinates (u, v).
    """
    def __init__(self, model_dir: Optional[Union[str, Path]] = None, device_id: int = 0):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_CHECKPOINT_DIR
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.box_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'

    def load_model(self):
        if self.model is not None:
            return
        print(f"[VisualGrounding] Loading RoboBrain2.5-8B-NV on {self.device}...")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            local_files_only=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.model_dir),
            dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            local_files_only=True
        )
        print(f"[VisualGrounding] Model ready in {time.time()-t0:.2f}s, VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    def detect_single(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        query: str,
        target_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Detects a single query in the image.
        Returns: {"box_norm": [...], "box_px": [x1, y1, x2, y2], "center_px": [u, v], "latency_s": ...}
        """
        results = self.detect_multi(image, [{"id": target_id or query, "query": query}])
        return results.get(target_id or query)

    def detect_multi(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        queries: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detects multiple targets sequentially in the image.
        queries format: [{"id": "red_cube", "query": "the red cube"}, ...]
        """
        self.load_model()

        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_w, orig_h = pil_img.size
        results = {}

        for item in queries:
            target_id = item["id"]
            query_text = item["query"]
            prompt = f"Detect {query_text} in the image and output its bounding box as [x1, y1, x2, y2] relative coordinates (0-1000)."

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt}
                ]
            }]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[pil_img], return_tensors="pt").to(self.model.device)

            t_infer = time.time()
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            dur = time.time() - t_infer

            raw = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            boxes = re.findall(self.box_pattern, raw)
            if boxes:
                # RoboBrain bbox format: [xmin, ymin, xmax, ymax] relative (0..1000)
                b = [int(v) for v in boxes[0]]
                xmin_rel, ymin_rel, xmax_rel, ymax_rel = b
                px1 = int(round(xmin_rel / 1000.0 * orig_w))
                py1 = int(round(ymin_rel / 1000.0 * orig_h))
                px2 = int(round(xmax_rel / 1000.0 * orig_w))
                py2 = int(round(ymax_rel / 1000.0 * orig_h))

                u_center = (px1 + px2) / 2.0
                v_center = (py1 + py2) / 2.0

                results[target_id] = {
                    "id": target_id,
                    "query": query_text,
                    "box_norm": b,
                    "box_px": [px1, py1, px2, py2],
                    "center_px": [round(u_center, 1), round(v_center, 1)],
                    "raw_output": raw.strip(),
                    "latency_s": round(dur, 3)
                }
                print(f"[VisualGrounding] '{target_id}' -> center=({u_center:.1f}, {v_center:.1f}), bbox={[px1, py1, px2, py2]} ({dur:.2f}s)")
            else:
                print(f"[VisualGrounding] Warning: No bbox detected for '{target_id}', raw: '{raw.strip()}'")

        return results

    @staticmethod
    def annotate_image(
        image: Union[Image.Image, np.ndarray, str, Path],
        detections: Dict[str, Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None
    ) -> Image.Image:
        """Draws bounding boxes and labels on the image."""
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        else:
            img = image.copy().convert("RGB")

        draw = ImageDraw.Draw(img)
        colors = ["red", "lime", "cyan", "yellow", "magenta", "orange"]

        for idx, (target_id, data) in enumerate(detections.items()):
            color = colors[idx % len(colors)]
            box = data["box_px"]
            u, v = data["center_px"]

            # Draw rectangle
            draw.rectangle(box, outline=color, width=3)
            # Draw center point
            draw.ellipse([u - 4, v - 4, u + 4, v + 4], fill=color, outline="white")
            label = f"{target_id} ({u:.0f}, {v:.0f})"
            draw.text((box[0], max(0, box[1] - 16)), label, fill=color)

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_p, quality=95)
            print(f"[VisualGrounding] Annotated image saved to {out_p}")

        return img
