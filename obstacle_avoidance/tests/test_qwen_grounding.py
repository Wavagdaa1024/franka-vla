import os
import sys
import json
import time
import re
from pathlib import Path
from PIL import Image, ImageDraw

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_DIR = Path(r"C:\Users\74727\Desktop\project\VLA_franka\checkpoints\Qwen3.5-9B")
IMAGE_PATH = Path(r"C:\Users\74727\Desktop\project\VLA_franka\tests\assets\front_camera.jpg")
OUTPUT_DIR = Path(r"C:\Users\74727\Desktop\project\VLA_franka\outputs\qwen_grounding_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_QUERIES = [
    {"id": "basket_01", "name": "棕色篮子", "query": "the brown woven basket"},
    {"id": "obj_01", "name": "紫色洋葱", "query": "the purple onion"},
    {"id": "obj_02", "name": "绿色尖椒", "query": "the green chili pepper"}
]

print(f"[Loading Qwen3.5-9B on GPU 1] from {MODEL_DIR}...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    str(MODEL_DIR),
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
    local_files_only=True
)
print(f"[Loaded Qwen] in {time.time() - t0:.2f}s. VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

raw_image = Image.open(str(IMAGE_PATH)).convert("RGB")
orig_w, orig_h = raw_image.size
annotated_image = raw_image.copy()
draw = ImageDraw.Draw(annotated_image)

colors = {
    "obj_01": "magenta",
    "obj_02": "lime",
    "basket_01": "cyan"
}

results = []
box_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'

for item in TARGET_QUERIES:
    target_id = item["id"]
    target_name = item["name"]
    query = item["query"]
    prompt = f"Detect {query} in the image and output its bounding box as [x1, y1, x2, y2] relative coordinates (0-1000). Output JSON or coordinate list directly."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(IMAGE_PATH)},
                {"type": "text", "text": prompt}
            ]
        },
        {"role": "assistant", "content": f"[{target_id}] bounding box: ["}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    inputs = processor(text=[text], images=[raw_image], return_tensors="pt").to(model.device)

    t_infer = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    dur = round(time.time() - t_infer, 4)

    raw_gen = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    raw_full = "[" + raw_gen.strip()
    boxes = re.findall(box_pattern, raw_full)
    
    parsed_boxes = []
    pixel_boxes = []
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        parsed_boxes.append([x1, y1, x2, y2])
        px1 = int(round(x1 / 1000.0 * orig_w))
        py1 = int(round(y1 / 1000.0 * orig_h))
        px2 = int(round(x2 / 1000.0 * orig_w))
        py2 = int(round(y2 / 1000.0 * orig_h))
        pixel_boxes.append([px1, py1, px2, py2])
        c = colors.get(target_id, "red")
        draw.rectangle([px1, py1, px2, py2], outline=c, width=3)
        draw.text((px1, max(0, py1 - 15)), f"{target_id} ({target_name})", fill=c)

    res_item = {
        "id": target_id,
        "name": target_name,
        "query": query,
        "latency_s": dur,
        "raw_output": raw_full,
        "norm_boxes": parsed_boxes,
        "pixel_boxes": pixel_boxes
    }
    results.append(res_item)
    print(f"[{target_id} | {target_name}] ({dur}s): raw='{raw_full}' -> pixel_boxes={pixel_boxes}")

annotated_path = OUTPUT_DIR / "front_camera_qwen_annotated.jpg"
annotated_image.save(str(annotated_path), quality=95)
print(f"\n[Annotated Image Saved]: {annotated_path}")

json_path = OUTPUT_DIR / "qwen_grounding_results.json"
json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[Results JSON Saved]: {json_path}")
