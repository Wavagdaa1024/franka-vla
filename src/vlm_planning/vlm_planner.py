import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Set

KNOWN_OBJECTS = {
    "obj_01": "紫洋葱",
    "obj_02": "绿尖椒",
    "red": "红色物体",
    "yellow": "黄色物体",
    "onion": "洋葱",
    "pepper": "尖椒",
}
KNOWN_CONTAINERS = {"basket", "basket_01", "box"}

def validate_plan_schema(
    plan: Any,
    allowed_objects: Optional[Set[str]] = None,
    allowed_containers: Optional[Set[str]] = None,
    allow_empty: bool = False
) -> Tuple[bool, str]:
    """Strict schema validator for planning output.
    Returns (is_valid, error_message).
    """
    if not isinstance(plan, dict):
        return False, "Plan must be a JSON dictionary"
    
    # Handle explicit rejection / no_action response
    status = plan.get("status")
    if status not in (None, "success", "rejected", "no_action"):
        return False, "Unsupported plan status"
    if status in ("rejected", "no_action"):
        if plan.get("steps") != []:
            return False, "Rejection/no_action requires an explicit empty steps list"
        if not isinstance(plan.get("reason"), str) or not plan["reason"].strip():
            return False, "Rejection/no_action requires a non-empty reason"
        return True, "ok: explicit rejection/no_action"

    steps = plan.get("steps")
    if steps is None or not isinstance(steps, list):
        return False, "Plan must contain a 'steps' list"

    if len(steps) == 0:
        if allow_empty:
            return True, "ok: empty steps allowed"
        return False, "Plan 'steps' list cannot be empty without explicit status='no_action'"

    if len(steps) > 8:
        return False, f"Steps count ({len(steps)}) exceeds maximum allowed (8)"

    held_in_plan = None
    for idx, s in enumerate(steps):
        if not isinstance(s, dict):
            return False, f"Step {idx} is not a dictionary"
        
        action = s.get("action")
        obj = s.get("object")
        target = s.get("target")

        if action not in ("pick", "place"):
            return False, f"Step {idx} has invalid action '{action}'. Only 'pick' and 'place' are allowed"
        
        if not obj or not isinstance(obj, str):
            return False, f"Step {idx} missing valid 'object' identifier"

        if allowed_objects is not None and obj not in allowed_objects:
            return False, f"Step {idx} references object '{obj}' not present in allowed scene objects"

        if action == "pick":
            if held_in_plan is not None:
                return False, f"Step {idx}: Invalid sequence - attempting to pick '{obj}' while already holding '{held_in_plan}'"
            held_in_plan = obj
        elif action == "place":
            if held_in_plan is None:
                return False, f"Step {idx}: Invalid sequence - attempting to place '{obj}' with empty gripper"
            if held_in_plan != obj:
                return False, f"Step {idx}: Object mismatch - holding '{held_in_plan}' but placing '{obj}'"
            if not target or not isinstance(target, str):
                return False, f"Step {idx}: 'place' action requires a non-empty 'target' container"
            if allowed_containers is not None and target not in allowed_containers:
                return False, f"Step {idx}: Target container '{target}' not in allowed containers"
            held_in_plan = None

    if held_in_plan is not None:
        return False, f"Plan ends with object '{held_in_plan}' still held in gripper (incomplete place)"

    return True, "ok"

def parse_and_validate_vlm_raw(
    raw: str,
    allowed_objects: Optional[Set[str]] = None,
    allowed_containers: Optional[Set[str]] = None
) -> Tuple[Dict[str, Any], bool, str]:
    """Parses raw text from VLM, extracts JSON, and strictly validates it.
    Does NOT use color-word fallback to fake success.
    """
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"steps": [], "status": "parse_error"}, False, "No JSON structure found in raw output"
    
    try:
        plan = json.loads(m.group(0))
    except Exception as e:
        return {"steps": [], "status": "json_syntax_error"}, False, f"Invalid JSON syntax: {e}"

    is_valid, msg = validate_plan_schema(plan, allowed_objects, allowed_containers)
    return plan, is_valid, msg

def build_planner_prompt(instruction, allowed_objects, allowed_containers):
    names = {"obj_01": "紫色洋葱", "obj_02": "绿色尖椒", "basket_01": "棕色篮子"}
    scene = "\n".join(f"- {key}: {names.get(key, key)}" for key in sorted(allowed_objects | allowed_containers))
    return (
        "你是机器人任务规划器。根据图片、许可对象表和指令，只输出合法 JSON。\n"
        f"许可对象表（不在表内不代表图像中不存在）：\n{scene}\n"
        "当前 pick/place 是规划表示，不代表已验收的独立真机技能。\n"
        '格式：{"steps": [{"action": "pick", "object": "obj_01"}, '
        '{"action": "place", "object": "obj_01", "target": "basket_01"}]}\n'
        '对象未获许可、歧义或无法执行时：{"status": "rejected", "reason": "具体原因", "steps": []}\n'
        f"指令：{instruction}"
    )


def plan_with_vlm(
    image: Path,
    instruction: str,
    model_name: str,
    gpu_id: Optional[int] = None,
    max_new_tokens: int = 256,
    allowed_objects: Optional[Set[str]] = None,
    allowed_containers: Optional[Set[str]] = None
) -> Tuple[Dict[str, Any], str, bool]:
    """Runs VLM inference with strict GPU isolation and schema verification.
    Returns: (plan_dict, raw_text, plan_valid_bool)
    """
    if gpu_id not in (0, 1):
        raise ValueError("Explicitly select physical GPU 0 or 1 before VLM inference")
    # Enforce GPU isolation before importing torch
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[VLM Planner] Isolated to physical GPU {gpu_id}. Visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    root = Path(__file__).resolve().parents[1] / "checkpoints"
    model_dir = root / ("Qwen3.5-9B" if model_name.lower().startswith("qwen") else "RoboBrain2.5-8B-NV")

    allowed_objects = {"obj_01", "obj_02"} if allowed_objects is None else set(allowed_objects)
    allowed_containers = {"basket_01"} if allowed_containers is None else set(allowed_containers)
    prompt = build_planner_prompt(instruction, allowed_objects, allowed_containers)

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # Index 0 within the isolated visible GPU space
        trust_remote_code=True,
        local_files_only=True
    )

    messages = [{"role": "user", "content": [{"type": "image", "image": str(image)}, {"type": "text", "text": prompt}]}]
    is_qwen = model_name.lower().startswith("qwen")
    if is_qwen:
        messages.append({"role": "assistant", "content": "```json\n{"})
        text = processor.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    else:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[Image.open(image).convert("RGB")], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    raw = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    if is_qwen:
        raw = "{\n" + raw.split("```")[0].strip()
    
    plan, is_valid, err_msg = parse_and_validate_vlm_raw(raw, allowed_objects, allowed_containers)
    if not is_valid:
        print(f"[VLM Planner Warning] Validation failed: {err_msg}")
    
    return plan, raw, is_valid
