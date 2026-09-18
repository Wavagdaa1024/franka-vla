import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def render_pair_image(front_bgr, wrist_bgr, title_cn, title_en, meta_str, font_title, font_meta, font_badge):
    h, w = 480, 640
    f_res = cv2.resize(front_bgr, (w, h))
    w_res = cv2.resize(wrist_bgr, (w, h))
    
    header_h = 76
    total_w = w * 2
    total_h = h + header_h
    
    # Base canvas
    canvas_bgr = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    
    # Header background (deep elegant dark navy #0f172a)
    canvas_bgr[:header_h, :] = (42, 23, 15)  # BGR for #0f172a
    
    # Top accent cyan line (4px)
    canvas_bgr[:4, :] = (212, 182, 6)  # BGR for #06b6d4
    
    # Images placement
    canvas_bgr[header_h:, :w] = f_res
    canvas_bgr[header_h:, w:] = w_res
    
    # Divider line between two views (subtle gray)
    cv2.line(canvas_bgr, (w, header_h), (w, total_h), (70, 70, 70), 2)
    # Divider below header
    cv2.line(canvas_bgr, (0, header_h), (total_w, header_h), (80, 80, 80), 1)
    
    # Convert to PIL for crisp typography
    img_pil = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Title on Header (Line 1, Y=14)
    full_title = f"{title_cn}  ({title_en})"
    draw.text((22, 12), full_title, font=font_title, fill=(255, 255, 255))
    
    # Meta Info on Header (Line 2, Y=44)
    draw.text((22, 46), meta_str, font=font_meta, fill=(148, 163, 184))
    
    # Overlay badges on camera frames
    # Front badge (Third-Person)
    badge_f_text = "[第三视角 Front Camera] 全局场景监控"
    # Measure badge size
    bbox_f = font_badge.getbbox(badge_f_text)
    bw_f = bbox_f[2] - bbox_f[0] + 20
    bh_f = bbox_f[3] - bbox_f[1] + 14
    
    # Draw translucent badge background for Front
    badge_f_x, badge_f_y = 16, header_h + 16
    draw.rectangle([badge_f_x, badge_f_y, badge_f_x + bw_f, badge_f_y + bh_f], fill=(15, 23, 42), outline=(6, 182, 212), width=2)
    draw.text((badge_f_x + 10, badge_f_y + 5), badge_f_text, font=font_badge, fill=(6, 182, 212))
    
    # Wrist badge (Eye-in-Hand)
    badge_w_text = "[腕部视角 Wrist Camera] 末端手眼特写"
    bbox_w = font_badge.getbbox(badge_w_text)
    bw_w = bbox_w[2] - bbox_w[0] + 20
    bh_w = bbox_w[3] - bbox_w[1] + 14
    
    badge_w_x, badge_w_y = w + 16, header_h + 16
    draw.rectangle([badge_w_x, badge_w_y, badge_w_x + bw_w, badge_w_y + bh_w], fill=(15, 23, 42), outline=(245, 158, 11), width=2)
    draw.text((badge_w_x + 10, badge_w_y + 5), badge_w_text, font=font_badge, fill=(245, 158, 11))
    
    # Convert back to BGR
    result_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return result_bgr

def main():
    dataset_root = Path(r"C:\Users\74727\Desktop\project\VLA_franka\dataset")
    out_dir = Path(r"C:\Users\74727\Desktop\project\extracted_camera_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    font_path = r"C:\Windows\Fonts\msyh.ttc"
    font_title = ImageFont.truetype(font_path, 21)
    font_meta = ImageFont.truetype(font_path, 14)
    font_badge = ImageFont.truetype(font_path, 15)
    
    targets = [
        # --- 1. Red Cube Full Sequence (teleop_pick_cube_15hz_001, Ep 0) ---
        {
            "ds": "teleop_pick_cube_15hz_001", "ep": 0, "frame": 25,
            "title_cn": "红方块抓放 · 阶段 1：初始就绪与俯冲接近",
            "title_en": "Approach & Reaching",
            "tag": "pair_01_red_cube_stage1_approach",
            "gripper_desc": "张开 (0.00)"
        },
        {
            "ds": "teleop_pick_cube_15hz_001", "ep": 0, "frame": 45,
            "title_cn": "红方块抓放 · 阶段 2：预抓取垂直居中对齐",
            "title_en": "Pre-Grasp Alignment & Centering",
            "tag": "pair_02_red_cube_stage2_pregrasp",
            "gripper_desc": "张开 (0.00)"
        },
        {
            "ds": "teleop_pick_cube_15hz_001", "ep": 0, "frame": 80,
            "title_cn": "红方块抓放 · 阶段 3：夹爪闭合接触抓紧",
            "title_en": "Gripper Contact & Grasp",
            "tag": "pair_03_red_cube_stage3_grasp",
            "gripper_desc": "夹紧锁定 (0.74)"
        },
        {
            "ds": "teleop_pick_cube_15hz_001", "ep": 0, "frame": 115,
            "title_cn": "红方块抓放 · 阶段 4：垂直提升与平移转运",
            "title_en": "Lift & Aerial Transfer",
            "tag": "pair_04_red_cube_stage4_lift",
            "gripper_desc": "夹紧保持 (0.71)"
        },
        {
            "ds": "teleop_pick_cube_15hz_001", "ep": 0, "frame": 145,
            "title_cn": "红方块抓放 · 阶段 5：目标篮内释放放置",
            "title_en": "Basket Entry & Release",
            "tag": "pair_05_red_cube_stage5_place",
            "gripper_desc": "松爪释放 (0.00)"
        },
        
        # --- 2. Red Cube Alternative Position (teleop_pick_cube_15hz_001, Ep 5) ---
        {
            "ds": "teleop_pick_cube_15hz_001", "ep": 5, "frame": 68,
            "title_cn": "红方块抓放 · 变位抓取：不同台面落点空间泛化",
            "title_en": "Alternative Position Grasp",
            "tag": "pair_06_red_cube_ep5_grasp",
            "gripper_desc": "夹紧锁定 (0.75)"
        },
        
        # --- 3. Multi-Cube Tasks (teleop_pick_cube_15hz_002) ---
        {
            "ds": "teleop_pick_cube_15hz_002", "ep": 11, "frame": 70,
            "title_cn": "蓝方块抓放 · 多目标场景下的手眼精准对准",
            "title_en": "Blue Cube Centering in Multi-Object Scene",
            "tag": "pair_07_blue_cube_align",
            "gripper_desc": "张开 (0.00)"
        },
        {
            "ds": "teleop_pick_cube_15hz_002", "ep": 22, "frame": 45,
            "title_cn": "橙方块抓放 · 相邻方块干扰下的局部目标锁定",
            "title_en": "Orange Cube Centering with Adjacent Red Cube",
            "tag": "pair_08_orange_cube_align",
            "gripper_desc": "张开 (0.00)"
        },
        
        # --- 4. Vegetable Tasks (teleop_pick_vegetables_15hz_001) ---
        {
            "ds": "teleop_pick_vegetables_15hz_001", "ep": 0, "frame": 95,
            "title_cn": "柔性异形物体 · 胡萝卜抓取特写（突破第三视角遮挡）",
            "title_en": "Flexible Object: Carrot Grasp Close-up",
            "tag": "pair_09_carrot_grasp",
            "gripper_desc": "夹紧 (0.58)"
        },
        {
            "ds": "teleop_pick_vegetables_15hz_001", "ep": 17, "frame": 90,
            "title_cn": "复杂曲面物体 · 青椒抓取特写（大曲率自适应接触）",
            "title_en": "Complex Curved Object: Pepper Grasp Close-up",
            "tag": "pair_10_pepper_grasp",
            "gripper_desc": "夹紧 (0.62)"
        },
    ]
    
    extracted_records = []
    
    for item in targets:
        ds_name = item["ds"]
        ep_idx = item["ep"]
        frame_in_ep = item["frame"]
        
        ds_dir = dataset_root / ds_name
        meta_df = pd.concat([pd.read_parquet(p) for p in (ds_dir / "meta" / "episodes").glob("**/*.parquet")])
        meta_row = meta_df[meta_df['episode_index'] == ep_idx].iloc[0]
        
        chunk_f = meta_row['videos/observation.images.front/chunk_index']
        file_f = meta_row['videos/observation.images.front/file_index']
        from_ts_f = float(meta_row['videos/observation.images.front/from_timestamp'])
        
        chunk_w = meta_row['videos/observation.images.wrist/chunk_index']
        file_w = meta_row['videos/observation.images.wrist/file_index']
        from_ts_w = float(meta_row['videos/observation.images.wrist/from_timestamp'])
        
        vid_f_path = ds_dir / "videos" / "observation.images.front" / f"chunk-{chunk_f:03d}" / f"file-{file_f:03d}.mp4"
        vid_w_path = ds_dir / "videos" / "observation.images.wrist" / f"chunk-{chunk_w:03d}" / f"file-{file_w:03d}.mp4"
        
        abs_frame_f = int(round(from_ts_f * 15)) + frame_in_ep
        abs_frame_w = int(round(from_ts_w * 15)) + frame_in_ep
        time_sec = frame_in_ep / 15.0
        
        cap_f = cv2.VideoCapture(str(vid_f_path))
        cap_w = cv2.VideoCapture(str(vid_w_path))
        
        cap_f.set(cv2.CAP_PROP_POS_FRAMES, abs_frame_f)
        ret_f, frame_f = cap_f.read()
        
        cap_w.set(cv2.CAP_PROP_POS_FRAMES, abs_frame_w)
        ret_w, frame_w = cap_w.read()
        
        cap_f.release()
        cap_w.release()
        
        if not ret_f or not ret_w:
            print(f"[ERROR] Failed to read frame for {item['tag']}")
            continue
            
        # Standalone files
        p_front = out_dir / f"{item['tag']}_front.jpg"
        p_wrist = out_dir / f"{item['tag']}_wrist.jpg"
        cv2.imwrite(str(p_front), frame_f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(p_wrist), frame_w, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Meta string
        total_ep_frames = meta_row['length']
        meta_str = f"数据集: {ds_name} | Episode: {ep_idx} | 帧号: Frame {frame_in_ep:03d}/{total_ep_frames} (t = {time_sec:.2f}s) | 夹爪状态: {item['gripper_desc']} | 机械臂: Franka Emika Panda"
        
        # Render pair
        pair_img = render_pair_image(
            frame_f, frame_w,
            item["title_cn"], item["title_en"],
            meta_str, font_title, font_meta, font_badge
        )
        p_pair = out_dir / f"{item['tag']}_pair.jpg"
        cv2.imwrite(str(p_pair), pair_img, [cv2.IMWRITE_JPEG_QUALITY, 96])
        
        extracted_records.append({
            "tag": item["tag"],
            "title_cn": item["title_cn"],
            "title_en": item["title_en"],
            "pair_file": str(p_pair)
        })
        print(f"Rendered: {item['tag']} -> {p_pair.name}")

    # Build sequence collage for Red Cube (first 5 pairs)
    pairs_seq = [cv2.imread(r["pair_file"]) for r in extracted_records[:5]]
    if len(pairs_seq) == 5:
        # Scale to 960 width each
        seq_scaled = [cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1]))) for img in pairs_seq]
        v_seq = np.vstack(seq_scaled)
        p_seq = out_dir / "red_cube_5stages_comparison_sequence.jpg"
        cv2.imwrite(str(p_seq), v_seq, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"Saved: {p_seq.name}")

    # Build multi-task collage (pairs 6 to 10)
    pairs_multi = [cv2.imread(r["pair_file"]) for r in extracted_records[5:]]
    if len(pairs_multi) == 5:
        multi_scaled = [cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1]))) for img in pairs_multi]
        v_multi = np.vstack(multi_scaled)
        p_multi = out_dir / "multitask_5objects_comparison_overview.jpg"
        cv2.imwrite(str(p_multi), v_multi, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"Saved: {p_multi.name}")

    print("\nAll 10 pairs and 2 composite sheets successfully created!")

if __name__ == "__main__":
    main()
