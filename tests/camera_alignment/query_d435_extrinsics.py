import pyrealsense2 as rs

ctx = rs.context()
for dev in ctx.devices:
    sn = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    print(f"\n================ Device: {name} (S/N: {sn}) ================")
    
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.depth)
    cfg.enable_stream(rs.stream.color)
    
    try:
        profile = pipe.start(cfg)
        depth_prof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        
        depth_intr = depth_prof.get_intrinsics()
        color_intr = color_prof.get_intrinsics()
        
        d2c_extr = depth_prof.get_extrinsics_to(color_prof)
        c2d_extr = color_prof.get_extrinsics_to(depth_prof)
        
        print("Depth Intrinsics:", depth_intr.width, "x", depth_intr.height, "ppx:", depth_intr.ppx, "ppy:", depth_intr.ppy, "fx:", depth_intr.fx, "fy:", depth_intr.fy)
        print("Color Intrinsics:", color_intr.width, "x", color_intr.height, "ppx:", color_intr.ppx, "ppy:", color_intr.ppy, "fx:", color_intr.fx, "fy:", color_intr.fy)
        print("Depth to Color Extrinsics translation (meters):", d2c_extr.translation)
        print(f"  -> dx = {d2c_extr.translation[0]*1000:.2f} mm, dy = {d2c_extr.translation[1]*1000:.2f} mm, dz = {d2c_extr.translation[2]*1000:.2f} mm")
        print("Color to Depth Extrinsics translation (meters):", c2d_extr.translation)
        print(f"  -> dx = {c2d_extr.translation[0]*1000:.2f} mm, dy = {c2d_extr.translation[1]*1000:.2f} mm, dz = {c2d_extr.translation[2]*1000:.2f} mm")
        pipe.stop()
    except Exception as e:
        print("Error reading pipeline:", e)
