import socket
import struct
import json
import numpy as np

host = "10.197.16.43"
port = 8765

print(f"Connecting to Franka at {host}:{port}...")
try:
    sock = socket.create_connection((host, port), timeout=5.0)
    hdr = sock.recv(4)
    if not hdr or len(hdr) < 4:
        print("[ERROR] No header received from Franka server!")
    else:
        size = struct.unpack("!I", hdr)[0]
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                break
            data.extend(chunk)
        msg = json.loads(data.decode("utf-8"))
        print("\n" + "=" * 65)
        print("  FRANKA CURRENT LOWEST POINT ROBOT STATE READOUT")
        print("=" * 65)
        q = msg.get("q", [])
        print(f"[*] Joint Angles (q, 7-DOF in rad):")
        for i, val in enumerate(q):
            print(f"    Joint {i}: {val:+.5f} rad ({np.degrees(val):+6.2f} deg)")

        O_T_EE = msg.get("O_T_EE", [])
        if len(O_T_EE) == 16:
            # libfranka O_T_EE is 4x4 transform in column-major order:
            # [0, 4,  8, 12] -> X is O_T_EE[12]
            # [1, 5,  9, 13] -> Y is O_T_EE[13]
            # [2, 6, 10, 14] -> Z is O_T_EE[14]
            # [3, 7, 11, 15] -> [0, 0, 0, 1]
            mat_col = np.array(O_T_EE).reshape((4, 4), order="F")
            mat_row = np.array(O_T_EE).reshape((4, 4), order="C")
            
            print(f"\n[*] End-Effector Transform Matrix (libfranka col-major):")
            print(mat_col)
            print(f"\n[*] End-Effector Position in Base Frame (O_T_EE translation):")
            print(f"    X = {mat_col[0, 3]:+.5f} m ({mat_col[0, 3]*100:+.2f} cm)")
            print(f"    Y = {mat_col[1, 3]:+.5f} m ({mat_col[1, 3]*100:+.2f} cm)")
            print(f"    Z = {mat_col[2, 3]:+.5f} m ({mat_col[2, 3]*100:+.2f} cm)  <--- LOWEST Z LIMIT")
            
            # Rotation matrix analysis
            R = mat_col[:3, :3]
            print(f"\n[*] End-Effector Orientation (Rotation Matrix):")
            print(R)
            # Roll Pitch Yaw
            # Assuming ZYX convention
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            singular = sy < 1e-6
            if not singular:
                x_r = np.arctan2(R[2, 1], R[2, 2])
                y_p = np.arctan2(-R[2, 0], sy)
                z_y = np.arctan2(R[1, 0], R[0, 0])
            else:
                x_r = np.arctan2(-R[1, 2], R[1, 1])
                y_p = np.arctan2(-R[2, 0], sy)
                z_y = 0
            print(f"    Euler Angles (RPY, deg): Roll={np.degrees(x_r):+.2f} deg, Pitch={np.degrees(y_p):+.2f} deg, Yaw={np.degrees(z_y):+.2f} deg")

        width = msg.get("gripper_width_m")
        print(f"\n[*] Gripper Width: {width} m" if width is not None else "\n[*] Gripper Width: N/A")
        print("=" * 65)

        # Save to json file
        with open(r"C:\Users\74727\Desktop\project\VLA_franka\outputs\robot_lowest_point.json", "w", encoding="utf-8") as f:
            json.dump({
                "q": q,
                "O_T_EE": O_T_EE,
                "x_m": float(mat_col[0, 3]) if len(O_T_EE)==16 else None,
                "y_m": float(mat_col[1, 3]) if len(O_T_EE)==16 else None,
                "z_m": float(mat_col[2, 3]) if len(O_T_EE)==16 else None,
                "gripper_width_m": width,
            }, f, indent=2)
        print("[Saved] Recorded state to outputs/robot_lowest_point.json")

    sock.close()
except Exception as e:
    print(f"[FAIL] Could not connect to Franka: {e}")
    # Try via ssh to franka-control if TCP server is not running
