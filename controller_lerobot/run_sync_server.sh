#!/bin/bash
# Install next to sync_franka.py and rollout_recording.py on the control PC.
set -e
ARGS=("$@")
set --
source /opt/ros/noetic/setup.bash
if [ -f /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash ]; then
    source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash
fi
cd "$(dirname "$(readlink -f "$0")")"
test -f sync_franka.py
test -f rollout_recording.py
test -f control_safety.py
test -f timestamped_feedback.py
echo "[Entry] $PWD/sync_franka.py (protocol v2)"
exec python3 -u sync_franka.py "${ARGS[@]}"
