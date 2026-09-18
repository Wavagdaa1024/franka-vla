@echo off
setlocal

echo ===============================================================================
echo   MOVE FRANKA TO GOLDEN CAMERA CALIBRATION POSE
echo ===============================================================================
echo   Target Machine: franka-control (10.197.16.43)
echo   Target Robot:   Franka Emika (172.16.0.2)
echo   Target Joints:  [0.0284, 0.7187, 0.0210, -1.7251, -0.0087, 2.4741, 0.8302] rad
echo ===============================================================================
echo   Sending command to franka-control over SSH...
echo.

ssh franka-control "bash /home/ssui/franka_ros_ws/move_to_camera_calib.sh"

if errorlevel 1 (
    echo.
    echo [ERROR] Motion execution encountered an error.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Franka has moved to the camera calibration position!
pause
