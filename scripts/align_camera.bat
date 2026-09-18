@echo off
setlocal
cd /d "%~dp0.."

set "PYTHONPATH=%~dp0..\src;%PYTHONPATH%"
call tests\camera_alignment\launch_align_camera.bat %*
exit /b %errorlevel%
