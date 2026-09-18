@echo off
setlocal
cd /d "%~dp0.."

set "PYTHONPATH=%~dp0..\src;%PYTHONPATH%"
set "PY=C:\Users\74727\miniconda3\envs\lerobot\python.exe"

%PY% src\franka_teleop\record_teleop.py %*
