@echo off
setlocal
cd /d "%~dp0\.."
set "PY=C:\Users\74727\miniconda3\envs\lerobot\python.exe"
set PYTHONPATH=%CD%;%CD%\lerobot\src;%PYTHONPATH%
%PY% franka_teleop\record_teleop.py %*
