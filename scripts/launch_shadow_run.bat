@echo off
setlocal
cd /d "%~dp0\.."
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=1
set "PY=C:\Users\74727\miniconda3\envs\lerobot\python.exe"
set PYTHONPATH=%CD%;%CD%\lerobot\src;%PYTHONPATH%
%PY% tests\shadow_run_pi05.py %*
