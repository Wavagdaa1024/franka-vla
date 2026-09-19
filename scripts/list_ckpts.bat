@echo off
setlocal
cd /d "%~dp0.."

set "PYTHONPATH=%~dp0..\src;%PYTHONPATH%"
set "PY=C:\Users\74727\miniconda3\envs\lerobot\python.exe"

if not exist "%PY%" (
    echo [ERROR] Canonical lerobot Python was not found: %PY%
    exit /b 2
)

"%PY%" scripts\python\list_ckpts.py %*
exit /b %errorlevel%
