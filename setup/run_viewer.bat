@echo off
title Mangosteen Edge AI Camera Viewer
echo Starting Mangosteen Live Camera Viewer...
python "%~dp0view_camera.py" %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Program exited with error. Make sure Tera Term is closed and ESP32 is connected.
    pause
)
