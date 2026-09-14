@echo off
setlocal
title Mangosteen Edge AI - Setup

echo ===========================================================
echo    Mangosteen Edge AI - Automated Setup
echo ===========================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH!
    echo Please install Python 3.10+ and check 'Add python.exe to PATH'.
    pause
    exit /b 1
)

python "%~dp0setup.py" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Setup encountered an issue. Please review the messages above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
pause
