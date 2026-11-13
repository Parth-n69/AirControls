@echo off
title AirControl - Touchless Hand Gesture Desktop Control
cd /d "%~dp0"
echo ========================================================
echo Starting AirControl...
echo If you do not have a webcam, the Default Camera will start.
echo ========================================================
python -m aircontrol.app
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%.
    pause
)

