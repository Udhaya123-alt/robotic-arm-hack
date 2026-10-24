@echo off
cd /d "%~dp0"
py -3 ".\gesture_control\gesture_camera.py"
if errorlevel 1 pause
