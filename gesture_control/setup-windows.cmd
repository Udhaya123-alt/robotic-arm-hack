@echo off
setlocal
cd /d "%~dp0"

py -3 -m pip install -r requirements-windows.txt
if errorlevel 1 goto :error

if not exist models mkdir models
if not exist models\hand_landmarker.task (
  powershell.exe -NoProfile -Command "Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile 'models\hand_landmarker.task'"
  if errorlevel 1 goto :error
)

echo.
echo Setup complete. Run ..\start-gesture-preview.cmd
exit /b 0

:error
echo.
echo Setup failed. Review the error above.
pause
exit /b 1
