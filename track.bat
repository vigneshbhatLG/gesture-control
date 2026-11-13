@echo off
rem Run the MediaPipe tracker with the project venv (double-click or type: track)
"%~dp0.venv\Scripts\python.exe" "%~dp0track.py"
if errorlevel 1 pause
