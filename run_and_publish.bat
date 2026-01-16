@echo off
REM run_and_publish.bat - Double-click to run monitoring and publish
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "run_and_publish.ps1"
pause
