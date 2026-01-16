@echo off
REM GeoSpec Daily Monitoring - Windows Batch Script
REM Runs Lambda_geo monitoring for all configured regions
REM Scheduled to run daily via Windows Task Scheduler

cd /d C:\GeoSpec\geospec_sprint

REM Activate virtual environment if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Set Python path
set PYTHONPATH=C:\GeoSpec\geospec_sprint

REM Run monitoring with live GPS data
echo ============================================
echo GeoSpec Daily Monitoring
echo Date: %date% %time%
echo ============================================

python -m monitoring.src.run_daily_live --date auto

REM Log completion
echo.
echo Monitoring complete: %date% %time%
echo Results saved to: C:\GeoSpec\geospec_sprint\monitoring\data
