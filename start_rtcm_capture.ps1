# start_rtcm_capture.ps1
# Start RTCM capture for 3 pilot stations (COSO, GOLD, JPLM)
# Run this in PowerShell to start 24-hour capture

$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  GeoSpec RTCM Capture - Phase A Pilot" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting 24-hour RTCM capture for 3 SoCal pilot stations:"
Write-Host "  - COSO00USA0 (9 km from P580)"
Write-Host "  - GOLD00USA0 (22 km from P592)"
Write-Host "  - JPLM00USA0 (96 km from P579)"
Write-Host ""

# Activate venv
$VenvPath = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    . $VenvPath
}

# Create log directory
$LogDir = Join-Path $ProjectRoot "monitoring\logs\rtcm"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# Start captures as background jobs
$Stations = @("COSO00USA0", "GOLD00USA0", "JPLM00USA0")

foreach ($Station in $Stations) {
    $LogFile = Join-Path $LogDir "$Station.log"

    Write-Host "Starting $Station capture..." -ForegroundColor Yellow

    Start-Process -FilePath "python" `
        -ArgumentList "-m monitoring.src.capture_rtcm --mount $Station --hours 24" `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $LogFile `
        -RedirectStandardError "$LogFile.err" `
        -WindowStyle Hidden

    Write-Host "  Log: $LogFile" -ForegroundColor Green
}

Write-Host ""
Write-Host "All captures started!" -ForegroundColor Green
Write-Host ""
Write-Host "To check status:" -ForegroundColor Cyan
Write-Host "  Get-Process python | Select-Object Id, StartTime"
Write-Host ""
Write-Host "To view logs:" -ForegroundColor Cyan
Write-Host "  Get-Content $LogDir\COSO00USA0.log -Tail 20"
Write-Host ""
Write-Host "Output will be in:" -ForegroundColor Cyan
Write-Host "  $ProjectRoot\monitoring\data\rtcm\{station}\{date}\"
