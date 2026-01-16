# GeoSpec Daily Monitoring - PowerShell Script
# Runs Lambda_geo monitoring for all configured regions
# Scheduled to run daily via Windows Task Scheduler

$ErrorActionPreference = "Continue"

# Configuration
$ProjectRoot = "C:\GeoSpec\geospec_sprint"
$LogDir = "$ProjectRoot\monitoring\logs"
$LogFile = "$LogDir\monitoring_$(Get-Date -Format 'yyyy-MM-dd').log"

# Create log directory if needed
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Start logging
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"============================================" | Tee-Object -FilePath $LogFile -Append
"GeoSpec Daily Monitoring" | Tee-Object -FilePath $LogFile -Append
"Started: $timestamp" | Tee-Object -FilePath $LogFile -Append
"============================================" | Tee-Object -FilePath $LogFile -Append

# Change to project directory
Set-Location $ProjectRoot

# Set Python path
$env:PYTHONPATH = $ProjectRoot

# Activate virtual environment if exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
}

# Run monitoring with live GPS data
try {
    python -m monitoring.src.run_daily_live --date auto 2>&1 | Tee-Object -FilePath $LogFile -Append
    $exitCode = $LASTEXITCODE
}
catch {
    "ERROR: $_" | Tee-Object -FilePath $LogFile -Append
    $exitCode = 1
}

# Log completion
$endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"" | Tee-Object -FilePath $LogFile -Append
"Monitoring complete: $endTime" | Tee-Object -FilePath $LogFile -Append
"Exit code: $exitCode" | Tee-Object -FilePath $LogFile -Append

# Check exit codes
if ($exitCode -eq 2) {
    "*** DATA ACQUISITION FAILED - IMMEDIATE ATTENTION REQUIRED ***" | Tee-Object -FilePath $LogFile -Append
    # Could add email/SMS notification here for data failures
}
elseif ($exitCode -eq 1) {
    "*** ELEVATED ALERT DETECTED ***" | Tee-Object -FilePath $LogFile -Append
    # Could add email notification here for elevated alerts
}

# Cleanup old logs (keep 30 days)
Get-ChildItem -Path $LogDir -Filter "monitoring_*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force

exit $exitCode
