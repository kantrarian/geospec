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

# Run monitoring with live GPS + Seismic data (Ensemble)
try {
    # Using run_ensemble_daily for full Multi-Method assessment
    # Default latency=2 allows for stable GPS/Seismic data arrival
    python -m monitoring.src.run_ensemble_daily --latency 2 2>&1 | Tee-Object -FilePath $LogFile -Append
    $exitCode = $LASTEXITCODE
}
catch {
    "ERROR: $_" | Tee-Object -FilePath $LogFile -Append
    $exitCode = 255 # Distinct error code for script failure
}

# Log completion
$endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"" | Tee-Object -FilePath $LogFile -Append
"Monitoring complete: $endTime" | Tee-Object -FilePath $LogFile -Append
"Exit code: $exitCode" | Tee-Object -FilePath $LogFile -Append

# Check exit codes (Mapped to Tiers)
if ($exitCode -eq 3) {
    "*** CRITICAL ALERT (TIER 3) DETECTED ***" | Tee-Object -FilePath $LogFile -Append
    # Trigger urgent notification logic here
}
elseif ($exitCode -eq 2) {
    "*** ELEVATED ALERT (TIER 2) DETECTED ***" | Tee-Object -FilePath $LogFile -Append
}
elseif ($exitCode -eq 1) {
    "*** WATCH ALERT (TIER 1) DETECTED ***" | Tee-Object -FilePath $LogFile -Append
}
elseif ($exitCode -eq 0) {
    "Status: Normal (Tier 0)" | Tee-Object -FilePath $LogFile -Append
}
else {
    "*** EXECUTION ERROR / DATA FAILURE DETECTED ***" | Tee-Object -FilePath $LogFile -Append
}

# Cleanup old logs (keep 30 days)
Get-ChildItem -Path $LogDir -Filter "monitoring_*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force

exit $exitCode
