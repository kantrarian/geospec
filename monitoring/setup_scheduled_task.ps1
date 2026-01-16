# GeoSpec Daily Monitoring - Scheduled Task Setup
# Run this script as Administrator to create the Windows scheduled task

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

Write-Host "Setting up GeoSpec Daily Monitoring scheduled task..." -ForegroundColor Cyan

# Task parameters
$TaskName = "GeoSpec\DailyMonitoring"
$TaskDescription = "Runs GeoSpec ensemble monitoring and publishes results to GitHub dashboard"
$ScriptPath = "C:\GeoSpec\geospec_sprint\run_and_publish.ps1"
$RunTime = "06:00"

# Check if script exists
if (-not (Test-Path $ScriptPath)) {
    Write-Error "Script not found: $ScriptPath"
    exit 1
}

# Create the action
$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPath`""

# Create the trigger (daily at 6 AM)
$Trigger = New-ScheduledTaskTrigger -Daily -At $RunTime

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -WakeToRun

# Create the principal (run as current user)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest

# Remove existing task if present
if (Get-ScheduledTask -TaskName "DailyMonitoring" -TaskPath "\GeoSpec\" -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName "DailyMonitoring" -TaskPath "\GeoSpec\" -Confirm:$false
}

# Register the task
try {
    Register-ScheduledTask -TaskName "DailyMonitoring" -TaskPath "\GeoSpec\" `
        -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal `
        -Description $TaskDescription

    Write-Host "`nScheduled task created successfully!" -ForegroundColor Green
    Write-Host "Task: $TaskName"
    Write-Host "Schedule: Daily at $RunTime"
    Write-Host "Script: $ScriptPath"
    Write-Host "`nTo view the task: Task Scheduler > GeoSpec > DailyMonitoring"
    Write-Host "To run manually: schtasks /Run /TN `"GeoSpec\DailyMonitoring`""
}
catch {
    Write-Error "Failed to create scheduled task: $_"
    exit 1
}

# Verify task was created
$Task = Get-ScheduledTask -TaskName "DailyMonitoring" -TaskPath "\GeoSpec\" -ErrorAction SilentlyContinue
if ($Task) {
    Write-Host "`nTask verification:" -ForegroundColor Cyan
    Write-Host "  State: $($Task.State)"
    Write-Host "  Next Run: $(($Task | Get-ScheduledTaskInfo).NextRunTime)"
}
