#!/usr/bin/env pwsh
# run_and_publish.ps1 - Run ensemble monitoring and publish results to GitHub
# Keeps source code private, only pushes data to public dashboard

param(
    [string]$Date = "auto"
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  GeoSpec Ensemble Monitoring - Run & Publish" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# 1. Activate virtual environment
Write-Host "[1/5] Activating Python environment..." -ForegroundColor Yellow
$VenvPath = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    . $VenvPath
} else {
    Write-Host "  Virtual environment not found at $VenvPath" -ForegroundColor Red
    Write-Host "  Run: py -3.11 -m venv .venv && .venv\Scripts\pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

# 2. Run ensemble monitoring
Write-Host "[2/5] Running ensemble monitoring..." -ForegroundColor Yellow
$MonitoringDir = Join-Path $RepoRoot "monitoring"
Push-Location $MonitoringDir

try {
    if ($Date -eq "auto") {
        python -m src.run_ensemble_daily
    } else {
        python -m src.run_ensemble_daily --date $Date
    }
    $MonitoringExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

# Exit codes: 0=normal, 1=preliminary, 2=elevated, 3=confirmed (all valid)
# Only fail on actual errors (exit code > 10)
if ($MonitoringExitCode -gt 10) {
    Write-Host "  Monitoring failed with exit code $MonitoringExitCode" -ForegroundColor Red
    exit $MonitoringExitCode
} elseif ($MonitoringExitCode -eq 3) {
    Write-Host "  CONFIRMED alerts detected (exit code 3)" -ForegroundColor Magenta
} elseif ($MonitoringExitCode -eq 2) {
    Write-Host "  Elevated signals detected (exit code 2)" -ForegroundColor Yellow
} else {
    Write-Host "  Monitoring completed (exit code $MonitoringExitCode)" -ForegroundColor Green
}

# 3. Copy results to docs/
Write-Host "[3/5] Copying results to docs/..." -ForegroundColor Yellow
$EnsembleDir = Join-Path $RepoRoot "monitoring\data\ensemble_results"
$DocsDir = Join-Path $RepoRoot "docs"

# Find latest ensemble file
$LatestFile = Get-ChildItem -Path $EnsembleDir -Filter "ensemble_*.json" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($LatestFile) {
    Copy-Item $LatestFile.FullName (Join-Path $DocsDir "ensemble_latest.json") -Force
    Write-Host "  Copied: $($LatestFile.Name) -> docs/ensemble_latest.json" -ForegroundColor Green

    # Extract date and summary for README
    $EnsembleData = Get-Content $LatestFile.FullName | ConvertFrom-Json
    $AssessmentDate = $EnsembleData.date
    $MaxRisk = $EnsembleData.summary.max_risk
    $MaxRegion = $EnsembleData.summary.max_risk_region
} else {
    Write-Host "  No ensemble results found" -ForegroundColor Red
    exit 1
}

# Copy dashboard CSV (the authoritative source with complete historical data)
# IMPORTANT: Use monitoring/dashboard/data.csv NOT ensemble_results/daily_states.csv
# The dashboard CSV has the correct schema and complete 30-day history
$CsvFile = Join-Path $RepoRoot "monitoring\dashboard\data.csv"
if (Test-Path $CsvFile) {
    Copy-Item $CsvFile (Join-Path $DocsDir "data.csv") -Force
    Write-Host "  Copied: monitoring/dashboard/data.csv -> docs/data.csv" -ForegroundColor Green
} else {
    Write-Host "  WARNING: Dashboard data.csv not found at $CsvFile" -ForegroundColor Red
}

# Copy validated events (track record) for public dashboard
$ValidatedFile = Join-Path $RepoRoot "monitoring\data\validated_events.json"
if (Test-Path $ValidatedFile) {
    Copy-Item $ValidatedFile (Join-Path $DocsDir "validated_events.json") -Force
    Write-Host "  Copied: monitoring/data/validated_events.json -> docs/validated_events.json" -ForegroundColor Green
} else {
    Write-Host "  WARNING: Validated events not found at $ValidatedFile (track record not available)" -ForegroundColor Yellow
}

# 4. Update README
Write-Host "[4/5] Updating README..." -ForegroundColor Yellow
$TierCounts = $EnsembleData.summary.tier_counts
$ReadmeContent = @"
# GeoSpec Ensemble Monitoring

**Last Update**: $AssessmentDate

## Current Status

| Metric | Value |
|--------|-------|
| Highest Risk Region | $MaxRegion |
| Risk Score | $([math]::Round($MaxRisk, 3)) |
| Regions Monitored | $($EnsembleData.summary.total_regions) |

### Tier Distribution

| Tier | Count |
|------|-------|
| NORMAL (0) | $($TierCounts.'0') |
| WATCH (1) | $($TierCounts.'1') |
| ELEVATED (2) | $($TierCounts.'2') |
| CRITICAL (3) | $($TierCounts.'3') |

## Dashboard

View the full dashboard: [GeoSpec Dashboard](https://kantrarian.github.io/geospec/)

---

*Research system - not for emergency use*
"@

$ReadmeContent | Out-File (Join-Path $RepoRoot "README.md") -Encoding UTF8
Write-Host "  README.md updated" -ForegroundColor Green

# 5. Commit and push
Write-Host "[5/5] Committing and pushing to GitHub..." -ForegroundColor Yellow
Push-Location $RepoRoot

try {
    git add docs/ensemble_latest.json docs/data.csv docs/validated_events.json README.md 2>$null

    $HasChanges = git diff --cached --quiet; $HasChanges = $LASTEXITCODE -ne 0

    if ($HasChanges) {
        git commit -m "Daily monitoring $AssessmentDate"
        git push origin master
        Write-Host "  Pushed to GitHub" -ForegroundColor Green
    } else {
        Write-Host "  No changes to commit" -ForegroundColor Yellow
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  Complete! Dashboard: https://kantrarian.github.io/geospec/" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
