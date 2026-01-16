# run_hybrid_pipeline.ps1
# GeoSpec Hybrid Pipeline Orchestrator
#
# Flow:
#   1. Check RTCM capture health
#   2. Call WSL to process RTCM -> positions (RTKLIB)
#   3. Run adapter to convert to NGL format
#   4. Push results to git
#
# Usage:
#   .\run_hybrid_pipeline.ps1
#   .\run_hybrid_pipeline.ps1 -Date "2026-01-11"
#   .\run_hybrid_pipeline.ps1 -SkipGit

param(
    [string]$Date = (Get-Date -Format "yyyy-MM-dd"),
    [switch]$SkipGit = $false,
    [switch]$SkipWSL = $false
)

$ErrorActionPreference = "Continue"
$ProjectRoot = "C:\GeoSpec\geospec_sprint"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  GeoSpec Hybrid Pipeline" -ForegroundColor Cyan
Write-Host "  Date: $Date" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Activate Python venv
$VenvPath = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    . $VenvPath
}

# Step 1: RTCM Health Check
Write-Host "[1/4] Running RTCM health check..." -ForegroundColor Yellow
try {
    python -m monitoring.src.rtcm_health_check 2>&1 | Out-Host
    Write-Host "      Health check complete" -ForegroundColor Green
} catch {
    Write-Host "      WARNING: Health check failed: $_" -ForegroundColor Red
}

# Step 2: WSL RTCM Processing
if (-not $SkipWSL) {
    Write-Host ""
    Write-Host "[2/4] Processing RTCM in WSL (RTKLIB)..." -ForegroundColor Yellow

    # Check if RTKLIB is installed
    $rtkCheck = wsl -d Ubuntu -- which convbin 2>&1
    if ($rtkCheck -notlike "*convbin*") {
        Write-Host "      WARNING: RTKLIB not installed in WSL" -ForegroundColor Red
        Write-Host "      Run in WSL: bash /mnt/c/GeoSpec/geospec_sprint/wsl/setup_rtklib.sh" -ForegroundColor Yellow
    } else {
        # Download IGS products
        Write-Host "      Downloading IGS products..." -ForegroundColor Gray
        wsl -d Ubuntu -- bash "/mnt/c/GeoSpec/geospec_sprint/wsl/download_igs_products.sh" $Date 2>&1 | Out-Host

        # Process RTCM files
        Write-Host "      Converting RTCM to positions..." -ForegroundColor Gray
        wsl -d Ubuntu -- bash "/mnt/c/GeoSpec/geospec_sprint/wsl/process_rtcm.sh" $Date 2>&1 | Out-Host
        Write-Host "      WSL processing complete" -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "[2/4] Skipping WSL processing (--SkipWSL)" -ForegroundColor Gray
}

# Step 3: Position Adapter (convert to NGL format)
Write-Host ""
Write-Host "[3/4] Running position adapter..." -ForegroundColor Yellow
$adapterScript = Join-Path $ProjectRoot "monitoring\src\position_adapter.py"
if (Test-Path $adapterScript) {
    try {
        python $adapterScript --date $Date 2>&1 | Out-Host
        Write-Host "      Adapter complete" -ForegroundColor Green
    } catch {
        Write-Host "      WARNING: Adapter failed: $_" -ForegroundColor Red
    }
} else {
    Write-Host "      Position adapter not yet implemented" -ForegroundColor Gray
    Write-Host "      (Phase C: monitoring/src/position_adapter.py)" -ForegroundColor Gray
}

# Step 4: Git Push
if (-not $SkipGit) {
    Write-Host ""
    Write-Host "[4/4] Pushing results to git..." -ForegroundColor Yellow
    Set-Location $ProjectRoot

    # Check for changes
    $status = git status --porcelain 2>&1
    if ($status) {
        git add monitoring/data/ 2>&1 | Out-Null
        $commitMsg = "data: RTCM + positions for $Date"
        git commit -m $commitMsg 2>&1 | Out-Host
        git push 2>&1 | Out-Host
        Write-Host "      Git push complete" -ForegroundColor Green
    } else {
        Write-Host "      No changes to commit" -ForegroundColor Gray
    }
} else {
    Write-Host ""
    Write-Host "[4/4] Skipping git push (--SkipGit)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Pipeline Complete" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output locations:" -ForegroundColor White
Write-Host "  RTCM:      $ProjectRoot\monitoring\data\rtcm\{station}\$Date\" -ForegroundColor Gray
Write-Host "  Positions: $ProjectRoot\monitoring\data\positions\{station}\$Date\" -ForegroundColor Gray
Write-Host "  Health:    $ProjectRoot\monitoring\data\rtcm\health.json" -ForegroundColor Gray
