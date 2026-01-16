# deploy_dashboard.ps1
# Deploys ensemble results to GitHub Pages dashboard
#
# Usage: .\deploy_dashboard.ps1
#
# Dashboard is served from docs/ folder via GitHub Pages:
# https://kantrarian.github.io/geospec/

$ErrorActionPreference = "Stop"

Write-Host "=== GeoSpec Dashboard Deployment ===" -ForegroundColor Cyan

# 1. Copy latest results to docs/ folder (GitHub Pages source)
Write-Host "`n1. Copying results to docs/ folder..."
Copy-Item "monitoring\dashboard\ensemble_latest.json" "docs\ensemble_latest.json" -Force
Write-Host "   - docs/ensemble_latest.json updated"

# Copy dated results if they exist
$today = (Get-Date).AddDays(-2).ToString("yyyy-MM-dd")  # Account for data latency
$datedFile = "monitoring\data\ensemble_results\ensemble_$today.json"
if (Test-Path $datedFile) {
    Copy-Item $datedFile "docs\ensemble_$today.json" -Force
    Write-Host "   - docs/ensemble_$today.json updated"
}

# 2. Stage, commit, and push
Write-Host "`n2. Committing to git..."
git add docs/ensemble_latest.json docs/ensemble_*.json
git commit -m "Dashboard update $(Get-Date -Format 'yyyy-MM-dd HH:mm')"

Write-Host "`n3. Pushing to GitHub..."
git push

Write-Host "`n=== Deployment Complete ===" -ForegroundColor Green
Write-Host "Dashboard URL: https://kantrarian.github.io/geospec/"
