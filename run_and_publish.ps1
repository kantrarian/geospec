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

# 0. Self-heal FIRST: pull latest source BEFORE any generation, so this run uses current remote
#    code. Fix 2026-07-30 (cayley, per grassmann diagnosis e313db2): the generators (r4 scorer at
#    step 3, etc.) previously ran on whatever src was on disk, while the ONLY pull was the step-5
#    push block AFTER generation -- so code landing on the remote between a runner's last manual
#    pull and the run (e.g. R4 v3, 98b44f7) lagged one full run (v3 source pulled at step 5, but
#    the step-3 record already generated on v2 -> published commit had v3 source + v2 record).
#    Pulling here closes that same-run code-freshness gap. The step-5 pull is KEPT for
#    post-generation push-safety. A transient failure here is non-fatal (native git exit does not
#    trip Stop) -> degrades to "run on current src", which self-heals next run, rather than aborting.
Write-Host "[0/5] Self-heal: pull latest source before run..." -ForegroundColor Yellow
Push-Location $RepoRoot
try {
    # --autostash: the runner's tree is chronically dirty (tracked monitoring/dashboard/data.csv
    # is written but not committed until step 5); a plain pull --rebase aborts (exit 128) on
    # unstaged changes. Mirrors the step-5 pull invocation.
    git pull --rebase --autostash origin master
} finally {
    Pop-Location
}
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

# 2a. Rolling baseline recalibration (amendment R3, registered 2026-07-29 -- see
# docs/AMENDMENT_2026-07-29_rolling_baseline.md). The lambda_geo baseline was previously a
# manual one-shot and went stale (static winter baseline -> full seasonal drift carried into
# the live ratio). Now: recalibrate weekly from a 90-day window LAGGED 30 days (end = today-30d)
# so a genuine precursory buildup cannot be absorbed into its own baseline.
$BaselineFile = Join-Path $MonitoringDir "data\baselines\lambda_geo_baselines.json"
$NeedsRecal = $true
if (Test-Path $BaselineFile) {
    $AgeDays = ((Get-Date) - (Get-Item $BaselineFile).LastWriteTime).TotalDays
    $NeedsRecal = $AgeDays -gt 7
}
if ($NeedsRecal) {
    $LagEnd = (Get-Date).AddDays(-30).ToString('yyyy-MM-dd')
    Write-Host "  [2a] Recalibrating lambda_geo baselines (90d window ending $LagEnd, R3)..." -ForegroundColor Yellow
    python -m src.calibrate_lambda_geo_baselines --days 90 --end-date $LagEnd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Recalibration failed; continuing with existing baselines" -ForegroundColor Red
    }
} else {
    Write-Host "  [2a] Baselines fresh (<7d), skipping recalibration" -ForegroundColor Green
}

# [2b] seismic_thd baseline rolling recal (INCIDENT 2026-07-31, D1): extend R3 rolling-recalibration to the
# THD component baselines. The THD baselines were a frozen 2026-01 one-shot (IU.COLA z=26-on-noise artifact);
# now recalibrated weekly on the same R3 terms (90d window ending today-30d) to a dated file that
# station_baselines.py loads newest-first. Non-fatal: a recal failure must never stop the daily publish (the
# ensemble staleness guard keeps any stale baseline fail-safe on absolute thresholds meanwhile).
$ThdBaselineDir = Join-Path $MonitoringDir "data\baselines"
$ThdNewest = Get-ChildItem -Path $ThdBaselineDir -Filter "thd_baselines_*.json" -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
$ThdNeedsRecal = $true
if ($ThdNewest) {
    $ThdAge = ((Get-Date) - $ThdNewest.LastWriteTime).TotalDays
    $ThdNeedsRecal = $ThdAge -gt 7
}
if ($ThdNeedsRecal) {
    Write-Host "  [2b] Recalibrating seismic_thd baselines (90d window ending today-30d, R3, incident 2026-07-31)..." -ForegroundColor Yellow
    python -m src.run_thd_recal --force
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  THD recalibration failed; continuing (staleness guard keeps stale baselines fail-safe)" -ForegroundColor Red
    }
} else {
    Write-Host "  [2b] THD baselines fresh (<7d), skipping recalibration" -ForegroundColor Green
}

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

# 2b. Amendment R4 prospective scorer (registered 2026-07-29; fail-open -- an R4
# scoring failure must never stop the daily monitoring publish)
Write-Host "  [2b] R4 prospective scorer..." -ForegroundColor Yellow
Push-Location $MonitoringDir
try {
    python -m src.r4_prospective_scorer
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  R4 scorer failed (exit $LASTEXITCODE); continuing without R4 update" -ForegroundColor Red
    }
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

# 3. Public record surfaces (asylum 2026-09-02, immutable revision model)
# The runner publishes each scored day CREATE-ONCE under
# docs/ensemble/<date>/<run_id>.json and DERIVES docs/ensemble_latest.json,
# docs/ensemble/current.json and docs/data.csv from the append-only index
# (monitoring/src/ensemble_revisions_cayley.py). This step no longer copies
# the newest LOCAL ensemble_*.json over the derived surface -- a stale local
# file with a newer mtime used to be able to overwrite it silently.
Write-Host "[3/5] Reading the published record surfaces in docs/..." -ForegroundColor Yellow
$DocsDir = Join-Path $RepoRoot "docs"
$LatestFile = Get-Item (Join-Path $DocsDir "ensemble_latest.json") -ErrorAction SilentlyContinue

if ($LatestFile) {
    # Extract date and summary for README (from the DERIVED surface)
    $EnsembleData = Get-Content $LatestFile.FullName | ConvertFrom-Json
    $AssessmentDate = $EnsembleData.date
    $MaxRisk = $EnsembleData.summary.max_risk
    $MaxRegion = $EnsembleData.summary.max_risk_region
    if ($EnsembleData.revision) {
        Write-Host "  Published revision: $($EnsembleData.revision.date)/$($EnsembleData.revision.run_id)" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: docs/ensemble_latest.json carries no revision block (pre-model record)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  No published ensemble record found at docs/ensemble_latest.json" -ForegroundColor Red
    exit 1
}

# Copy dashboard CSV (the authoritative source with complete historical data)
# IMPORTANT: Use monitoring/dashboard/data.csv NOT ensemble_results/daily_states.csv
# The dashboard CSV has the correct schema and complete 30-day history
$CsvFile = Join-Path $RepoRoot "monitoring\dashboard\data.csv"
$DocsCSV = Join-Path $DocsDir "data.csv"
if (Test-Path $CsvFile) {
    $SrcRows = (Get-Content $CsvFile | Measure-Object -Line).Lines
    if (Test-Path $DocsCSV) {
        $DstRows = (Get-Content $DocsCSV | Measure-Object -Line).Lines
    } else {
        $DstRows = 0
    }
    if ($SrcRows -lt $DstRows) {
        Write-Host "  ABORT: monitoring/dashboard/data.csv ($SrcRows rows) is smaller than docs/data.csv ($DstRows rows) -- would destroy history. Fix the source before re-running." -ForegroundColor Red
        exit 11
    }
    Copy-Item $CsvFile $DocsCSV -Force
    Write-Host "  Copied: monitoring/dashboard/data.csv -> docs/data.csv ($SrcRows rows)" -ForegroundColor Green
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

# 3b. Regenerate the public atlas page (presentation only).
# FAIL-SOFT (codex atlas fix 1): under $ErrorActionPreference='Stop',
# a native command writing stderr becomes a terminating
# NativeCommandError (the same PS 5.1 behavior the git hardening in
# step 5 documents), so the generator MUST run under 'Continue' with
# its exit code captured immediately and EAP restored in finally.
# $AtlasOk gates staging; a failure logs, skips, and the previous
# atlas.html stands (it displays its own data date honestly).
Write-Host "[3b/5] Regenerating docs/atlas.html..." -ForegroundColor Yellow
$AtlasOk = $false
$AtlasExit = -1
$PrevEAPAtlas = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    cmd /c exit 1 | Out-Null   # pin LASTEXITCODE so a launch failure can never inherit a stale 0
    python (Join-Path $RepoRoot "monitoring\generate_atlas.py") 2>&1 | ForEach-Object { "$_" } | Write-Host
    $AtlasExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PrevEAPAtlas
}
if ($AtlasExit -eq 0) {
    Write-Host "  Atlas regenerated" -ForegroundColor Green
    $AtlasOk = $true
} else {
    Write-Host "  WARNING: atlas generation failed (exit $AtlasExit); keeping previous atlas.html unstaged" -ForegroundColor Yellow
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
    # HARDENING (grassmann 2026-08-10, root cause of the 2026-08-08..10 publish stall):
    # the script runs under $ErrorActionPreference='Stop'. In PS 5.1 a NATIVE command that
    # writes to stderr AND exits non-zero is escalated to a terminating NativeCommandError
    # that aborts the whole run. The old `git add <list> 2>$null` hit exactly this the day
    # `monitoring/receipts` was added to the list (4025794) while that dir did not yet exist:
    # `fatal: pathspec 'monitoring/receipts' did not match any files` (exit 128) killed the
    # run BEFORE `git commit` -- and `2>$null` does NOT prevent it (it hides the text, not the
    # error record). So the daily data generated but never published. Two guards below:
    #   (1) stage only paths that actually exist -> no missing-pathspec fatal, ever;
    #   (2) drop to 'Continue' around the native git calls so a non-zero git exit degrades
    #       gracefully (logged) instead of aborting mid-publish. Explicit exit-code checks
    #       keep push-safety. Restored in finally.
    # W4: monitoring/dashboard/data.csv (the authoritative source) is committed daily to keep
    # the public copy fresh AND the tree clean. monitoring/receipts holds the R6 §1
    # server-stamped publication receipts (P2 item 1), written POST-PUSH (step 5b) so each
    # lands in git on the NEXT run; the .gitkeep makes the dir exist so it always stages.
    $PrevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'

    $Candidates = @(
        'docs/ensemble', 'docs/ensemble_latest.json', 'docs/data.csv', 'docs/validated_events.json',
        'docs/r4_prospective_record.json', 'docs/r5_daily.json',
        'monitoring/dashboard/data.csv', 'monitoring/receipts', 'README.md'
    )
    # codex atlas fix 1: atlas stages ONLY on a verified exit-0
    # generation this run -- never on mere file existence
    if ($AtlasOk) { $Candidates += 'docs/atlas.html' }
    $ToAdd = @($Candidates | Where-Object { Test-Path (Join-Path $RepoRoot $_) })
    if ($ToAdd.Count -gt 0) { git add -- $ToAdd 2>&1 | Write-Host }

    git diff --cached --quiet
    $HasChanges = $LASTEXITCODE -ne 0

    if ($HasChanges) {
        git commit -m "Daily monitoring $AssessmentDate" 2>&1 | Write-Host
        # Sync before pushing: remote may have advanced (e.g. registered amendments).
        # Rebase keeps the daily data commit on top; without this, the push is rejected
        # and the daily update silently fails. (Added 2026-07-29 with amendment R2.)
        # --autostash: the runner's tree is chronically dirty (tracked
        # monitoring/dashboard/data.csv is written but never committed by this flow), and a
        # plain pull --rebase ABORTS on unstaged changes (exit 128) without stopping the
        # script -- making the sync silently inert (grassmann, empirical, 2026-07-29).
        git pull --rebase --autostash origin master 2>&1 | Write-Host
        git push origin master 2>&1 | Write-Host
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Pushed to GitHub" -ForegroundColor Green
        } else {
            Write-Host "  Push failed (exit $LASTEXITCODE); the commit is local and will push on the next run's rebase" -ForegroundColor Red
        }
    } else {
        Write-Host "  No changes to commit" -ForegroundColor Yellow
    }
} finally {
    $ErrorActionPreference = $PrevEAP
    Pop-Location
}

# 5b. R6 §1 server-stamped publication receipt (P2 item 1). POST-PUSH: query the GitHub Pages build record and,
# if it is a BUILT build for a daily-monitoring commit, write monitoring/receipts/<day>.json (committed next run).
# Fail-open -- a receipt failure must NEVER break the daily publish; a receipt-less day degrades conservatively
# (23:59:59Z ceiling, hit-ineligible) and self-heals when Pages has genuinely deployed that commit. Never backfills.
Write-Host "[5b] Publication receipt (R6 §1 server-stamped hit-clock)..." -ForegroundColor Yellow
Push-Location $RepoRoot
try {
    python src\build_daily_receipt.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Receipt step exited $LASTEXITCODE; continuing (fail-open)" -ForegroundColor Red
    }
} catch {
    Write-Host "  Receipt step error (fail-open): $_" -ForegroundColor Red
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  Complete! Dashboard: https://kantrarian.github.io/geospec/" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
