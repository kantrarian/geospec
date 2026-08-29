# Lock test for codex atlas fix 1: the 3b atlas step is fail-soft
# under $ErrorActionPreference='Stop' (the production condition).
# Extracts the REAL step block from run_and_publish.ps1 -- never a
# re-typed copy that could drift -- and executes it in a sandbox
# with a fake generator. Run: powershell -File monitoring\test_atlas_failsoft.ps1
$ErrorActionPreference = 'Stop'
$Fails = @()
function Check([string]$Name, [bool]$Ok, [string]$Detail = "") {
    if ($Ok) { Write-Host "  [PASS] $Name" }
    else { Write-Host "  [FAIL] $Name -- $Detail"; $script:Fails += $Name }
}

$ScriptPath = Join-Path (Split-Path $PSScriptRoot -Parent) "run_and_publish.ps1"
$Src = Get-Content $ScriptPath -Raw

# ---- extract the real 3b block ----
$StartMark = "# 3b. Regenerate the public atlas page"
$EndMark = "# 4. Update README"
$iStart = $Src.IndexOf($StartMark)
$iEnd = $Src.IndexOf($EndMark)
Check "X0 the 3b block exists before step 4" ($iStart -ge 0 -and $iEnd -gt $iStart)
$Block = $Src.Substring($iStart, $iEnd - $iStart)

function Invoke-AtlasStep([string]$FakeGenBody) {
    $RepoRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("atlas_fs_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
    New-Item -ItemType Directory -Path (Join-Path $RepoRoot "monitoring") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $RepoRoot "docs") -Force | Out-Null
    Set-Content -Path (Join-Path $RepoRoot "monitoring\generate_atlas.py") -Value $FakeGenBody -Encoding utf8
    Set-Content -Path (Join-Path $RepoRoot "docs\atlas.html") -Value "PRIOR-ATLAS-BYTES" -Encoding utf8
    $PriorHash = (Get-FileHash (Join-Path $RepoRoot "docs\atlas.html") -Algorithm SHA256).Hash
    $AtlasOk = $null
    $Sentinel = $false
    # production condition: EAP=Stop around the block, exactly as in
    # run_and_publish.ps1
    $ErrorActionPreference = 'Stop'
    Invoke-Expression $Block
    $Sentinel = $true
    $AfterHash = (Get-FileHash (Join-Path $RepoRoot "docs\atlas.html") -Algorithm SHA256).Hash
    $EAPAfter = $ErrorActionPreference
    Remove-Item -Recurse -Force $RepoRoot
    return @{ Sentinel = $Sentinel; AtlasOk = $AtlasOk;
              Unchanged = ($PriorHash -eq $AfterHash); EAP = $EAPAfter }
}

# ---- failing generator: stderr + exit 7 (the NativeCommandError
# shape that terminated the original revision) ----
$FailGen = "import sys`nsys.stderr.write('ATLAS_VALIDATE_REFUSED: injected\n')`nsys.exit(7)"
$r = Invoke-AtlasStep $FailGen
Check "X1 stderr + exit 7 does NOT terminate the publisher (sentinel reached)" $r.Sentinel
Check "X2 a failed generation yields AtlasOk=false" ($r.AtlasOk -eq $false)
Check "X3 the prior atlas bytes are unchanged" $r.Unchanged
Check "X4 ErrorActionPreference is restored to Stop" ($r.EAP -eq 'Stop')

# ---- exit-0 control ----
$OkGen = "import sys`nprint('atlas: ok')`nsys.exit(0)"
$r0 = Invoke-AtlasStep $OkGen
Check "X5 an exit-0 generation yields AtlasOk=true" ($r0.AtlasOk -eq $true)

# ---- staging gate: atlas only enters candidates on AtlasOk ----
Check "X6 the static candidates list does NOT carry docs/atlas.html" (-not ($Src -match "'docs/atlas\.html',"))
Check "X7 the AtlasOk-gated append exists" ($Src -match "if \(\`$AtlasOk\) \{ \`$Candidates \+= 'docs/atlas\.html' \}")
$Candidates = @('docs/x.json')
$AtlasOk = $false
if ($AtlasOk) { $Candidates += 'docs/atlas.html' }
$NotStaged = -not ($Candidates -contains 'docs/atlas.html')
$AtlasOk = $true
if ($AtlasOk) { $Candidates += 'docs/atlas.html' }
Check "X8 the gate stages atlas ONLY when AtlasOk" ($NotStaged -and ($Candidates -contains 'docs/atlas.html'))

Write-Host ""
if ($Fails.Count -gt 0) {
    Write-Host "ATLAS FAIL-SOFT LOCK-TEST FAILURES ($($Fails.Count)): $($Fails -join ', ')"
    exit 1
}
Write-Host "ATLAS FAIL-SOFT LOCK TESTS: ALL PASS (sandboxed; the real repo untouched)"
exit 0
