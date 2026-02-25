$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

# Always run from repo root
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

Write-Host "[smoke] repo: $RepoRoot"
python --version
python -m pip --version

Write-Host "[smoke] editable install (dev only)"
python -m pip install -U pip
python -m pip install -e ".[dev]" --no-build-isolation
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# -----------------------------
# Detect matplotlib availability (non-fatal)
# -----------------------------
$HasMatplotlib = $false
$oldEA = $ErrorActionPreference
$ErrorActionPreference = "Continue"
python -c "import matplotlib" *> $null
$ErrorActionPreference = $oldEA

if ($LASTEXITCODE -eq 0) { $HasMatplotlib = $true }
Write-Host "[smoke] matplotlib available: $HasMatplotlib"

Write-Host "[smoke] doctor"
ganmg doctor
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$RUN_ID = "smoke"
$SEED  = "123"
$T     = "1000"
$TMIN  = "300"
$TMAX  = "1200"
$NT    = "10"

# Clean previous smoke run (idempotent)
$SmokeDir = Join-Path "runs" $RUN_ID
if (Test-Path $SmokeDir) {
  Write-Host "[smoke] removing existing $SmokeDir"
  Remove-Item -Recurse -Force $SmokeDir
}

Write-Host "[smoke] generate"
ganmg generate --run-id $RUN_ID --seed $SEED
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[smoke] analyze"
ganmg analyze --run-id $RUN_ID --T $T
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[smoke] sweep"
ganmg sweep --run-id $RUN_ID --T-min $TMIN --T-max $TMAX --nT $NT
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[smoke] assert CSV outputs"
if (!(Test-Path ("runs\{0}\inputs\results.csv" -f $RUN_ID))) { throw "Missing runs\$RUN_ID\inputs\results.csv" }
if (!(Test-Path ("runs\{0}\outputs\thermo_vs_T.csv" -f $RUN_ID))) { throw "Missing runs\$RUN_ID\outputs\thermo_vs_T.csv" }

Write-Host "[smoke] OK"