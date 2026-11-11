# Reproduce the whole study (Windows PowerShell). Needs: gcc with OpenMP (MinGW-w64), uv (or Python 3.11+ with numpy pandas matplotlib).
# Inputs (~360 MB) are written OUTSIDE the project folder by default so that cloud-sync tools do not touch them while timing.
param(
  [string]$DataDir = "C:\bench_a3\data",
  [string]$Python = "uv run --with numpy --with pandas --with matplotlib python"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = $Python.Split(" ")
function Py { & $py[0] $py[1..($py.Length-1)] @args; if ($LASTEXITCODE -ne 0) { throw "python step failed" } }

.\build.ps1
.\scripts\collect_environment.ps1
Py scripts/check_vectorisation.py
Py scripts/generate_inputs.py --outdir $DataDir `
   --check "..\..\Assignment01_AlejandroSantana_Materials\data\inputs\manifest.json" `
   --check "..\..\assigment-2\Assignment02_AlejandroSantana_Materials\data\inputs\manifest.json"
Py tests/validate.py --data-dir $DataDir > results\validation.csv
if (Test-Path results\raw_samples.csv) { throw "results\ already contains measurements: move or delete it before a full re-run" }
Py scripts/run_benchmarks.py --data-dir $DataDir        # P1_threads V1_simd P2_schedule P3_affinity P4_ecores
Py scripts/analyze.py
Py report/build_report.py
