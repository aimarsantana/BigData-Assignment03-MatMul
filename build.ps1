# Build the two variants of the C implementation with the exact flags used in the study (PowerShell, MinGW-w64 gcc).
#   build/matmul3_scalar.exe : no SIMD vectorisation (scalar SSE2 code)
#   build/matmul3_avx2.exe   : '#pragma omp simd' on the inner loops, AVX2 (4 x float64 per instruction), no FMA
# The vectoriser report of each build is written to build/vec_report_<variant>.txt.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force build | Out-Null
$common = "-O2 -std=c11 -Wall -Wextra -fopenmp"
$variants = [ordered]@{
  "scalar" = "-fno-tree-vectorize"
  "avx2"   = "-mavx2 -ffp-contract=off -DUSE_SIMD"
}
foreach ($v in $variants.Keys) {
  $extra = $variants[$v]
  $flags = "$common $extra"
  # the flag string is passed to the program through a generated header (avoids shell quoting issues)
  Set-Content -Encoding ascii "build/cflags_$v.h" "#define CFLAGS_STR `"$flags -lpsapi`""
  $gccArgs = @($common.Split(" ")) + @($extra.Split(" ")) + @("-include", "build/cflags_$v.h",
            "-fopt-info-vec-optimized=build/vec_report_$v.txt", "-o", "build/matmul3_$v.exe", "src/c/matmul3.c", "-lpsapi")
  & gcc @gccArgs
  if ($LASTEXITCODE -ne 0) { throw "gcc failed ($v)" }
  Write-Host "built build/matmul3_$v.exe with: gcc $flags -lpsapi"
}
