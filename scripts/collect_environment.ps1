# Record the hardware / software environment of the measurements (writes results/environment.txt
# and results/topology.csv, the logical-processor order used by the 'pcores' placement policy).
param([string]$Out = (Join-Path $PSScriptRoot "..\results\environment.txt"))
$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
$cpu = Get-CimInstance Win32_Processor
$cs  = Get-CimInstance Win32_ComputerSystem
$os  = Get-CimInstance Win32_OperatingSystem
$mem = Get-CimInstance Win32_PhysicalMemory
$lines = @()
$lines += "date_utc: " + (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$lines += "cpu_model: " + $cpu.Name.Trim()
$lines += "physical_cores: " + $cpu.NumberOfCores
$lines += "logical_processors: " + $cpu.NumberOfLogicalProcessors
$topo = & (Join-Path $root "build\matmul3_scalar.exe") topology
$p = ($topo | Where-Object { $_ -match ',P$' }).Count
$e = ($topo | Where-Object { $_ -match ',E$' }).Count
$s = ($topo | Where-Object { $_ -match ',S$' }).Count
$lines += "core_types (GetLogicalProcessorInformationEx): $p performance cores with 2 SMT threads each, $e efficiency cores without SMT ($($p + $s + $e) logical processors)"
$lines += "cpu_max_clock_mhz (reported by Windows): " + $cpu.MaxClockSpeed
$lines += "l2_cache_kb (Win32_Processor): " + $cpu.L2CacheSize
$lines += "l3_cache_kb (Win32_Processor): " + $cpu.L3CacheSize
$lines += "ram_total_gb: " + [math]::Round($cs.TotalPhysicalMemory / 1GB, 2)
$lines += "ram_modules: " + (($mem | ForEach-Object { "$([math]::Round($_.Capacity/1GB))GB@$($_.ConfiguredClockSpeed)MT/s" }) -join ", ")
$lines += "os: " + $os.Caption + " " + $os.Version + " build " + $os.BuildNumber
$lines += "power_scheme: " + [regex]::Match((powercfg /getactivescheme | Out-String), '\(([^)]*)\)').Groups[1].Value
$lines += "machine_type: desktop (no battery)"
$lines += "gcc: " + ((& gcc --version)[0])
$ompver = (& (Join-Path $root "build\matmul3_scalar.exe") dense 2>$null | Select-String 'INFO,all,openmp,') -replace 'INFO,all,openmp,', ''
$lines += "openmp_runtime: libgomp (bundled with gcc), _OPENMP = $ompver"
$lines += "python (uv): " + ((& uv run --with numpy --with pandas --with matplotlib python -c "import sys,numpy,pandas,matplotlib;print(sys.version.split()[0],'numpy',numpy.__version__,'pandas',pandas.__version__,'matplotlib',matplotlib.__version__)"))
$lines += "build_flags scalar: gcc -O2 -std=c11 -Wall -Wextra -fopenmp -fno-tree-vectorize -lpsapi (see build.ps1)"
$lines += "build_flags avx2:   gcc -O2 -std=c11 -Wall -Wextra -fopenmp -mavx2 -ffp-contract=off -DUSE_SIMD -lpsapi"
$lines += "omp_environment: OMP_* variables removed by the harness; threads, schedule and dynamic=false set by the program"
$lines | Set-Content -Encoding utf8 $Out
$topo | Set-Content -Encoding utf8 (Join-Path (Split-Path $Out) "topology.csv")
$lines
