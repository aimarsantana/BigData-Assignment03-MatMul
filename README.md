# Assignment 3 — Parallel Matrix Multiplication and (optional) Vectorisation (C, OpenMP, AVX2)

**Big Data · Grado en Ciencia e Ingeniería de Datos · ULPGC · 2026-2027**
Author: Aimar Daniel Alejandro Santana

Research question:

* *How does the number of OpenMP threads affect the runtime, speed-up, efficiency and memory of a
  shared-memory dense `float64` multiplication on a hybrid 8 P-core + 8 E-core CPU, for which matrix
  sizes does parallel execution help, and how much does SIMD vectorisation (AVX2) add on top of it?*

The report is `../Assignment03_AlejandroSantana_Report.pdf` (source: `report/build_report.py`).

## What is in this folder

```
src/c/matmul3.c                all kernels (C, float64), one source file, two builds:
                                 ijk  basic triple loop of Assignment 1 (context only)
                                 seq  sequential baseline = 'unroll4' kernel of Assignment 2 (ikj, k unrolled by 4)
                                 par  the same row kernel, rows of C distributed over OpenMP threads
build.ps1                      builds build/matmul3_scalar.exe (no SIMD) and build/matmul3_avx2.exe (AVX2)
scripts/make_config.py         writes config/benchmark_config.json (sizes, thread counts, experiments, budgets)
scripts/generate_inputs.py     shared inputs (same seeds and SHA-256 as Assignments 1 and 2)
scripts/run_benchmarks.py      benchmark harness (one fresh process per configuration, budgets, stop rule)
scripts/check_vectorisation.py vectoriser reports + objdump instruction counts -> results/disasm_summary.csv
scripts/analyze.py             statistics, speed-up / efficiency, figures/*.png|pdf, report/generated/*.html
scripts/collect_environment.ps1  hardware / software description -> results/environment.txt, topology.csv
tests/validate.py              correctness tests (known answers, identity, zero, odd sizes, more threads
                               than rows, NumPy reference, bit-exactness, repeated runs for races)
report/build_report.py         assembles the English report (HTML) and prints it to PDF with Chrome/Edge
results/                       raw_samples.csv, runs.csv, summary.csv, speedup.csv, validation.csv, logs/, ...
figures/                       all figures (PNG + PDF)
config/benchmark_config.json   sizes, thread counts, repetitions, warm-ups, budgets, experiments
run_all.ps1                    full reproduction (Windows PowerShell)
```

## Reused material (Assignments 1 and 2)

* `seq` is the `unroll4` kernel of `Assignment02_AlejandroSantana_Materials/src/c/matmul2.c` (fastest
  sequential method of Assignment 2). The loop body is unchanged; it was moved into one function
  `row_unroll4` (one row of C) that both `seq` and `par` call, and C is zeroed row by row inside it
  (the same total work as the single `memset` of Assignment 2), so that the sequential and the
  parallel versions run exactly the same code per row.
* `ijk` is the C triple loop of Assignment 1 (`matmul.c`), used only to put the speed-ups in context.
* The inputs are the Assignment 1/2 matrices: NumPy `default_rng` (PCG64), seeds 20261012 (A) and
  20261013 (B), uniform in [-1, 1). `generate_inputs.py --check <manifest>` compares the SHA-256 of
  every regenerated file with the earlier manifests (all files present there matched, see
  `data/inputs/manifest.json`, key `checked_against`).

## Requirements

* Windows 10/11, **gcc with OpenMP** (MinGW-w64; tested: WinLibs gcc 16.1.0 UCRT, POSIX threads,
  `winget install BrechtSanders.WinLibs.POSIX.UCRT --version 16.1.0-14.0.0-r2`; libgomp, OpenMP 5.2 / `_OPENMP` 202111).
  The AVX2 build needs a CPU with AVX2.
* **Python 3.11+** with `numpy`, `pandas`, `matplotlib` (tested: 3.11.9, numpy 2.4.6, pandas 3.1.1,
  matplotlib 3.11.2). With [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`) no installation is needed:
  `uv run --with numpy --with pandas --with matplotlib python <script>`.
* Google Chrome or Microsoft Edge (only to print the report to PDF).
* About 0.4 GB of free disk space for the inputs and 1 GB of free RAM.

## Reproduce everything

```powershell
.\run_all.ps1      # build -> environment -> vectorisation evidence -> inputs -> validation -> benchmarks -> analysis -> report
```

Step by step (inputs are stored outside the project folder, here `C:\bench_a3\data`, so that cloud
sync does not interfere with the timings):

```powershell
.\build.ps1                                                    # two builds, see below
.\scripts\collect_environment.ps1                              # results\environment.txt, results\topology.csv
uv run python scripts/check_vectorisation.py                   # results\disasm_summary.csv, results\vec_report_*.txt
uv run --with numpy python scripts/generate_inputs.py --outdir C:\bench_a3\data `
        --check ..\..\Assignment01_AlejandroSantana_Materials\data\inputs\manifest.json `
        --check ..\..\assigment-2\Assignment02_AlejandroSantana_Materials\data\inputs\manifest.json
uv run --with numpy python tests/validate.py --data-dir C:\bench_a3\data > results\validation.csv
uv run python scripts/run_benchmarks.py --data-dir C:\bench_a3\data       # all experiments (~35 min)
uv run --with numpy --with pandas --with matplotlib python scripts/analyze.py
uv run --with numpy --with pandas --with matplotlib python report/build_report.py
```

`run_benchmarks.py` only needs the standard library; `--results DIR` redirects the output and
experiment ids (e.g. `P1_threads`) select a subset.

Build flags (`build.ps1`):

| Build | Flags |
|---|---|
| scalar | `gcc -O2 -std=c11 -Wall -Wextra -fopenmp -fno-tree-vectorize -lpsapi` |
| avx2   | `gcc -O2 -std=c11 -Wall -Wextra -fopenmp -mavx2 -ffp-contract=off -DUSE_SIMD -lpsapi` (`#pragma omp simd` on the inner loops) |

Running one configuration by hand:

```powershell
build\matmul3_scalar.exe dense par A_n2048.bin B_n2048.bin 2048 1 7 0 16 dynamic pcores [C_out.bin]
#   dense <ijk|seq|par> A.bin B.bin n warmup reps batch threads <static|dynamic> <pcores|ecores|none> [C_out.bin]
#   batch 0 = automatic (a sample lasts ~20 ms, otherwise one multiplication); sequential methods need threads = 1
build\matmul3_scalar.exe topology        # logical processors in the order used by the 'pcores' policy
```

Every program prints `SAMPLE,<method>,<n>,<phase>,<index>,<batch>,<ns per multiplication>,<checksum>`
for each warm-up and measured run, `INFO,...` lines (build, flags, OpenMP version, requested/actual
threads, placement map) and `MEM,...` lines with the peak working set and peak commit charge.

## Experiments

| Id | Build | Sizes n | Threads p | Schedule | Placement | Purpose |
|---|---|---|---|---|---|---|
| P1_threads | scalar | 64 … 4096 (9 sizes) | 1,2,4,6,8,10,12,16,20,24,32,48 + seq (+ ijk ≤ 2048) | static | P-cores first | main study |
| V1_simd | avx2 | same | same + seq | static | P-cores first | vectorisation extension |
| P2_schedule | scalar | 1024, 2048, 4096 | same | dynamic,1 | P-cores first | load balance on the hybrid CPU |
| P3_affinity | scalar | 2048 | 1,2,4,8,12,16,24 + seq | static | none (OS) | effect of pinning |
| P4_ecores | scalar | 2048 | 1,2,4,8 + seq | static | E-cores first | E-core speed |

Placement `pcores`: thread t runs on slot t of the order *8 P-cores (first SMT thread) → 8 E-cores →
8 second SMT threads of the P-cores*, so p ≤ 8 uses only P-cores, 8 < p ≤ 16 adds E-cores, 16 < p ≤ 24
adds SMT siblings, and p > 24 oversubscribes the 24 logical processors (thread t → slot t mod 24).
Each thread pins itself with `SetThreadAffinityMask`; after the measurements every thread reports the
processor it runs on (`placement_map`, `placement_ok` in `results/runs.csv`).

## Experimental protocol (summary — details in the report)

* Timer: `QueryPerformanceCounter`. Timed region: the kernel including zeroing of C (inside the
  parallel loop for `par`); C is allocated, inputs loaded and the thread team created and pinned
  outside the timer. One fresh process per configuration; process priority class HIGH.
* Warm-up: 2 runs for n ≤ 512, 1 run above (kept in the raw data, excluded from statistics).
* Repetitions: 10 (n ≤ 512), 7 (n ≤ 2048), 5 (n > 2048). Short kernels are batched automatically.
* Budget: 120 s per sample, 8 GB peak working set (stop rule per series; none was reached).
* Metrics: median and IQR of the time; S(p) = T_seq / T_p and E(p) = S(p) / p from **medians**,
  where T_seq is the `seq` run of the same build; memory = peak working set of the process
  (`GetProcessMemoryInfo`).
* No numerical library runs in any timed region; OpenMP is configured by the program itself
  (`omp_set_dynamic(0)`, `omp_set_num_threads(p)`, `omp_set_schedule`), and the harness removes any
  `OMP_*` environment variable, so the reported thread count is the actual one (checked: `threads_actual`).
* Correctness: `tests/validate.py` against NumPy with |ΔC| ≤ 1e-9 + 1e-9·|C| **and** bit-exact
  equality with the scalar sequential baseline for every method, thread count (1–48), schedule,
  placement and build; repeated parallel runs must give identical checksums. In the benchmarks the
  checksum of every sample of every configuration is compared per n (`results/checksum_check.csv`).

## AI-use disclosure

Generative AI (Claude, Anthropic) was used as described in the "AI-use disclosure" section of the
report. All measurements in `results/` were produced by running the code in this folder.

Repository: https://github.com/aimarsantana/BigData-Assignment03-MatMul
