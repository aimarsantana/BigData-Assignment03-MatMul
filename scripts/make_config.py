#!/usr/bin/env python3
"""Write config/benchmark_config.json (the single source of truth for inputs, thread counts and experiments).

The file is generated here only to avoid typing the long lists by hand; the JSON is committed and is
what generate_inputs.py and run_benchmarks.py read.
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SIZES = [64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
# 1..8 = one thread per P-core, 10..16 = + E-cores, 20..24 = + second SMT thread of the P-cores,
# 32 and 48 = oversubscription (more threads than the 24 logical processors)
THREADS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48]


def main():
    cfg = {
        "description": "Assignment 3 benchmark configuration (read by generate_inputs.py and run_benchmarks.py).",
        "numeric_type": "float64 (IEEE-754 binary64)",
        "kernel": "row_unroll4 (ikj order, k unrolled by 4) = the 'unroll4' kernel of Assignment 2",
        "budget": {
            "max_seconds_per_sample": 120,
            "memory_budget_peak_working_set_bytes": 8000000000,
            "stop_rule": "if a series (experiment, build, method, threads, schedule, placement) exceeds the time or memory budget at size n, larger sizes of that series are NOT_ATTEMPTED",
        },
        "repetitions": {"n<=512": 10, "n<=2048": 7, "n>2048": 5},
        "warmup_runs": {"n<=512": 2, "n>512": 1},
        "batch": 0,
        "inputs": {"dense_sizes": SIZES},
        "thread_counts": THREADS,
        "experiments": [
            {"id": "P1_threads", "build": "scalar", "sizes": SIZES, "threads": THREADS, "schedule": "static",
             "placement": "pcores", "sequential": ["seq", "ijk"], "ijk_max_n": 2048,
             "purpose": "main study: thread count x matrix size, scalar code, static schedule, P-cores first"},
            {"id": "V1_simd", "build": "avx2", "sizes": SIZES, "threads": THREADS, "schedule": "static",
             "placement": "pcores", "sequential": ["seq"],
             "purpose": "optional vectorisation extension: same grid with the AVX2 build"},
            {"id": "P2_schedule", "build": "scalar", "sizes": [1024, 2048, 4096], "threads": THREADS,
             "schedule": "dynamic", "placement": "pcores", "sequential": [],
             "purpose": "load balance on the hybrid CPU: dynamic,1 instead of static (compare with P1)"},
            {"id": "P3_affinity", "build": "scalar", "sizes": [2048], "threads": [1, 2, 4, 8, 12, 16, 24],
             "schedule": "static", "placement": "none", "sequential": ["seq"],
             "purpose": "no thread pinning: the Windows scheduler places the threads (compare with P1)"},
            {"id": "P4_ecores", "build": "scalar", "sizes": [2048], "threads": [1, 2, 4, 8],
             "schedule": "static", "placement": "ecores", "sequential": ["seq"],
             "purpose": "the same thread counts on efficiency cores only (core-type speed ratio)"},
        ],
    }
    path = os.path.join(ROOT, "config", "benchmark_config.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
