#!/usr/bin/env python3
"""Supplementary checks for two unusual results of the main runs (run AFTER run_benchmarks.py).

C1 core_position: is the speed-up inflated because the baseline runs on logical processor 0 (which
   also serves most interrupts)? The sequential AVX2 and scalar kernels (n = 1024) are run on the
   first SMT thread of four different P-cores (logical processors 0, 2, 4, 6). The program is started
   with 'placement none' inside 'cmd /c start /affinity <mask>', so the whole process is confined
   to one logical processor by the operating system.
C2 repeat_n128: is the S(1) = 1.56 of the AVX2 build at n = 128 reproducible in fresh processes?
   seq and par (p = 1) of both builds are run alternately, 3 processes each.
C3 per_thread_rate: AVX2 par n = 1024 with p = 1, 2, 4 pinned to P-cores (same as V1), 3 processes each,
   to check whether the superlinear S(2) of V1 is reproducible.

Each process: 1 warm-up, 7 measured repetitions; the median of each process is recorded.
Output: results/supplementary_checks.csv (raw per-process medians) and printed summary.
Usage: python scripts/supplementary_checks.py [--data-dir DIR]
"""
import argparse
import csv
import os
import statistics
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(build, method, n, threads, placement, data, affinity=None):
    exe = os.path.join(ROOT, "build", f"matmul3_{build}.exe")
    A, B = (os.path.join(data, f"{x}_n{n}.bin") for x in "AB")
    args = [exe, "dense", method, A, B, str(n), "1", "7", "0", str(threads), "static", placement]
    if affinity is not None:
        args = ["cmd", "/c", "start", "", "/b", "/wait", "/affinity", format(1 << affinity, "X")] + args
    out = subprocess.run(args, capture_output=True, text=True, check=True, cwd=ROOT).stdout
    t = [float(l.split(",")[6]) / 1e9 for l in out.splitlines() if l.startswith("SAMPLE") and ",measured," in l]
    cpu = [l.split(",")[3] for l in out.splitlines() if "cpu_at_end" in l]
    return statistics.median(t), (cpu[0] if cpu else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "inputs"))
    args = ap.parse_args()
    rows = []
    for rnd in range(3):
        for build in ("avx2", "scalar"):
            for cpu in (0, 2, 4, 6):
                t, at = run(build, "seq", 1024, 1, "none", args.data_dir, affinity=cpu)
                rows.append(dict(check="C1_core_position", build=build, method="seq", n=1024, threads=1,
                                 where=f"logical {cpu} (ran on {at})", round=rnd, median_s=t))
        for build in ("avx2", "scalar"):
            for method in ("seq", "par"):
                t, _ = run(build, method, 128, 1, "pcores", args.data_dir)
                rows.append(dict(check="C2_repeat_n128", build=build, method=method, n=128, threads=1,
                                 where="pcores", round=rnd, median_s=t))
        for p in (1, 2, 4):
            t, _ = run("avx2", "par", 1024, p, "pcores", args.data_dir)
            rows.append(dict(check="C3_per_thread_rate", build="avx2", method="par", n=1024, threads=p,
                             where="pcores", round=rnd, median_s=t))
    with open(os.path.join(ROOT, "results", "supplementary_checks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r["check"], r["build"], r["method"], r["n"], r["threads"], r["where"].split(" (")[0]), []).append(r["median_s"])
    for k, v in groups.items():
        print(k, " ".join(f"{x * 1e3:.3f}ms" for x in v))


if __name__ == "__main__":
    main()
