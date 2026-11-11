#!/usr/bin/env python3
"""Correctness tests for the sequential and parallel kernels of Assignment 3 (run BEFORE benchmarking).

Reference: NumPy matmul (OpenBLAS), used only for checking, never as a student implementation.
Tolerance (float64): |C - C_ref| <= atol + rtol*|C_ref| with atol = rtol = 1e-9.
  Justification: for entries in [-1, 1] the rounding error of a length-n dot product is bounded by
  n*u*sum|a_ik b_kj| <= n^2*u (u = 2^-53 = 1.1e-16) -> 1.2e-10 for n = 1024. 1e-9 is above this bound
  yet far below any plausible logic error (products are O(0.1 - 1)).
Stronger check: every method, thread count, schedule, placement and build must equal the scalar
sequential baseline BIT FOR BIT, because all of them add the products of each C[i][j] in increasing k,
with no FMA (see matmul3.c). A data race or a wrong row partition would break this equality.
Race check: the parallel kernel is run repeatedly (dynamic schedule, oversubscribed) and every repetition
must give the same checksum.

Output: CSV on stdout (test, build, method, n, threads, schedule, placement, max_abs_err_vs_numpy,
bitwise_equal_to_seq, result). Exit code 1 if any test fails.

Usage: python tests/validate.py [--data-dir DIR]   (the data dir enables tests on benchmark inputs)
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILDS = ("scalar", "avx2")
ATOL = RTOL = 1e-9
FAILED = []
PASSED = [0]


def exe(build):
    return os.path.join(ROOT, "build", f"matmul3_{build}.exe")


def run(build, args):
    r = subprocess.run([exe(build)] + [str(a) for a in args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{build} {args}: {r.stderr}")
    return r.stdout


def multiply(tmp, build, method, pa, pb, n, threads=1, schedule="static", placement="pcores", reps=1):
    po = os.path.join(tmp, "C.bin")
    out = run(build, ["dense", method, pa, pb, n, 0, reps, 1, threads, schedule, placement, po])
    C = np.fromfile(po, "<f8").reshape(n, n)
    return C, out


def record(test, build, method, n, threads, schedule, placement, ok, err="", bit=""):
    print(f"{test},{build},{method},{n},{threads},{schedule},{placement},{err},{bit},{'PASS' if ok else 'FAIL'}")
    if ok:
        PASSED[0] += 1
    else:
        FAILED.append((test, build, method, n, threads, schedule, placement))


def check(test, A, B, configs, tmp):
    """configs: list of (build, method, threads, schedule, placement). The first result of the
    scalar 'seq' method is the bitwise reference."""
    n = A.shape[0]
    pa, pb = os.path.join(tmp, "A.bin"), os.path.join(tmp, "B.bin")
    A.astype("<f8").tofile(pa)
    B.astype("<f8").tofile(pb)
    ref_np = A @ B
    ref_seq, _ = multiply(tmp, "scalar", "seq", pa, pb, n)
    for build, method, p, sched, place in configs:
        C, out = multiply(tmp, build, method, pa, pb, n, p, sched, place)
        err = float(np.max(np.abs(C - ref_np))) if C.size else 0.0
        bit = bool(np.array_equal(C, ref_seq))
        ok = bool(np.allclose(C, ref_np, rtol=RTOL, atol=ATOL)) and bit
        if method == "par" and place != "none":
            ok = ok and "INFO,par,placement_ok,1" in out
        if method == "par":
            ok = ok and f"INFO,par,threads_actual,{p}" in out
        record(test, build, method, n, p, sched if method == "par" else "none", place, ok, f"{err:.3e}", str(bit))


def all_configs(threads, schedules=("static", "dynamic"), placements=("pcores",), ijk=True):
    cfgs = []
    for b in BUILDS:
        cfgs.append((b, "seq", 1, "static", "pcores"))
        if ijk and b == "scalar":
            cfgs.append((b, "ijk", 1, "static", "pcores"))
        for p in threads:
            for s in schedules:
                for pl in placements:
                    cfgs.append((b, "par", p, s, pl))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(424242)
    print("test,build,method,n,threads,schedule,placement,max_abs_err_vs_numpy,bitwise_equal_to_seq,result")
    with tempfile.TemporaryDirectory() as tmp:
        # --- hand-computed 3x3 example
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float)
        B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], float)
        assert np.array_equal(A @ B, [[30, 24, 18], [84, 69, 54], [138, 114, 90]])
        check("known_3x3", A, B, all_configs((1, 2, 3, 4)), tmp)
        # --- identity and zero matrices
        X = rng.uniform(-1, 1, (37, 37))
        check("identity_right", X, np.eye(37), all_configs((1, 4, 24)), tmp)
        check("identity_left", np.eye(37), X, all_configs((1, 4, 24)), tmp)
        check("zero", X, np.zeros((37, 37)), all_configs((1, 4, 24)), tmp)
        # --- sizes not divisible by the unroll factor / thread count, more threads than rows
        for n in (1, 2, 3, 5, 7, 33, 100, 127, 130):
            A, B = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
            check(f"random_n{n}", A, B, all_configs((1, 2, 3, 7, 8, 16, 24, 48), placements=("pcores", "none")), tmp)
        # --- placement policies
        A, B = rng.uniform(-1, 1, (64, 64)), rng.uniform(-1, 1, (64, 64))
        check("placement_ecores", A, B, [(b, "par", p, "static", "ecores") for b in BUILDS for p in (1, 4, 8, 12)], tmp)
        # --- race check: repeated parallel runs must give identical checksums
        n = 256
        A, B = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
        pa, pb = os.path.join(tmp, "A.bin"), os.path.join(tmp, "B.bin")
        A.tofile(pa)
        B.tofile(pb)
        for b in BUILDS:
            for p, s in ((24, "dynamic"), (48, "dynamic"), (16, "static")):
                _, out = multiply(tmp, b, "par", pa, pb, n, p, s, "pcores", reps=20)
                sums = {l.split(",")[7] for l in out.splitlines() if l.startswith("SAMPLE")}
                record("repeat20_same_checksum", b, "par", n, p, s, "pcores", len(sums) == 1, "", "")
        # --- benchmark inputs (if available)
        if args.data_dir:
            for n in (512, 1024):
                A = np.fromfile(os.path.join(args.data_dir, f"A_n{n}.bin"), "<f8").reshape(n, n)
                B = np.fromfile(os.path.join(args.data_dir, f"B_n{n}.bin"), "<f8").reshape(n, n)
                check(f"benchmark_n{n}", A, B, all_configs((1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48), ijk=(n == 512)), tmp)
    print(f"# {PASSED[0]} passed, {len(FAILED)} failure(s)", file=sys.stderr)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
