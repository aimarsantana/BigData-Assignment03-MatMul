#!/usr/bin/env python3
"""Evidence that vectorisation does (avx2 build) or does not (scalar build) occur.

1. The gcc vectoriser reports written by build.ps1 (-fopt-info-vec-optimized) are copied to
   results/vec_report_<build>.txt.
2. The two executables are disassembled with objdump and the floating-point instructions of the
   functions that contain the row kernel (matmul_seq and the OpenMP outlined body of matmul_par)
   are counted by class:
     scalar_sse2  mulsd / addsd                (1 double per instruction)
     packed_sse2  mulpd / addpd                (2 doubles, 128-bit xmm)
     packed_avx   vmulpd / vaddpd on ymm       (4 doubles, 256-bit)
     scalar_avx   vmulsd / vaddsd              (1 double, VEX encoded)
     fma          vfmadd* / vfmsub* ...        (must be 0: no FMA contraction)
   Output: results/disasm_summary.csv

Usage: python scripts/check_vectorisation.py   (needs objdump from the same MinGW-w64 toolchain)
"""
import csv
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FUNCS = ("matmul_seq", "matmul_par._omp_fn.0")
CLASSES = {
    "scalar_sse2": re.compile(r"\s(mulsd|addsd)\s"),
    "packed_sse2": re.compile(r"\s(mulpd|addpd)\s"),
    "packed_avx_ymm": re.compile(r"\s(vmulpd|vaddpd)\s.*%ymm"),
    "packed_avx_xmm": re.compile(r"\s(vmulpd|vaddpd)\s(?!.*%ymm)"),
    "scalar_avx": re.compile(r"\s(vmulsd|vaddsd)\s"),
    "fma": re.compile(r"\sv?f(n)?m(add|sub)"),
}


def functions(disasm):
    """split objdump output into {function name: [lines]}"""
    out, cur = {}, None
    for line in disasm.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur:
            out[cur].append(line)
    return out


def main():
    objdump = shutil.which("objdump")
    if not objdump:
        sys.exit("objdump not found (it is part of the MinGW-w64 binutils)")
    rows = []
    for build in ("scalar", "avx2"):
        src = os.path.join(ROOT, "build", f"vec_report_{build}.txt")
        dst = os.path.join(ROOT, "results", f"vec_report_{build}.txt")
        text = open(src).read() if os.path.exists(src) else ""
        with open(dst, "w") as f:
            f.write(text if text.strip() else "(no loop was vectorised in this build)\n")
        dis = subprocess.run([objdump, "-d", "--no-show-raw-insn", os.path.join(ROOT, "build", f"matmul3_{build}.exe")],
                             capture_output=True, text=True, check=True).stdout
        fns = functions(dis)
        for fn in FUNCS:
            body = fns.get(fn, [])
            row = {"build": build, "function": fn, "instructions": len(body)}
            for name, rx in CLASSES.items():
                row[name] = sum(1 for l in body if rx.search(l + " "))
            rows.append(row)
            print(row)
    with open(os.path.join(ROOT, "results", "disasm_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
