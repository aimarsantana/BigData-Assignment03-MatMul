#!/usr/bin/env python3
"""Generate the shared dense input matrices of Assignment 3.

They are the SAME matrices as in Assignments 1 and 2: NumPy PCG64 (default_rng), seeds 20261012 (A)
and 20261013 (B), uniform in [-1, 1), float64, raw little-endian row-major files A_n<n>.bin /
B_n<n>.bin. The SHA-256 of every file is written to <outdir>/manifest.json and, with --check, compared
with the manifest of an earlier assignment (Assignment 1 stores the hashes under "files",
Assignment 2 under "dense"; both formats are accepted, several --check options are allowed).

Usage: python scripts/generate_inputs.py [--outdir DIR] [--check manifest.json ...]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

SEED_A, SEED_B = 20261012, 20261013          # identical to Assignments 1 and 2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def dense_matrix(n, seed):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, n)).astype("<f8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "benchmark_config.json"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data", "inputs"))
    ap.add_argument("--check", action="append", default=[], help="earlier manifest.json to compare hashes with")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "inputs", "manifest.json"),
                    help="where to write the manifest (committed copy)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    os.makedirs(args.outdir, exist_ok=True)
    earlier = {}
    for path in args.check:
        m = json.load(open(path))
        earlier[path] = m.get("files") or m.get("dense") or {}
    manifest = {"numpy_version": np.__version__, "generator": "numpy.random.default_rng (PCG64)",
                "seeds": {"A": SEED_A, "B": SEED_B}, "distribution": "uniform [-1, 1)", "dtype": "float64 little-endian, row-major",
                "files": {}, "checked_against": {}}
    for n in cfg["inputs"]["dense_sizes"]:
        for name, seed in (("A", SEED_A), ("B", SEED_B)):
            m = dense_matrix(n, seed)
            key = f"{name}_n{n}.bin"
            m.tofile(os.path.join(args.outdir, key))
            digest = hashlib.sha256(m.tobytes()).hexdigest()
            manifest["files"][key] = digest
            notes = []
            for path, files in earlier.items():
                if key in files:
                    same = files[key] == digest
                    manifest["checked_against"].setdefault(path.replace("\\", "/"), {})[key] = same
                    notes.append("matches" if same else "DIFFERS from")
            print(f"{key}: " + (", ".join(f"{t} earlier manifest" for t in notes) if notes else "new size"), file=sys.stderr)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    bad = [k for d in manifest["checked_against"].values() for k, v in d.items() if not v]
    if bad:
        sys.exit(f"hash mismatch with an earlier assignment: {bad}")


if __name__ == "__main__":
    main()
