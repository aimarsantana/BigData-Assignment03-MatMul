#!/usr/bin/env python3
"""Benchmark harness for Assignment 3 (Windows, OpenMP threads).

Every configuration (experiment, build, method, n, threads, schedule, placement) runs in ONE fresh
process of build/matmul3_<build>.exe that
  - loads the shared input files (outside the timer),
  - creates and pins its OpenMP thread team (outside the timer),
  - performs the declared warm-up runs (kept in the raw data, excluded from statistics),
  - performs the measured repetitions (a sample = `batch` multiplications, reported per multiplication;
    the batch is chosen automatically so that a sample lasts ~20 ms, and is 1 for long kernels),
  - reports where its threads ran and its peak working set / peak commit charge before exiting.

For one matrix size, all methods and thread counts of an experiment run back to back, so the speed-up
S(p) = T_seq / T_p compares measurements taken minutes apart in the same session.

Budgets (config/benchmark_config.json): if no new SAMPLE line appears within the per-sample budget
the process is killed and the configuration is recorded as TIMEOUT. A series that times out, fails to
allocate or exceeds the memory budget is not attempted at larger sizes (NOT_ATTEMPTED). No execution
time is invented for those rows.

Outputs (results/, appended so that experiments can be run one at a time):
  raw_samples.csv  every warm-up and measured sample
  runs.csv         one row per process: status, threads, placement check, peak memory, wall time
  info.csv         build / compiler / OpenMP information printed by the programs
  logs/*.log       raw stdout/stderr of every process
"""
import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SAMPLE_COLS = ["experiment", "build", "method", "n", "threads", "schedule", "placement", "numeric_type",
               "phase", "index", "batch", "ns_per_multiplication", "checksum"]
RUN_COLS = ["experiment", "build", "method", "n", "threads", "schedule", "placement", "numeric_type",
            "status", "warmup_runs", "measured_reps", "threads_actual", "placement_ok", "placement_map",
            "peak_working_set_kib", "peak_commit_kib", "process_wall_s", "exit_code", "note"]


def exe(build):
    return os.path.join(ROOT, "build", f"matmul3_{build}.exe")


def reps_for(cfg, n):
    r = cfg["repetitions"]
    return r["n<=512"] if n <= 512 else (r["n<=2048"] if n <= 2048 else r["n>2048"])


def warmup_for(cfg, n):
    w = cfg["warmup_runs"]
    return w["n<=512"] if n <= 512 else w["n>512"]


def run_one(cmd, budget_s, log_path):
    """Run a process and stream its stdout. Returns (status, lines, wall_s, exit_code)."""
    t_start = time.monotonic()
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OMP_SCHEDULE", "OMP_PROC_BIND", "OMP_PLACES", "OMP_DYNAMIC", "GOMP_CPU_AFFINITY"):
        env.pop(var, None)          # the program sets threads and schedule itself; no external override
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=ROOT, env=env)
    q = queue.Queue()

    def reader():
        for line in p.stdout:
            q.put(line.rstrip("\n"))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    lines, status, last = [], "OK", time.monotonic()
    while True:
        remaining = budget_s - (time.monotonic() - last)
        if remaining <= 0:
            status = "TIMEOUT"
            p.kill()
            break
        try:
            item = q.get(timeout=min(remaining, 2.0))
        except queue.Empty:
            continue
        if item is None:
            break
        lines.append(item)
        if item.startswith("SAMPLE"):
            last = time.monotonic()
    p.wait()
    wall = time.monotonic() - t_start
    if status == "OK" and p.returncode != 0:
        status = "MEMORY_FAILURE" if p.returncode == 3 else f"ERROR({p.returncode})"
    with open(log_path, "w") as f:
        f.write(" ".join(cmd) + "\n" + "\n".join(lines) + "\n")
    return status, lines, wall, p.returncode


class Recorder:
    def __init__(self, res):
        os.makedirs(os.path.join(res, "logs"), exist_ok=True)
        self.res = res
        self.files = {}
        for name, cols in (("raw_samples.csv", SAMPLE_COLS), ("runs.csv", RUN_COLS), ("info.csv", ["build", "key", "value"])):
            path = os.path.join(res, name)
            new = not os.path.exists(path)
            f = open(path, "a", newline="")
            w = csv.writer(f)
            if new:
                w.writerow(cols)
            self.files[name] = (f, w)
        self.info_seen = set()

    def row(self, name, values):
        f, w = self.files[name]
        w.writerow(values)
        f.flush()

    def close(self):
        for f, _ in self.files.values():
            f.close()


def parse(lines):
    samples, mem, info = [], {}, {}
    for ln in lines:
        parts = ln.split(",")
        if parts[0] == "SAMPLE":
            samples.append(parts[1:])
        elif parts[0] == "MEM":
            mem[parts[3]] = parts[4]
        elif parts[0] == "INFO":
            info[f"{parts[1]}.{parts[2]}"] = ",".join(parts[3:])
    return samples, mem, info


STATIC_INFO = ("all.build", "all.compiler", "all.cflags", "all.openmp", "all.logical_processors", "all.priority")


def execute(rec, cfg, stopped, job):
    budget = cfg["budget"]["max_seconds_per_sample"]
    mem_budget = cfg["budget"]["memory_budget_peak_working_set_bytes"]
    base = [job["experiment"], job["build"], job["method"], job["n"], job["threads"], job["schedule"],
            job["placement"], "float64"]
    label = (f"[{job['experiment']}] {job['build']} {job['method']} n={job['n']} p={job['threads']} "
             f"{job['schedule']} {job['placement']}")
    if job["series"] in stopped:
        rec.row("runs.csv", base + ["NOT_ATTEMPTED", job["warm"], job["reps"], "", "", "", "", "", "", "", "",
                                    "series stopped by an earlier failure/over-budget run"])
        print(f"{label}: not attempted", flush=True)
        return
    name = (f"{job['experiment']}_{job['build']}_{job['method']}_n{job['n']}_p{job['threads']}_"
            f"{job['schedule']}_{job['placement']}")
    status, lines, wall, ec = run_one(job["cmd"], budget, os.path.join(rec.res, "logs", name + ".log"))
    samples, mem, info = parse(lines)
    for key in STATIC_INFO:
        if key in info and (job["build"], key, info[key]) not in rec.info_seen:
            rec.info_seen.add((job["build"], key, info[key]))
            rec.row("info.csv", [job["build"], key, info[key]])
    for s in samples:
        method, n, phase, idx, batch, ns, chk = s
        rec.row("raw_samples.csv", [job["experiment"], job["build"], method, n, job["threads"], job["schedule"],
                                    job["placement"], "float64", phase, idx, batch, ns, chk])
    ws = mem.get("peak_working_set_kib", "")
    note = ""
    if status == "OK" and ws not in ("", "-1") and int(ws) * 1024 > mem_budget:
        status, note = "OVER_MEMORY_BUDGET", f"peak working set above {mem_budget} bytes"
    actual = info.get("par.threads_actual", "1")
    pl_ok = info.get("par.placement_ok", "")
    pl_map = info.get("par.placement_map", info.get(f"{job['method']}.cpu_at_end", ""))
    rec.row("runs.csv", base + [status, job["warm"], job["reps"], actual, pl_ok, pl_map, ws,
                                mem.get("peak_commit_kib", ""), f"{wall:.3f}", ec, note])
    if status != "OK":
        stopped.add(job["series"])
    print(f"{label}: {status}, peak WS {ws} KiB, wall {wall:.1f}s", flush=True)


def make_job(cfg, exp, data, method, n, threads, schedule, placement):
    warm, reps = warmup_for(cfg, n), reps_for(cfg, n)
    A, B = (os.path.join(data, f"{x}_n{n}.bin") for x in "AB")
    sched = schedule if method == "par" else "none"
    return dict(experiment=exp["id"], build=exp["build"], method=method, n=n, threads=threads, schedule=sched,
                placement=placement, warm=warm, reps=reps,
                series=(exp["id"], exp["build"], method, threads, sched, placement),
                cmd=[exe(exp["build"]), "dense", method, A, B, str(n), str(warm), str(reps), str(cfg["batch"]),
                     str(threads), schedule, placement])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "benchmark_config.json"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "inputs"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("experiments", nargs="*", help="experiment ids to run (default: all)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    rec = Recorder(args.results)
    stopped = set()
    for exp in cfg["experiments"]:
        if args.experiments and exp["id"] not in args.experiments:
            continue
        for n in exp["sizes"]:
            jobs = []
            for method in exp["sequential"]:
                if method == "ijk" and n > exp.get("ijk_max_n", n):
                    continue
                jobs.append(make_job(cfg, exp, args.data_dir, method, n, 1, "static", exp["placement"]))
            for p in exp["threads"]:
                jobs.append(make_job(cfg, exp, args.data_dir, "par", n, p, exp["schedule"], exp["placement"]))
            for job in jobs:
                execute(rec, cfg, stopped, job)
    rec.close()


if __name__ == "__main__":
    sys.exit(main())
