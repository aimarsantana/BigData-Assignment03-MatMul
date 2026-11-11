#!/usr/bin/env python3
"""Analysis of the Assignment 3 measurements: summary statistics, speed-up, efficiency, figures and report tables.

Input : results/raw_samples.csv, results/runs.csv (written by run_benchmarks.py)
Output: results/summary.csv          median / quartiles / min / max / std of every configuration
        results/speedup.csv          S(p) = T_seq / T_p and E(p) = S(p) / p for every parallel configuration
        results/checksum_check.csv   agreement of the result checksums of all configurations of each n
        results/key_numbers.json     numbers quoted in the report
        figures/*.png, *.pdf         all figures
        report/generated/*.html      tables included in the report

Statistics: central value = median of the measured samples (warm-up excluded); variability = interquartile
range (IQR, linear interpolation quartiles). Time per multiplication in seconds.
Speed-up: S(p) = median T_seq / median T_p, where T_seq is the sequential baseline ('seq', the same row
kernel without OpenMP) of the SAME build, measured in the same experiment and session:
  P1, V1: their own seq run (P-core);  P2: the seq run of P1 (same build, same session, run just before);
  P3: its own seq run (unpinned);      P4: the P-core seq run of P1 (so S shows what E-cores deliver).
Vectorisation effect: T(scalar build) / T(avx2 build) at the same n, p, schedule and placement.
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KEY = ["experiment", "build", "method", "n", "threads", "schedule", "placement"]
# Okabe-Ito colour-blind safe palette, one colour per matrix size
SIZE_COL = {64: "#999999", 128: "#56B4E9", 256: "#E69F00", 512: "#009E73", 1024: "#0072B2", 1536: "#F0E442",
            2048: "#D55E00", 3072: "#CC79A7", 4096: "#000000"}
N_PLOT = [128, 256, 512, 1024, 2048, 4096]
DRAM_PEAK_GBS = 2 * 8 * 5600e6 / 1e9      # dual-channel DDR5-5600, 64-bit channels (theoretical)
P_CORES, E_CORES, LOGICAL = 8, 8, 24


# ----------------------------------------------------------------------------- loading
def load(results):
    s = pd.read_csv(os.path.join(results, "raw_samples.csv"), keep_default_na=False)
    r = pd.read_csv(os.path.join(results, "runs.csv"), keep_default_na=False)
    s["seconds"] = pd.to_numeric(s["ns_per_multiplication"]) / 1e9
    return s, r


def summarise(s, r):
    m = s[s["phase"] == "measured"]
    g = m.groupby(KEY)["seconds"]
    out = g.agg(reps="count", median_s="median", min_s="min", max_s="max", mean_s="mean", std_s="std").reset_index()
    out["q1_s"] = g.quantile(0.25).values
    out["q3_s"] = g.quantile(0.75).values
    out["iqr_s"] = out["q3_s"] - out["q1_s"]
    out["rel_iqr"] = out["iqr_s"] / out["median_s"]
    out["batch"] = m.groupby(KEY)["batch"].first().values
    runs = r.copy()
    for c in ("peak_working_set_kib", "peak_commit_kib", "process_wall_s"):
        runs[c] = pd.to_numeric(runs[c], errors="coerce")
    full = runs[KEY + ["status", "warmup_runs", "measured_reps", "threads_actual", "placement_ok", "peak_working_set_kib",
                       "peak_commit_kib", "process_wall_s", "note"]].merge(out.drop(columns=["reps"]), on=KEY, how="left")
    full["reps"] = full["measured_reps"]
    full["gflops"] = 2.0 * full["n"].astype(float) ** 3 / full["median_s"] / 1e9
    return full


# ----------------------------------------------------------------------------- formatting helpers
def fmt_t(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "&ndash;"
    if x >= 100:
        return f"{x:.0f} s"
    if x >= 1:
        return f"{x:.2f} s"
    if x >= 1e-3:
        return f"{x * 1e3:.2f} ms"
    return f"{x * 1e6:.1f} &micro;s"


def fmt_mem(kib):
    if kib is None or (isinstance(kib, float) and math.isnan(kib)):
        return "&ndash;"
    mb = kib / 1024
    return f"{mb / 1024:.2f} GiB" if mb >= 1024 else f"{mb:.1f} MiB"


def html_table(header, rows, caption, cls=""):
    h = "".join(f"<th>{c}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="{cls}"><caption>{caption}</caption><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def save_fig(fig, name, figdir):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def style(ax, xlabel, ylabel, title=None, logx=False, logy=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def thread_axis(ax, pmax=48):
    """x axis for thread counts with the hardware regions shaded."""
    ax.axvspan(0.5, P_CORES + 0.5, color="#0072B2", alpha=0.05, lw=0)
    ax.axvspan(P_CORES + 0.5, P_CORES + E_CORES + 0.5, color="#009E73", alpha=0.07, lw=0)
    ax.axvspan(P_CORES + E_CORES + 0.5, LOGICAL + 0.5, color="#E69F00", alpha=0.07, lw=0)
    ax.axvspan(LOGICAL + 0.5, pmax + 2, color="#999999", alpha=0.08, lw=0)
    ax.set_xlim(0, pmax + 2)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 32, 48])


def region_labels(ax, y):
    for x, t in ((4.5, "P-cores"), (12.5, "+E-cores"), (20.5, "+SMT"), (36, "oversubscribed")):
        ax.text(x, y, t, ha="center", va="top", fontsize=7, color="#555", transform=ax.get_xaxis_transform())


def cell(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    return q


def ok(df):
    return df[(df["status"] == "OK") & df["median_s"].notna()]


def one(df, col="median_s"):
    return float(df[col].iloc[0]) if len(df) else float("nan")


# ----------------------------------------------------------------------------- analysis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--figdir", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--gendir", default=os.path.join(ROOT, "report", "generated"))
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.gendir, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7.5, "figure.dpi": 110})

    s, r = load(args.results)
    S = summarise(s, r)
    S.to_csv(os.path.join(args.results, "summary.csv"), index=False)
    key = {}
    G = ok(S)

    # ---------------------------------------------------------------- checksum agreement (all configurations per n)
    m = s[s["phase"] == "measured"]
    rows = []
    for n, g in m.groupby("n"):
        rows.append(dict(n=n, configurations=int(g.groupby(KEY).ngroups), samples=int(len(g)),
                         distinct_checksums=int(g["checksum"].nunique())))
    chk = pd.DataFrame(rows)
    chk.to_csv(os.path.join(args.results, "checksum_check.csv"), index=False)
    key["checksum_sizes"] = int(len(chk))
    key["checksum_samples"] = int(chk["samples"].sum())
    key["checksum_configurations"] = int(chk["configurations"].sum())
    key["checksum_disagreements"] = int((chk["distinct_checksums"] != 1).sum())

    # ---------------------------------------------------------------- placement verification
    par_runs = r[(r["method"] == "par") & (r["status"] == "OK")]
    pinned = par_runs[par_runs["placement"] != "none"]
    key["pinned_runs"] = int(len(pinned))
    key["pinned_runs_placement_ok"] = int((pinned["placement_ok"].astype(str) == "1").sum())
    key["threads_actual_mismatch"] = int((par_runs["threads_actual"].astype(int) != par_runs["threads"].astype(int)).sum())

    # ---------------------------------------------------------------- speed-up table
    def tseq(exp, build, n):
        src = {"P2_schedule": "P1_threads", "P4_ecores": "P1_threads"}.get(exp, exp)
        return one(cell(G, experiment=src, build=build, method="seq", n=n))

    par = G[G["method"] == "par"].copy()
    par["t_seq"] = [tseq(e, b, n) for e, b, n in zip(par["experiment"], par["build"], par["n"])]
    par["speedup"] = par["t_seq"] / par["median_s"]
    par["efficiency"] = par["speedup"] / par["threads"]
    # spread of the speed-up from the quartiles of T_p (T_seq taken at its median)
    par["speedup_q1"] = par["t_seq"] / par["q3_s"]
    par["speedup_q3"] = par["t_seq"] / par["q1_s"]
    par.to_csv(os.path.join(args.results, "speedup.csv"), index=False,
               columns=KEY + ["reps", "median_s", "iqr_s", "t_seq", "speedup", "speedup_q1", "speedup_q3", "efficiency"])

    p1 = cell(par, experiment="P1_threads")
    v1 = cell(par, experiment="V1_simd")
    seq1 = cell(G, experiment="P1_threads", method="seq")
    seqv = cell(G, experiment="V1_simd", method="seq")
    ijk1 = cell(G, experiment="P1_threads", method="ijk")

    # ---------------------------------------------------------------- Fig 1: runtime vs threads (P1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n in N_PLOT:
        g = p1[p1["n"] == n].sort_values("threads")
        if g.empty:
            continue
        ax.errorbar(g["threads"], g["median_s"], yerr=[g["median_s"] - g["q1_s"], g["q3_s"] - g["median_s"]],
                    color=SIZE_COL[n], marker="o", ms=3.5, lw=1.2, capsize=2, label=f"n = {n}")
        ax.scatter([0.5], [one(cell(seq1, n=n))], color=SIZE_COL[n], marker="_", s=120, lw=2)
    thread_axis(ax)
    style(ax, "threads p", "time per multiplication (s)",
          "Scalar build, static schedule, P-cores first (dash at left = sequential baseline)", logy=True)
    region_labels(ax, 0.995)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save_fig(fig, "fig_runtime_threads", args.figdir)

    # ---------------------------------------------------------------- Fig 2/3: speed-up and efficiency (P1)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for n in N_PLOT:
        g = p1[p1["n"] == n].sort_values("threads")
        if g.empty:
            continue
        axs[0].plot(g["threads"], g["speedup"], color=SIZE_COL[n], marker="o", ms=3.5, lw=1.2, label=f"n = {n}")
        axs[1].plot(g["threads"], g["efficiency"], color=SIZE_COL[n], marker="o", ms=3.5, lw=1.2, label=f"n = {n}")
    xs = np.array([1, 24])
    axs[0].plot(xs, xs, ls=":", color="grey", label="ideal S = p")
    axs[1].axhline(1, ls=":", color="grey")
    for ax, yl, t in ((axs[0], "speed-up S(p) = T_seq / T_p", "Speed-up (scalar build, static)"),
                      (axs[1], "efficiency E(p) = S(p) / p", "Parallel efficiency")):
        thread_axis(ax)
        style(ax, "threads p", yl, t)
        region_labels(ax, 0.995)
    axs[0].set_ylim(0, 16)
    axs[1].set_ylim(0, 1.25)
    axs[0].legend(ncol=2, loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.tight_layout()
    save_fig(fig, "fig_speedup_efficiency", args.figdir)

    # ---------------------------------------------------------------- Fig 4: when does parallelism help (S vs n)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    pal = {1: "#999999", 2: "#56B4E9", 8: "#0072B2", 16: "#009E73", 24: "#D55E00", 48: "#CC79A7"}
    for p, c in pal.items():
        g = p1[p1["threads"] == p].sort_values("n")
        if g.empty:
            continue
        ax.plot(g["n"], g["speedup"], color=c, marker="o", ms=3.5, lw=1.2, label=f"p = {p}")
    ax.axhline(1, color="grey", lw=0.8)
    style(ax, "matrix size n", "speed-up over the sequential baseline", "Speed-up versus matrix size (scalar, static)",
          logx=True, logy=True)
    ax.legend(ncol=3)
    save_fig(fig, "fig_speedup_vs_n", args.figdir)

    # ---------------------------------------------------------------- Fig 5: schedule (P1 static vs P2 dynamic)
    p2 = cell(par, experiment="P2_schedule")
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, n in zip(axs, (1024, 2048, 4096)):
        for df, lab, c, mk in ((p1, "static (one block of rows per thread)", "#0072B2", "o"),
                               (p2, "dynamic,1 (rows handed out one by one)", "#D55E00", "s")):
            g = df[df["n"] == n].sort_values("threads")
            if g.empty:
                continue
            ax.errorbar(g["threads"], g["speedup"], yerr=[g["speedup"] - g["speedup_q1"], g["speedup_q3"] - g["speedup"]],
                        color=c, marker=mk, ms=3.5, lw=1.2, capsize=2, label=lab)
        thread_axis(ax)
        style(ax, "threads p", "speed-up S(p)" if n == 1024 else "", f"n = {n}")
        region_labels(ax, 0.995)
    axs[0].legend(loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.tight_layout()
    save_fig(fig, "fig_schedule", args.figdir)

    # ---------------------------------------------------------------- Fig 6: placement (n = 2048)
    p3, p4 = cell(par, experiment="P3_affinity"), cell(par, experiment="P4_ecores")
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for df, lab, c, mk in ((cell(p1, n=2048), "pinned, P-cores first (P1)", "#0072B2", "o"),
                           (cell(p2, n=2048), "pinned, P-cores first, dynamic (P2)", "#D55E00", "s"),
                           (cell(p3, n=2048), "not pinned, OS scheduler (P3)", "#009E73", "^"),
                           (cell(p4, n=2048), "pinned, E-cores only (P4)", "#CC79A7", "D")):
        g = df.sort_values("threads")
        g = g[g["threads"] <= 24]
        if g.empty:
            continue
        ax.errorbar(g["threads"], g["speedup"], yerr=[g["speedup"] - g["speedup_q1"], g["speedup_q3"] - g["speedup"]],
                    color=c, marker=mk, ms=3.5, lw=1.2, capsize=2, label=lab)
    thread_axis(ax, 24)
    ax.set_xticks([1, 2, 4, 8, 12, 16, 20, 24])
    ax.set_xlim(0, 25)
    style(ax, "threads p", "speed-up over the P-core sequential baseline", "Thread placement and schedule, n = 2048")
    ax.legend(loc="upper left")
    save_fig(fig, "fig_placement", args.figdir)

    # ---------------------------------------------------------------- Fig 7: vectorisation (V1 vs P1)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for n, c in ((256, "#E69F00"), (1024, "#0072B2"), (2048, "#D55E00"), (4096, "#000000")):
        a, b = cell(p1, n=n).set_index("threads"), cell(v1, n=n).set_index("threads")
        common = sorted(set(a.index) & set(b.index))
        if not common:
            continue
        axs[0].plot(common, a.loc[common, "gflops"], color=c, marker="o", ms=3, lw=1.1, ls="--", label=f"n = {n}, scalar")
        axs[0].plot(common, b.loc[common, "gflops"], color=c, marker="s", ms=3, lw=1.4, label=f"n = {n}, AVX2")
        axs[1].plot(common, a.loc[common, "median_s"] / b.loc[common, "median_s"], color=c, marker="o", ms=3.5, lw=1.2,
                    label=f"n = {n}")
    axs[1].axhline(1, color="grey", lw=0.8)
    for ax in axs:
        thread_axis(ax)
        region_labels(ax, 0.995)
    style(axs[0], "threads p", "throughput (GFLOP/s, 2n$^3$ / median time)", "Scalar (dashed) vs AVX2 (solid)")
    style(axs[1], "threads p", "vectorisation gain T_scalar / T_AVX2 (same p)", "Effect of SIMD at equal thread count")
    axs[0].legend(ncol=2, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, 0.93))
    axs[1].legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig_simd", args.figdir)

    # ---------------------------------------------------------------- Fig 8: memory-bandwidth view
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for df, lab, ls, mk in ((p1, "scalar", "--", "o"), (v1, "AVX2", "-", "s")):
        for n in (1024, 2048, 4096):
            g = df[df["n"] == n].sort_values("threads")
            if g.empty:
                continue
            ax.plot(g["threads"], 8.0 * n ** 3 / g["median_s"] / 1e9, color=SIZE_COL[n], ls=ls, marker=mk, ms=3,
                    label=f"n = {n}, {lab}")
    ax.axhline(DRAM_PEAK_GBS, color="red", lw=0.9, ls=":")
    ax.text(47, DRAM_PEAK_GBS * 1.03, f"DRAM peak {DRAM_PEAK_GBS:.1f} GB/s (theoretical)", ha="right", fontsize=7, color="red")
    thread_axis(ax)
    region_labels(ax, 0.995)
    style(ax, "threads p", "B-stream rate 8n$^3$ / time (GB/s)", "Rate at which the kernel streams B (n reads of B per product)")
    ax.legend(ncol=2, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, 0.93))
    save_fig(fig, "fig_bandwidth", args.figdir)

    # ---------------------------------------------------------------- Fig 9: memory
    mem = S[(S["status"] == "OK") & (S["experiment"] == "P1_threads")]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for n in (512, 1024, 2048, 4096):
        g = mem[(mem["n"] == n) & (mem["method"] == "par")].sort_values("threads")
        if g.empty:
            continue
        ax.plot(g["threads"], g["peak_working_set_kib"] / 1024, color=SIZE_COL[n], marker="o", ms=3, label=f"n = {n}")
        ax.axhline(3 * 8 * n * n / 2 ** 20, color=SIZE_COL[n], ls=":", lw=0.8)
    thread_axis(ax)
    style(ax, "threads p", "peak working set (MiB)", "Peak working set vs threads (dotted: 3 x 8n$^2$ bytes)", logy=True)
    ax.legend(ncol=2)
    save_fig(fig, "fig_memory", args.figdir)

    # ---------------------------------------------------------------- tables
    T_COLS = [1, 2, 4, 8, 12, 16, 24, 32, 48]
    rows = []
    for n in sorted(seq1["n"].unique()):
        row = [str(n)]
        ij = cell(ijk1, n=n)
        allijk = cell(S, experiment="P1_threads", method="ijk", n=n)
        row.append(fmt_t(one(ij)) if len(ij) else ("&ndash;" if allijk.empty else allijk["status"].iloc[0].lower()))
        sq = cell(seq1, n=n).iloc[0]
        row.append(f"{fmt_t(sq['median_s'])}<br><span class='iqr'>{fmt_t(sq['iqr_s'])}</span>")
        for p in T_COLS:
            g = cell(p1, n=n, threads=p)
            row.append("&ndash;" if g.empty else f"{fmt_t(one(g))}<br><span class='iqr'>{fmt_t(one(g, 'iqr_s'))}</span>")
        rows.append(row)
    open(os.path.join(args.gendir, "table_time.html"), "w", encoding="utf8").write(html_table(
        ["n", "ijk (A1)", "seq"] + [f"p={p}" for p in T_COLS], rows,
        "Table 3. Scalar build (P1): median time per multiplication, IQR below in grey (10 repetitions for n&nbsp;&le;&nbsp;512, "
        "7 for n&nbsp;&le;&nbsp;2048, 5 above). seq = sequential baseline; p = threads of the parallel version (static schedule, "
        "P-cores first). ijk = basic algorithm of Assignment 1, context only, run up to n&nbsp;=&nbsp;2048.", "small"))

    rows = []
    for n in sorted(p1["n"].unique()):
        row = [str(n)]
        for p in T_COLS:
            g = cell(p1, n=n, threads=p)
            row.append("&ndash;" if g.empty else f"{one(g, 'speedup'):.2f} / {one(g, 'efficiency'):.2f}")
        rows.append(row)
    open(os.path.join(args.gendir, "table_speedup.html"), "w", encoding="utf8").write(html_table(
        ["n"] + [f"p={p}" for p in T_COLS], rows,
        "Table 4. Speed-up S(p) / efficiency E(p) of the scalar build (P1), from median times. p = 1 shows the OpenMP overhead "
        "(S(1) &lt; 1 means the parallel version with one thread is slower than the sequential one).", "small"))

    rows = []
    for n in (1024, 2048, 4096):
        for p in (8, 12, 16, 20, 24, 32, 48):
            a, b = cell(p1, n=n, threads=p), cell(p2, n=n, threads=p)
            if a.empty or b.empty:
                continue
            rows.append([str(n), str(p), f"{one(a, 'speedup'):.2f}", f"{one(b, 'speedup'):.2f}",
                         f"{one(a) / one(b):.2f}&times;"])
    open(os.path.join(args.gendir, "table_schedule.html"), "w", encoding="utf8").write(html_table(
        ["n", "p", "S static", "S dynamic,1", "T static / T dynamic"], rows,
        "Table 5. Static versus dynamic scheduling (scalar build, P-cores first; P1 versus P2).", "small"))

    rows = []
    for p in (1, 2, 4, 8, 12, 16, 24):
        cells_ = [str(p)]
        for df in (cell(p1, n=2048), cell(p2, n=2048), cell(p3, n=2048), cell(p4, n=2048)):
            g = df[df["threads"] == p]
            cells_.append("&ndash;" if g.empty else f"{fmt_t(one(g))} ({one(g, 'speedup'):.2f})")
        rows.append(cells_)
    open(os.path.join(args.gendir, "table_placement.html"), "w", encoding="utf8").write(html_table(
        ["p", "P-cores first, static (P1)", "P-cores first, dynamic (P2)", "not pinned, static (P3)", "E-cores first, static (P4)"],
        rows, "Table 6. n&nbsp;=&nbsp;2048, scalar build: median time and, in parentheses, speed-up over the sequential baseline "
              "on a P-core (P3 uses its own unpinned sequential run). P4 with p&nbsp;&le;&nbsp;8 runs on E-cores only.", "small"))

    rows = []
    for n in sorted(seqv["n"].unique()):
        a, b = cell(seq1, n=n), cell(seqv, n=n)
        best_a = cell(p1, n=n).sort_values("median_s").head(1)
        best_b = cell(v1, n=n).sort_values("median_s").head(1)
        if a.empty or b.empty or best_a.empty or best_b.empty:
            continue
        rows.append([str(n), fmt_t(one(a)), fmt_t(one(b)), f"{one(a) / one(b):.2f}&times;",
                     f"{fmt_t(one(best_a))} (p={int(best_a['threads'].iloc[0])})",
                     f"{fmt_t(one(best_b))} (p={int(best_b['threads'].iloc[0])})",
                     f"{one(best_b, 'speedup'):.1f}&times;", f"{one(a) / one(best_b):.1f}&times;"])
    open(os.path.join(args.gendir, "table_simd.html"), "w", encoding="utf8").write(html_table(
        ["n", "seq scalar", "seq AVX2", "SIMD gain (1 thread)", "best scalar parallel", "best AVX2 parallel",
         "S thread-only (AVX2)", "combined vs seq scalar"], rows,
        "Table 7. Vectorisation extension (V1 versus P1). The SIMD gain compares the two builds at one thread; "
        "&lsquo;S thread-only&rsquo; is the speed-up of the best AVX2 parallel run over the AVX2 sequential run, so the two effects are "
        "reported separately; the last column is their combination.", "small"))

    rows = []
    for n in (256, 1024, 2048, 4096):
        row = [str(n), fmt_mem(one(cell(S, experiment="P1_threads", method="seq", n=n), "peak_working_set_kib"))]
        for p in (1, 8, 24, 48):
            row.append(fmt_mem(one(cell(S, experiment="P1_threads", method="par", n=n, threads=p), "peak_working_set_kib")))
        row.append(f"{3 * 8 * n * n / 2 ** 20:.1f} MiB")
        rows.append(row)
    open(os.path.join(args.gendir, "table_memory.html"), "w", encoding="utf8").write(html_table(
        ["n", "seq", "par p=1", "par p=8", "par p=24", "par p=48", "3 &times; 8n&sup2;"], rows,
        "Table 8. Peak working set of the benchmark process (GetProcessMemoryInfo), scalar build, P1.", "small"))

    # ---------------------------------------------------------------- key numbers
    def sp(df, n, p, col="speedup"):
        return one(cell(df, n=n, threads=p), col)

    for n in (64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096):
        for p in (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48):
            key[f"S_p{p}_n{n}"] = sp(p1, n, p)
            key[f"Sv_p{p}_n{n}"] = sp(v1, n, p)
        key[f"tseq_n{n}"] = one(cell(seq1, n=n))
        key[f"tseqv_n{n}"] = one(cell(seqv, n=n))
        g = cell(p1, n=n)
        if len(g):
            b = g.loc[g["speedup"].idxmax()]
            key[f"best_S_n{n}"], key[f"best_p_n{n}"] = float(b["speedup"]), int(b["threads"])
        g = cell(v1, n=n)
        if len(g):
            b = g.loc[g["speedup"].idxmax()]
            key[f"best_Sv_n{n}"], key[f"best_pv_n{n}"] = float(b["speedup"]), int(b["threads"])
            key[f"best_tv_n{n}"] = float(b["median_s"])
            key[f"combined_n{n}"] = key[f"tseq_n{n}"] / float(g["median_s"].min())
        key[f"simd_seq_gain_n{n}"] = key[f"tseq_n{n}"] / key[f"tseqv_n{n}"]
    for n in (1024, 2048, 4096):
        for p in (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48):
            key[f"S_dyn_p{p}_n{n}"] = sp(p2, n, p)
    for p in (1, 2, 4, 8, 12, 16, 24):
        key[f"S_unpinned_p{p}_n2048"] = sp(p3, 2048, p)
    for p in (1, 2, 4, 8):
        key[f"S_ecores_p{p}_n2048"] = sp(p4, 2048, p)
    key["tseq_unpinned_n2048"] = one(cell(G, experiment="P3_affinity", method="seq", n=2048))
    key["tseq_ecore_n2048"] = one(cell(G, experiment="P4_ecores", method="seq", n=2048))
    key["ecore_speed_ratio_n2048"] = key["tseq_n2048"] / key["tseq_ecore_n2048"]
    for n in (1024, 2048, 4096):
        key[f"ijk_n{n}"] = one(cell(ijk1, n=n))
        for p in (1, 8, 16, 24):
            key[f"bstream_gbs_p{p}_n{n}"] = 8.0 * n ** 3 / one(cell(p1, n=n, threads=p)) / 1e9
            key[f"bstream_gbs_avx2_p{p}_n{n}"] = 8.0 * n ** 3 / one(cell(v1, n=n, threads=p)) / 1e9
    key["dram_peak_gbs"] = DRAM_PEAK_GBS
    key["median_rel_iqr"] = float(G["rel_iqr"].median())
    key["max_rel_iqr"] = float(G["rel_iqr"].max())
    key["p95_rel_iqr"] = float(G["rel_iqr"].quantile(0.95))
    key["n_configurations"] = int(len(S))
    key["n_ok"] = int((S["status"] == "OK").sum())
    key["n_samples_measured"] = int((s["phase"] == "measured").sum())
    key["status_counts"] = S["status"].value_counts().to_dict()
    key["largest_ok_n"] = int(G["n"].max())
    for n in (256, 1024, 4096):
        key[f"ws_seq_n{n}_mib"] = one(cell(S, experiment="P1_threads", method="seq", n=n), "peak_working_set_kib") / 1024
        key[f"ws_p24_n{n}_mib"] = one(cell(S, experiment="P1_threads", method="par", n=n, threads=24), "peak_working_set_kib") / 1024
        key[f"ws_p48_n{n}_mib"] = one(cell(S, experiment="P1_threads", method="par", n=n, threads=48), "peak_working_set_kib") / 1024
    non_ok = S[S["status"] != "OK"]
    non_ok.to_csv(os.path.join(args.results, "non_ok_runs.csv"), index=False)
    with open(os.path.join(args.results, "key_numbers.json"), "w") as f:
        json.dump(key, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in key.items() if not k.startswith(("S_p", "Sv_p"))}, indent=1, default=float))


if __name__ == "__main__":
    main()
