#!/usr/bin/env python3
"""Assemble the Assignment 3 report (HTML, figures embedded) and print it to PDF with Chrome/Edge.

Tables come from report/generated/*.html and figures from figures/*.png (both written by
scripts/analyze.py). The narrative quotes numbers from results/key_numbers.json, results/validation.csv,
results/disasm_summary.csv and results/supplementary_checks.csv; numbers written literally in the text were
read from those files and from the generated tables.

Usage: python report/build_report.py [--no-pdf]
"""
import argparse
import base64
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEN = os.path.join(ROOT, "report", "generated")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
OUT_HTML = os.path.join(ROOT, "report", "Assignment03_AlejandroSantana_Report.html")
OUT_PDF = os.path.join(ROOT, "report", "Assignment03_AlejandroSantana_Report.pdf")
REPO_URL = "https://github.com/aimarsantana/BigData-Assignment03-MatMul"

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm;
        @bottom-center { content: counter(page) " / " counter(pages); font: 8pt Georgia, serif; color: #555; } }
body { font: 10pt/1.38 Georgia, 'Times New Roman', serif; color: #111; background: #fff; max-width: 100%; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 4pt; line-height: 1.2; }
.sub { text-align: center; font-size: 11pt; margin: 2pt 0; }
.auth { text-align: center; font-size: 9.5pt; margin: 8pt 0 12pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 4pt; border-bottom: 0.6pt solid #888; padding-bottom: 1pt; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 10pt 0 3pt; break-after: avoid; }
p { margin: 4pt 0; text-align: justify; hyphens: auto; }
.abstract { margin: 8pt 22pt; font-size: 9.5pt; }
table { border-collapse: collapse; margin: 8pt auto; font: 8.5pt/1.25 'Segoe UI', Arial, sans-serif; break-inside: avoid; }
table.small { font-size: 7.3pt; }
caption { caption-side: top; font: 8.5pt/1.3 Georgia, serif; text-align: left; margin-bottom: 3pt; font-weight: normal; }
th, td { border-top: 0.5pt solid #888; border-bottom: 0.5pt solid #888; padding: 2pt 5pt; text-align: right; }
th { background: #f0f0f0; text-align: center; }
td:first-child, th:first-child { text-align: center; }
.iqr { color: #777; font-size: 0.88em; }
figure { margin: 8pt 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.5pt; text-align: left; margin-top: 2pt; }
code { font: 8.8pt Consolas, 'Courier New', monospace; }
ul, ol { margin: 3pt 0 3pt 16pt; padding: 0; } li { margin: 1.5pt 0; text-align: justify; }
.env td { text-align: left; } .env td:first-child { font-weight: bold; }
.ref li { text-align: left; font-size: 8.8pt; }
"""


def fig(name, caption, width="100%"):
    data = base64.b64encode(open(os.path.join(FIG, name + ".png"), "rb").read()).decode()
    return f'<figure><img style="width:{width}" src="data:image/png;base64,{data}"><figcaption>{caption}</figcaption></figure>'


def table(name):
    return open(os.path.join(GEN, name), encoding="utf8").read()


def supplementary():
    """table of the supplementary checks (median over the 3 processes, and min-max across processes)."""
    path = os.path.join(RES, "supplementary_checks.csv")
    groups = {}
    for r in csv.DictReader(open(path)):
        k = (r["check"], r["build"], r["method"], r["n"], r["threads"], r["where"].split(" (")[0])
        groups.setdefault(k, []).append(float(r["median_s"]) * 1e3)
    rows = "".join(
        f"<tr><td>{c.split('_', 1)[0]}</td><td>{b}</td><td>{m}</td><td>{n}</td><td>{p}</td><td>{w}</td>"
        f"<td>{statistics.median(v):.3f}</td><td>{min(v):.3f} &ndash; {max(v):.3f}</td></tr>"
        for (c, b, m, n, p, w), v in groups.items())
    return ('<table class="small"><caption>Table 9. Supplementary checks (<code>scripts/supplementary_checks.py</code>): '
            'median over 3 fresh processes of the per-process median time (ms), and the range across the processes. '
            'C1: the sequential kernel confined by the OS to one of four P-cores; C2: n&nbsp;=&nbsp;128, seq versus par with one thread; '
            'C3: AVX2 parallel kernel, n&nbsp;=&nbsp;1024.</caption><thead><tr><th>check</th><th>build</th><th>method</th><th>n</th>'
            '<th>p</th><th>where</th><th>median (ms)</th><th>range (ms)</th></tr></thead><tbody>' + rows + "</tbody></table>"), groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    k = json.load(open(os.path.join(RES, "key_numbers.json")))
    val = list(csv.DictReader(open(os.path.join(RES, "validation.csv"), encoding="utf-8-sig")))
    n_val = len(val)
    n_pass = sum(1 for r in val if r["result"] == "PASS")
    n_bit = sum(1 for r in val if r["bitwise_equal_to_seq"] == "True")
    max_err = max(float(r["max_abs_err_vs_numpy"]) for r in val if r["max_abs_err_vs_numpy"] and r["test"].startswith("benchmark"))
    dis = {(r["build"], r["function"]): r for r in csv.DictReader(open(os.path.join(RES, "disasm_summary.csv")))}
    supp_table, supp = supplementary()
    c1 = {(b, w): statistics.median(v) for (c, b, m, n, p, w), v in supp.items() if c == "C1_core_position"}
    c2 = {(b, m): v for (c, b, m, n, p, w), v in supp.items() if c == "C2_repeat_n128"}
    c3 = {int(p): statistics.median(v) for (c, b, m, n, p, w), v in supp.items() if c == "C3_per_thread_rate"}
    c1_spread_avx = (max(v for (b, w), v in c1.items() if b == "avx2") / min(v for (b, w), v in c1.items() if b == "avx2") - 1) * 100
    c1_spread_sc = (max(v for (b, w), v in c1.items() if b == "scalar") / min(v for (b, w), v in c1.items() if b == "scalar") - 1) * 100
    r_e = k["ecore_speed_ratio_n2048"]
    e_p4 = k["S_ecores_p4_n2048"] / 4
    e_p8 = k["S_ecores_p8_n2048"] / 8

    def S(p, n):
        return k[f"S_p{p}_n{n}"]

    def Sv(p, n):
        return k[f"Sv_p{p}_n{n}"]

    def Sd(p, n):
        return k[f"S_dyn_p{p}_n{n}"]

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Assignment 3 - Parallel Matrix Multiplication and Vectorisation</title><style>{CSS}</style></head><body>

<h1>When Do More Threads Help? OpenMP Parallel Matrix Multiplication on a Hybrid P-core/E-core CPU, with an AVX2 Vectorisation Extension</h1>
<div class="sub">Big Data &ndash; Individual Assignment 3: Parallel Matrix Multiplication and Optional Vectorisation</div>
<div class="auth"><b>Aimar Daniel Alejandro Santana</b><br>Grado en Ciencia e Ingenier&iacute;a de Datos, Universidad de Las Palmas de Gran Canaria<br>
Repository: {REPO_URL}<br>November 2026</div>

<div class="abstract"><b>Abstract.</b> We study how the number of threads affects a shared-memory dense binary64 matrix multiplication in C on one
machine with a hybrid CPU (Intel Core i7-13700K: 8 performance cores with 2 hardware threads each and 8 efficiency cores, 24 logical processors).
The sequential baseline is the fastest kernel of Assignment 2 (ikj order, k loop unrolled by 4); the parallel version runs the same row kernel and distributes
the rows of C over OpenMP threads, pinned P-cores first, then E-cores, then the second hardware thread of each P-core. All {k['n_configurations']} configurations
(n&nbsp;=&nbsp;64&ndash;4096, 1&ndash;48 threads, two schedules, three placements, scalar and AVX2 builds) give bit-identical results. With one thread the OpenMP version
costs at most 6&nbsp;% more than the sequential one. On the 8 P-cores the speed-up is nearly linear for 256&nbsp;&le;&nbsp;n&nbsp;&le;&nbsp;1536 (S(8)&nbsp;=&nbsp;6.7&ndash;7.8), lower once B
no longer fits in the 30&nbsp;MB L3 cache (4.2&ndash;4.8 for n&nbsp;&ge;&nbsp;2048), and negligible for n&nbsp;=&nbsp;64. Adding E-cores with a static schedule <em>reduces</em> the speed-up
(n&nbsp;=&nbsp;2048: {S(8, 2048):.2f} with 8 threads, {S(12, 2048):.2f} with 12) because every thread gets the same number of rows and an E-core is {1 / r_e:.1f}&times; slower;
a dynamic schedule removes this drop and reaches {Sd(24, 2048):.1f}&times; with 24 threads (static: {S(24, 2048):.1f}&times;). AVX2 vectorisation, separated from the thread effect, is worth
1.9&ndash;3.1&times; when the data fit in the private caches (n&nbsp;&le;&nbsp;512) but only 1.1&ndash;1.2&times; for n&nbsp;&ge;&nbsp;1024. Memory use does not depend on the thread count
(+&nbsp;&asymp;30&nbsp;KiB per thread). Conclusions are limited to this kernel, compiler and machine.</div>

<h2>1. Introduction and objective</h2>
<p>Assignment 1 compared a basic triple loop in three languages; Assignment 2 showed that a sequential C kernel becomes 16&ndash;24&times; faster through loop reordering
and unrolling, and that it then becomes limited by memory access rather than arithmetic. A modern processor offers two further sources of speed: several cores
(thread-level parallelism) and SIMD units inside each core (data-level parallelism). This study measures what the first one buys for the Assignment 2 kernel,
on a desktop processor whose cores are of two different types, and separates it from the second one, which we evaluate as the optional extension. Matrices are
n&times;n; n is the side length; p is the number of threads.</p>
<p><b>Research question.</b> <i>How does the thread count p affect the runtime, speed-up, efficiency and memory of an OpenMP parallel dense binary64 multiplication
on a hybrid 8&nbsp;P-core + 8&nbsp;E-core CPU, for which matrix sizes does parallel execution help, and how much does AVX2 vectorisation add on top of it?</i></p>
<p><b>Predictions</b> (formulated before the main runs; a single short probe at n&nbsp;=&nbsp;2048 with 1, 8, 16 and 24 threads, used to choose the experiment list, had already shown that 16 threads were slower than 8, which motivated H3):</p>
<ol>
<li><b>H1.</b> On P-cores only (p&nbsp;&le;&nbsp;8) the speed-up is close to p (E(p)&nbsp;&ge;&nbsp;0.8) for medium sizes, because the rows of C are independent and the parallel overhead is O(p) against O(n&sup3;) work.</li>
<li><b>H2.</b> For small matrices (n&nbsp;&le;&nbsp;128) the cost of starting and synchronising the threads is comparable to the work, so the speed-up stays low and falls with more threads.</li>
<li><b>H3.</b> With a static schedule, adding the slower E-cores (8&nbsp;&lt;&nbsp;p&nbsp;&le;&nbsp;16) lowers the speed-up, because the slowest thread determines the time; a dynamic schedule recovers it.</li>
<li><b>H4.</b> When B no longer fits in the last-level cache (n&nbsp;&ge;&nbsp;2048), the efficiency falls even on P-cores, because the threads share the cache and memory bandwidth.</li>
<li><b>H5.</b> AVX2 (4 doubles per instruction) gives a large gain (2&ndash;4&times;) only while the data are cache resident, and the gain shrinks when the kernel is bandwidth limited, including when many threads run.</li>
</ol>

<h2>2. Background</h2>
<p><b>Shared-memory parallelism.</b> In C=AB every row of C depends only on one row of A and the whole of B, so the rows can be computed concurrently without
communication; only the read-only operands are shared. OpenMP [1, 2] expresses this with <code>#pragma omp parallel for</code>: a team of p threads is created (and kept in a pool),
and the iterations are divided by a <i>schedule</i>. <code>static</code> gives each thread one contiguous block of about n/p iterations; <code>dynamic,1</code> hands out
one iteration at a time to whichever thread is free, adding a small synchronisation cost per iteration but balancing unequal speeds. With T<sub>seq</sub> the
sequential time and T<sub>p</sub> the time with p threads, the speed-up is S(p)&nbsp;=&nbsp;T<sub>seq</sub>/T<sub>p</sub> and the efficiency E(p)&nbsp;=&nbsp;S(p)/p; Amdahl&rsquo;s law [3]
bounds S(p) by the serial fraction of the work, which here is tiny (thread start-up, zeroing is parallel too).</p>
<p><b>Hardware limits.</b> Parallel threads share the last-level cache and the memory bandwidth. The roofline model [4] bounds the attainable throughput by
min(peak compute, bandwidth&nbsp;&times;&nbsp;arithmetic intensity). The ikj/unroll4 kernel reads all of B (8n&sup2; bytes) for every row of C, i.e. 8n&sup3; bytes per product
for 2n&sup3; flops (0.25 flop/byte), so it is bandwidth-bound unless B is reused from cache [5, 6]. <b>SMT</b> (Hyper-Threading) lets two threads share one core&rsquo;s
execution units, which helps mainly when a thread stalls on memory [5]. <b>Hybrid CPUs</b> such as Intel&rsquo;s 13th generation combine large performance cores with
smaller efficiency cores; Windows reports them through the <code>EfficiencyClass</code> of each core [7, 8]. Equal work per thread then means unequal finishing times.
<b>SIMD vectorisation</b> applies one instruction to several data: an AVX2 <code>vmulpd</code> on a 256-bit ymm register processes 4 doubles, against 1 for scalar SSE2
<code>mulsd</code> [8]. A loop over j in C[i][j]&nbsp;+=&nbsp;a&middot;B[k][j] vectorises without changing the order of the additions of any C[i][j], provided no fused multiply-add (FMA) is generated.</p>

<h2>3. Methodology</h2>
<h3>3.1 Implementations</h3>
<table class="env"><caption>Table 1. Methods (all in <code>src/c/matmul3.c</code>, C11, binary64, own code; no numerical library).</caption>
<thead><tr><th>Method</th><th>Description</th></tr></thead><tbody>
<tr><td>seq</td><td><b>Sequential baseline</b>: the <code>unroll4</code> kernel of Assignment 2 (ikj order, k loop unrolled by 4, row-pointer storage <code>double**</code>), moved unchanged into a function <code>row_unroll4</code> that computes one row of C (including zeroing it).</td></tr>
<tr><td>par</td><td><b>Parallel version</b>: <code>#pragma omp parallel for schedule(runtime)</code> over the rows, each iteration calls the same <code>row_unroll4</code>. Schedule <code>static</code> (default) or <code>dynamic,1</code>. Threads write disjoint rows of C; A and B are read-only; there is no lock, atomic or reduction in the kernel.</td></tr>
<tr><td>ijk</td><td>Basic triple loop of Assignment 1 (context only, run up to n&nbsp;=&nbsp;2048; it is not the baseline of S(p)).</td></tr></tbody></table>
<p>Both seq and par use the same algorithm, numeric type, storage layout, input files and timing boundary; the only difference is the distribution of rows over threads.
The OpenMP team is created, sized (<code>omp_set_dynamic(0)</code>, <code>omp_set_num_threads(p)</code>) and pinned before the first warm-up, outside the timer, and the harness removes all
<code>OMP_*</code> environment variables, so the actual thread count is the requested one (checked in every run: {k['threads_actual_mismatch']} mismatches). No library creates internal threads in any timed region.</p>
<p><b>Two builds (vectorisation extension).</b> The same source is compiled twice with gcc 16.1.0: <i>scalar</i> (<code>-O2 -fopenmp -fno-tree-vectorize</code>) and <i>AVX2</i> (<code>-O2 -fopenmp -mavx2 -ffp-contract=off</code>, plus
<code>#pragma omp simd</code> on the two inner j loops). Evidence that vectorisation occurs only in the AVX2 build: (i) the gcc vectoriser report (<code>-fopt-info-vec-optimized</code>) lists the j loops as
&ldquo;vectorized using 32 byte vectors&rdquo; for AVX2 and nothing for the scalar build; (ii) in the disassembly (<code>objdump -d</code>, <code>results/disasm_summary.csv</code>) the functions
<code>matmul_seq</code> and the OpenMP body <code>matmul_par._omp_fn.0</code> contain {dis[('scalar', 'matmul_seq')]['scalar_sse2']} scalar <code>mulsd/addsd</code> and no packed instruction in the scalar build, and
{dis[('avx2', 'matmul_seq')]['packed_avx_ymm']} <code>vmulpd/vaddpd</code> on ymm registers (main loop), {dis[('avx2', 'matmul_seq')]['packed_avx_xmm']} on xmm (epilogue) and {dis[('avx2', 'matmul_seq')]['scalar_avx']} scalar ones (remainder) in the AVX2 build;
(iii) neither build contains an FMA instruction ({dis[('avx2', 'matmul_seq')]['fma']}), so the arithmetic, and the results, are identical. The two builds are compared at equal p, schedule and placement, so the SIMD effect is separated from the thread effect.</p>

<h3>3.2 Inputs</h3>
<p>The dense matrices of Assignments 1 and 2: NumPy PCG64 (<code>default_rng</code>), seeds 20261012 (A) and 20261013 (B), U[&minus;1,1), binary64, little-endian row-major files,
n&nbsp;&isin;&nbsp;{{64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096}}. The SHA-256 of every regenerated file equals the manifest of Assignment 2, and that of Assignment 1 for the sizes it contains (<code>data/inputs/manifest.json</code>).
Every method reads the same files.</p>

<h3>3.3 Environment and thread placement</h3>
<table class="env"><caption>Table 2. Measurement environment (<code>results/environment.txt</code>, <code>results/topology.csv</code>).</caption><tbody>
<tr><td>CPU</td><td>Intel Core i7-13700K (desktop): 8 P-cores &times; 2 hardware threads + 8 E-cores &times; 1 = 16 physical cores, 24 logical processors; L2 2&nbsp;MB per P-core and 4&nbsp;MB per cluster of 4 E-cores (24&nbsp;MB total reported); L3 30&nbsp;MB shared. Turbo and frequency not controlled.</td></tr>
<tr><td>Memory / OS</td><td>31.7&nbsp;GB RAM (2&times;16&nbsp;GB DDR5-5600, theoretical peak {k['dram_peak_gbs']:.1f}&nbsp;GB/s); Windows 11 Home build 26200; &ldquo;Balanced&rdquo; power plan; process priority HIGH.</td></tr>
<tr><td>Toolchain</td><td>gcc 16.1.0 (WinLibs MinGW-w64 UCRT, POSIX threads) with libgomp (<code>_OPENMP</code>&nbsp;=&nbsp;202111); Python 3.11.9, NumPy 2.4.6, pandas 3.1.1, matplotlib 3.11.2 (inputs, validation, analysis only).</td></tr>
<tr><td>Placement</td><td>Logical processors 0&ndash;15 are the 8 P-cores (pairs 0/1, 2/3, &hellip;), 16&ndash;23 the E-cores. Policy <i>pcores</i>: thread t &rarr; slot t mod 24 of the order 0,2,&hellip;,14 (P-cores) | 16&ndash;23 (E-cores) | 1,3,&hellip;,15 (second SMT threads); so p&nbsp;&le;&nbsp;8 uses P-cores only, p&nbsp;=&nbsp;10&ndash;16 adds E-cores, p&nbsp;=&nbsp;20&ndash;24 adds SMT, p&nbsp;=&nbsp;32,&nbsp;48 oversubscribes. Policy <i>ecores</i>: E-cores first. Policy <i>none</i>: no affinity. Each thread pins itself with <code>SetThreadAffinityMask</code>; at the end of each run every thread reports its processor: {k['pinned_runs_placement_ok']} of {k['pinned_runs']} pinned runs ran exactly where intended. The sequential baseline runs on the same logical processor as thread 0 (logical 0).</td></tr>
</tbody></table>
<p>Experiments (<code>config/benchmark_config.json</code>): <b>P1</b> scalar build, all sizes, p&nbsp;&isin;&nbsp;{{1,2,4,6,8,10,12,16,20,24,32,48}}, static, pcores (+ seq, + ijk for n&nbsp;&le;&nbsp;2048); <b>V1</b> the same grid with the AVX2 build;
<b>P2</b> as P1 with <code>dynamic,1</code> for n&nbsp;&isin;&nbsp;{{1024, 2048, 4096}}; <b>P3</b> n&nbsp;=&nbsp;2048 without pinning; <b>P4</b> n&nbsp;=&nbsp;2048 on E-cores first (p&nbsp;&le;&nbsp;8). For a given n all configurations of an experiment run back to back, so T<sub>seq</sub> and T<sub>p</sub> come from the same session.</p>

<h3>3.4 Correctness and safe access to shared data</h3>
<p><code>tests/validate.py</code> ran {n_val} checks before benchmarking (<b>{n_pass} pass, {n_val - n_pass} fail</b>): a hand-computed 3&times;3 product, identity (left and right) and zero matrices, sizes 1, 2, 3, 5, 7, 33, 100, 127, 130
(not multiples of the unroll factor or of p, and fewer rows than threads), 1&ndash;48 threads, both schedules, all placements, both builds, and the benchmark inputs at n&nbsp;=&nbsp;512 and 1024. The reference is NumPy (OpenBLAS), used only for checking, with
tolerance |&Delta;C|&nbsp;&le;&nbsp;10<sup>&minus;9</sup>&nbsp;+&nbsp;10<sup>&minus;9</sup>|C|, above the rounding bound n&sup2;u&nbsp;&asymp;&nbsp;1.2&times;10<sup>&minus;10</sup> for n&nbsp;=&nbsp;1024 [9] (largest error on the benchmark inputs {max_err:.1e}). The stronger test is that
every configuration must equal the scalar sequential result <em>bit for bit</em> ({n_bit} comparisons, all equal): a data race or a wrong row partition would break it. As a race check, the parallel kernel was run 20 times in one process with
16 (static), 24 and 48 threads (dynamic) in both builds, and all repetitions gave identical checksums. During the benchmarks, the checksum of all {k['checksum_samples']} measured samples of the {k['checksum_configurations']} configurations was compared per n: {k['checksum_disagreements']} disagreements.</p>

<h3>3.5 Measurement protocol</h3>
<ul>
<li><b>Timer boundary (kernel only).</b> The multiplication including zeroing of C (row by row, inside the parallel loop for par). Input loading, allocation of C, creation and pinning of the thread team, and the result checksum are outside the timer. Timer: <code>QueryPerformanceCounter</code> (monotonic).</li>
<li><b>Repetitions.</b> One fresh process per configuration; warm-up 2 runs (n&nbsp;&le;&nbsp;512) or 1 run (excluded from statistics, kept in the raw data); 10 measured repetitions for n&nbsp;&le;&nbsp;512, 7 for n&nbsp;&le;&nbsp;2048, 5 above. Short kernels are batched automatically so that a sample lasts &asymp;20&nbsp;ms; the time per multiplication is reported.</li>
<li><b>Statistics.</b> Median and interquartile range (IQR). <b>S(p) and E(p) are computed from median times</b>; T<sub>seq</sub> is the seq run of the same build (P2 and P4 use the seq run of P1, measured in the same session just before). All {k['n_samples_measured']} measured samples are in <code>results/raw_samples.csv</code>; none was removed.</li>
<li><b>Budgets.</b> 120&nbsp;s per sample and 8&nbsp;GB peak working set, with a stop rule per series. All {k['n_ok']} of {k['n_configurations']} configurations completed within the budget; the largest tested size is n&nbsp;=&nbsp;{k['largest_ok_n']} (also the largest attempted), chosen so that the whole campaign fits in about 45 minutes.</li>
<li><b>Memory metric.</b> Peak working set (<code>PeakWorkingSetSize</code>) of the process, read by the process itself through <code>GetProcessMemoryInfo</code> [10]; it includes the runtime, the loaded inputs, the thread stacks and all temporaries.</li></ul>

<h2>4. Results</h2>
<p>The median relative IQR over all configurations is {100 * k['median_rel_iqr']:.1f}&nbsp;% (95th percentile {100 * k['p95_rel_iqr']:.0f}&nbsp;%); the largest spreads (up to {100 * k['max_rel_iqr']:.0f}&nbsp;%) occur in oversubscribed runs (p&nbsp;=&nbsp;32, 48), where threads are time-sliced, and at n&nbsp;&le;&nbsp;128.</p>
<h3>4.1 Runtime, speed-up and efficiency (P1, scalar build, static schedule)</h3>
{table('table_time.html')}
{fig('fig_runtime_threads', 'Figure 1. Time per multiplication versus threads (median, bars = IQR, mostly smaller than the markers). The short dash at the left of each curve is the sequential baseline. Shaded regions: P-cores only, P-cores + E-cores, + SMT threads, oversubscribed.', '82%')}
{fig('fig_speedup_efficiency', 'Figure 2. Speed-up S(p) and efficiency E(p) of the scalar build with the static schedule (from medians).')}
{table('table_speedup.html')}
<p><b>Overhead.</b> With one thread the OpenMP version takes 0.94&ndash;1.00 times the sequential time (S(1)&nbsp;=&nbsp;{S(1, 64):.2f} at n&nbsp;=&nbsp;64, &ge;&nbsp;0.96 from n&nbsp;=&nbsp;128): the cost of the parallel loop with one thread is a few microseconds per product.
<b>P-cores (p&nbsp;&le;&nbsp;8).</b> For 256&nbsp;&le;&nbsp;n&nbsp;&le;&nbsp;1536, S(8)&nbsp;=&nbsp;{S(8, 256):.2f}&ndash;{S(8, 1024):.2f} and E(8)&nbsp;=&nbsp;0.83&ndash;0.97; for n&nbsp;&ge;&nbsp;2048 it falls to {S(8, 2048):.2f} (2048), {S(8, 3072):.2f} (3072) and {S(8, 4096):.2f} (4096), with an efficiency of at most 0.82 already at p&nbsp;=&nbsp;2.
<b>E-cores (p&nbsp;=&nbsp;10&ndash;16).</b> At every size n&nbsp;&ge;&nbsp;128 the speed-up <em>drops</em> when the first E-cores join (n&nbsp;=&nbsp;1024: {S(8, 1024):.2f} &rarr; {S(10, 1024):.2f} at p&nbsp;=&nbsp;10) and recovers only partly by p&nbsp;=&nbsp;16.
<b>SMT (p&nbsp;=&nbsp;20&ndash;24)</b> raises S again ({S(24, 1024):.2f} at n&nbsp;=&nbsp;1024). <b>Oversubscription:</b> p&nbsp;=&nbsp;32 gives the best static result for n&nbsp;&ge;&nbsp;1024 (e.g. {S(32, 2048):.2f} at n&nbsp;=&nbsp;2048); p&nbsp;=&nbsp;48 is worse again.
For context, the basic ijk loop of Assignment 1 takes {k['ijk_n2048']:.1f}&nbsp;s at n&nbsp;=&nbsp;2048, {k['ijk_n2048'] / k['tseq_n2048']:.0f}&times; the sequential baseline.</p>

<h3>4.2 When does parallel execution help? (matrix size)</h3>
{fig('fig_speedup_vs_n', 'Figure 3. Speed-up versus n for selected thread counts (scalar, static). Below the horizontal line the parallel version is slower than the sequential one.', '72%')}
<p>At n&nbsp;=&nbsp;64 (37&nbsp;&micro;s sequential) the best speed-up is {k['best_S_n64']:.2f} (p&nbsp;=&nbsp;{k['best_p_n64']}) and from p&nbsp;=&nbsp;10 the parallel version is slower than the sequential one (S(24)&nbsp;=&nbsp;{S(24, 64):.2f}). At n&nbsp;=&nbsp;128 (0.29&nbsp;ms) 8 P-cores give {S(8, 128):.2f}, but more
threads again lose ({S(24, 128):.2f} with 24). From n&nbsp;=&nbsp;256 on, every p&nbsp;&ge;&nbsp;2 is faster than the baseline. The largest speed-ups (10&ndash;11) are reached for n&nbsp;=&nbsp;1024&ndash;1536, and they decrease for larger matrices.</p>

<h3>4.3 Static versus dynamic scheduling (P2)</h3>
{fig('fig_schedule', 'Figure 4. Speed-up with the static and the dynamic,1 schedule (P-cores first, scalar build); bars = speed-up computed from the quartiles of T<sub>p</sub>.')}
{table('table_schedule.html')}
<p>Up to p&nbsp;=&nbsp;8 the two schedules differ by at most 9&nbsp;% (within 3&nbsp;% at p&nbsp;=&nbsp;8; n&nbsp;=&nbsp;1024&ndash;4096). Once E-cores participate (p&nbsp;=&nbsp;10&ndash;24), dynamic scheduling is 1.07&ndash;1.6&times; faster than static at n&nbsp;=&nbsp;1024 and 1.3&ndash;2.1&times; at n&nbsp;&ge;&nbsp;2048 (n&nbsp;=&nbsp;2048, p&nbsp;=&nbsp;12: S&nbsp;=&nbsp;{Sd(12, 2048):.2f} versus {S(12, 2048):.2f}), and the
speed-up then grows monotonically up to p&nbsp;=&nbsp;24: {Sd(24, 1024):.1f} (n&nbsp;=&nbsp;1024), {Sd(24, 2048):.1f} (2048) and {Sd(24, 4096):.1f} (4096). With dynamic scheduling oversubscription brings nothing (n&nbsp;=&nbsp;4096: {Sd(32, 4096):.1f} at p&nbsp;=&nbsp;32, {Sd(48, 4096):.1f} at 48).</p>

<h3>4.4 Thread placement and core types (P3, P4; n&nbsp;=&nbsp;2048)</h3>
{table('table_placement.html')}
{fig('fig_placement', 'Figure 5. n&nbsp;=&nbsp;2048: speed-up over the P-core sequential baseline for the placement and schedule variants.', '72%')}
<p>The sequential kernel takes {k['tseq_n2048']:.2f}&nbsp;s on a P-core and {k['tseq_ecore_n2048']:.2f}&nbsp;s on an E-core, i.e. one E-core delivers {r_e:.2f} of a P-core. E-cores scale poorly inside one cluster: 2 E-cores give {k['S_ecores_p2_n2048']:.2f} and 4 E-cores (one cluster sharing a 4&nbsp;MB L2)
only {k['S_ecores_p4_n2048']:.2f}; 8 E-cores (two clusters) give {k['S_ecores_p8_n2048']:.2f}, so per E-core {e_p4:.2f} and {e_p8:.2f} P-core equivalents at p&nbsp;=&nbsp;4 and 8. Without pinning (P3), the static schedule behaves like the dynamic one up to 16 threads (S(16)&nbsp;=&nbsp;{k['S_unpinned_p16_n2048']:.2f} versus {Sd(16, 2048):.2f}) and is
better than the pinned static runs from p&nbsp;=&nbsp;8, but it falls behind the pinned dynamic version at p&nbsp;=&nbsp;24 ({k['S_unpinned_p24_n2048']:.2f} versus {Sd(24, 2048):.2f}).</p>

<h3>4.5 Vectorisation extension (V1 versus P1)</h3>
{table('table_simd.html')}
{fig('fig_simd', 'Figure 6. Left: throughput 2n<sup>3</sup>/T of the scalar (dashed) and AVX2 (solid) builds. Right: vectorisation gain T<sub>scalar</sub>/T<sub>AVX2</sub> at the same thread count, schedule and placement.')}
<p>With one thread, AVX2 is 1.9&ndash;3.1&times; faster for n&nbsp;=&nbsp;64&ndash;512 ({k['simd_seq_gain_n256']:.2f}&times; at n&nbsp;=&nbsp;256), but only {k['simd_seq_gain_n1024']:.2f}&times; at 1024 and 1.11&ndash;1.21&times; for n&nbsp;&ge;&nbsp;1536, far from the factor 4 of the vector width.
The thread speed-ups of the AVX2 build (relative to the AVX2 baseline) follow the same pattern as the scalar ones (static drop with E-cores, best at p&nbsp;=&nbsp;32), reaching {k['best_Sv_n1024']:.1f}&times; at n&nbsp;=&nbsp;1024. At equal p, the SIMD gain at n&nbsp;=&nbsp;256 falls from 3.0&times; (p&nbsp;=&nbsp;1) to {k['tseq_n256'] / k['tseqv_n256'] * Sv(24, 256) / S(24, 256):.2f}&times; (p&nbsp;=&nbsp;24),
while for n&nbsp;&ge;&nbsp;2048 it stays between 0.9 and 1.6&times;. The best combined result is {k['combined_n512']:.1f}&times; over the scalar sequential baseline at n&nbsp;=&nbsp;512 and {k['combined_n1024']:.1f}&times; at n&nbsp;=&nbsp;1024 (11.2&nbsp;ms, {2 * 1024 ** 3 / k['best_tv_n1024'] / 1e9:.0f}&nbsp;GFLOP/s).</p>

<h3>4.6 Memory and bandwidth</h3>
{table('table_memory.html')}
{fig('fig_bandwidth', 'Figure 7. Rate at which the kernel streams B, 8n<sup>3</sup> bytes (one read of B per row of C) divided by the median time. Values above the theoretical DRAM bandwidth (red) are only possible if part of B is served by the caches.', '76%')}
<p><b>Memory.</b> The peak working set equals the analytic size of A, B and C (3&middot;8n&sup2; bytes) plus a constant of about 5&nbsp;MiB, and it is independent of the thread count: from the sequential run to 48 threads it grows by {k['ws_p48_n4096_mib'] - k['ws_seq_n4096_mib']:.1f}&nbsp;MiB at n&nbsp;=&nbsp;4096 (about 30&nbsp;KiB per thread: touched stack pages and runtime structures).
The parallel version allocates no per-thread copy of any matrix. <b>Bandwidth.</b> The single-thread kernel streams B at {k['bstream_gbs_p1_n1024']:.0f}&nbsp;GB/s (n&nbsp;=&nbsp;1024, B&nbsp;=&nbsp;8&nbsp;MiB, L3 resident) and {k['bstream_gbs_p1_n4096']:.0f}&nbsp;GB/s (n&nbsp;=&nbsp;4096, B&nbsp;=&nbsp;128&nbsp;MiB). With 8 P-core threads the aggregate
rate is {k['bstream_gbs_p8_n1024']:.0f}&nbsp;GB/s at n&nbsp;=&nbsp;1024 but {k['bstream_gbs_p8_n4096']:.0f}&nbsp;GB/s at n&nbsp;=&nbsp;4096, already above the {k['dram_peak_gbs']:.1f}&nbsp;GB/s DRAM peak.</p>

<h3>4.7 Unusual results and supplementary checks</h3>
{supp_table}
<p>Two results needed checking rather than trust: <b>superlinear speed-up</b> of the AVX2 build at n&nbsp;=&nbsp;1024 (S(2)&nbsp;=&nbsp;{Sv(2, 1024):.2f}, S(4)&nbsp;=&nbsp;{Sv(4, 1024):.2f}, E&nbsp;&gt;&nbsp;1; the scalar build shows only {S(2, 1024):.2f}), and <b>S(1)&nbsp;=&nbsp;{Sv(1, 128):.2f}</b> of the AVX2 build at n&nbsp;=&nbsp;128.
(C1) The baseline runs on logical processor 0, which serves more interrupts; confining the sequential process to P-cores 0, 2, 4 or 6 changes its time by at most {c1_spread_avx:.1f}&nbsp;% (AVX2) and {c1_spread_sc:.1f}&nbsp;% (scalar), so the core choice does not explain a 22&nbsp;% excess.
(C3) The superlinear speed-up is reproducible in fresh processes (median S(2)&nbsp;=&nbsp;{c3[1] / c3[2]:.2f}, S(4)&nbsp;=&nbsp;{c3[1] / c3[4]:.2f} relative to p&nbsp;=&nbsp;1).
(C2) At n&nbsp;=&nbsp;128 the AVX2 times differ between processes by up to {max(max(v) / min(v) for (b, m), v in c2.items() if b == 'avx2'):.1f}&times; for <em>both</em> seq and par (seq {min(c2[('avx2', 'seq')]):.3f}&ndash;{max(c2[('avx2', 'seq')]):.3f}&nbsp;ms), while they are stable inside a process; the scalar sequential runs vary by less than 1&nbsp;%.
The n&nbsp;=&nbsp;128 AVX2 ratio is therefore a between-process effect (one process per configuration), not an OpenMP effect, and the AVX2 speed-ups at n&nbsp;&le;&nbsp;128 are not reliable.</p>

<h2>5. Discussion</h2>
<p><b>H1 (P-cores) is supported for 256&nbsp;&le;&nbsp;n&nbsp;&le;&nbsp;1536</b> (E(8)&nbsp;=&nbsp;0.83&ndash;0.97) and refuted for n&nbsp;&ge;&nbsp;2048 (E(8)&nbsp;=&nbsp;0.53&ndash;0.59), which is <b>H4</b>. The drop coincides with B (32&nbsp;MiB at n&nbsp;=&nbsp;2048) exceeding the 30&nbsp;MB L3 cache, as in Assignment 2 where blocking started to help at the same point. The bandwidth view supports
a memory explanation: the sequential kernel already streams B at {k['bstream_gbs_p1_n4096']:.0f}&nbsp;GB/s at n&nbsp;=&nbsp;4096, about a third of the DRAM peak, so eight threads that each read B independently would need several times the available bandwidth. The measured aggregate rate is above the DRAM peak, which is only possible if threads reuse lines of B
that another thread has just brought into the shared L3 (with the static schedule all threads traverse B in the same k order, starting at the same time). The kernel is therefore limited by the shared L3 and DRAM, not by arithmetic; tiling B for the caches (Assignment 2) would be the remedy. We did not measure cache or memory counters, so this is an interpretation consistent with the timings and the bound, not a direct observation.</p>
<p><b>H2 (small matrices) is supported.</b> The fork&ndash;join cost of an OpenMP parallel loop is microseconds and grows with p; at n&nbsp;=&nbsp;64 the whole product takes 37&nbsp;&micro;s, so parallel execution barely pays (best 1.18&times;) and loses from 10 threads; at n&nbsp;=&nbsp;128 it pays only up to the 8 P-cores. Parallel execution helps when each thread receives at least a few hundred microseconds of work and the data fit in the caches.</p>
<p><b>H3 (static schedule on a hybrid CPU) is supported, and quantitatively.</b> With the static schedule every thread receives n/p rows and the product ends when the slowest thread ends. If one E-core works at a fraction s of a P-core, the time is that of n/p rows at speed s, so S(p)&nbsp;&asymp;&nbsp;p&middot;s whenever an E-core is in the team. With s&nbsp;=&nbsp;{r_e:.2f} (one E-core alone, measured in P4 at n&nbsp;=&nbsp;2048) the model predicts {10 * r_e:.1f}, {12 * r_e:.1f} and {16 * r_e:.1f} for p&nbsp;=&nbsp;10, 12 and 16; at n&nbsp;=&nbsp;1024 we measured {S(10, 1024):.2f}, {S(12, 1024):.2f} and {S(16, 1024):.2f} (close for 12 and 16; for p&nbsp;=&nbsp;10 the model, with s taken at another size, is 20&nbsp;% low). At n&nbsp;=&nbsp;2048 the E-cores slow down further when a cluster is shared (s&nbsp;=&nbsp;{e_p4:.2f} per E-core with 4 per cluster), and the model gives 12&nbsp;&times;&nbsp;{e_p4:.2f}&nbsp;=&nbsp;{12 * e_p4:.1f} for p&nbsp;=&nbsp;12, against {S(12, 2048):.2f} measured.
Dynamic scheduling lets the P-cores take more rows and removes the drop (Section 4.3). The unpinned runs (P3) behave like the dynamic ones up to p&nbsp;=&nbsp;16, which is consistent with the Windows scheduler moving the threads that are left on E-cores to P-cores that have finished, although we did not trace thread migrations. The &ldquo;best&rdquo; static result at p&nbsp;=&nbsp;32 has the same cause: the placement map shows that
threads 24&ndash;31 are placed on the P-cores, so each P-core hosts three threads and each E-core one; the share of rows per core then approximates the ratio of their speeds. This is an accidental load balance, with a higher run-to-run spread (Section 4), and the dynamic schedule obtains more without oversubscription. <b>SMT</b> adds 6&ndash;64&nbsp;% with the dynamic schedule (n&nbsp;=&nbsp;1024: {Sd(16, 1024):.1f} &rarr; {Sd(24, 1024):.1f}; n&nbsp;=&nbsp;2048: {Sd(16, 2048):.1f} &rarr; {Sd(24, 2048):.1f}), more at the sizes where the kernel waits for memory, consistent with the second hardware thread hiding memory stalls of a bandwidth-bound kernel.</p>
<p><b>H5 (vectorisation) is supported.</b> The gain is 2&ndash;3&times; while A, B and C fit in the private L1/L2 caches (n&nbsp;&le;&nbsp;256&ndash;512: B is 0.5&ndash;2&nbsp;MiB) and 1.1&ndash;1.2&times; once B must come from L3 or memory: SIMD shortens the arithmetic but not the data movement, so the kernel moves from compute-bound to bandwidth-bound and the roofline [4] caps the benefit. For the same reason the SIMD gain shrinks when threads are added at small n (the shared caches and the parallel overhead become dominant),
and the two effects do not multiply: at n&nbsp;=&nbsp;1024, 1.22&times; (SIMD) and {k['best_S_n1024']:.1f}&times; (threads, scalar) combine to {k['combined_n1024']:.1f}&times;, not {1.22 * k['best_S_n1024']:.1f}&times;. The superlinear AVX2 speed-up at n&nbsp;=&nbsp;1024 fits this picture: a single AVX2 thread reaches only {k['bstream_gbs_avx2_p1_n1024']:.0f}&nbsp;GB/s from L3, below what two cores obtain together
(per-core limits on outstanding cache misses would be one explanation); the cause is not the core position (C1) nor noise (C3), but without hardware counters it remains unresolved.</p>
<p><b>Memory.</b> Threads add no data structures; the memory cost of parallelism here is negligible, unlike in a distributed setting (Assignment 4) where operands must be copied.</p>
<p><b>Limitations.</b> (i) One machine, one compiler (gcc 16.1, <code>-O2</code>), one run order, Balanced power plan; turbo frequencies fall when more cores are active, which lowers S(p) and was not controlled. (ii) No hardware performance counters: cache, bandwidth and migration explanations are hypotheses consistent with the data. (iii) One process per configuration;
check C2 shows that between-process variation can exceed within-process variation for small AVX2 kernels, so the reported IQR understates it there. (iv) 5&ndash;7 repetitions for n&nbsp;&ge;&nbsp;1024. (v) The speed-up uses the median of T<sub>seq</sub> and T<sub>p</sub>; their uncertainties are shown only as the quartile range of T<sub>p</sub>. (vi) The kernel is not cache-blocked, so the scaling limit at large n is partly a property of this kernel;
a blocked parallel kernel or a BLAS library would scale differently. (vii) The core-type results are specific to this hybrid processor and to Windows&rsquo; scheduler.</p>

<h2>6. Conclusions</h2>
<ol>
<li>Within the tested scope (i7-13700K, Windows 11, gcc 16.1 <code>-O2</code>, binary64, kernel-only timing), the OpenMP row-parallel version is correct and bit-identical to its sequential counterpart for 1&ndash;48 threads; its overhead with one thread is at most 6&nbsp;%.</li>
<li>Parallel execution helps from n&nbsp;&asymp;&nbsp;128&ndash;256 on; on the 8 P-cores the speed-up is nearly linear while the data fit in the 30&nbsp;MB L3 (up to 7.8 for n&nbsp;=&nbsp;1024) and saturates around 4.2&ndash;4.8 when they do not.</li>
<li>On a hybrid CPU the thread count alone is not the right control variable: with a static schedule, adding E-cores makes the product slower (S&nbsp;&asymp;&nbsp;p&middot;s), while a dynamic schedule turns all 24 logical processors into a speed-up of 11&ndash;12 (n&nbsp;=&nbsp;1024&ndash;2048) and 9.8 (n&nbsp;=&nbsp;4096).</li>
<li>AVX2 vectorisation gives 2&ndash;3&times; for cache-resident matrices and only 1.1&ndash;1.2&times; for larger ones; combined with threads the best result is {k['combined_n512']:.0f}&times; (n&nbsp;=&nbsp;512) over the scalar sequential kernel. Memory does not grow with threads.</li></ol>
<p>Unresolved: the exact cause of the superlinear AVX2 speed-up, the frequency and migration behaviour behind each curve (hardware counters were not available), and how a cache-blocked parallel kernel would scale.</p>

<h2>AI-use disclosure</h2>
<p>In line with the course policy, the AI assistant Claude (Anthropic) was used to plan the experimental design, draft and debug the C code, the benchmark and analysis scripts and the tests, and to draft and revise the English of this report. All numbers in the report were produced by running the code in the repository
(tables and figures are generated by <code>scripts/analyze.py</code>); no result was generated by the AI tool. The author is responsible for understanding the code, checking the measurements and references, and defending the conclusions.</p>

<h2>Reproducibility materials</h2>
<p style="text-align:left">Folder <code>Assignment03_AlejandroSantana_Materials</code> (and the repository {REPO_URL}): <code>src/c/matmul3.c</code>, <code>build.ps1</code> (exact flags of both builds), <code>README.md</code> with build and run commands,
<code>config/benchmark_config.json</code> (sizes, thread counts, schedules, placements, repetitions, budgets), the input generator and manifest, raw samples, per-run logs with the thread placement map (<code>results/</code>), validation output, vectorisation evidence, environment and topology description, the supplementary checks, and the scripts that produce all tables and figures. <code>run_all.ps1</code> rebuilds, validates, measures, analyses and regenerates this report.
Reused: the unroll4 kernel of Assignment 2 (sequential baseline), the ijk loop of Assignment 1, and the input matrices of Assignments 1&ndash;2 (same seeds, verified hashes).</p>

<h2>References</h2>
<ol class="ref">
<li>OpenMP Architecture Review Board, <i>OpenMP Application Programming Interface, Version 5.2</i>, Nov. 2021.</li>
<li>R. Chandra, L. Dagum, D. Kohr, D. Maydan, J. McDonald, and R. Menon, <i>Parallel Programming in OpenMP</i>. San Francisco, CA, USA: Morgan Kaufmann, 2001.</li>
<li>G. M. Amdahl, &ldquo;Validity of the single processor approach to achieving large scale computing capabilities,&rdquo; in <i>Proc. AFIPS Spring Joint Computer Conf.</i>, 1967, pp. 483&ndash;485.</li>
<li>S. Williams, A. Waterman, and D. Patterson, &ldquo;Roofline: An insightful visual performance model for multicore architectures,&rdquo; <i>Commun. ACM</i>, vol. 52, no. 4, pp. 65&ndash;76, 2009.</li>
<li>J. L. Hennessy and D. A. Patterson, <i>Computer Architecture: A Quantitative Approach</i>, 6th ed. Cambridge, MA, USA: Morgan Kaufmann, 2017.</li>
<li>U. Drepper, &ldquo;What every programmer should know about memory,&rdquo; Red Hat, Inc., Tech. Rep., Nov. 2007.</li>
<li>Microsoft, &ldquo;GetLogicalProcessorInformationEx function,&rdquo; &ldquo;PROCESSOR_RELATIONSHIP structure&rdquo; and &ldquo;SetThreadAffinityMask function,&rdquo; Microsoft Learn, Windows API documentation.</li>
<li>Intel Corporation, <i>Intel 64 and IA-32 Architectures Optimization Reference Manual</i>, Santa Clara, CA, USA.</li>
<li>N. J. Higham, <i>Accuracy and Stability of Numerical Algorithms</i>, 2nd ed. Philadelphia, PA, USA: SIAM, 2002.</li>
<li>Microsoft, &ldquo;PROCESS_MEMORY_COUNTERS structure (psapi.h)&rdquo; and &ldquo;GetProcessMemoryInfo function,&rdquo; Microsoft Learn, Windows API documentation.</li>
<li>Free Software Foundation, &ldquo;Options that control optimization&rdquo; (<code>-ftree-vectorize</code>, <code>-ffp-contract</code>) and &ldquo;Developer options&rdquo; (<code>-fopt-info</code>), in <i>Using the GNU Compiler Collection (GCC)</i>.</li>
<li>K. Goto and R. A. van de Geijn, &ldquo;Anatomy of high-performance matrix multiplication,&rdquo; <i>ACM Trans. Math. Softw.</i>, vol. 34, no. 3, Art. 12, 2008.</li></ol>
</body></html>"""
    open(OUT_HTML, "w", encoding="utf8").write(html)
    print("wrote", OUT_HTML)
    if args.no_pdf:
        return
    browsers = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
    exe = next((b for b in browsers if os.path.exists(b)), shutil.which("chrome") or shutil.which("msedge"))
    if not exe:
        sys.exit("No Chrome/Edge found: open the HTML file and print it to PDF manually.")
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}",
                    "file:///" + OUT_HTML.replace("\\", "/")], check=True, timeout=180)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
