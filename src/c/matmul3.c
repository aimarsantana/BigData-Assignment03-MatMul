/*
 * Assignment 3 - shared-memory parallel dense multiplication (OpenMP) and optional SIMD
 * vectorisation, C, float64.
 *
 * Methods (all use the same storage: contiguous row-major block + row pointers, as in A1/A2):
 *   ijk  basic triple loop of Assignment 1, sequential. Context only (much slower algorithm);
 *        it is NOT the baseline of the speed-up.
 *   seq  sequential baseline = the 'unroll4' kernel of Assignment 2 (ikj order, k unrolled by 4).
 *   par  the SAME row kernel as seq, rows of C distributed over OpenMP threads
 *        (#pragma omp parallel for, schedule chosen at run time: static or dynamic,1).
 * seq and par call one shared function (row_unroll4) per row of C, so the arithmetic, the loop
 * order and the summation order of every C[i][j] are identical; only the distribution of rows
 * over threads differs. Each thread writes only its own rows of C; A and B are read-only, so
 * there is no shared write and no synchronisation inside the kernel.
 *
 * Vectorisation (optional extension) is a BUILD variant of this same file (see build.ps1):
 *   scalar build : gcc -O2 -fopenmp -fno-tree-vectorize            (SSE2 scalar mulsd/addsd)
 *   avx2 build   : gcc -O2 -fopenmp -mavx2 -ffp-contract=off -DUSE_SIMD
 *                  (#pragma omp simd on the inner j loops -> 4-wide vmulpd/vaddpd on ymm registers)
 * No FMA contraction is allowed in either build, and vectorising the j loop does not reorder the
 * additions of any C[i][j], so both builds are expected to give bit-identical results.
 *
 * Thread placement (Windows, hybrid CPU): the logical processors are read with
 * GetLogicalProcessorInformationEx and ordered by a placement policy:
 *   pcores  one thread per performance core first, then the efficiency cores, then the second
 *           SMT thread of each performance core (thread t -> order[t mod #logical])
 *   ecores  efficiency cores first, then performance cores, then SMT siblings
 *   none    no affinity: the operating system schedules the threads
 * Each OpenMP thread pins itself (SetThreadAffinityMask) in an untimed parallel region; after the
 * measurements every thread reports the processor it runs on (INFO ... placement lines).
 *
 * Usage:
 *   matmul3 dense <ijk|seq|par> A.bin B.bin n warmup reps batch threads schedule placement [C_out.bin]
 *     schedule: static | dynamic (par only)   placement: pcores | ecores | none
 *     batch = multiplications per timed sample; 0 = automatic (a sample lasts ~20 ms)
 *   matmul3 topology                 print the logical processor layout
 *
 * Timer: QueryPerformanceCounter (monotonic). Timed region: the kernel, including zeroing of C
 * (row by row, inside the parallel loop for par); C is allocated outside the timer; the thread team
 * is created and pinned before the first warm-up run.
 * Memory: peak working set (PeakWorkingSetSize) and peak commit charge of the process (MEM lines).
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_SIMD
#define SIMD_LOOP _Pragma("omp simd")
#define BUILD_NAME "avx2"
#else
#define SIMD_LOOP
#define BUILD_NAME "scalar"
#endif

/* ---------------------------------------------------------------- utilities */

static long long now_ns(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER c;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    long long sec = c.QuadPart / freq.QuadPart, rem = c.QuadPart % freq.QuadPart;
    return sec * 1000000000LL + rem * 1000000000LL / freq.QuadPart;
}

static void die(const char *msg, int code) {
    fprintf(stderr, "%s\n", msg);
    exit(code);
}

static void print_memory(const char *method, long long n) {
    PROCESS_MEMORY_COUNTERS pmc;
    memset(&pmc, 0, sizeof pmc);
    pmc.cb = sizeof pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        printf("MEM,%s,%lld,peak_working_set_kib,%llu\n", method, n,
               (unsigned long long)(pmc.PeakWorkingSetSize / 1024));
        printf("MEM,%s,%lld,peak_commit_kib,%llu\n", method, n,
               (unsigned long long)(pmc.PeakPagefileUsage / 1024));
    } else {
        printf("MEM,%s,%lld,peak_working_set_kib,-1\n", method, n);
    }
}

static void print_build_info(void) {
    printf("INFO,all,build,%s\n", BUILD_NAME);
#ifdef __VERSION__
    printf("INFO,all,compiler,gcc %s\n", __VERSION__);
#endif
#ifdef CFLAGS_STR
    printf("INFO,all,cflags,%s\n", CFLAGS_STR);
#endif
    printf("INFO,all,openmp,%d\n", _OPENMP);
}

/* batch == 0 asks for an automatic batch: the first warm-up run is executed with batch 1 and its
 * duration is used to choose the number of multiplications per sample so that one sample lasts
 * about 20 ms (at most 10000, at least 1). The batch is printed in every SAMPLE line. */
static int auto_batch(long long ns_single) {
    long long b = ns_single > 0 ? 20000000LL / ns_single : 10000;
    return b < 1 ? 1 : (b > 10000 ? 10000 : (int)b);
}

/* ------------------------------------------------------------ processor topology */

#define MAX_CPUS 64

typedef struct {
    int count;                 /* logical processors in the order of the policy */
    int cpu[MAX_CPUS];         /* logical processor number (group 0) */
    char kind[MAX_CPUS];       /* 'P' first SMT thread of a P-core, 'E' E-core, 'S' second SMT thread */
} placement_t;

/* Logical processors of group 0 classified by core type. EfficiencyClass is higher for the
 * performance cores; on a non-hybrid CPU every core has the same class and is treated as 'P'. */
static placement_t build_placement(const char *policy) {
    placement_t pl;
    int pfirst[MAX_CPUS], ecore[MAX_CPUS], sib[MAX_CPUS], np = 0, ne = 0, ns = 0, maxclass = 0;
    DWORD len = 0;
    memset(&pl, 0, sizeof pl);
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    char *buf = malloc(len);
    if (!buf || !GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                  (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len))
        die("GetLogicalProcessorInformationEx failed", 8);
    for (DWORD off = 0; off < len;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
        if (p->Processor.EfficiencyClass > maxclass) maxclass = p->Processor.EfficiencyClass;
        off += p->Size;
    }
    for (DWORD off = 0; off < len;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
        KAFFINITY m = p->Processor.GroupMask[0].Mask;
        int first = 1, perf = (p->Processor.EfficiencyClass == maxclass);
        for (int b = 0; b < MAX_CPUS && b < (int)(8 * sizeof(KAFFINITY)); b++) {
            if (!(m & ((KAFFINITY)1 << b))) continue;
            if (!perf) ecore[ne++] = b;
            else if (first) pfirst[np++] = b;
            else sib[ns++] = b;
            first = 0;
        }
        off += p->Size;
    }
    free(buf);
    int *grp[3]; int cnt[3]; char kd[3];
    if (strcmp(policy, "ecores") == 0) {
        grp[0] = ecore; cnt[0] = ne; kd[0] = 'E';
        grp[1] = pfirst; cnt[1] = np; kd[1] = 'P';
    } else {
        grp[0] = pfirst; cnt[0] = np; kd[0] = 'P';
        grp[1] = ecore; cnt[1] = ne; kd[1] = 'E';
    }
    grp[2] = sib; cnt[2] = ns; kd[2] = 'S';
    for (int g = 0; g < 3; g++)
        for (int i = 0; i < cnt[g]; i++) {
            pl.cpu[pl.count] = grp[g][i];
            pl.kind[pl.count] = kd[g];
            pl.count++;
        }
    return pl;
}

static int pin_current_thread(const placement_t *pl, int slot) {
    int cpu = pl->cpu[slot % pl->count];
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu) != 0;
}

/* Create the OpenMP team (untimed) and pin every thread according to the policy. */
static void setup_team(int threads, const char *placement, const placement_t *pl) {
    int pinned = 0, actual = 0;
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    #pragma omp parallel reduction(+:pinned)
    {
        #pragma omp single
        actual = omp_get_num_threads();
        if (strcmp(placement, "none") != 0) pinned += pin_current_thread(pl, omp_get_thread_num());
    }
    printf("INFO,par,threads_requested,%d\n", threads);
    printf("INFO,par,threads_actual,%d\n", actual);
    printf("INFO,par,threads_pinned,%d\n", pinned);
    if (actual != threads) die("OpenMP did not create the requested number of threads", 9);
}

/* Report where every thread of the team runs (after the measurements, untimed). For pinned runs
 * this checks that the pinning survived all parallel regions of the run. */
static void report_placement(int threads, const char *placement, const placement_t *pl) {
    int cpu_of[1024];
    int t_max = threads < 1024 ? threads : 1024;
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        volatile double spin = 0.0;
        for (int r = 0; r < 200000; r++) spin += r;      /* let the scheduler settle the thread */
        (void)spin;
        if (t < t_max) cpu_of[t] = (int)GetCurrentProcessorNumber();
    }
    int ok = 1;
    printf("INFO,par,placement_map,");
    for (int t = 0; t < t_max; t++) {
        int want = pl->cpu[t % pl->count];
        char kind = '?';
        for (int s = 0; s < pl->count; s++) if (pl->cpu[s] == cpu_of[t]) kind = pl->kind[s];
        printf("%s%d:%d%c", t ? " " : "", t, cpu_of[t], kind);
        if (strcmp(placement, "none") != 0 && cpu_of[t] != want) ok = 0;
    }
    printf("\n");
    if (strcmp(placement, "none") != 0) printf("INFO,par,placement_ok,%d\n", ok);
}

/* ------------------------------------------------------------ dense matrices */

static double **alloc_matrix(int n) {
    double *block = calloc((size_t)n * n, sizeof(double));
    double **rows = malloc((size_t)n * sizeof(double *));
    if (!block || !rows) die("allocation failed (dense matrix)", 3);
    for (int i = 0; i < n; i++) rows[i] = block + (size_t)i * n;
    return rows;
}

static void free_matrix(double **m) {
    free(m[0]);
    free(m);
}

static double **read_matrix(const char *path, int n) {
    double **m = alloc_matrix(n);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(4); }
    size_t got = fread(m[0], sizeof(double), (size_t)n * n, f);
    fclose(f);
    if (got != (size_t)n * n) die("short read on dense input", 5);
    return m;
}

static void write_matrix(const char *path, double **m, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(6); }
    fwrite(m[0], sizeof(double), (size_t)n * n, f);
    fclose(f);
}

/* Context method (Assignment 1): straightforward i-j-k triple loop, scalar accumulator. */
static void matmul_ijk(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += A[i][k] * B[k][j];
            }
            C[i][j] = s;
        }
    }
}

/* One row of C = row i of A times B: the 'unroll4' kernel of Assignment 2 (ikj order, k loop
 * unrolled by 4). The four products are added to the running value of C[i][j] one after the
 * other, in increasing k, so every C[i][j] keeps the summation order of the basic triple loop.
 * In the avx2 build the j loops carry '#pragma omp simd': 4 consecutive j are processed per
 * instruction, which does not change the order of the additions of any single C[i][j]. */
static inline void row_unroll4(const double *Ai, double *const *B, double *Ci, int n) {
    memset(Ci, 0, (size_t)n * sizeof(double));
    int k = 0;
    for (; k + 3 < n; k += 4) {
        const double a0 = Ai[k], a1 = Ai[k + 1], a2 = Ai[k + 2], a3 = Ai[k + 3];
        const double *B0 = B[k], *B1 = B[k + 1], *B2 = B[k + 2], *B3 = B[k + 3];
        SIMD_LOOP
        for (int j = 0; j < n; j++) {
            double c = Ci[j];
            c += a0 * B0[j];
            c += a1 * B1[j];
            c += a2 * B2[j];
            c += a3 * B3[j];
            Ci[j] = c;
        }
    }
    for (; k < n; k++) {
        const double a = Ai[k];
        const double *Bk = B[k];
        SIMD_LOOP
        for (int j = 0; j < n; j++) Ci[j] += a * Bk[j];
    }
}

/* Sequential baseline. */
static void matmul_seq(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) row_unroll4(A[i], B, C[i], n);
}

/* Parallel version: the rows of C are distributed over the team (schedule set by omp_set_schedule:
 * static = one contiguous block of about n/p rows per thread, dynamic,1 = rows handed out one at a
 * time). Rows are independent, so threads never write the same memory. */
static void matmul_par(double **A, double **B, double **C, int n) {
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++) row_unroll4(A[i], B, C[i], n);
}

/* Row-by-row sum of all entries (same order for every method). */
static double checksum_dense(double **C, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        double rs = 0.0;
        for (int j = 0; j < n; j++) rs += C[i][j];
        total += rs;
    }
    return total;
}

/* --------------------------------------------------------------- drivers */

typedef void (*dense_kernel)(double **, double **, double **, int);

static int run_dense(int argc, char **argv) {
    if (argc < 12) die("dense: need method A.bin B.bin n warmup reps batch threads schedule placement [C_out.bin]", 2);
    const char *method = argv[2], *a_path = argv[3], *b_path = argv[4];
    int n = atoi(argv[5]), warmup = atoi(argv[6]), reps = atoi(argv[7]);
    int batch = atoi(argv[8]), threads = atoi(argv[9]);
    const char *schedule = argv[10], *placement = argv[11];
    const char *out_path = argc > 12 ? argv[12] : NULL;
    dense_kernel kernel;
    int parallel = 0;
    if (strcmp(method, "ijk") == 0) kernel = matmul_ijk;
    else if (strcmp(method, "seq") == 0) kernel = matmul_seq;
    else if (strcmp(method, "par") == 0) { kernel = matmul_par; parallel = 1; }
    else die("unknown method", 2);
    if (n < 1 || threads < 1) die("n and threads must be >= 1", 2);
    if (strcmp(placement, "pcores") && strcmp(placement, "ecores") && strcmp(placement, "none"))
        die("placement must be pcores, ecores or none", 2);
    if (strcmp(schedule, "static") && strcmp(schedule, "dynamic")) die("schedule must be static or dynamic", 2);
    if (!parallel && threads != 1) die("sequential methods run with threads = 1", 2);

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    placement_t pl = build_placement(placement);
    printf("INFO,all,logical_processors,%d\n", pl.count);
    printf("INFO,all,priority,high\n");
    if (parallel) {
        omp_set_schedule(strcmp(schedule, "dynamic") == 0 ? omp_sched_dynamic : omp_sched_static,
                         strcmp(schedule, "dynamic") == 0 ? 1 : 0);
        setup_team(threads, placement, &pl);
        printf("INFO,par,schedule,%s\n", schedule);
    } else if (strcmp(placement, "none") != 0) {
        pin_current_thread(&pl, 0);       /* the same logical processor as thread 0 of par */
    }
    printf("INFO,%s,placement,%s\n", method, placement);

    double **A = read_matrix(a_path, n);
    double **B = read_matrix(b_path, n);
    double **C = alloc_matrix(n);    /* allocated outside the timer */
    int auto_mode = (batch == 0);
    if (auto_mode && warmup < 1) warmup = 1;
    int cur = auto_mode ? 1 : batch;
    for (int r = 0; r < warmup + reps; r++) {
        const char *phase = r < warmup ? "warmup" : "measured";
        int idx = r < warmup ? r : r - warmup;
        long long t0 = now_ns();
        for (int b = 0; b < cur; b++) kernel(A, B, C, n);
        long long t1 = now_ns();
        double cs = checksum_dense(C, n); /* consumes the result, outside the timer */
        printf("SAMPLE,%s,%d,%s,%d,%d,%.1f,%.17g\n", method, n, phase, idx, cur, (double)(t1 - t0) / cur, cs);
        fflush(stdout);
        if (auto_mode && r == 0) cur = auto_batch(t1 - t0);
    }
    if (parallel) report_placement(threads, placement, &pl);
    else printf("INFO,%s,cpu_at_end,%d\n", method, (int)GetCurrentProcessorNumber());
    print_memory(method, n);
    if (out_path) write_matrix(out_path, C, n);
    free_matrix(A); free_matrix(B); free_matrix(C);
    return 0;
}

static int run_topology(void) {
    placement_t pl = build_placement("pcores");
    printf("slot,logical_processor,kind\n");
    for (int s = 0; s < pl.count; s++) printf("%d,%d,%c\n", s, pl.cpu[s], pl.kind[s]);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s dense|topology ...\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "topology") == 0) return run_topology();
    print_build_info();
    if (strcmp(argv[1], "dense") == 0) return run_dense(argc, argv);
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
