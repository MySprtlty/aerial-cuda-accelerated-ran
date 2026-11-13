// stream_log.c (2026-11-13) - LD_PRELOAD shim that lists every CUDA stream a process creates, in creation order
// (diagnosis of the green-context DL late slots: which streams could share a host->GPU connection with the UL order stream).
// It only records; the wrapped call is passed through unchanged. Output: one line per stream in $STREAM_LOG (appended).
// build (container c_aerial_troy): gcc -O2 -shared -fPIC -o /tmp/stream_log.so stream_log.c -ldl -lpthread
// use  : LD_PRELOAD=/tmp/stream_log.so STREAM_LOG=/tmp/<name>.txt <program>
//        (the L1 is started through `sudo -E`, which drops LD_PRELOAD: pass it behind sudo with the launcher's own option,
//         L1_EXTRA_ARGS="--gdb_script='env LD_PRELOAD=/tmp/stream_log_w.so'" and L1_ENV="-e STREAM_LOG=/tmp/<name>.txt")
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <sys/syscall.h>
typedef void* strm_t;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int n_created = 0;
static FILE* lf = NULL;
static void note(const char* api, strm_t s, unsigned flags, int prio, int rc, void* ra)
{
    Dl_info di; memset(&di, 0, sizeof di); dladdr(ra, &di);
    void* ctx = NULL; int (*getcur)(void**) = (int (*)(void**))dlsym(RTLD_DEFAULT, "cuCtxGetCurrent"); if(getcur) getcur(&ctx);
    const char* mod = di.dli_fname ? di.dli_fname : "?"; const char* base = strrchr(mod, '/'); if(base) mod = base + 1;
    pthread_mutex_lock(&mu);
    if(!lf) { const char* p = getenv("STREAM_LOG"); lf = fopen(p ? p : "/tmp/stream_log.txt", "a"); }
    if(lf) {
        fprintf(lf, "%d pid=%d %s rc=%d stream=%p ctx=%p flags=%u prio=%d caller=%s+0x%lx near=%s\n", n_created++, (int)getpid(), api, rc, s, ctx, flags, prio,
                mod, (unsigned long)((char*)ra - (char*)di.dli_fbase), di.dli_sname ? di.dli_sname : "?");
        fflush(lf);
    }
    pthread_mutex_unlock(&mu);
}
int cudaStreamCreate(strm_t* s)
{
    static int (*real)(strm_t*); if(!real) real = (int (*)(strm_t*))dlsym(RTLD_NEXT, "cudaStreamCreate");
    int rc = real(s); note("cudaStreamCreate", s ? *s : NULL, 0, 0, rc, __builtin_return_address(0)); return rc;
}
int cudaStreamCreateWithFlags(strm_t* s, unsigned flags)
{
    static int (*real)(strm_t*, unsigned); if(!real) real = (int (*)(strm_t*, unsigned))dlsym(RTLD_NEXT, "cudaStreamCreateWithFlags");
    int rc = real(s, flags); note("cudaStreamCreateWithFlags", s ? *s : NULL, flags, 0, rc, __builtin_return_address(0)); return rc;
}
int cudaStreamCreateWithPriority(strm_t* s, unsigned flags, int prio)
{
    static int (*real)(strm_t*, unsigned, int); if(!real) real = (int (*)(strm_t*, unsigned, int))dlsym(RTLD_NEXT, "cudaStreamCreateWithPriority");
    int rc = real(s, flags, prio); note("cudaStreamCreateWithPriority", s ? *s : NULL, flags, prio, rc, __builtin_return_address(0)); return rc;
}
int cuStreamCreate(strm_t* s, unsigned flags)
{
    static int (*real)(strm_t*, unsigned); if(!real) real = (int (*)(strm_t*, unsigned))dlsym(RTLD_NEXT, "cuStreamCreate");
    int rc = real(s, flags); note("cuStreamCreate", s ? *s : NULL, flags, 0, rc, __builtin_return_address(0)); return rc;
}
int cuStreamCreateWithPriority(strm_t* s, unsigned flags, int prio)
{
    static int (*real)(strm_t*, unsigned, int); if(!real) real = (int (*)(strm_t*, unsigned, int))dlsym(RTLD_NEXT, "cuStreamCreateWithPriority");
    int rc = real(s, flags, prio); note("cuStreamCreateWithPriority", s ? *s : NULL, flags, prio, rc, __builtin_return_address(0)); return rc;
}
int cuGreenCtxStreamCreate(strm_t* s, void* gctx, unsigned flags, int prio)
{
    static int (*real)(strm_t*, void*, unsigned, int); if(!real) real = (int (*)(strm_t*, void*, unsigned, int))dlsym(RTLD_NEXT, "cuGreenCtxStreamCreate");
    int rc = real(s, gctx, flags, prio); note("cuGreenCtxStreamCreate", s ? *s : NULL, flags, prio, rc, __builtin_return_address(0)); return rc;
}

// ---- wait graph + parked-wait sampler (added 2026-11-13) ----------------------------------------------------------------------
// Which stream waits (cudaStreamWaitEvent) for an event of which other stream, how often, and how often the event was NOT complete
// when the wait was queued (= a wait that parks in the stream's host->GPU connection). cudaEventRecord is wrapped only to remember
// the stream an event was last recorded on. A sampler thread (own CPU: $STREAM_LOG_CPU, default 47; period $STREAM_LOG_PERIOD_NS,
// default 100000) polls the parked waits with cudaEventQuery and counts, per 100-us phase of the 5-ms TDD period (CLOCK_REALTIME mod
// 5 ms), how many DISTINCT streams have a parked wait at that instant, in total and per source stream. Output: $STREAM_LOG.waits,
// rewritten every 10 s and at exit:  W = wait table, R = source streams, H = histogram of the parked-stream count per phase,
// S = sum of parked streams per phase and source, D = parked time per source (100-us bins).
#include <sched.h>
#include <stdatomic.h>
#include <time.h>
#define NEV 8192
#define NW 2048
#define NACT 512
#define NSRC 64
#define NPH 50
#define NCNT 65
#define CUDA_NOT_READY 600
static atomic_flag sp = ATOMIC_FLAG_INIT;
static void lk(void) { while(atomic_flag_test_and_set_explicit(&sp, memory_order_acquire)) ; }
static void ul(void) { atomic_flag_clear_explicit(&sp, memory_order_release); }
struct act_t { void* w; void* ev; void* ctx; void* src; long t; };
static struct { void* ev; void* strm; } evt[NEV];
static struct { void* w; void* src; void* ra; unsigned long n, pend, err; } wt[NW];
static struct act_t act[NACT];
static int n_act = 0, n_srcs = 0;
static void* srcs[NSRC];
static unsigned long n_waits = 0, act_drop = 0, act_aged = 0, hist[NPH][NCNT], src_sum[NPH][NSRC], n_samp[NPH], dur[NSRC][32];
static long now_rt(void) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); return t.tv_sec * 1000000000L + t.tv_nsec; }
static unsigned hp(void* p, unsigned n) { unsigned long x = (unsigned long)p; x ^= x >> 17; x *= 0x9E3779B97F4A7C15UL; return (unsigned)(x >> 40) % n; }
static void ev_set(void* ev, void* s) { unsigned i = hp(ev, NEV); for(unsigned k = 0; k < NEV; k++, i = (i + 1) % NEV) { if(evt[i].ev == ev || !evt[i].ev) { evt[i].ev = ev; evt[i].strm = s; return; } } }
static void* ev_get(void* ev) { unsigned i = hp(ev, NEV); for(unsigned k = 0; k < NEV; k++, i = (i + 1) % NEV) { if(evt[i].ev == ev) return evt[i].strm; if(!evt[i].ev) return NULL; } return NULL; }
static int src_idx(void* s) { for(int i = 0; i < n_srcs; i++) if(srcs[i] == s) return i; if(n_srcs < NSRC - 1) { srcs[n_srcs] = s; return n_srcs++; } return NSRC - 1; }
static void dump_waits(void)
{
    char path[512]; const char* p = getenv("STREAM_LOG"); snprintf(path, sizeof path, "%s.waits", p ? p : "/tmp/stream_log.txt");
    FILE* f = fopen(path, "w"); if(!f) return;
    fprintf(f, "# pid=%d waits=%lu parked_list_overflow=%lu aged_out=%lu\n# W waiting_stream source_stream calls pending_at_queue query_errors caller\n", (int)getpid(), n_waits, act_drop, act_aged);
    for(unsigned i = 0; i < NW; i++) if(wt[i].n) {
        Dl_info di; memset(&di, 0, sizeof di); dladdr(wt[i].ra, &di); const char* mod = di.dli_fname ? di.dli_fname : "?"; const char* b = strrchr(mod, '/'); if(b) mod = b + 1;
        fprintf(f, "W %p %p %lu %lu %lu %s+0x%lx\n", wt[i].w, wt[i].src, wt[i].n, wt[i].pend, wt[i].err, mod, (unsigned long)((char*)wt[i].ra - (char*)di.dli_fbase));
    }
    for(int i = 0; i < n_srcs; i++) fprintf(f, "R %d %p\n", i, srcs[i]);
    for(int ph = 0; ph < NPH; ph++) { fprintf(f, "H %d %lu", ph, n_samp[ph]); for(int c = 0; c < NCNT; c++) fprintf(f, " %lu", hist[ph][c]); fprintf(f, "\n"); }
    for(int ph = 0; ph < NPH; ph++) { fprintf(f, "S %d", ph); for(int i = 0; i < n_srcs; i++) fprintf(f, " %lu", src_sum[ph][i]); fprintf(f, "\n"); }
    for(int i = 0; i < n_srcs; i++) { fprintf(f, "D %d", i); for(int b = 0; b < 32; b++) fprintf(f, " %lu", dur[i][b]); fprintf(f, "\n"); }
    fclose(f);
}
static void dump_calls(void);
static void dump_trace(void);
static void at_exit_dump(void) { dump_waits(); dump_calls(); dump_trace(); }
static void* sampler(void* arg)
{
    (void)arg;
    int cpu = getenv("STREAM_LOG_CPU") ? atoi(getenv("STREAM_LOG_CPU")) : 47; long period = getenv("STREAM_LOG_PERIOD_NS") ? atol(getenv("STREAM_LOG_PERIOD_NS")) : 100000;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    struct sched_param prm; memset(&prm, 0, sizeof prm); pthread_setschedparam(pthread_self(), SCHED_OTHER, &prm);
    int (*query)(void*) = (int (*)(void*))dlsym(RTLD_NEXT, "cudaEventQuery"); int (*setctx)(void*) = (int (*)(void*))dlsym(RTLD_DEFAULT, "cuCtxSetCurrent");
    static struct act_t loc[NACT]; static char done[NACT];
    struct timespec next; clock_gettime(CLOCK_MONOTONIC, &next); void* cur = NULL; unsigned long k = 0;
    for(;;) {
        next.tv_nsec += period; while(next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; next.tv_sec++; }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
        lk(); int n = n_act; memcpy(loc, act, (size_t)n * sizeof(act[0])); ul();
        long tr = now_rt(); int ph = (int)((tr % 5000000L) / 100000L), nd = 0, ns = 0;
        for(int i = 0; i < n; i++) {
            if(loc[i].ctx != cur && setctx && loc[i].ctx) { setctx(loc[i].ctx); cur = loc[i].ctx; }
            int rc = query ? query(loc[i].ev) : 0; done[i] = (rc != CUDA_NOT_READY) ? 1 : ((tr - loc[i].t > 20000000L) ? 2 : 0); nd += done[i] != 0;
        }
        for(int i = 0; i < n; i++) {
            if(done[i]) continue;
            int dup_w = 0, dup_ws = 0;   // a stream counts once in total, and once per source it is parked on
            for(int j = 0; j < i; j++) if(!done[j] && loc[j].w == loc[i].w) { dup_w = 1; if(loc[j].src == loc[i].src) dup_ws = 1; }
            if(!dup_w) ns++;
            if(!dup_ws) src_sum[ph][src_idx(loc[i].src)]++;
        }
        hist[ph][ns < NCNT ? ns : NCNT - 1]++; n_samp[ph]++;
        if(nd) {
            lk();
            for(int i = 0; i < n; i++) if(done[i]) for(int j = 0; j < n_act; j++) if(act[j].w == loc[i].w && act[j].ev == loc[i].ev && act[j].t == loc[i].t) { act[j] = act[--n_act]; break; }
            ul();
            for(int i = 0; i < n; i++) { if(done[i] == 1) { long b = (tr - loc[i].t) / 100000L; dur[src_idx(loc[i].src)][b < 32 ? b : 31]++; } else if(done[i] == 2) act_aged++; }
        }
        if((++k % 100000) == 0) dump_waits();
    }
    return NULL;
}
// ---- call-duration table (added 2026-11-13) ---------------------------------------------------------------------------------
// CPU time spent INSIDE every wrapped CUDA call, per (API, call site): does a submission (launch / wait / record / copy) or a
// synchronous call block the calling thread, where, and at which phase of the 5-ms TDD period? Output: $STREAM_LOG.calls at exit:
//   C = one line per (API, call site): calls, mean / max CPU time, calls >= 100 us and >= 500 us
//   P = phase histogram (100-us bins of the 5-ms period, call start) of the calls >= $STREAM_LOG_LONG_NS (default 200000)
//   L = the first 6000 such long calls: API, call site, stream, start (ns, CLOCK_REALTIME), duration (us), CPU core
// $STREAM_LOG_SAMPLER=0 switches the parked-wait sampler thread off (the wait table W is still written).
// cudaEventQuery / cudaStreamQuery (polled in CPU spin loops by the L1) are only recorded when one call takes >= 20 us.
// Call trace: $STREAM_LOG_TRACE_N > 0 keeps every wrapped call (start, duration, API, call site, stream, thread) from
// $STREAM_LOG_TRACE_AFTER_S (default 30) seconds after the first cudaStreamWaitEvent until N calls are stored; a helper thread on
// CPU $STREAM_LOG_CPU writes them to $STREAM_LOG.trace (binary, 40 bytes per call) and the names to $STREAM_LOG.trace.sym.
#define NCT 1024
#define NLONG 6000
enum { A_WAITEV, A_EVREC, A_LAUNCH, A_GRAPH, A_MEMCPYA, A_MEMSETA, A_MEMCPY, A_SSYNC, A_ESYNC, A_EQUERY, A_SQUERY, A_CULAUNCH, A_DSYNC, A_N };
static const char* api_nm[A_N] = { "cudaStreamWaitEvent", "cudaEventRecord", "cudaLaunchKernel", "cudaGraphLaunch", "cudaMemcpyAsync", "cudaMemsetAsync", "cudaMemcpy",
                                   "cudaStreamSynchronize", "cudaEventSynchronize", "cudaEventQuery", "cudaStreamQuery", "cuLaunchKernel", "cudaDeviceSynchronize" };
static struct { int api; void* ra; unsigned long n, sum, max, n100, n500, ph[NPH]; } ct[NCT];
static struct { int api; void* ra; void* strm; long t0; unsigned dur_us; int cpu; } lg[NLONG];
static unsigned long n_lg = 0, n_lg_drop = 0;
static long long_ns = -1;
struct tr_t { long t0; unsigned dur; int api; void* ra; void* strm; int tid; int pad; };   // 40 bytes
static struct tr_t* tr = NULL;
static unsigned long tr_cap = 0, tr_n = 0;
static long tr_from = 0x7fffffffffffffffL;
static int tr_done = 0;
#define NTH 256
static struct { int tid; char nm[16]; } th_tab[NTH];
static int n_th = 0;
static __thread int my_tid = 0;
static int tid_get(void)
{
    if(!my_tid) { my_tid = (int)syscall(SYS_gettid); lk(); if(n_th < NTH) { th_tab[n_th].tid = my_tid; th_tab[n_th].nm[0] = 0; pthread_getname_np(pthread_self(), th_tab[n_th].nm, 16); n_th++; } ul(); }
    return my_tid;
}
static void site(void* ra, char* out, size_t n);
static void dump_trace(void)
{
    lk(); int was = tr_done; tr_done = 1; ul(); if(was || !tr) return;
    unsigned long n = __atomic_load_n(&tr_n, __ATOMIC_RELAXED); if(n > tr_cap) n = tr_cap;
    char path[512], sb[256]; const char* p = getenv("STREAM_LOG"); snprintf(path, sizeof path, "%s.trace", p ? p : "/tmp/stream_log.txt");
    FILE* f = fopen(path, "wb"); if(!f) return; fwrite(tr, sizeof(struct tr_t), n, f); fclose(f);
    snprintf(path, sizeof path, "%s.trace.sym", p ? p : "/tmp/stream_log.txt"); f = fopen(path, "w"); if(!f) return;
    fprintf(f, "# pid=%d calls=%lu entry_bytes=%zu\n", (int)getpid(), n, sizeof(struct tr_t));
    for(int i = 0; i < n_th; i++) fprintf(f, "T %d %s\n", th_tab[i].tid, th_tab[i].nm);
    for(int a = 0; a < A_N; a++) fprintf(f, "N %d %s\n", a, api_nm[a]);
    static void* seen[8192]; memset(seen, 0, sizeof seen);
    for(unsigned long i = 0; i < n; i++) {
        unsigned h = hp(tr[i].ra, 8192);
        for(unsigned k = 0; k < 8192; k++, h = (h + 1) % 8192) { if(seen[h] == tr[i].ra) break; if(!seen[h]) { seen[h] = tr[i].ra; site(tr[i].ra, sb, sizeof sb); fprintf(f, "A %p %s\n", tr[i].ra, sb); break; } }
    }
    fclose(f);
}
static void* tr_writer(void* a)
{
    (void)a; const char* e = getenv("STREAM_LOG_CPU"); cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(e ? atoi(e) : 47, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    struct sched_param sp = { 0 }; pthread_setschedparam(pthread_self(), SCHED_OTHER, &sp);
    for(;;) { usleep(200000); if(__atomic_load_n(&tr_n, __ATOMIC_RELAXED) >= tr_cap) break; }
    dump_trace(); return NULL;
}
static void tr_start(void)   // called once, at the first cudaStreamWaitEvent
{
    const char* e = getenv("STREAM_LOG_TRACE_N"); unsigned long n = e ? strtoul(e, NULL, 10) : 0; if(!n) return;
    struct tr_t* m = (struct tr_t*)calloc(n, sizeof(struct tr_t)); if(!m) return;
    memset(m, 0, n * sizeof(struct tr_t));   // touch the pages now, not in the measured calls
    e = getenv("STREAM_LOG_TRACE_AFTER_S"); tr_cap = n; tr = m; tr_from = now_rt() + (e ? atol(e) : 30) * 1000000000L;
    pthread_t th; if(pthread_create(&th, NULL, tr_writer, NULL) == 0) pthread_detach(th);
}
static void call_rec(int api, void* ra, void* strm, long t0, long d)
{
    if(t0 >= tr_from) {
        unsigned long i = __atomic_fetch_add(&tr_n, 1, __ATOMIC_RELAXED);
        if(i < tr_cap) { tr[i].t0 = t0; tr[i].dur = d > 0xffffffffL ? 0xffffffffu : (unsigned)d; tr[i].api = api; tr[i].ra = ra; tr[i].strm = strm; tr[i].tid = tid_get(); }
    }
    if(long_ns < 0) long_ns = getenv("STREAM_LOG_LONG_NS") ? atol(getenv("STREAM_LOG_LONG_NS")) : 200000;
    int cpu = (d >= long_ns) ? sched_getcpu() : 0;
    lk();
    unsigned i = hp((void*)((unsigned long)ra * 31UL + (unsigned long)api), NCT);
    for(unsigned k = 0; k < NCT; k++, i = (i + 1) % NCT) {
        if(!ct[i].n || (ct[i].api == api && ct[i].ra == ra)) {
            ct[i].api = api; ct[i].ra = ra; ct[i].n++; ct[i].sum += (unsigned long)d; if((unsigned long)d > ct[i].max) ct[i].max = (unsigned long)d;
            ct[i].n100 += d >= 100000; ct[i].n500 += d >= 500000;
            if(d >= long_ns) ct[i].ph[(t0 % 5000000L) / 100000L]++;
            break;
        }
    }
    if(d >= long_ns) { if(n_lg < NLONG) { lg[n_lg].api = api; lg[n_lg].ra = ra; lg[n_lg].strm = strm; lg[n_lg].t0 = t0; lg[n_lg].dur_us = (unsigned)(d / 1000); lg[n_lg].cpu = cpu; n_lg++; } else n_lg_drop++; }
    ul();
}
static void site(void* ra, char* out, size_t n)
{
    Dl_info di; memset(&di, 0, sizeof di); dladdr(ra, &di); const char* mod = di.dli_fname ? di.dli_fname : "?"; const char* b = strrchr(mod, '/'); if(b) mod = b + 1;
    snprintf(out, n, "%s+0x%lx", mod, (unsigned long)((char*)ra - (char*)di.dli_fbase));
}
static void dump_calls(void)
{
    char path[512], sb[256]; const char* p = getenv("STREAM_LOG"); snprintf(path, sizeof path, "%s.calls", p ? p : "/tmp/stream_log.txt");
    FILE* f = fopen(path, "w"); if(!f) return;
    fprintf(f, "# pid=%d long_ns=%ld long_calls=%lu (+%lu not kept)\n# C api caller calls mean_us max_us n_ge_100us n_ge_500us\n", (int)getpid(), long_ns, n_lg, n_lg_drop);
    for(unsigned i = 0; i < NCT; i++) if(ct[i].n) {
        site(ct[i].ra, sb, sizeof sb);
        fprintf(f, "C %s %s %lu %.2f %.1f %lu %lu\n", api_nm[ct[i].api], sb, ct[i].n, ct[i].sum / 1000.0 / ct[i].n, ct[i].max / 1000.0, ct[i].n100, ct[i].n500);
        unsigned long t = 0; for(int b = 0; b < NPH; b++) t += ct[i].ph[b];
        if(t) { fprintf(f, "P %s %s", api_nm[ct[i].api], sb); for(int b = 0; b < NPH; b++) fprintf(f, " %lu", ct[i].ph[b]); fprintf(f, "\n"); }
    }
    for(unsigned long i = 0; i < n_lg; i++) { site(lg[i].ra, sb, sizeof sb); fprintf(f, "L %s %s %p %ld %u %d\n", api_nm[lg[i].api], sb, lg[i].strm, lg[i].t0, lg[i].dur_us, lg[i].cpu); }
    fclose(f);
}
#define TIMED_Q(api, obj, call) do { void* ra_ = __builtin_return_address(0); long t0_ = now_rt(); int rc_ = (call); long d_ = now_rt() - t0_; if(d_ >= 20000) call_rec(api, ra_, (void*)(obj), t0_, d_); return rc_; } while(0)
#define TIMED(api, strm, call) do { void* ra_ = __builtin_return_address(0); long t0_ = now_rt(); int rc_ = (call); call_rec(api, ra_, (void*)(strm), t0_, now_rt() - t0_); return rc_; } while(0)
typedef struct { unsigned x, y, z; } dim3_t;
int cudaLaunchKernel(const void* fn, dim3_t g, dim3_t b, void** args, size_t sh, strm_t s)
{
    static int (*real)(const void*, dim3_t, dim3_t, void**, size_t, strm_t); if(!real) real = (int (*)(const void*, dim3_t, dim3_t, void**, size_t, strm_t))dlsym(RTLD_NEXT, "cudaLaunchKernel");
    TIMED(A_LAUNCH, s, real(fn, g, b, args, sh, s));
}
int cuLaunchKernel(void* fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned sh, strm_t s, void** params, void** extra)
{
    static int (*real)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, strm_t, void**, void**);
    if(!real) real = (int (*)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, strm_t, void**, void**))dlsym(RTLD_NEXT, "cuLaunchKernel");
    TIMED(A_CULAUNCH, s, real(fn, gx, gy, gz, bx, by, bz, sh, s, params, extra));
}
int cudaGraphLaunch(void* gexec, strm_t s)
{
    static int (*real)(void*, strm_t); if(!real) real = (int (*)(void*, strm_t))dlsym(RTLD_NEXT, "cudaGraphLaunch");
    TIMED(A_GRAPH, s, real(gexec, s));
}
int cudaMemcpyAsync(void* dst, const void* src, size_t n, int kind, strm_t s)
{
    static int (*real)(void*, const void*, size_t, int, strm_t); if(!real) real = (int (*)(void*, const void*, size_t, int, strm_t))dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    TIMED(A_MEMCPYA, s, real(dst, src, n, kind, s));
}
int cudaMemsetAsync(void* dst, int v, size_t n, strm_t s)
{
    static int (*real)(void*, int, size_t, strm_t); if(!real) real = (int (*)(void*, int, size_t, strm_t))dlsym(RTLD_NEXT, "cudaMemsetAsync");
    TIMED(A_MEMSETA, s, real(dst, v, n, s));
}
int cudaMemcpy(void* dst, const void* src, size_t n, int kind)
{
    static int (*real)(void*, const void*, size_t, int); if(!real) real = (int (*)(void*, const void*, size_t, int))dlsym(RTLD_NEXT, "cudaMemcpy");
    TIMED(A_MEMCPY, NULL, real(dst, src, n, kind));
}
int cudaStreamSynchronize(strm_t s)
{
    static int (*real)(strm_t); if(!real) real = (int (*)(strm_t))dlsym(RTLD_NEXT, "cudaStreamSynchronize");
    TIMED(A_SSYNC, s, real(s));
}
int cudaEventSynchronize(void* ev)
{
    static int (*real)(void*); if(!real) real = (int (*)(void*))dlsym(RTLD_NEXT, "cudaEventSynchronize");
    TIMED(A_ESYNC, ev, real(ev));
}
int cudaDeviceSynchronize(void)
{
    static int (*real)(void); if(!real) real = (int (*)(void))dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    TIMED(A_DSYNC, NULL, real());
}
int cudaEventQuery(void* ev)
{
    static int (*real)(void*); if(!real) real = (int (*)(void*))dlsym(RTLD_NEXT, "cudaEventQuery");
    TIMED_Q(A_EQUERY, ev, real(ev));
}
int cudaStreamQuery(strm_t s)
{
    static int (*real)(strm_t); if(!real) real = (int (*)(strm_t))dlsym(RTLD_NEXT, "cudaStreamQuery");
    TIMED_Q(A_SQUERY, s, real(s));
}
int cudaEventRecord(void* ev, strm_t s)
{
    static int (*real)(void*, strm_t); if(!real) real = (int (*)(void*, strm_t))dlsym(RTLD_NEXT, "cudaEventRecord");
    lk(); ev_set(ev, s); ul();
    TIMED(A_EVREC, s, real(ev, s));
}
int cudaEventRecordWithFlags(void* ev, strm_t s, unsigned flags)
{
    static int (*real)(void*, strm_t, unsigned); if(!real) real = (int (*)(void*, strm_t, unsigned))dlsym(RTLD_NEXT, "cudaEventRecordWithFlags");
    lk(); ev_set(ev, s); ul();
    TIMED(A_EVREC, s, real(ev, s, flags));
}
int cudaStreamWaitEvent(strm_t s, void* ev, unsigned flags)
{
    static int (*real)(strm_t, void*, unsigned); static int (*query)(void*); static int (*getcur)(void**); static int hooked = 0, samp = -1;
    if(!real) { query = (int (*)(void*))dlsym(RTLD_NEXT, "cudaEventQuery"); getcur = (int (*)(void**))dlsym(RTLD_DEFAULT, "cuCtxGetCurrent"); real = (int (*)(strm_t, void*, unsigned))dlsym(RTLD_NEXT, "cudaStreamWaitEvent"); }
    if(samp < 0) samp = getenv("STREAM_LOG_SAMPLER") ? atoi(getenv("STREAM_LOG_SAMPLER")) : 1;
    int rc = query ? query(ev) : 0, pending = (rc == CUDA_NOT_READY), err = (rc != 0 && rc != CUDA_NOT_READY), start = 0;   // state of the event when the wait is queued
    void* ctx = NULL; long t = 0; if(pending && samp) { if(getcur) getcur(&ctx); t = now_rt(); }
    void* ra = __builtin_return_address(0);
    lk();
    if(!hooked) { hooked = 1; start = 1; }
    void* src = ev_get(ev); unsigned i = hp((void*)((unsigned long)s ^ ((unsigned long)src << 7) ^ ((unsigned long)ra << 13)), NW);
    for(unsigned k = 0; k < NW; k++, i = (i + 1) % NW) { if(!wt[i].n || (wt[i].w == s && wt[i].src == src && wt[i].ra == ra)) { wt[i].w = s; wt[i].src = src; wt[i].ra = ra; wt[i].n++; wt[i].pend += pending; wt[i].err += err; break; } }
    if(pending && samp) { if(n_act < NACT) { act[n_act].w = s; act[n_act].ev = ev; act[n_act].ctx = ctx; act[n_act].src = src; act[n_act].t = t; n_act++; } else act_drop++; }
    n_waits++;
    ul();
    if(start) { atexit(at_exit_dump); tr_start(); if(samp) { pthread_t th; if(pthread_create(&th, NULL, sampler, NULL) == 0) pthread_detach(th); } }
    TIMED(A_WAITEV, s, real(s, ev, flags));
}
