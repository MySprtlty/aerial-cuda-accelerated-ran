// conn_probe.cu (2026-11-13) - which CUDA streams wait behind a long kernel of ANOTHER stream?
// Question behind it: the green-context L1 creates more streams than CUDA_DEVICE_MAX_CONNECTIONS (32) and its DL kernels
// are held until a ~0.5-ms busy-wait UL order kernel ends. This probe measures, outside the L1, how streams of one process
// (primary context or green contexts derived from it, as cuphydriver does) share the host->GPU connections:
//   map  : a busy-wait kernel on stream a, a tiny kernel on every other stream -> which streams finish only after the
//          busy-wait kernel (the "blocked" set), per a, repeated with a reversed launch order (static or dynamic mapping?)
//   busy : K streams run a busy-wait kernel at the same time (and W more only wait for an event) -> are tiny kernels on
//          the remaining streams held once K (+W) reaches the connection count?
//   fifo : on one blocked pair (a, j): three busy-wait kernels launched one after another on a, a tiny kernel on j in
//          between -> behind which of them does the tiny kernel run?
//   graph: P streams each launch ONE CUDA graph (root -> B parallel branches of D kernels -> join), either parked behind an
//          event of a long busy-wait kernel on stream 0 (parked 1) or running at once with a long first kernel per branch
//          (parked 0) -> how many connections does a graph hold (number of held tiny kernels on the remaining streams)?
//   dgraph: P streams each launch a CHAIN of device graphs as PuschRx does with pusch_workCancelMode 2: host graph H (one launcher
//          kernel) -> G0 -> G1 -> ... -> G[C-1], each started from the GPU by cudaGraphLaunch(tail launch or fire-and-forget) in the
//          last kernel of its parent; parked behind an event (L -1) or with the long kernels in G[L] -> held tiny kernels as in graph
// build (container c_aerial_troy): nvcc -O2 -std=c++17 -arch=sm_90 conn_probe.cu -o /tmp/conn_probe -lcuda
// run  : CUDA_DEVICE_MAX_CONNECTIONS=<n> /tmp/conn_probe map <streams> <green ctxs, 0 = primary> <spin_us> [prio list]
//        CUDA_DEVICE_MAX_CONNECTIONS=<n> /tmp/conn_probe fifo <streams> <green ctxs> <a> <j>
//        CUDA_DEVICE_MAX_CONNECTIONS=<n> /tmp/conn_probe busy <streams> <green ctxs> <K busy streams> [spin_us] [W waiting streams]
//        CUDA_DEVICE_MAX_CONNECTIONS=<n> /tmp/conn_probe graph <streams> <green ctxs> <B branches> [P graphs] [parked 0|1] [spin_us] [D kernels per branch]
//        CUDA_DEVICE_MAX_CONNECTIONS=<n> /tmp/conn_probe dgraph <streams> <green ctxs> <C chain length> [P chains] [L long level, -1 = parked] [spin_us]
//                                        [opt: 1 upload before every launch, 2 H with the DEVICE_LAUNCH flag, 4 fire-and-forget, 8 no sync after the uploads]
//                                        [settle_us before the tiny kernels] [B branches per graph]
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = nullptr; cuGetErrorString(r_, &s_); fprintf(stderr, "%s -> %s (line %d)\n", #x, s_ ? s_ : "?", __LINE__); exit(1); } } while (0)
#define CR(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { fprintf(stderr, "%s -> %s (line %d)\n", #x, cudaGetErrorString(r_), __LINE__); exit(1); } } while (0)

__global__ void spin(unsigned long long iters, unsigned* sink) { unsigned x = 1; for (unsigned long long i = 0; i < iters; i++) x = x * 1664525u + 1013904223u; *sink = x; }
__global__ void tiny(unsigned* sink) { *sink = 1; }
// device-side graph launch (as cuPHY's kDeviceGraphLauncher / deviceGraphLaunchKernel) and a completion counter for the last graph of a chain
__global__ void dgl(cudaGraphExec_t gx, int ff) { if (threadIdx.x == 0 && blockIdx.x == 0) cudaGraphLaunch(gx, ff ? cudaStreamGraphFireAndForget : cudaStreamGraphTailLaunch); }
__global__ void inc(unsigned* cnt) { *cnt = *cnt + 1; }

struct Strm { cudaStream_t s = nullptr; CUcontext ctx = nullptr; cudaEvent_t ev = nullptr; unsigned* sink = nullptr; int gc = -1; int prio = 0; };
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::vector<Strm> st; static double iters_per_us = 0;
static void launch_spin(int a, double us) { CK(cuCtxSetCurrent(st[a].ctx)); spin<<<1, 1, 0, st[a].s>>>((unsigned long long)(us * iters_per_us), st[a].sink); }
// PROBE_REAL_CTX=1: the <green ctxs> count creates REGULAR contexts instead (run under MPS: does every context have its own connections?);
// PROBE_SPLIT=k: streams 0..k-1 live in context 0, the others in the remaining contexts (holders in one context, free streams in another)
// PROBE_QUEUED=1: every busy stream gets a SECOND kernel queued behind the running one (the queued one has to wait for the first:
// that wait sits in the stream's host->GPU connection and holds whatever another stream puts into the same connection afterwards)
static bool queued = getenv("PROBE_QUEUED") && atoi(getenv("PROBE_QUEUED")) > 0;
static void launch_busy(int a, double us) { launch_spin(a, us); if (queued) launch_spin(a, 200); }
static void launch_tiny(int j) { CK(cuCtxSetCurrent(st[j].ctx)); tiny<<<1, 1, 0, st[j].s>>>(st[j].sink); CR(cudaEventRecord(st[j].ev, st[j].s)); }
static bool done(int j) { CK(cuCtxSetCurrent(st[j].ctx)); return cudaEventQuery(st[j].ev) == cudaSuccess; }
static void sync_all() { for (auto& x : st) { CK(cuCtxSetCurrent(x.ctx)); CR(cudaStreamSynchronize(x.s)); } }

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: conn_probe map <streams> <green ctxs> <spin_us> [prio,prio,...] | fifo <streams> <green ctxs> <a> <j>\n"); return 2; }
    std::string mode = argv[1]; int M = atoi(argv[2]), G = atoi(argv[3]);
    std::vector<int> prios; if (mode == "map" && argc > 5) { char* p = strtok(argv[5], ","); while (p) { prios.push_back(atoi(p)); p = strtok(nullptr, ","); } }
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev, 0)); int nsm = 0; CK(cuDeviceGetAttribute(&nsm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    CUcontext prim; CK(cuDevicePrimaryCtxRetain(&prim, dev)); CK(cuCtxSetCurrent(prim));
    std::vector<CUcontext> ctxs;
    if (G <= 0) ctxs.push_back(prim);
    else if (getenv("PROBE_REAL_CTX") && atoi(getenv("PROBE_REAL_CTX")) > 0) {   // G REGULAR contexts (cuCtxCreate) in this process - meant for runs under MPS
        for (int g = 0; g < G; g++) { CUcontext c; CK(cuCtxCreate(&c, 0, dev)); ctxs.push_back(c); }
    }
    else {   // G green contexts of nsm/G SMs each (rounded down to a multiple of 8), disjoint
        CUdevResource rest{}; CK(cuDeviceGetDevResource(dev, &rest, CU_DEV_RESOURCE_TYPE_SM)); unsigned per = (unsigned)(nsm / G) / 8 * 8; if (per < 8) per = 8;
        // ONE split call into G groups: a remainder cannot be split again unless it went through a green context first
        std::vector<CUdevResource> parts(G); CUdevResource rem{}; unsigned ng = (unsigned)G; CK(cuDevSmResourceSplitByCount(parts.data(), &ng, &rest, &rem, 0, per));
        for (unsigned g = 0; g < ng; g++) {
            CUdevResourceDesc d; CK(cuDevResourceGenerateDesc(&d, &parts[g], 1)); CUgreenCtx gc; CK(cuGreenCtxCreate(&gc, d, dev, CU_GREEN_CTX_DEFAULT_STREAM));
            CUcontext c; CK(cuCtxFromGreenCtx(&c, gc)); ctxs.push_back(c);
        }
    }
    st.resize(M); int split = getenv("PROBE_SPLIT") ? atoi(getenv("PROBE_SPLIT")) : 0;
    for (int i = 0; i < M; i++) {   // stream i lives in context i mod G; created in index order
        Strm& x = st[i]; x.gc = i % (int)ctxs.size();
        if (split > 0 && ctxs.size() > 1) x.gc = i < split ? 0 : 1 + (i - split) % ((int)ctxs.size() - 1);   // PROBE_SPLIT=k: streams 0..k-1 in context 0, the rest in the others
        x.ctx = ctxs[x.gc]; CK(cuCtxSetCurrent(x.ctx));
        x.prio = prios.empty() ? 0 : prios[i % prios.size()];
        CR(cudaStreamCreateWithPriority(&x.s, cudaStreamNonBlocking, x.prio)); CR(cudaEventCreateWithFlags(&x.ev, cudaEventDisableTiming)); CR(cudaMalloc(&x.sink, 4));
    }
    const char* mc = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    printf("# conn_probe %s: %d SMs, CUDA_DEVICE_MAX_CONNECTIONS=%s, %d streams, %d %s ctx(s)%s%s\n", mode.c_str(), nsm, mc ? mc : "unset(8)", M, G, getenv("PROBE_REAL_CTX") && atoi(getenv("PROBE_REAL_CTX")) > 0 ? "regular" : "green", prios.empty() ? "" : ", priorities cycled", queued ? ", a second kernel queued behind every busy one" : "");
    // calibrate the busy-wait loop on stream 0 (twice: the first launch pays the module load)
    for (int r = 0; r < 2; r++) { CK(cuCtxSetCurrent(st[0].ctx)); double t = now_us(); spin<<<1, 1, 0, st[0].s>>>(2000000ULL, st[0].sink); CR(cudaStreamSynchronize(st[0].s)); iters_per_us = 2000000.0 / (now_us() - t); }
    for (int j = 0; j < M; j++) launch_tiny(j); sync_all();   // every stream has run once
    printf("# busy-wait loop: %.0f iterations per us\n", iters_per_us);

    if (mode == "map") {
        double spin_us = atof(argv[4]); int stable = 0, total = 0; std::vector<std::vector<int>> first(M);
        for (int round = 0; round < 2; round++) {   // round 1 launches the tiny kernels in REVERSE stream order
            for (int a = 0; a < M; a++) {
                std::vector<double> fin(M, -1); double t0 = now_us(); launch_busy(a, spin_us);
                if (round == 0) { for (int j = 0; j < M; j++) if (j != a) launch_tiny(j); } else { for (int j = M - 1; j >= 0; j--) if (j != a) launch_tiny(j); }
                double tl = now_us() - t0; int left = M - 1;
                while (left > 0 && now_us() - t0 < spin_us * 4 + 20000) for (int j = 0; j < M; j++) if (j != a && fin[j] < 0 && done(j)) { fin[j] = now_us() - t0; left--; }
                sync_all();
                std::vector<int> blk; double worst_free = 0; for (int j = 0; j < M; j++) if (j != a) { if (fin[j] < 0 || fin[j] > spin_us * 0.5) blk.push_back(j); else if (fin[j] > worst_free) worst_free = fin[j]; }
                if (round == 0) first[a] = blk; else { total++; if (blk == first[a]) stable++; }
                printf("%s a=%2d (ctx %d, prio %d): launches took %4.0f us, free streams done by %5.0f us, blocked %zu:", round ? "rev" : "fwd", a, st[a].gc, st[a].prio, tl, worst_free, blk.size());
                for (int j : blk) printf(" %d", j); printf("\n");
            }
        }
        printf("# blocked set identical in both launch orders: %d of %d\n", stable, total);
    } else if (mode == "busy") {   // K streams run a busy-wait kernel, W streams only WAIT for an event of stream 0; are the other streams' tiny kernels held?
        int K = atoi(argv[4]); double spin_us = argc > 5 ? atof(argv[5]) : 2000; int W = argc > 6 ? atoi(argv[6]) : 0;
        if (K < 1 || K + W >= M) { fprintf(stderr, "busy: need 1 <= K and K + W < streams\n"); return 2; }
        cudaEvent_t gate; CK(cuCtxSetCurrent(st[0].ctx)); CR(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));
        for (int rep = 0; rep < 3; rep++) {
            std::vector<double> fin(M, -1); double t0 = now_us();
            for (int k = 0; k < K; k++) launch_busy(k, spin_us);
            CK(cuCtxSetCurrent(st[0].ctx)); CR(cudaEventRecord(gate, st[0].s));
            for (int w = K; w < K + W; w++) { CK(cuCtxSetCurrent(st[w].ctx)); CR(cudaStreamWaitEvent(st[w].s, gate, 0)); launch_tiny(w); }
            for (int j = K + W; j < M; j++) launch_tiny(j);
            double tl = now_us() - t0; int left = M - K - W;
            while (left > 0 && now_us() - t0 < spin_us * 4 + 20000) for (int j = K + W; j < M; j++) if (fin[j] < 0 && done(j)) { fin[j] = now_us() - t0; left--; }
            sync_all();
            std::vector<int> blk; std::vector<double> fr; for (int j = K + W; j < M; j++) { if (fin[j] < 0 || fin[j] > spin_us * 0.5) blk.push_back(j); else fr.push_back(fin[j]); }
            double fmax = 0; for (double f : fr) if (f > fmax) fmax = f; double bmin = 1e18, bmax = 0; for (int j : blk) { double f = fin[j] < 0 ? 1e9 : fin[j]; if (f < bmin) bmin = f; if (f > bmax) bmax = f; }
            printf("busy K=%d W=%d spin %.0f us: launches took %4.0f us | free %zu (done by %5.0f us) | blocked %zu", K, W, spin_us, tl, fr.size(), fmax, blk.size());
            if (!blk.empty()) { printf(" (done %5.0f..%5.0f us):", bmin, bmax); for (size_t i = 0; i < blk.size() && i < 24; i++) printf(" %d", blk[i]); }
            printf("\n");
        }
    } else if (mode == "graph") {   // P graphs (root -> B branches x D kernels -> join), parked behind stream 0's event or running at once
        int B = atoi(argv[4]), P = argc > 5 ? atoi(argv[5]) : 1, parked = argc > 6 ? atoi(argv[6]) : 1; double spin_us = argc > 7 ? atof(argv[7]) : 2000; int D = argc > 8 ? atoi(argv[8]) : 2;
        if (B < 1 || P < 1 || D < 1 || P + 1 >= M) { fprintf(stderr, "graph: need B >= 1, D >= 1, 1 <= P and P + 1 < streams\n"); return 2; }
        // parked: every branch kernel is short (20 us); running: the FIRST kernel of every branch spins for spin_us (the rest of the branch waits for it)
        std::vector<cudaGraphExec_t> gx(P + 1); static unsigned long long it_long, it_short; it_long = (unsigned long long)(spin_us * iters_per_us); it_short = (unsigned long long)(20 * iters_per_us);
        for (int p = 1; p <= P; p++) {
            CK(cuCtxSetCurrent(st[p].ctx)); cudaGraph_t g; CR(cudaGraphCreate(&g, 0));
            cudaKernelNodeParams kp{}; kp.gridDim = dim3(1); kp.blockDim = dim3(1); kp.sharedMemBytes = 0; kp.extra = nullptr;
            void* a_tiny[1] = {&st[p].sink}; void* a_long[2] = {&it_long, &st[p].sink}; void* a_short[2] = {&it_short, &st[p].sink};
            cudaGraphNode_t root, join; kp.func = (void*)tiny; kp.kernelParams = a_tiny; CR(cudaGraphAddKernelNode(&root, g, nullptr, 0, &kp));
            std::vector<cudaGraphNode_t> tails;
            for (int b = 0; b < B; b++) { cudaGraphNode_t prev = root;
                for (int d = 0; d < D; d++) { cudaGraphNode_t n; kp.func = (void*)spin; kp.kernelParams = (!parked && d == 0) ? a_long : a_short; CR(cudaGraphAddKernelNode(&n, g, &prev, 1, &kp)); prev = n; }
                tails.push_back(prev); }
            kp.func = (void*)tiny; kp.kernelParams = a_tiny; CR(cudaGraphAddKernelNode(&join, g, tails.data(), tails.size(), &kp));
            CR(cudaGraphInstantiate(&gx[p], g, 0));
            CR(cudaGraphLaunch(gx[p], st[p].s)); CR(cudaStreamSynchronize(st[p].s));   // first launch outside the measurement
        }
        cudaEvent_t gate; CK(cuCtxSetCurrent(st[0].ctx)); CR(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));
        for (int rep = 0; rep < 3; rep++) {
            std::vector<double> fin(M, -1); double t0 = now_us();
            if (parked) { launch_spin(0, spin_us); CK(cuCtxSetCurrent(st[0].ctx)); CR(cudaEventRecord(gate, st[0].s)); }
            for (int p = 1; p <= P; p++) { CK(cuCtxSetCurrent(st[p].ctx)); if (parked) CR(cudaStreamWaitEvent(st[p].s, gate, 0)); CR(cudaGraphLaunch(gx[p], st[p].s)); }
            for (int j = P + 1; j < M; j++) launch_tiny(j);
            double tl = now_us() - t0; int left = M - P - 1;
            while (left > 0 && now_us() - t0 < spin_us * 4 + 20000) for (int j = P + 1; j < M; j++) if (fin[j] < 0 && done(j)) { fin[j] = now_us() - t0; left--; }
            sync_all(); double tend = now_us() - t0;
            int nb = 0; double fmax = 0, bmin = 1e18, bmax = 0; for (int j = P + 1; j < M; j++) { double f = fin[j] < 0 ? 1e9 : fin[j]; if (f > spin_us * 0.5) { nb++; if (f < bmin) bmin = f; if (f > bmax) bmax = f; } else if (f > fmax) fmax = f; }
            printf("graph B=%d D=%d P=%d %s spin %.0f us: launches took %4.0f us | free %d (done by %5.0f us) | held %d of %d", B, D, P, parked ? "parked" : "running", spin_us, tl, M - P - 1 - nb, fmax, nb, M - P - 1);
            if (nb) printf(" (done %5.0f..%5.0f us)", bmin, bmax);
            printf(" | all idle at %5.0f us\n", tend);
        }
    } else if (mode == "dgraph") {   // P chains H -> G0 -> ... -> G[C-1] of device graphs (GPU-side tail launch), parked behind stream 0's event (L -1) or long in G[L]
        int C = atoi(argv[4]), P = argc > 5 ? atoi(argv[5]) : 1, L = argc > 6 ? atoi(argv[6]) : -1; double spin_us = argc > 7 ? atof(argv[7]) : 2000;
        int opt = argc > 8 ? atoi(argv[8]) : 1; double settle_us = argc > 9 ? atof(argv[9]) : 300; int B = argc > 10 ? atoi(argv[10]) : 2;
        if (C < 1 || B < 1 || P < 1 || L >= C || P + 1 >= M) { fprintf(stderr, "dgraph: need C >= 1, B >= 1, L < C, 1 <= P and P + 1 < streams\n"); return 2; }
        static unsigned long long it_long, it_short; static int ff; it_long = (unsigned long long)(spin_us * iters_per_us); it_short = (unsigned long long)(20 * iters_per_us); ff = (opt & 4) ? 1 : 0;
        std::vector<std::vector<cudaGraphExec_t>> gx(P + 1, std::vector<cudaGraphExec_t>(C)); std::vector<cudaGraphExec_t> hx(P + 1); std::vector<unsigned*> cnt(P + 1, nullptr); unsigned launches = 0;
        auto upload = [&](int p) { CK(cuCtxSetCurrent(st[p].ctx)); for (int c = C - 1; c >= 0; c--) CR(cudaGraphUpload(gx[p][c], st[p].s)); CR(cudaGraphUpload(hx[p], st[p].s)); };
        for (int p = 1; p <= P; p++) {
            CK(cuCtxSetCurrent(st[p].ctx)); CR(cudaMalloc(&cnt[p], 4)); CR(cudaMemset(cnt[p], 0, 4));
            cudaKernelNodeParams kp{}; kp.gridDim = dim3(1); kp.blockDim = dim3(1); kp.sharedMemBytes = 0; kp.extra = nullptr;
            void* a_tiny[1] = {&st[p].sink}; void* a_long[2] = {&it_long, &st[p].sink}; void* a_short[2] = {&it_short, &st[p].sink}; void* a_cnt[1] = {&cnt[p]};
            for (int c = C - 1; c >= 0; c--) {   // bottom-up: the launcher kernel of G[c] needs the executable of G[c+1]
                cudaGraph_t g; CR(cudaGraphCreate(&g, 0)); cudaGraphNode_t root, last; kp.func = (void*)tiny; kp.kernelParams = a_tiny; CR(cudaGraphAddKernelNode(&root, g, nullptr, 0, &kp));
                std::vector<cudaGraphNode_t> tails;
                for (int b = 0; b < B; b++) { cudaGraphNode_t prev = root;
                    for (int d = 0; d < 2; d++) { cudaGraphNode_t n; kp.func = (void*)spin; kp.kernelParams = (c == L && d == 0) ? a_long : a_short; CR(cudaGraphAddKernelNode(&n, g, &prev, 1, &kp)); prev = n; }
                    tails.push_back(prev); }
                void* a_dgl[2] = {c < C - 1 ? (void*)&gx[p][c + 1] : nullptr, &ff};
                if (c < C - 1) { kp.func = (void*)dgl; kp.kernelParams = a_dgl; }
                else { kp.func = (void*)inc; kp.kernelParams = a_cnt; }
                CR(cudaGraphAddKernelNode(&last, g, tails.data(), tails.size(), &kp));
                CR(cudaGraphInstantiate(&gx[p][c], g, cudaGraphInstantiateFlagDeviceLaunch));
            }
            cudaGraph_t h; CR(cudaGraphCreate(&h, 0)); cudaGraphNode_t n; void* a_dgl[2] = {&gx[p][0], &ff}; kp.func = (void*)dgl; kp.kernelParams = a_dgl; CR(cudaGraphAddKernelNode(&n, h, nullptr, 0, &kp));
            CR(cudaGraphInstantiate(&hx[p], h, (opt & 2) ? cudaGraphInstantiateFlagDeviceLaunch : 0));
            upload(p); CR(cudaGraphLaunch(hx[p], st[p].s)); CR(cudaStreamSynchronize(st[p].s));   // first launch outside the measurement
        }
        launches++; { double tw = now_us(); while (now_us() - tw < spin_us * 2 + 5000) {} }   // fire-and-forget chains are not covered by the stream synchronize
        cudaEvent_t gate; CK(cuCtxSetCurrent(st[0].ctx)); CR(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));
        for (int rep = 0; rep < 3; rep++) {
            std::vector<double> fin(M, -1);
            if (opt & 1) { for (int p = 1; p <= P; p++) upload(p); if (!(opt & 8)) for (int p = 1; p <= P; p++) { CK(cuCtxSetCurrent(st[p].ctx)); CR(cudaStreamSynchronize(st[p].s)); } }
            double t0 = now_us();
            if (L < 0) { launch_spin(0, spin_us); CK(cuCtxSetCurrent(st[0].ctx)); CR(cudaEventRecord(gate, st[0].s)); }
            for (int p = 1; p <= P; p++) { CK(cuCtxSetCurrent(st[p].ctx)); if (L < 0) CR(cudaStreamWaitEvent(st[p].s, gate, 0)); CR(cudaGraphLaunch(hx[p], st[p].s)); }
            launches++; { double tw = now_us(); while (now_us() - tw < settle_us) {} }   // let every chain reach G[L]
            double ts = now_us() - t0; for (int j = P + 1; j < M; j++) launch_tiny(j);
            double tl = now_us() - t0; int left = M - P - 1;
            while (left > 0 && now_us() - t0 < spin_us * 4 + 20000) for (int j = P + 1; j < M; j++) if (fin[j] < 0 && done(j)) { fin[j] = now_us() - t0; left--; }
            sync_all(); double tend = now_us() - t0; { double tw = now_us(); while (now_us() - tw < 3000) {} }
            int ok = 0; for (int p = 1; p <= P; p++) { unsigned v = 0; CK(cuCtxSetCurrent(st[p].ctx)); CR(cudaMemcpy(&v, cnt[p], 4, cudaMemcpyDeviceToHost)); if (v == launches) ok++; }
            int nb = 0; double fmax = 0, bmin = 1e18, bmax = 0; for (int j = P + 1; j < M; j++) { double f = fin[j] < 0 ? 1e9 : fin[j]; if (f > spin_us * 0.5) { nb++; if (f < bmin) bmin = f; if (f > bmax) bmax = f; } else if (f > fmax) fmax = f; }
            printf("dgraph C=%d B=%d P=%d L=%d opt=%d spin %.0f us: tiny kernels at %4.0f..%4.0f us | free %d (done by %5.0f us) | held %d of %d", C, B, P, L, opt, spin_us, ts, tl, M - P - 1 - nb, fmax, nb, M - P - 1);
            if (nb) printf(" (done %5.0f..%5.0f us)", bmin, bmax);
            printf(" | streams idle at %5.0f us | chains complete %d of %d\n", tend, ok, P);
        }
    } else {   // fifo: A1 (1400 us) at 0, A2 (500 us) at +500, A3 (500 us) at +1000 on stream a; one tiny kernel on stream j at +x
        int a = atoi(argv[4]), j = argc > 5 ? atoi(argv[5]) : 0; const double xs[] = {300, 700, 1200, 1700};
        for (double x : xs) for (int rep = 0; rep < 3; rep++) {
            double t0 = now_us(), tj = -1, fin = -1, ta[3] = {0, 500, 1000}; int na = 0; bool sent = false;
            while (fin < 0 && now_us() - t0 < 50000) {
                double t = now_us() - t0;
                if (na < 3 && t >= ta[na]) { launch_spin(a, na == 0 ? 1400 : 500); na++; }
                if (!sent && t >= x) { launch_tiny(j); tj = now_us() - t0; sent = true; }
                if (sent && done(j)) fin = now_us() - t0;
            }
            CK(cuCtxSetCurrent(st[a].ctx)); CR(cudaStreamSynchronize(st[a].s)); double end = now_us() - t0; sync_all();
            printf("fifo a=%d j=%d: tiny launched at %5.0f us, done at %5.0f us | A1 ends ~1400, A2 ~1900, A3 ~2400 (stream a idle at %5.0f us)\n", a, j, tj, fin, end);
        }
    }
    return 0;
}
