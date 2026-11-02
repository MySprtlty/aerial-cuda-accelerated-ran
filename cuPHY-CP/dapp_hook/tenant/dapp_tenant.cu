// dapp_tenant.cu (2026-11-01): YOLO tenant thread inside cuphycontroller, see dapp_tenant.hpp.
//
// Two inference pipelines share the slot clock: the "quiet" pipeline runs the batch-8 segment engines (DAPP_TENANT_DIR) on
// the largest free partition whenever no PUSCH is in flight, the optional "busy" pipeline runs the batch-1 segment engines
// (DAPP_TENANT_BUSY_DIR, low DRAM traffic) on the small gap that stays free while PUSCH + PDSCH are in flight. Work is
// counted in tiles of a 4K request (8 tiles = one request): the quiet pipeline finishes a request per inference, the busy
// pipeline per 8 inferences; per-request latency is first-segment-enqueue to last-tile-completion (closed loop: there is
// always a next request). Environment:
//   DAPP_TENANT=1                       enable
//   DAPP_TENANT_DIR=<dir>               quiet set: seg<k>_sm<c>.engine (default models/seg26n_fine under /opt/nvidia/cuBB)
//   DAPP_TENANT_INPUT=<file>            quiet set input (fp16 CHW, default models/seg26n/input_uhd_b8_fp16.bin; synthetic when missing)
//   DAPP_TENANT_BUSY_DIR=<dir>          busy set (batch-1 segments); unset = no busy pipeline (the old single-pipeline behaviour)
//   DAPP_TENANT_BUSY_INPUT=<file>       busy set input (default models/seg26n_b1/input_uhd_b1_fp16.bin)
//   DAPP_TENANT_BUSY_BATCH=1            tiles per busy-set inference, DAPP_TENANT_BATCH=8 for the quiet set
//   DAPP_TENANT_CLASSES=40,48,64,88     SM classes with engines (either set; a set uses the classes it has files for)
//   DAPP_TENANT_CPU=45                  core of the tenant thread (isolated), DAPP_TENANT_PRIO=0 (SCHED_FIFO priority, 0 = none)
//   DAPP_TENANT_LOG=/tmp/dapp_tenant.log      periodic statistics (every DAPP_TENANT_LOG_S seconds, default 10)
//   DAPP_TENANT_CSV=/tmp/dapp_tenant_slots.csv per wall-clock slot ("" = off)
//   DAPP_TENANT_LOOKAHEAD=1             segments allowed in flight beyond the one being enqueued (per pipeline)
//   DAPP_TENANT_GUARD_US=60             near the slot end the next segment must also fit the next slot's busy set
//   DAPP_TENANT_UL_BACK=4 DAPP_TENANT_DL_FWD=2   busy window: PUSCH of slots w-2..w-UL_BACK (+ UL control of w-1), PDSCH of w+1..w+DL_FWD
//   DAPP_TENANT_UL_PAUSE=0              1: the quiet pipeline never runs while PUSCH / UL control is in flight (busy pipeline unaffected)
//   DAPP_TENANT_DL_PAUSE=0              1: nothing runs while a PDSCH slot is in flight
//   DAPP_TENANT_DL_MAX_CLASS=0          >0: while a PDSCH slot is in flight use at most this class
//   DAPP_TENANT_UNKNOWN=prev            busy set of a slot whose plan has not arrived: prev = same slot of the previous frame
//                                       (fallback: worst case), max = worst case (largest profiles)
//   DAPP_TENANT_FIXED=<c>               debug: always run the quiet set at class c on a plain partition split from the full device
//   DAPP_TENANT_FIXED_GAP=P,N           with DAPP_TENANT_FIXED=<c>: the static context is the class-c context of gap (P, N) instead
//                                       (e.g. 88,40 = the Average busy-window gap: disjoint from PUSCH 44 / PDSCH 40 / order, overlaps PUSCH >= 60)
//   DAPP_TENANT_LAYOUT=union|gap        union (default, 2026-11-02): class contexts = union of 8-SM groups disjoint from the PUSCH profile
//                                       (+ DAPP_TENANT_FB_GUARD larger fallback profiles, default 0), PUCCH/PRACH, the PDSCH profile and the UL order
//                                       context, all measured by %smid at start-up (log: '==== union layout'); gap = the 2026-11-01 construction
//   DAPP_AGENT_QOS=1                    refuse a request whose estimate (wait + decode + class time) exceeds its qos.deadline_ms (fail/overload)
//   DAPP_AGENT_GRAPH_CHUNK=3            nodes per CUDA-graph piece of a request (0 = whole graph); DAPP_AGENT_CHUNK_AHEAD=2 pieces queued ahead
//                                       (the GPU serves a channel's queued work without yielding: queued YOLO work = cuPHY's worst wait, docs/PLAN §11)
//   DAPP_TENANT_ONE_PER_GAP=0           1: create only the largest fitting class per (P, N) (fewer contexts / streams)
//   DAPP_TENANT_PAUSE_US=20             polling period while paused
//   DAPP_TENANT_HORIZON_MAX=200         a segment that outlives the slot must fit every slot it spans (up to this many)
//   DAPP_AGENT=1                        xApp agent (docs/XAPP_INTERFACE.md): DAPP_AGENT_PORT=9810, DAPP_AGENT_UDP=host:port, DAPP_AGENT_CPU,
//                                       DAPP_AGENT_MAX_QUEUE=8, DAPP_AGENT_CELLS, DAPP_AGENT_CONF=0.25, DAPP_AGENT_IOU=0.7;
//                                       DAPP_TENANT_MODELS=26n=/dir,...,26x=/dir (default models/full26<m>); DAPP_TENANT_LOOP=1 keeps the
//                                       closed-loop benchmark pipelines running as well; DAPP_TENANT_FREERUN=1 = wall-clock slots without an L1
//   DAPP_TENANT_STRESS=spin|l2|hbm|l2w|hbmw|tc  diagnosis (l2w/hbmw = writes, tc = tensor-core mma in registers): run a synthetic kernel instead of TensorRT with the same slot logic / contexts:
//                                       spin = compute only (no memory), l2 = reads of an L2-resident buffer (DAPP_TENANT_STRESS_MB, 24),
//                                       hbm = reads of a buffer far larger than L2 (DAPP_TENANT_STRESS_MB, 1024); DAPP_TENANT_STRESS_US (200)
//                                       = target duration of one launch ("segment"); 30 launches = one "inference";
//                                       DAPP_TENANT_STRESS_BLOCKS (0 = 4 x class) and DAPP_TENANT_STRESS_GAP_US (host gap between launches) scale the intensity
#include "dapp_tenant.hpp"
#include "dapp_agent.hpp"
#include "dapp_json.hpp"
#include "dapp_nvjpeg.h"
#include "dapp_hook/dapp_sizer.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>
#include <time.h>

uint64_t cuphydriver_dapp_profile_count(int channel, int profile);
uint64_t cuphydriver_dapp_fallback_count(int channel);

namespace nv {
namespace dapp {
namespace tenant {
namespace {

struct Logger : nvinfer1::ILogger {
    void log(Severity s, const char* m) noexcept override { if (s <= Severity::kWARNING) { std::fprintf(stderr, "[dapp_tenant trt] %s\n", m); } }
};
Logger g_trt_logger;

__global__ void k_f2h(const float* __restrict__ in, __half* __restrict__ out, size_t n) { const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; if (i < n) out[i] = __float2half(in[i]); }
__global__ void k_smid(unsigned* hist) { unsigned smid; asm("mov.u32 %0, %%smid;" : "=r"(smid)); if (threadIdx.x == 0) atomicAdd(&hist[smid], 1u); const long long t0 = clock64(); while (clock64() - t0 < 20000) { } }   // union layout: which physical SMs a context runs on
__global__ void k_spin(long long cycles) { const long long t0 = clock64(); while (clock64() - t0 < cycles) { } }
__global__ void k_read(const float4* __restrict__ buf, size_t n, int iters, float* sink) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (int it = 0; it < iters; it++) {
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) { const float4 v = buf[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
    }
    if (acc.x + acc.y + acc.z + acc.w == 123456.789f) { sink[0] = acc.x; }
}
__global__ void k_write(float4* __restrict__ buf, size_t n, int iters) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (int it = 0; it < iters; it++) {
        const float4 v = make_float4(static_cast<float>(it), 1.f, 2.f, 3.f);
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) { buf[i] = v; }
    }
}
// tensor-core stressor: each warp multiplies 16x16x16 fp16 fragments held in registers (no memory traffic after the first load)
__global__ void k_tc(const __half* __restrict__ a, const __half* __restrict__ b, float* sink, int iters) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> fa;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> fb;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> fc;
    nvcuda::wmma::load_matrix_sync(fa, a, 16); nvcuda::wmma::load_matrix_sync(fb, b, 16); nvcuda::wmma::fill_fragment(fc, 0.f);
    for (int it = 0; it < iters; it++) { nvcuda::wmma::mma_sync(fc, fa, fb, fc); }
    if (fc.x[0] == 123456.789f) { sink[0] = fc.x[0]; }
}

inline int64_t now_real_ns() { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec; }
inline int64_t now_mono_ns() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec; }
inline const char* envs(const char* n, const char* d) { const char* v = std::getenv(n); return (v && *v) ? v : d; }
inline int envi(const char* n, int d) { const char* v = std::getenv(n); return (v && *v) ? std::atoi(v) : d; }
inline std::vector<int> env_list(const char* n, const std::vector<int>& d) {
    const char* v = std::getenv(n); if (!v || !*v) { return d; }
    std::vector<int> out; std::string t; for (const char* p = v;; ++p) { if (*p == ',' || *p == ' ' || *p == '\0') { if (!t.empty()) { out.push_back(std::atoi(t.c_str())); t.clear(); } if (*p == '\0') break; } else t += *p; }
    return out;
}
std::vector<char> slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); return std::vector<char>((std::istreambuf_iterator<char>(f)), {}); }
size_t numel(const nvinfer1::Dims& d) { size_t n = 1; for (int i = 0; i < d.nbDims; i++) { n *= static_cast<size_t>(d.d[i]); } return n; }
bool file_exists(const std::string& p) { std::ifstream f(p); return static_cast<bool>(f); }
std::string sm_ranges(const std::set<int>& s) { if (s.empty()) return "{}"; std::string o; int a = *s.begin(), prev = a; for (int x : s) { if (x == a || x == prev + 1) { prev = x; continue; } o += std::to_string(a) + "-" + std::to_string(prev) + ","; a = prev = x; } o += std::to_string(a) + "-" + std::to_string(prev); return o; }
int sm_inter(const std::set<int>& a, const std::set<int>& b) { int n = 0; for (int x : a) { if (b.count(x)) n++; } return n; }
double pct(std::vector<double> v, double p) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))]; }

#define CKD(x) do { CUresult rc_ = (x); if (rc_ != CUDA_SUCCESS) { const char* s_ = nullptr; cuGetErrorString(rc_, &s_); err = std::string(#x) + ": " + (s_ ? s_ : "?"); return false; } } while (0)
#define CKR(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { err = std::string(#x) + ": " + cudaGetErrorString(e_); return false; } } while (0)

struct Part {                      // one tenant green context
    int         P = 132;           // PUSCH complement (prefix) this gap is cut from; 132 = no PUSCH
    int         N = 0;             // PDSCH prefix removed from it
    int         cls = 0;           // engine class (SM count of the context)
    int         gap = 0;           // size of the gap the context was cut from
    CUgreenCtx  g = nullptr;
    CUcontext   ctx = nullptr;
    cudaStream_t s = nullptr;
    std::vector<cudaStream_t> aux;  // TensorRT auxiliary streams inside this green context (2026-11-02): TRT's own aux streams would be created in the
                                    // current (primary) context at the first enqueue and run their kernels on ANY SM, including cuPHY's
    std::vector<int> ids;           // measured physical SM ids of this context (union layout, 2026-11-02)
};
struct Seg { std::vector<nvinfer1::ICudaEngine*> eng; std::vector<nvinfer1::IExecutionContext*> exec; };
// bind the engine's auxiliary streams to the partition's own streams before every enqueue (the execution context is shared by all partitions of a class)
static inline void bind_aux(Seg& sg, int ci, const Part& p) { const int na = sg.eng[ci]->getNbAuxStreams(); if (na > 0 && !p.aux.empty()) sg.exec[ci]->setAuxStreams(const_cast<cudaStream_t*>(p.aux.data()), std::min<int>(na, static_cast<int>(p.aux.size()))); }

// one segmented network (a batch size) with its own buffers and events
struct EngineSet {
    std::string name, dir, input, model;          // model: "26n".."26x" for the agent's sets
    int batch = 8, nseg = 0;
    bool loaded = false;
    bool in_half = false, out_half = false; int k_out = 300, net = 1280;   // "images" / "output0" types, head rows per slot, input size
    std::map<std::pair<int, int>, cudaGraphExec_t> graphs;   // (segment, partition) -> captured enqueueV3 (2026-11-02): one cudaGraphLaunch per segment instead of hundreds of kernel launches
    std::map<std::pair<int, int>, std::vector<cudaGraphExec_t>> chunks;   // (segment, partition) -> the same graph cut into DAPP_AGENT_GRAPH_CHUNK-node pieces (2026-11-02 13:00): the GPU
                                                   // serves a channel's queued work without yielding to other channels (nsys: cuPHY streams waited exactly until the YOLO graph ended),
                                                   // so at most DAPP_AGENT_CHUNK_AHEAD pieces (~0.5 ms) are ever queued on the tenant stream
    std::vector<cudaEvent_t> chunk_ev; int chunk_log_n = 0;
    std::vector<int> classes;                     // classes with engine files, ascending
    std::vector<Seg> segs;                        // segs[k].exec[i] for classes[i]
    std::map<std::string, void*> buf; std::map<std::string, size_t> bytes;
    std::vector<cudaEvent_t> ev_start, ev_end;
    std::vector<std::vector<double>> seg_ema;     // [class idx][seg] GPU ms (EMA)
    std::vector<int> ema_n;                       // [class idx] samples seen (the first 5 use alpha 0.5: the validation seed is taken during L1 start-up)
    int cls_index(int c) const { for (size_t i = 0; i < classes.size(); i++) { if (classes[i] == c) return static_cast<int>(i); } return -1; }
};
// one inference pipeline over a set (closed loop: always a next request of `tiles_per_request` tiles)
struct Pipe {
    EngineSet* set = nullptr;
    int k = 0, prev_part = -1, tiles_left = 8;
    int64_t t_inf_start = 0, t_req_start = 0;
    uint64_t inferences = 0, requests = 0, segments = 0, switches = 0;
    std::vector<double> inf_lat, req_lat;         // ms, cleared at every log line
    std::vector<uint64_t> cls_hist;
};

struct Tenant {
    ~Tenant() { stop.store(true); if (th.joinable()) th.join(); if (agent_on) srv.stop(); }   // L1 exit handler -> std::exit -> static destructors: never leave a joinable thread
    // config
    std::string log_file, csv_file;
    std::vector<int> classes;      // ascending union
    int cpu = 45, prio = 0, lookahead = 1, guard_us = 60, ul_back = 4, dl_fwd = 2, fixed = 0, pause_us = 20, log_s = 10, dl_pause = 0, dl_max_class = 0, ul_pause = 0, tiles_per_request = 8;
    bool unknown_prev = true;
    int horizon_max = 200;            // DAPP_TENANT_HORIZON_MAX: longest horizon considered (slots)
    int pred_frames = 8;              // DAPP_TENANT_PRED_FRAMES: frames to look back for an unplanned slot's prediction
    int srv_port = 9810;
    // ---- xApp agent (2026-11-02, docs/XAPP_INTERFACE.md): requests from the Resource Allocator, per-slot estimates, SM reports
    bool agent_on = false, loop_on = true, freerun = false; int64_t freerun_t0 = 0;
    agent::Server srv; std::map<std::string, EngineSet> models; std::vector<std::string> model_names;
    void* njd = nullptr; float* d_dec = nullptr; size_t d_dec_n = 0; double dec_ema = 6.0; cudaStream_t dec_s = nullptr; float conf_thr = 0.25f, iou_thr = 0.7f;   // dec_s: decode on a primary-context stream (nvjpeg + green-context streams hung the thread, k1_26x_av4)
    agent::Request cur; bool have_req = false; EngineSet* aset = nullptr; Pipe apipe; int arois[64] = {0}; int awh[2] = {0, 0};
    int64_t a_t_start = 0; double a_dec_ms = 0.0; std::vector<int> a_cls; int a_pidx = -1; int64_t a_est_end_ns = 0;
    std::mutex rep_m; struct DoneRec { int64_t t_end_ns; std::string model; double sm_mean; }; std::deque<DoneRec> done;
    double yolo_sm_sum = 0.0, ran_sm_sum = 0.0; uint64_t slot_samples = 0; std::string last_estimate; uint64_t a_ok = 0, a_fail = 0;
    bool a_skip_decode = false, a_skip_infer = false, a_use_graph = true;
    unsigned char* h_jpeg = nullptr; size_t h_jpeg_cap = 0; void* h_out = nullptr; size_t h_out_cap = 0;   // pinned staging (GH200: a GPU copy engine touching pageable memory goes through ATS; keep cuPHY out of that)   // bisection knobs (DAPP_AGENT_SKIP_DECODE / _SKIP_INFER): request path without the decode / without the TRT enqueue
    int fixed_gap_P = 0, fixed_gap_N = 0, fixed_gap_part = -1, one_per_gap = 0;
    // union layout (2026-11-02, DAPP_TENANT_LAYOUT=union|gap): class contexts avoid the UL order / UL control contexts and the next
    // DAPP_TENANT_FB_GUARD larger PUSCH profiles (pool-busy fallbacks); DAPP_AGENT_QOS=1: refuse a request whose estimate misses its deadline
    int fb_guard = 1; std::string layout = "union"; bool qos_on = true; std::string layout_report;
    int chunk_n = 6, chunk_ahead = 2; bool a_chunking = false; int a_chunk_next = 0, a_chunk_k = 0; const std::vector<cudaGraphExec_t>* a_chunks = nullptr; const Part* a_chunk_part = nullptr;
    int prof_s = 0; int64_t prof_t0 = 0; bool prof_on = false, prof_done = false;   // DAPP_AGENT_PROFILE_S: cudaProfilerStart at the first request, Stop after that many seconds (nsys --capture-range=cudaProfilerApi)
    // stress mode (diagnosis)
    std::string stress; int stress_us = 200, stress_mb = 0, stress_blocks = 0, stress_gap_us = 0; __half* stress_tc_a = nullptr;
    void stress_launch(const Part& p) {
        const int blocks = stress_blocks > 0 ? stress_blocks : p.cls * 4;
        if (stress == "spin") { k_spin<<<blocks, 256, 0, p.s>>>(spin_cycles); }
        else if (stress == "l2" || stress == "hbm") { k_read<<<blocks, 256, 0, p.s>>>(stress_buf, stress_n, read_iters, stress_sink); }
        else if (stress == "l2w" || stress == "hbmw") { k_write<<<blocks, 256, 0, p.s>>>(stress_buf, stress_n, read_iters); }
        else if (stress == "tc") { k_tc<<<blocks, 256, 0, p.s>>>(stress_tc_a, stress_tc_a + 256, stress_sink, read_iters); }
    } float4* stress_buf = nullptr; float* stress_sink = nullptr; size_t stress_n = 0; long long spin_cycles = 0; int read_iters = 1;
    // device
    CUdevice dev = 0; CUcontext prim = nullptr; int nsm = 132; CUdevResource all{};
    std::vector<int> pusch_sm, pdsch_sm;         // actual partition counts (ascending) from cuphydriver
    std::vector<Part> parts;                      // every (P, N, class) context
    std::map<std::pair<int, int>, std::vector<int>> part_of;  // (P, N) -> parts, largest class first (empty = nothing fits)
    nvinfer1::IRuntime* rt = nullptr;
    EngineSet quiet, busy;
    Pipe qpipe, bpipe;
    // thread
    std::thread th; std::atomic<bool> stop{false};
    uint64_t pauses = 0; int64_t pause_ns = 0;

    bool make_ctx(Part& p, const CUdevResource& r, std::string& err) { CUdevResourceDesc d; CKD(cuDevResourceGenerateDesc(&d, const_cast<CUdevResource*>(&r), 1)); return make_ctx_desc(p, d, err); }
    bool make_ctx_desc(Part& p, CUdevResourceDesc d, std::string& err) {
        CKD(cuGreenCtxCreate(&p.g, d, dev, CU_GREEN_CTX_DEFAULT_STREAM)); CKD(cuCtxFromGreenCtx(&p.ctx, p.g));
        CKD(cuCtxSetCurrent(p.ctx)); CKR(cudaStreamCreateWithFlags(&p.s, cudaStreamNonBlocking));
        for (int i = 0; i < envi("DAPP_TENANT_AUX", 8); i++) { cudaStream_t a = nullptr; CKR(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking)); p.aux.push_back(a); }
        CKD(cuCtxSetCurrent(prim));
        return true;
    }
    bool split(const CUdevResource& in, unsigned count, CUdevResource* res, CUdevResource* rem, std::string& err) { unsigned ng = 1; CKD(cuDevSmResourceSplitByCount(res, &ng, const_cast<CUdevResource*>(&in), rem, 0, count)); return true; }
    bool via_ctx(const CUdevResource& r, CUdevResource& out, std::string& err) {   // a remainder cannot be re-split directly (915): go through a temporary green context
        CUdevResourceDesc d; CKD(cuDevResourceGenerateDesc(&d, const_cast<CUdevResource*>(&r), 1)); CUgreenCtx g = nullptr; CKD(cuGreenCtxCreate(&g, d, dev, CU_GREEN_CTX_DEFAULT_STREAM));
        CKD(cuGreenCtxGetDevResource(g, &out, CU_DEV_RESOURCE_TYPE_SM)); cuGreenCtxDestroy(g); return true;
    }

    // ---- union layout (2026-11-02): class contexts = union of 8-SM groups disjoint from everything cuPHY may run -------------
    // Every cuphydriver context (context.cpp) is a prefix or a suffix of the deterministic split order of the device (measured
    // with physical %smid by tools/gc_bench/gc_nest.cu and gc_union.cu, no MPS): PDSCH N = [0, N), PUSCH y = [132 - y, 132),
    // PUCCH + PRACH = the last pucch + prach, UL order = [C - order, C) with C = 132 - yaml PUSCH (mode 0) or [0, order) (mode 1);
    // the device split once into 8-SM groups gives group i = [8 i, 8 i + 8) in the same order, and outputs of ONE split call may be
    // combined by cuDevResourceGenerateDesc. For every (PUSCH profile, PDSCH profile) the class context is the union of the groups
    // disjoint from: that PUSCH profile and the next fb_guard larger ones (pool-busy fallbacks), PUCCH + PRACH, the PDSCH profile
    // and the UL order context; groups farthest from the fallback profiles first. The gap layout (below) had the 1-cell order
    // slice inside the YOLO contexts: every long-kernel request starved the packet-wait order kernel -> L1 recovery
    // (docs/PLAN_2026-11-02.md §9). Inside cuphycontroller the process is an MPS client, where %smid is virtual (0..n-1 per
    // context), so the sets are positions; the %smid self-check runs only when MPS is off (stand-alone dapp_tenant_main).
    struct SmSet { std::string name; std::set<int> ids; int sm = 0; };
    bool probe_stream(cudaStream_t st, std::set<int>& ids, unsigned* hist, std::string& err) {
        std::vector<unsigned> h(nsm);
        for (int rep = 0; rep < 4; rep++) { CKR(cudaMemsetAsync(hist, 0, nsm * sizeof(unsigned), st)); k_smid<<<4 * nsm, 64, 0, st>>>(hist); CKR(cudaMemcpyAsync(h.data(), hist, nsm * sizeof(unsigned), cudaMemcpyDeviceToHost, st)); CKR(cudaStreamSynchronize(st)); for (int i = 0; i < nsm; i++) { if (h[i]) ids.insert(i); } }
        return true;
    }
    bool probe_res(const CUdevResource& r, std::set<int>& ids, unsigned* hist, std::string& err) {   // temporary green context on r
        CUdevResourceDesc d; CKD(cuDevResourceGenerateDesc(&d, const_cast<CUdevResource*>(&r), 1));
        CUgreenCtx g = nullptr; CUcontext c = nullptr; CKD(cuGreenCtxCreate(&g, d, dev, CU_GREEN_CTX_DEFAULT_STREAM)); CKD(cuCtxFromGreenCtx(&c, g)); CKD(cuCtxSetCurrent(c));
        cudaStream_t st = nullptr; CKR(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        const bool ok = probe_stream(st, ids, hist, err);
        cudaStreamDestroy(st); cuCtxSetCurrent(prim); cuGreenCtxDestroy(g); return ok;
    }
    bool build_partitions_union(std::string& err) {
        int tmp[16]; int n = cuphydriver_dapp_profile_sms(0, tmp, 16); pusch_sm.assign(tmp, tmp + n);
        n = cuphydriver_dapp_profile_sms(1, tmp, 16); pdsch_sm.assign(tmp, tmp + n);
        if (fixed > 0 && fixed_gap_P == 0) {
            Part p; p.cls = fixed; p.gap = fixed; CUdevResource r{}, rem{}; if (!split(all, fixed, &r, &rem, err)) return false; if (!make_ctx(p, r, err)) return false;
            parts.push_back(p); part_of[{132, 0}] = {0}; return true;
        }
        if (pusch_sm.empty() || pdsch_sm.empty()) { err = "cuphydriver reports no PUSCH/PDSCH profiles (DAPP_PUSCH_PROFILES / DAPP_PDSCH_PROFILES unset?)"; return false; }
        int ord[2] = {0, 0}; const int no = cuphydriver_dapp_profile_sms(2, ord, 2); const int order_sm = no > 0 ? ord[0] : 0, order_mode = no > 1 ? ord[1] : 0;
        int ulc[2] = {0, 0}; const int nu = cuphydriver_dapp_profile_sms(3, ulc, 2); const int ulctrl_sm = nu > 1 ? ulc[0] + ulc[1] : 0;
        if (order_sm <= 0 || ulctrl_sm <= 0) { err = "cuphydriver does not report the UL order / UL control contexts (profile_sms channels 2, 3): old cuphydriver build?"; return false; }
        int mps = 0; cuDeviceGetAttribute(&mps, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, dev); const bool verify = (mps == 0) && envi("DAPP_TENANT_PROBE", 1) != 0;
        unsigned* hist = nullptr; if (verify) CKR(cudaMalloc(&hist, nsm * sizeof(unsigned)));
        auto interval = [](int a, int b) { std::set<int> o; for (int i = a; i < b; i++) o.insert(i); return o; };
        // the device in 8-SM groups (one split call); group i = positions [8 i, 8 i + 8), the remainder = [8 ng, nsm)
        unsigned ng = static_cast<unsigned>(nsm / 8); std::vector<CUdevResource> groups(ng); CUdevResource grem{};
        CKD(cuDevSmResourceSplitByCount(groups.data(), &ng, &all, &grem, 0, 8)); groups.resize(ng);
        std::vector<std::set<int>> gids(ng), gphys(ng); std::set<int> rphys;
        for (unsigned i = 0; i < ng; i++) {
            if (groups[i].sm.smCount != 8) { err = "group " + std::to_string(i) + " has " + std::to_string(groups[i].sm.smCount) + " SMs"; return false; }
            gids[i] = interval(8 * i, 8 * i + 8);
            if (verify) { if (!probe_res(groups[i], gphys[i], hist, err)) return false; if (gphys[i].size() != 8) { err = "layout self-check: group " + std::to_string(i) + " measured " + std::to_string(gphys[i].size()) + " SMs"; return false; } }
        }
        if (verify && grem.sm.smCount > 0) { if (!probe_res(grem, rphys, hist, err)) return false; }
        // physical SM set of a position set (self-check): the set must be a union of whole groups (and/or the whole remainder)
        auto phys_of = [&](const std::set<int>& pos, std::set<int>& out) -> bool {
            out.clear();
            for (unsigned i = 0; i < ng; i++) { bool all_in = true, any = false; for (int q = 8 * static_cast<int>(i); q < 8 * static_cast<int>(i) + 8; q++) { if (pos.count(q)) any = true; else all_in = false; } if (any && !all_in) return false; if (all_in) out.insert(gphys[i].begin(), gphys[i].end()); }
            bool all_in = true, any = false; for (int q = 8 * static_cast<int>(ng); q < nsm; q++) { if (pos.count(q)) any = true; else all_in = false; } if (any && !all_in) return false; if (any) out.insert(rphys.begin(), rphys.end());
            return true; };
        std::vector<SmSet> sets;
        auto add_set = [&](const std::string& name, int a, int b, const CUdevResource* r) -> bool {   // positions [a, b); r = the same resource for the optional %smid check
            SmSet x; x.name = name; x.ids = interval(a, b); x.sm = b - a;
            if (verify && r) { std::set<int> m, e; if (!probe_res(*r, m, hist, err)) return false; if (!phys_of(x.ids, e) || m != e) { err = "layout self-check: " + name + " positions [" + sm_ranges(x.ids) + "] measured smid [" + sm_ranges(m) + "] expected [" + sm_ranges(e) + "]"; return false; } }
            sets.push_back(x); return true; };
        const int C = nsm - pusch_sm.back(); CUdevResource yaml_result{};
        for (int y : pusch_sm) { CUdevResource r{}, rem{}; if (!split(all, nsm - y, &r, &rem, err)) return false; if (static_cast<int>(rem.sm.smCount) != y) { err = "PUSCH shadow split gives " + std::to_string(rem.sm.smCount) + " SMs for profile " + std::to_string(y); return false; } if (!add_set("PUSCH" + std::to_string(y), nsm - y, nsm, &rem)) return false; if (y == pusch_sm.back()) yaml_result = r; }
        for (int N : pdsch_sm) { CUdevResource r{}, rem{}; if (!split(all, N, &r, &rem, err)) return false; if (!add_set("PDSCH" + std::to_string(N), 0, N, &r)) return false; }
        { CUdevResource r{}, rem{}; if (!split(all, nsm - ulctrl_sm, &r, &rem, err)) return false; if (!add_set("ULCTRL", nsm - ulctrl_sm, nsm, &rem)) return false; }   // PUCCH + PRACH = remainder of cuphydriver's prep split
        if (order_mode == 1) { CUdevResource r{}, rem{}; if (!split(all, order_sm, &r, &rem, err)) return false; if (!add_set("ORDER", 0, order_sm, &r)) return false; }
        else { CUdevResource comp{}, r{}, rem{}; if (!via_ctx(yaml_result, comp, err)) return false; if (!split(comp, static_cast<unsigned>(C - order_sm), &r, &rem, err)) return false; if (static_cast<int>(rem.sm.smCount) != order_sm) { err = "ORDER shadow split gives " + std::to_string(rem.sm.smCount) + " SMs, cuphydriver has " + std::to_string(order_sm); return false; } if (!add_set("ORDER", C - order_sm, C, &rem)) return false; }
        auto find = [&](const std::string& nm) -> const SmSet& { for (const SmSet& s : sets) { if (s.name == nm) return s; } return sets.front(); };
        char line[2048];
        layout_report += verify ? "==== union layout (positions, %smid-verified):" : "==== union layout (split-order positions; MPS client, %smid virtual):";
        for (const SmSet& s : sets) { std::snprintf(line, sizeof line, " %s %d [%s]", s.name.c_str(), s.sm, sm_ranges(s.ids).c_str()); layout_report += line; }
        std::snprintf(line, sizeof line, " | %u groups of 8 + %u left, fb_guard %d\n", ng, grem.sm.smCount, fb_guard); layout_report += line;
        std::vector<std::vector<int>> ladders;   // per engine set: the classes of `classes` that have an engine file in its directory
        { std::vector<std::string> dirs; if (loop_on) { dirs.push_back(quiet.dir); if (!busy.dir.empty()) dirs.push_back(busy.dir); } for (const std::string& m : model_names) dirs.push_back(models[m].dir);
          for (const std::string& d : dirs) { std::vector<int> lad; for (int c : classes) { char fp[512]; std::snprintf(fp, sizeof fp, "%s/seg00_sm%d.engine", d.c_str(), c); if (file_exists(fp)) lad.push_back(c); } if (lad.empty()) lad = classes; ladders.push_back(lad); }
          if (!stress.empty() || ladders.empty()) ladders.push_back(classes); }
        std::vector<int> Ps; for (int y : pusch_sm) { Ps.push_back(nsm - y); } Ps.push_back(nsm);
        for (size_t pi = 0; pi < Ps.size(); pi++) {
            const int P = Ps[pi];
            for (int N : pdsch_sm) {
                std::set<int> busy = find("ORDER").ids; { const SmSet& d = find("PDSCH" + std::to_string(N)); busy.insert(d.ids.begin(), d.ids.end()); }
                std::string guarded;
                if (P < nsm) {
                    for (size_t j = pi; j < pusch_sm.size() && j <= pi + static_cast<size_t>(std::max(fb_guard, 0)); j++) { const SmSet& u = find("PUSCH" + std::to_string(pusch_sm[j])); busy.insert(u.ids.begin(), u.ids.end()); if (j > pi) guarded += " PUSCH" + std::to_string(pusch_sm[j]); }
                    const SmSet& u = find("ULCTRL"); busy.insert(u.ids.begin(), u.ids.end());
                }
                std::set<int> anyfb; for (size_t j = pi + 1; j < pusch_sm.size(); j++) { const SmSet& u = find("PUSCH" + std::to_string(pusch_sm[j])); anyfb.insert(u.ids.begin(), u.ids.end()); }
                std::vector<unsigned> free; for (unsigned g = 0; g < ng; g++) { if (sm_inter(gids[g], busy) == 0) free.push_back(g); }
                std::stable_sort(free.begin(), free.end(), [&](unsigned a, unsigned b) { return (sm_inter(gids[a], anyfb) == 0) > (sm_inter(gids[b], anyfb) == 0); });
                const int gsz = static_cast<int>(free.size()) * 8;
                std::vector<int> fits;
                if (one_per_gap) {   // only the classes some engine set will actually use here: the largest class of each set's ladder that fits
                    for (const std::vector<int>& lad : ladders) { int best = 0; for (int c : lad) { if (c % 8 == 0 && c <= gsz && c > best) best = c; } if (best > 0 && std::find(fits.begin(), fits.end(), best) == fits.end()) fits.push_back(best); }
                } else { for (int c : classes) { if (c % 8 == 0 && c <= gsz) fits.push_back(c); } }
                std::sort(fits.begin(), fits.end(), [](int a, int b) { return a > b; });
                if (fixed_gap_P > 0) { fits.clear(); if (P == fixed_gap_P && N == fixed_gap_N && fixed <= gsz && fixed % 8 == 0) fits.push_back(fixed); }
                std::vector<int> idx;
                for (int c : fits) {
                    std::vector<CUdevResource> sel; for (int k = 0; k < c / 8; k++) sel.push_back(groups[free[k]]);
                    CUdevResourceDesc d; CKD(cuDevResourceGenerateDesc(&d, sel.data(), static_cast<unsigned>(sel.size())));
                    Part p; p.P = P; p.N = N; p.cls = c; p.gap = gsz; if (!make_ctx_desc(p, d, err)) return false;
                    std::set<int> ids; for (int k = 0; k < c / 8; k++) ids.insert(gids[free[k]].begin(), gids[free[k]].end());
                    if (verify) { std::set<int> m, e; CKD(cuCtxSetCurrent(p.ctx)); const bool okp = probe_stream(p.s, m, hist, err); cuCtxSetCurrent(prim); if (!okp) return false; if (!phys_of(ids, e) || m != e) { err = "layout self-check: class " + std::to_string(c) + " context measured smid [" + sm_ranges(m) + "] expected [" + sm_ranges(e) + "]"; return false; } }
                    p.ids.assign(ids.begin(), ids.end());
                    std::string ov; for (const SmSet& s : sets) { const int k = sm_inter(ids, s.ids); if (k) ov += " " + s.name + "(" + std::to_string(k) + ")"; }
                    const bool ok = static_cast<int>(ids.size()) == c && sm_inter(ids, busy) == 0;
                    std::string phys; if (verify) { std::set<int> e; phys_of(ids, e); phys = " smid [" + sm_ranges(e) + "]"; }
                    std::snprintf(line, sizeof line, "  (P %d, N %d) free %d SMs%s%s -> class %d positions [%s]%s overlaps:%s%s\n", P, N, gsz, guarded.empty() ? "" : " guarding", guarded.c_str(), c, sm_ranges(ids).c_str(), phys.c_str(), ov.empty() ? " none" : ov.c_str(), ok ? "" : "  <-- LAYOUT VIOLATION"); layout_report += line;
                    if (!ok) { err = "union layout violated for class " + std::to_string(c) + " at (P " + std::to_string(P) + ", N " + std::to_string(N) + "): " + ov; return false; }
                    idx.push_back(static_cast<int>(parts.size())); parts.push_back(p);
                }
                if (fits.empty()) { std::snprintf(line, sizeof line, "  (P %d, N %d) free %d SMs%s%s -> no class fits (pause)\n", P, N, gsz, guarded.empty() ? "" : " guarding", guarded.c_str()); layout_report += line; }
                part_of[{P, N}] = idx;
                if (fixed_gap_P > 0 && !idx.empty()) fixed_gap_part = idx.front();
            }
        }
        if (hist) cudaFree(hist);
        if (fixed_gap_P > 0 && fixed_gap_part < 0) { err = "DAPP_TENANT_FIXED_GAP: class " + std::to_string(fixed) + " does not fit gap (" + std::to_string(fixed_gap_P) + ", " + std::to_string(fixed_gap_N) + ")"; return false; }
        return true;
    }
    bool build_partitions(std::string& err) { return layout == "gap" ? build_partitions_gap(err) : build_partitions_union(err); }

    // ---- gap layout (2026-11-01, DAPP_TENANT_LAYOUT=gap): for every (PUSCH complement P, PDSCH prefix N) one context per class that fits the gap
    bool build_partitions_gap(std::string& err) {
        int tmp[16]; int n = cuphydriver_dapp_profile_sms(0, tmp, 16); pusch_sm.assign(tmp, tmp + n);
        n = cuphydriver_dapp_profile_sms(1, tmp, 16); pdsch_sm.assign(tmp, tmp + n);
        if (fixed > 0 && fixed_gap_P == 0) {
            Part p; p.cls = fixed; p.gap = fixed; CUdevResource r{}, rem{}; if (!split(all, fixed, &r, &rem, err)) return false; if (!make_ctx(p, r, err)) return false;
            parts.push_back(p); part_of[{132, 0}] = {0}; return true;
        }
        if (pusch_sm.empty() || pdsch_sm.empty()) { err = "cuphydriver reports no PUSCH/PDSCH profiles (DAPP_PUSCH_PROFILES / DAPP_PDSCH_PROFILES unset?)"; return false; }
        std::vector<int> Ps; for (int y : pusch_sm) { Ps.push_back(nsm - y); } Ps.push_back(nsm);
        for (int P : Ps) {
            for (int N : pdsch_sm) {
                CUdevResource gap{};
                if (P < nsm && P - N < classes.front()) { part_of[{P, N}] = {}; continue; }
                std::string e2; bool ok = true;
                if (P >= nsm) { CUdevResource r{}; ok = split(all, N, &r, &gap, e2); }
                else { CUdevResource pre{}, rem{}; ok = split(all, P, &pre, &rem, e2); CUdevResource pre2{}; if (ok) ok = via_ctx(pre, pre2, e2); CUdevResource a{}; if (ok) ok = split(pre2, N, &a, &gap, e2); }
                if (!ok) { std::fprintf(stderr, "[dapp_tenant] gap (P %d, N %d): %s -> no partition\n", P, N, e2.c_str()); part_of[{P, N}] = {}; continue; }
                const int gsz = static_cast<int>(gap.sm.smCount);
                std::vector<int> fits; for (int c : classes) { if (c <= gsz) fits.push_back(c); }
                std::sort(fits.begin(), fits.end(), [](int a, int b) { return a > b; });
                if (one_per_gap && !fits.empty()) { fits.resize(1); }
                if (fixed_gap_P > 0) { fits.clear(); if (P == fixed_gap_P && N == fixed_gap_N && fixed <= gsz) fits.push_back(fixed); }
                std::vector<int> idx;
                for (int c : fits) {
                    Part p; p.P = P; p.N = N; p.cls = c; p.gap = gsz;
                    if (c == gsz) { if (!make_ctx(p, gap, err)) return false; }
                    else { CUdevResource g2{}; if (!via_ctx(gap, g2, err)) return false; CUdevResource cc{}, crem{}; if (!split(g2, c, &cc, &crem, err)) return false; if (!make_ctx(p, cc, err)) return false; }
                    idx.push_back(static_cast<int>(parts.size())); parts.push_back(p);
                }
                part_of[{P, N}] = idx;
                if (fixed_gap_P > 0 && !idx.empty()) fixed_gap_part = idx.front();
            }
        }
        if (fixed_gap_P > 0 && fixed_gap_part < 0) { err = "DAPP_TENANT_FIXED_GAP: class " + std::to_string(fixed) + " does not fit gap (" + std::to_string(fixed_gap_P) + ", " + std::to_string(fixed_gap_N) + ")"; return false; }
        return true;
    }

    // ---- network sets ------------------------------------------------------------------------------------------------
    bool load_set(EngineSet& s, std::string& err) {
        if (!rt) { rt = nvinfer1::createInferRuntime(g_trt_logger); if (!rt) { err = "createInferRuntime failed"; return false; } }
        for (int c : classes) { char p[512]; std::snprintf(p, sizeof p, "%s/seg00_sm%d.engine", s.dir.c_str(), c); if (file_exists(p)) s.classes.push_back(c); }
        if (s.classes.empty()) { err = "no seg00_sm<c>.engine in " + s.dir; return false; }
        // engines of one set must share the I/O tensor layout (one buffer per tensor name): keep the classes whose seg00 engine has the same
        // tensor signature as the largest class (an older export left e.g. a 32-SM 26n engine with another input size) (2026-11-02 14:20)
        { std::map<int, std::string> sig;
          for (int c : s.classes) { char p[512]; std::snprintf(p, sizeof p, "%s/seg00_sm%d.engine", s.dir.c_str(), c); auto blob = slurp(p); if (blob.empty()) continue;
              CKD(cuCtxSetCurrent(prim)); nvinfer1::ICudaEngine* e = rt->deserializeCudaEngine(blob.data(), blob.size()); if (!e) continue;
              std::string g; for (int i = 0; i < e->getNbIOTensors(); i++) { const char* n = e->getIOTensorName(i); const nvinfer1::Dims d = e->getTensorShape(n); g += n; g += ':'; for (int j = 0; j < d.nbDims; j++) { g += std::to_string(d.d[j]); g += 'x'; } g += std::to_string(static_cast<int>(e->getTensorDataType(n))); g += ';'; }
              sig[c] = g; delete e; }
          const std::string ref = sig.count(s.classes.back()) ? sig[s.classes.back()] : std::string();
          std::vector<int> keep; for (int c : s.classes) { if (sig.count(c) && sig[c] == ref) keep.push_back(c); else std::fprintf(stderr, "dapp_tenant: %s: class %d engine skipped (tensor layout differs from class %d)\n", s.dir.c_str(), c, s.classes.back()); }
          s.classes = keep; if (s.classes.empty()) { err = "no usable seg00_sm<c>.engine in " + s.dir; return false; } }
        for (s.nseg = 0;; s.nseg++) { char p[512]; std::snprintf(p, sizeof p, "%s/seg%02d_sm%d.engine", s.dir.c_str(), s.nseg, s.classes[0]); if (!file_exists(p)) break; }
        s.segs.resize(s.nseg);
        for (int k = 0; k < s.nseg; k++) {
            for (size_t ci = 0; ci < s.classes.size(); ci++) {
                char p[512]; std::snprintf(p, sizeof p, "%s/seg%02d_sm%d.engine", s.dir.c_str(), k, s.classes[ci]);
                auto blob = slurp(p); if (blob.empty()) { err = std::string("missing ") + p; return false; }
                CUcontext c0 = prim; for (const Part& q : parts) { if (q.cls == s.classes[ci]) { c0 = q.ctx; break; } }
                CKD(cuCtxSetCurrent(c0));
                nvinfer1::ICudaEngine* e = rt->deserializeCudaEngine(blob.data(), blob.size()); if (!e) { err = std::string("deserialize failed ") + p; return false; }
                nvinfer1::IExecutionContext* x = e->createExecutionContext(); if (!x) { err = std::string("execution context failed ") + p; return false; }
                for (int i = 0; i < e->getNbIOTensors(); i++) {
                    const char* n = e->getIOTensorName(i); nvinfer1::DataType dt = e->getTensorDataType(n);
                    size_t esz = dt == nvinfer1::DataType::kHALF ? 2 : dt == nvinfer1::DataType::kINT64 ? 8 : (dt == nvinfer1::DataType::kBOOL || dt == nvinfer1::DataType::kINT8 || dt == nvinfer1::DataType::kUINT8) ? 1 : 4;
                    size_t nb = numel(e->getTensorShape(n)) * esz;
                    if (!s.buf.count(n)) { void* d = nullptr; CKR(cudaMalloc(&d, nb)); CKR(cudaMemset(d, 0, nb)); s.buf[n] = d; s.bytes[n] = nb; }
                    else if (s.bytes[n] != nb) { err = std::string("tensor size mismatch ") + n + " in " + s.dir; return false; }
                    x->setTensorAddress(n, s.buf[n]);
                }
                s.segs[k].eng.push_back(e); s.segs[k].exec.push_back(x);
                std::fprintf(stderr, "dapp_tenant: %s: TensorRT aux streams %d\n", p, e->getNbAuxStreams());
                CKD(cuCtxSetCurrent(prim));
            }
        }
        if (!s.buf.count("images") || !s.buf.count("output0")) { err = "network in " + s.dir + " has no 'images' input / 'output0' output"; return false; }
        { nvinfer1::ICudaEngine* e0 = s.segs[0].eng[0]; nvinfer1::ICudaEngine* eL = s.segs[s.nseg - 1].eng[0];
          s.in_half = e0->getTensorDataType("images") == nvinfer1::DataType::kHALF; s.out_half = eL->getTensorDataType("output0") == nvinfer1::DataType::kHALF;
          const nvinfer1::Dims di = e0->getTensorShape("images"); if (di.nbDims == 4) { s.batch = di.d[0]; s.net = di.d[3]; }
          const nvinfer1::Dims dout = eL->getTensorShape("output0"); if (dout.nbDims == 3) s.k_out = dout.d[1]; }
        { const size_t n = s.bytes["images"] / (s.in_half ? 2 : 4); auto b = slurp(s.input);   // benchmark input: fp16 file (converted when the engine wants fp32)
          std::vector<__half> h(n); if (b.size() == n * 2) { std::memcpy(h.data(), b.data(), n * 2); } else { for (size_t i = 0; i < n; i++) h[i] = __float2half(((i * 7919u) % 256) / 255.0f); }
          if (s.in_half) { CKR(cudaMemcpy(s.buf["images"], h.data(), n * 2, cudaMemcpyHostToDevice)); }
          else { std::vector<float> f(n); for (size_t i = 0; i < n; i++) f[i] = __half2float(h[i]); CKR(cudaMemcpy(s.buf["images"], f.data(), n * 4, cudaMemcpyHostToDevice)); } }
        s.ev_start.resize(s.nseg); s.ev_end.resize(s.nseg);
        for (int k = 0; k < s.nseg; k++) { CKR(cudaEventCreate(&s.ev_start[k])); CKR(cudaEventCreate(&s.ev_end[k])); }
        s.seg_ema.assign(s.classes.size(), std::vector<double>(s.nseg, 0.0)); s.ema_n.assign(s.classes.size(), 0);
        s.loaded = true;
        return true;
    }
    bool d2h(void* dst, const void* src, size_t nb) {   // never the legacy default stream inside cuphycontroller (implicit device-wide sync)
        cudaStream_t ds = dec_s ? dec_s : (a_pidx >= 0 && a_pidx < static_cast<int>(parts.size()) ? parts[a_pidx].s : (parts.empty() ? nullptr : parts.front().s));
        if (ds) { if (cudaMemcpyAsync(dst, src, nb, cudaMemcpyDeviceToHost, ds) != cudaSuccess) return false; return cudaStreamSynchronize(ds) == cudaSuccess; }
        return cudaMemcpy(dst, src, nb, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    bool read_output(EngineSet& s, std::vector<float>& out) {
        const size_t nb = s.bytes["output0"];
        void* stage = (h_out && nb <= h_out_cap) ? h_out : nullptr;
        if (s.out_half) { const __half* h; std::vector<__half> tmp; if (stage) { if (!d2h(stage, s.buf["output0"], nb)) return false; h = static_cast<const __half*>(stage); } else { tmp.resize(nb / 2); if (!d2h(tmp.data(), s.buf["output0"], nb)) return false; h = tmp.data(); }
            out.resize(nb / 2); for (size_t i = 0; i < out.size(); i++) out[i] = __half2float(h[i]); }
        else { out.resize(nb / 4); if (stage) { if (!d2h(stage, s.buf["output0"], nb)) return false; std::memcpy(out.data(), stage, nb); } else if (!d2h(out.data(), s.buf["output0"], nb)) return false; }
        return true;
    }
    // one full inference of a set on a given partition, synchronous; detections (score > 0.25) and latency
    bool run_once(EngineSet& s, const Part& p, int& dets, double& ms, std::string& err) {
        const int ci = s.cls_index(p.cls); if (ci < 0) { err = s.name + ": no engines for class " + std::to_string(p.cls); return false; }
        const int64_t t0 = now_mono_ns();
        for (int k = 0; k < s.nseg; k++) { bind_aux(s.segs[k], ci, p); if (!s.segs[k].exec[ci]->enqueueV3(p.s)) { err = s.name + ": enqueue failed seg " + std::to_string(k); return false; } }
        CKR(cudaStreamSynchronize(p.s)); ms = (now_mono_ns() - t0) / 1e6;
        std::vector<float> out; if (!read_output(s, out)) { err = "output copy failed"; return false; }
        dets = 0; for (size_t r = 0; r + 6 <= out.size(); r += 6) { if (out[r + 4] > 0.25f) dets++; }
        return true;
    }

    // ---- slot logic -----------------------------------------------------------------------------------------------------
    static uint32_t key_of(int64_t n) { return sizer_key(static_cast<uint16_t>((n / kSzSlotsPerFrame) & 1023), static_cast<uint16_t>(n % kSzSlotsPerFrame)); }
    // plan entry of slot n, or of the same slot one frame earlier when n's plan has not arrived (DAPP_TENANT_UNKNOWN=prev)
    const SlotPlanEntry* plan_of(int64_t n, bool& predicted) const {
        predicted = false;
        if (n >= 0) { const SlotPlanEntry& e = slot_plan_entry(n); if (e.key.load(std::memory_order_acquire) == key_of(n)) return &e; }
        // 2026-11-02: look back up to DAPP_TENANT_PRED_FRAMES frames (a 26x request spans 60+ slots, far beyond the planned horizon)
        if (unknown_prev) { for (int k = 1; k <= pred_frames; k++) { const int64_t m = n - k * kSzSlotsPerFrame; if (m < 0) break; const SlotPlanEntry& e = slot_plan_entry(m); if (e.key.load(std::memory_order_acquire) == key_of(m)) { predicted = true; return &e; } } }
        return nullptr;
    }
    // busy partitions of wall-clock slot w: PUSCH prefix P (132 = none) and DL prefix N; known = every plan was present
    void busy_of(int64_t w, int& P, int& N, bool& known, bool& dl_active, bool& ul_active) const {
        int ul = 0, dl = 0; known = true; bool pred;
        for (int b = 1; b <= ul_back; b++) {
            const SlotPlanEntry* e = plan_of(w - b, pred);
            if (!e) { known = false; if (freerun) continue; ul = std::max(ul, b == 1 ? pusch_sm.front() : pusch_sm.back()); continue; }   // freerun (no L1): unknown = idle
            if (!e->ul_any) continue;
            if (b == 1) { ul = std::max(ul, pusch_sm.front()); }                   // only UL control runs this early (inside the smallest PUSCH partition)
            else { ul = std::max(ul, e->ul_prof >= 0 ? static_cast<int>(e->ul_sm) : pusch_sm.front()); }
        }
        for (int f = 1; f <= dl_fwd; f++) {
            const SlotPlanEntry* e = plan_of(w + f, pred);
            if (!e) { known = false; if (freerun) continue; dl = std::max(dl, pdsch_sm.back()); continue; }
            if (!e->dl_any) continue;
            dl = std::max(dl, e->dl_prof >= 0 ? static_cast<int>(e->dl_sm) : pdsch_sm.front());
        }
        P = ul > 0 ? nsm - ul : nsm; N = dl > 0 ? dl : pdsch_sm.front(); dl_active = dl > 0; ul_active = ul > 0;
    }
    // largest class of the set that fits (P, N), subject to the policy
    int part_for(int P, int N, bool dl_act, bool ul_act, const EngineSet& s, bool apply_ul_pause) const {
        if (dl_act && dl_pause) return -1;
        if (ul_act && ul_pause && apply_ul_pause) return -1;
        auto it = part_of.find({P, N}); if (it == part_of.end()) return -1;
        for (int i : it->second) { if (s.cls_index(parts[i].cls) < 0) continue; if (!dl_act || dl_max_class <= 0 || parts[i].cls <= dl_max_class) return i; }
        return -1;
    }
    bool current_slot(int64_t& w, int64_t& slot_end_ns) const {
        const int64_t id = g_plan_ref_id.load(std::memory_order_acquire); const int64_t t0 = g_plan_ref_t0_ns.load(std::memory_order_relaxed);
        if (id < 0 || t0 == 0) {
            if (!freerun) return false;                      // DAPP_TENANT_FREERUN=1 (standalone tests without an L1): wall-clock slots, plans unknown
            const int64_t d = now_real_ns() - freerun_t0; w = d / 500000; slot_end_ns = freerun_t0 + (w + 1) * 500000; return true;
        }
        const int64_t now = now_real_ns(); const int64_t d = now - t0;
        const int64_t k = (d >= 0) ? d / 500000 : -((-d + 499999) / 500000);
        w = id + k; slot_end_ns = t0 + (k + 1) * 500000; return true;
    }

    // ---- advance one pipeline by one segment on part pidx -------------------------------------------------------------
    bool step(Pipe& pp, int pidx, std::FILE* lf) {
        EngineSet& s = *pp.set; const Part& p = parts[pidx]; const int ci = s.cls_index(p.cls);
        const int k = pp.k;
        if (k > lookahead) { cudaEventSynchronize(s.ev_end[k - 1 - lookahead]); }
        if (k == 0) { pp.t_inf_start = now_mono_ns(); if (pp.tiles_left == tiles_per_request) pp.t_req_start = pp.t_inf_start; }
        if (pp.prev_part >= 0 && pp.prev_part != pidx && k > 0) { cudaStreamWaitEvent(p.s, s.ev_end[k - 1], 0); pp.switches++; }
        cudaEventRecord(s.ev_start[k], p.s);
        if (!stress.empty()) {
            if (stress_gap_us > 0) { const auto t_gap = std::chrono::steady_clock::now() + std::chrono::microseconds(stress_gap_us); while (std::chrono::steady_clock::now() < t_gap) { } }
            stress_launch(p);
        } else if (bind_aux(s.segs[k], ci, p), !s.segs[k].exec[ci]->enqueueV3(p.s)) { if (lf) { std::fprintf(lf, "%s: enqueue failed seg %d class %d\n", s.name.c_str(), k, p.cls); std::fflush(lf); } return false; }
        cudaEventRecord(s.ev_end[k], p.s);
        pp.prev_part = pidx; pp.segments++; pp.cls_hist[ci]++;
        if (k > lookahead) { const int j = k - 1 - lookahead; float ms = 0; if (cudaEventElapsedTime(&ms, s.ev_start[j], s.ev_end[j]) == cudaSuccess) { s.seg_ema[ci][j] = s.seg_ema[ci][j] == 0.0 ? ms : 0.9 * s.seg_ema[ci][j] + 0.1 * ms; } }
        if (++pp.k == s.nseg) {
            cudaEventSynchronize(s.ev_end[s.nseg - 1]);
            for (int j = std::max(0, s.nseg - 1 - lookahead); j < s.nseg; j++) { float ms = 0; if (cudaEventElapsedTime(&ms, s.ev_start[j], s.ev_end[j]) == cudaSuccess) { s.seg_ema[ci][j] = s.seg_ema[ci][j] == 0.0 ? ms : 0.9 * s.seg_ema[ci][j] + 0.1 * ms; } }
            const int64_t now = now_mono_ns();
            pp.inferences++; pp.inf_lat.push_back((now - pp.t_inf_start) / 1e6); pp.k = 0; pp.prev_part = -1;
            pp.tiles_left -= s.batch;
            if (pp.tiles_left <= 0) { pp.requests++; pp.req_lat.push_back((now - pp.t_req_start) / 1e6); pp.tiles_left = tiles_per_request; }
        }
        return true;
    }

    // partition for the next segment k of set s at wall slot w (busy sets of w, plus every slot the segment will span)
    int choose_part(EngineSet& s, int k, int64_t w, int64_t slot_end, bool& known, bool apply_ul_pause, int* gap_now = nullptr) {
        int P, N; bool dla, ula; busy_of(w, P, N, known, dla, ula);
        if (gap_now) *gap_now = P - N;
        double est_us = 0.0; for (size_t ci = 0; ci < s.classes.size(); ci++) { est_us = std::max(est_us, s.seg_ema[ci][k] * 1000.0); }
        const int64_t span_ns = static_cast<int64_t>((guard_us + est_us) * 1000.0) - (slot_end - now_real_ns());
        const int H = span_ns > 0 ? static_cast<int>(span_ns / 500000) + 1 : 0;
        for (int j = 1; j <= std::min(H, horizon_max); j++) { int P2, N2; bool kn2, dla2, ula2; busy_of(w + j, P2, N2, kn2, dla2, ula2); known = known && kn2; P = std::min(P, P2); N = std::max(N, N2); dla = dla || dla2; ula = ula || ula2; }
        return part_for(P, N, dla, ula, s, apply_ul_pause);
    }
    // ---- agent: estimates --------------------------------------------------------------------------------------------
    double set_time_ms(const EngineSet& s, int ci) const { double t = 0.0; for (int k = 0; k < s.nseg; k++) t += s.seg_ema[ci][k]; return t; }
    std::string estimate_json(int64_t w, int64_t slot_end, bool clock) {
        const int64_t now = now_real_ns();
        double inflight_ms = 0.0; if (have_req && a_est_end_ns > now) inflight_ms = (a_est_end_ns - now) / 1e6;
        const int qd = static_cast<int>(srv.queue_depth());
        int ul = 0, dl = 0, gap_now = 0, gap_min = 1 << 30; bool known = true;
        if (clock) { int P, N; bool dla, ula; busy_of(w, P, N, known, dla, ula); ul = nsm - P; dl = N; gap_now = P - N; }
        json::Obj e; e.kv("type", "estimate").kv("t_ns", static_cast<long long>(now)).kv("sfn", clock ? static_cast<int>((w / kSzSlotsPerFrame) & 1023) : -1).kv("slot", clock ? static_cast<int>(w % kSzSlotsPerFrame) : -1)
         .kv("ul_sm", ul).kv("dl_sm", dl).kv("queue", qd).kv("inflight", have_req ? 1 : 0).kv("known", known);
        std::string ms = "{", sm = "{"; bool first = true;
        for (const std::string& m : model_names) {
            EngineSet& s = models[m]; if (!s.loaded) continue;
            int pidx = -1; bool kn = true;
            if (clock) pidx = choose_part(s, 0, w, slot_end, kn, false); else { auto it = part_of.find({nsm - pusch_sm.back(), pdsch_sm.back()}); if (it != part_of.end()) for (int i : it->second) { if (s.cls_index(parts[i].cls) >= 0) { pidx = i; break; } } }
            double t = 0.0; int c = 0;
            if (pidx >= 0) { c = parts[pidx].cls; t = set_time_ms(s, s.cls_index(c)); }
            else { c = s.classes.front(); t = set_time_ms(s, 0) + 0.5; }   // paused now: assume the smallest class after the window
            gap_min = std::min(gap_min, c);
            const double est = inflight_ms + qd * t + dec_ema + t;
            if (!first) { ms += ','; sm += ','; } first = false;
            ms += json::quote(m) + ':' + json::num(est); sm += json::quote(m) + ':' + std::to_string(c);
        }
        ms += '}'; sm += '}';
        e.kv("yolo_sm_now", gap_now).kv("yolo_sm_min", gap_min == (1 << 30) ? 0 : gap_min).raw("est_ms", ms).raw("est_sm", sm);
        return e.str();
    }
    std::string report_json(int window_ms) {
        std::lock_guard<std::mutex> g(rep_m);
        const int64_t now = now_real_ns(); const int64_t from = now - static_cast<int64_t>(window_ms) * 1000000LL;
        while (!done.empty() && done.front().t_end_ns < from) done.pop_front();
        double sum = 0.0; std::map<std::string, std::pair<int, double>> by; std::string per = "[";
        for (size_t i = 0; i < done.size(); i++) { sum += done[i].sm_mean; by[done[i].model].first++; by[done[i].model].second += done[i].sm_mean; if (i) per += ','; per += json::num(done[i].sm_mean); }
        per += ']';
        json::Obj o; o.kv("type", "sm_report").kv("t_ns", static_cast<long long>(now)).kv("window_ms", window_ms).kv("requests", static_cast<int>(done.size()));
        if (done.empty()) o.null("sm_mean"); else o.kv("sm_mean", sum / done.size());
        o.raw("sm_mean_per_request", per);
        std::string bm = "{"; bool first = true; for (auto& kv : by) { if (!first) bm += ','; first = false; bm += json::quote(kv.first) + ":{\"n\":" + std::to_string(kv.second.first) + ",\"sm_mean\":" + json::num(kv.second.second / kv.second.first) + "}"; } bm += '}';
        o.raw("sm_by_model", bm);
        if (slot_samples > 0) { o.kv("yolo_sm_time_mean", yolo_sm_sum / slot_samples).kv("ran_sm_time_mean", ran_sm_sum / slot_samples).kv("slots", static_cast<long long>(slot_samples)); } else { o.null("yolo_sm_time_mean"); o.null("ran_sm_time_mean"); }
        yolo_sm_sum = 0.0; ran_sm_sum = 0.0; slot_samples = 0;
        return o.str();
    }
    // ---- agent: one request -------------------------------------------------------------------------------------------
    void fail_request(const char* reason) {
        agent::Result r; r.id = cur.id; r.client = cur.client; r.model = cur.model; r.status = "fail"; r.reason = reason; r.t_recv_ns = cur.t_recv_ns; r.t_end_ns = now_real_ns(); r.latency_ms = (r.t_end_ns - cur.t_recv_ns) / 1e6;
        srv.push_result(r); have_req = false; aset = nullptr; a_fail++; a_chunking = false;
    }
    bool begin_request(int64_t w, int64_t slot_end, bool clock, std::FILE* lf) {
        auto it = models.find(cur.model);
        if (it == models.end() || !it->second.loaded) { fail_request("no_engine"); return false; }
        aset = &it->second; EngineSet& s = *aset;
        bool known = true;
        if (clock) a_pidx = choose_part(s, 0, w, slot_end, known, false);
        else { a_pidx = -1; auto pt = part_of.find({nsm - pusch_sm.back(), pdsch_sm.back()}); if (pt != part_of.end()) for (int i : pt->second) { if (s.cls_index(parts[i].cls) >= 0) { a_pidx = i; break; } } }
        if (a_pidx < 0) return false;                       // paused now: keep the request, retry next loop
        const Part& p = parts[a_pidx];
        if (prof_s > 0 && !prof_on && !prof_done) { cudaProfilerStart(); prof_on = true; prof_t0 = now_real_ns(); if (lf) { std::fprintf(lf, "agent: cudaProfilerStart at req %llu (%d s)\n", static_cast<unsigned long long>(cur.id), prof_s); std::fflush(lf); } }
        if (qos_on && cur.deadline_ms > 0) {   // docs/XAPP_INTERFACE.md: a request that certainly cannot meet its deadline is refused at once (fail / overload)
            const double est = (now_real_ns() - cur.t_recv_ns) / 1e6 + dec_ema + set_time_ms(s, s.cls_index(p.cls));
            if (est > cur.deadline_ms) { if (lf) { std::fprintf(lf, "agent: req %llu %s refused: estimate %.1f ms > deadline %.1f ms (class %d)\n", static_cast<unsigned long long>(cur.id), cur.model.c_str(), est, cur.deadline_ms, p.cls); std::fflush(lf); } fail_request("overload"); return false; }
        }
        a_t_start = now_real_ns();
        if (!njd) { fail_request("decode_error"); if (lf) { std::fprintf(lf, "agent: no nvjpeg decoder\n"); std::fflush(lf); } return false; }
        float* target = s.in_half ? d_dec : static_cast<float*>(s.buf["images"]);
        float t_ms[3] = {0, 0, 0};
        cudaStream_t ds = dec_s ? dec_s : p.s;
        const unsigned char* jsrc = cur.jpeg.data(); if (h_jpeg && cur.jpeg.size() <= h_jpeg_cap) { std::memcpy(h_jpeg, cur.jpeg.data(), cur.jpeg.size()); jsrc = h_jpeg; }
        const int st = a_skip_decode ? 0 : njd_decode_tiles(njd, jsrc, cur.jpeg.size(), target, s.net, s.batch, s.batch > 1 ? 1 : 0, ds, t_ms, awh, arois);
        if (a_skip_decode) { awh[0] = 3840; awh[1] = 2160; for (int i = 0; i < 4 * s.batch; i++) arois[i] = (i % 4 < 2) ? 0 : (i % 4 == 2 ? 3840 : 2160); }
        if (st != 0) { fail_request("decode_error"); if (lf) { std::fprintf(lf, "agent: decode failed (%d) req %llu %zu bytes\n", st, static_cast<unsigned long long>(cur.id), cur.jpeg.size()); std::fflush(lf); } return false; }
        if (s.in_half && !a_skip_decode) { const size_t n = s.bytes["images"] / 2; k_f2h<<<static_cast<unsigned>((n + 255) / 256), 256, 0, ds>>>(d_dec, static_cast<__half*>(s.buf["images"]), n); }
        if (!a_skip_decode) cudaStreamSynchronize(ds);
        a_dec_ms = (now_real_ns() - a_t_start) / 1e6; dec_ema = 0.9 * dec_ema + 0.1 * a_dec_ms;
        apipe = Pipe(); apipe.set = &s; apipe.cls_hist.assign(s.classes.size(), 0); apipe.tiles_left = s.batch; a_cls.clear();
        a_est_end_ns = now_real_ns() + static_cast<int64_t>(set_time_ms(s, s.cls_index(p.cls)) * 1e6);
        return true;
    }
    // enqueue segment k of the agent request on partition pidx (no blocking sync; completion is polled)
    // cut a captured graph into pieces of chunk_n nodes in topological order; each piece is a clone with the other nodes removed
    // (edges to earlier pieces are implied by stream order, edges to later pieces cannot exist in topological order)
    bool chunk_graph(cudaGraph_t g, int chunk, std::vector<cudaGraphExec_t>& out, cudaStream_t st, std::string& err) {
        size_t n = 0; CKR(cudaGraphGetNodes(g, nullptr, &n)); if (n == 0) return true;
        std::vector<cudaGraphNode_t> nodes(n); CKR(cudaGraphGetNodes(g, nodes.data(), &n));
        // TensorRT graphs carry edge data (programmatic dependent launch), which the legacy cudaGraphGetEdges refuses ("semantically lossy"): read the topology with _v2
        size_t ne = 0; CKR(cudaGraphGetEdges_v2(g, nullptr, nullptr, nullptr, &ne)); std::vector<cudaGraphNode_t> from(ne), to(ne); std::vector<cudaGraphEdgeData> ed(ne); if (ne) CKR(cudaGraphGetEdges_v2(g, from.data(), to.data(), ed.data(), &ne));
        std::map<cudaGraphNode_t, int> idx; for (size_t i = 0; i < n; i++) idx[nodes[i]] = static_cast<int>(i);
        std::vector<int> indeg(n, 0); std::vector<std::vector<int>> adj(n);
        for (size_t e = 0; e < ne; e++) { adj[idx[from[e]]].push_back(idx[to[e]]); indeg[idx[to[e]]]++; }
        std::vector<int> order; std::deque<int> q; for (size_t i = 0; i < n; i++) { if (indeg[i] == 0) q.push_back(static_cast<int>(i)); }
        while (!q.empty()) { const int u = q.front(); q.pop_front(); order.push_back(u); for (int v : adj[u]) { if (--indeg[v] == 0) q.push_back(v); } }
        if (order.size() != n) { err = "graph is not a DAG"; return false; }
        for (size_t c0 = 0; c0 < n; c0 += chunk) {
            std::set<int> keep(order.begin() + c0, order.begin() + std::min(n, c0 + chunk));
            cudaGraph_t cg = nullptr; CKR(cudaGraphClone(&cg, g));
            for (size_t i = 0; i < n; i++) { if (keep.count(static_cast<int>(i))) continue; cudaGraphNode_t cn = nullptr; CKR(cudaGraphNodeFindInClone(&cn, nodes[i], cg)); CKR(cudaGraphDestroyNode(cn)); }
            cudaGraphExec_t ge = nullptr; CKR(cudaGraphInstantiate(&ge, cg, 0)); cudaGraphDestroy(cg); cudaGraphUpload(ge, st); out.push_back(ge);
        }
        return true;
    }
    bool agent_enqueue(int pidx, std::FILE* lf) {
        EngineSet& s = *aset; const Part& p = parts[pidx]; const int ci = s.cls_index(p.cls); const int k = apipe.k;
        if (k == 0) apipe.t_inf_start = now_mono_ns();
        if (apipe.prev_part >= 0 && apipe.prev_part != pidx && k > 0) { cudaStreamWaitEvent(p.s, s.ev_end[k - 1], 0); apipe.switches++; }
        cudaEventRecord(s.ev_start[k], p.s);
        auto git = s.graphs.find({k, pidx}); auto cit = s.chunks.find({k, pidx});
        if (!a_skip_infer && a_use_graph && cit != s.chunks.end() && !cit->second.empty()) {
            a_chunks = &cit->second; a_chunk_part = &p; a_chunk_k = k; a_chunk_next = 0; a_chunking = true;
            apipe.prev_part = pidx; apipe.segments++; apipe.cls_hist[ci]++; a_cls.push_back(p.cls); apipe.k++;
            return agent_pump(lf);                    // ev_end[k] is recorded by the pump after the last piece
        }
        if (!a_skip_infer && a_use_graph && git != s.graphs.end()) { if (cudaGraphLaunch(git->second, p.s) != cudaSuccess) { if (lf) { std::fprintf(lf, "agent: graph launch failed seg %d part %d\n", k, pidx); std::fflush(lf); } fail_request("internal"); return false; } }
        else { bind_aux(s.segs[k], ci, p);
        if (!a_skip_infer && !s.segs[k].exec[ci]->enqueueV3(p.s)) { if (lf) { std::fprintf(lf, "agent: enqueue failed seg %d class %d\n", k, p.cls); std::fflush(lf); } fail_request("internal"); return false; }
        }
        cudaEventRecord(s.ev_end[k], p.s);
        apipe.prev_part = pidx; apipe.segments++; apipe.cls_hist[ci]++; a_cls.push_back(p.cls); apipe.k++;
        return true;
    }
    // chunked graph launch: keep at most chunk_ahead pieces queued on the partition stream; called every loop iteration while a segment is in flight
    bool agent_pump(std::FILE* lf) {
        if (!a_chunking || !aset) return true;
        EngineSet& s = *aset; const Part& p = *a_chunk_part; const int total = static_cast<int>(a_chunks->size());
        while (a_chunk_next < total) {
            if (a_chunk_next >= chunk_ahead && cudaEventQuery(s.chunk_ev[a_chunk_next - chunk_ahead]) != cudaSuccess) return true;
            if (cudaGraphLaunch((*a_chunks)[a_chunk_next], p.s) != cudaSuccess) { if (lf) { std::fprintf(lf, "agent: chunk %d launch failed seg %d\n", a_chunk_next, a_chunk_k); std::fflush(lf); } a_chunking = false; fail_request("internal"); return false; }
            cudaEventRecord(s.chunk_ev[a_chunk_next], p.s); a_chunk_next++;
        }
        cudaEventRecord(s.ev_end[a_chunk_k], p.s); a_chunking = false; return true;
    }
    void finish_request(std::FILE* lf) {
        EngineSet& s = *aset; const int64_t t_end = now_real_ns();
        for (int k = 0; k < s.nseg; k++) { float ms = 0; const int ci = s.cls_index(a_cls[k]); if (cudaEventElapsedTime(&ms, s.ev_start[k], s.ev_end[k]) == cudaSuccess) { const double al = (ci < static_cast<int>(s.ema_n.size()) && s.ema_n[ci] < 5) ? 0.5 : 0.1; s.seg_ema[ci][k] = s.seg_ema[ci][k] == 0.0 ? ms : (1.0 - al) * s.seg_ema[ci][k] + al * ms; } }
        if (!a_cls.empty()) { const int ci0 = s.cls_index(a_cls.front()); if (ci0 >= 0 && ci0 < static_cast<int>(s.ema_n.size())) s.ema_n[ci0]++; }
        if (lf && s.chunk_log_n < 3 && a_pidx >= 0) { auto cit = s.chunks.find({0, a_pidx}); if (cit != s.chunks.end() && !cit->second.empty()) {
            std::vector<double> d; float ms = 0; if (cudaEventElapsedTime(&ms, s.ev_start[0], s.chunk_ev[0]) == cudaSuccess) d.push_back(ms);
            for (size_t j = 1; j < cit->second.size(); j++) { if (cudaEventElapsedTime(&ms, s.chunk_ev[j - 1], s.chunk_ev[j]) == cudaSuccess) d.push_back(ms); }
            if (!d.empty()) { std::fprintf(lf, "agent: %s class %d: %zu graph pieces of %d nodes, piece ms p50 %.2f p90 %.2f max %.2f (queued at most %d)\n", s.model.c_str(), parts[a_pidx].cls, cit->second.size(), chunk_n, pct(d, 0.5), pct(d, 0.9), *std::max_element(d.begin(), d.end()), chunk_ahead); std::fflush(lf); s.chunk_log_n++; } } }
        agent::Result r; r.id = cur.id; r.client = cur.client; r.model = cur.model; r.t_recv_ns = cur.t_recv_ns; r.t_start_ns = a_t_start; r.t_end_ns = t_end;
        r.wait_ms = (a_t_start - cur.t_recv_ns) / 1e6; r.decode_ms = a_dec_ms; r.infer_ms = (now_mono_ns() - apipe.t_inf_start) / 1e6; r.latency_ms = (t_end - cur.t_recv_ns) / 1e6;
        r.sm = a_cls.empty() ? 0 : a_cls.front(); double sm_sum = 0; for (int c : a_cls) sm_sum += c; r.sm_mean = a_cls.empty() ? 0.0 : sm_sum / a_cls.size(); r.tiles = s.batch;
        std::vector<float> out;
        if (read_output(s, out)) {
            // the end-to-end head: per slot k_out rows [x1 y1 x2 y2 conf cls] in net pixels, sorted by score; map through the slot ROI, class-aware NMS between slots
            struct Cand { float x1, y1, x2, y2, conf; int cls, slot; }; std::vector<Cand> cand;
            const float fw = static_cast<float>(awh[0]), fh = static_cast<float>(awh[1]);
            for (int sl = 0; sl < s.batch; sl++) {
                const int* roi = &arois[4 * sl]; bool dup = false; for (int t = 0; t < sl && !dup; t++) dup = std::equal(roi, roi + 4, &arois[4 * t]); if (dup) continue;
                const float scale = std::min(static_cast<float>(s.net) / roi[2], static_cast<float>(s.net) / roi[3]);
                const int nw = static_cast<int>(std::lround(roi[2] * scale)), nh = static_cast<int>(std::lround(roi[3] * scale)); const float ox = static_cast<float>((s.net - nw) / 2), oy = static_cast<float>((s.net - nh) / 2);
                for (int k = 0; k < s.k_out; k++) {
                    const float* q = out.data() + (static_cast<size_t>(sl) * s.k_out + k) * 6; if (q[4] < conf_thr) break;
                    auto mx = [&](float v) { return std::min(std::max((v - ox) / scale + roi[0], 0.f), fw); }; auto my = [&](float v) { return std::min(std::max((v - oy) / scale + roi[1], 0.f), fh); };
                    cand.push_back(Cand{mx(q[0]), my(q[1]), mx(q[2]), my(q[3]), q[4], static_cast<int>(q[5] + 0.5f), sl});
                }
            }
            std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.conf > b.conf; });
            std::vector<char> dead(cand.size(), 0); std::string dj = "["; int nd = 0;
            for (size_t i = 0; i < cand.size() && nd < 300; i++) {
                if (dead[i]) continue;
                for (size_t j = i + 1; j < cand.size(); j++) {
                    if (dead[j] || cand[j].slot == cand[i].slot || cand[j].cls != cand[i].cls) continue;
                    const float ix = std::max(0.f, std::min(cand[i].x2, cand[j].x2) - std::max(cand[i].x1, cand[j].x1)), iy = std::max(0.f, std::min(cand[i].y2, cand[j].y2) - std::max(cand[i].y1, cand[j].y1));
                    const float inter = ix * iy, uni = (cand[i].x2 - cand[i].x1) * (cand[i].y2 - cand[i].y1) + (cand[j].x2 - cand[j].x1) * (cand[j].y2 - cand[j].y1) - inter;
                    if (uni > 0.f && inter / uni > iou_thr) dead[j] = 1;
                }
                if (cur.want_dets) { char b[160]; std::snprintf(b, sizeof b, "%s{\"cls\":%d,\"conf\":%.3f,\"xyxy\":[%.1f,%.1f,%.1f,%.1f]}", nd ? "," : "", cand[i].cls, cand[i].conf, cand[i].x1, cand[i].y1, cand[i].x2, cand[i].y2); dj += b; }
                nd++;
            }
            r.detections = nd; if (cur.want_dets) r.dets_json = dj + "]"; r.status = "success";
        } else { r.status = "fail"; r.reason = "internal"; }
        srv.push_result(r);
        { std::lock_guard<std::mutex> g(rep_m); done.push_back(DoneRec{t_end, cur.model, r.sm_mean}); while (done.size() > 10000) done.pop_front(); }
        if (r.status == "success") a_ok++; else a_fail++;
        have_req = false; aset = nullptr; a_pidx = -1;
    }

    // ---- main loop ----------------------------------------------------------------------------------------------------
    void run() {
        pthread_setname_np(pthread_self(), "dapp_tenant");
        cuCtxSetCurrent(prim);
        std::FILE* lf = log_file.empty() ? nullptr : std::fopen(log_file.c_str(), "a");
        if (agent_on) {
            njd = njd_create(0); if (!njd) njd = njd_create(1);
            // DAPP_AGENT_DEC_PART=1 (2026-11-02 08:45): no separate decode stream; the decode's CUDA kernels (nvjpeg post-processing, k_f2h) run on
            // the chosen partition's stream, i.e. inside the gap SMs. A primary-context stream runs its kernels on ANY SM, including the SMs of
            // cuPHY's order / PUCCH / PRACH / DL-ctrl green contexts (1-cell runs: every request put the L1 into recovery).
            if (envi("DAPP_AGENT_DEC_PART", 1)) dec_s = nullptr;
            else if (cudaStreamCreateWithFlags(&dec_s, cudaStreamNonBlocking) != cudaSuccess) dec_s = nullptr;
            // warm-up: decode the largest images we expect (DAPP_AGENT_WARMUP, comma list) so that every nvjpeg / NVJPG buffer reaches
            // its final size now; a later growth would cudaFree + cudaMalloc on the request path and stall cuPHY (gdb 04:30)
            if (!h_jpeg) { h_jpeg_cap = static_cast<size_t>(envi("DAPP_AGENT_JPEG_MB", 32)) << 20; if (cudaHostAlloc(reinterpret_cast<void**>(&h_jpeg), h_jpeg_cap, cudaHostAllocDefault) != cudaSuccess) { h_jpeg = nullptr; h_jpeg_cap = 0; } }
            if (!h_out) { size_t mx = 0; for (auto& kv : models) { if (kv.second.loaded && kv.second.bytes.count("output0")) mx = std::max(mx, kv.second.bytes["output0"]); } if (mx && cudaHostAlloc(&h_out, mx, cudaHostAllocDefault) == cudaSuccess) h_out_cap = mx; else h_out = nullptr; }
            if (njd && !models.empty() && models[model_names.front()].loaded) {
                EngineSet& s0 = models[model_names.front()];
                std::stringstream ws(envs("DAPP_AGENT_WARMUP", "/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/img4k/img4k_0.jpg,/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/img4k/uhd_0_q95.jpg")); std::string wf;
                while (std::getline(ws, wf, ',')) {
                    if (wf.empty()) continue; auto b = slurp(wf); if (b.empty()) { if (lf) { std::fprintf(lf, "warm-up: cannot read %s\n", wf.c_str()); } continue; }
                    float t_ms[3] = {0, 0, 0}; int wh[2] = {0, 0}; int rois[64];
                    float* target = s0.in_half ? d_dec : static_cast<float*>(s0.buf["images"]);
                    const unsigned char* wsrc = reinterpret_cast<const unsigned char*>(b.data()); if (h_jpeg && b.size() <= h_jpeg_cap) { std::memcpy(h_jpeg, b.data(), b.size()); wsrc = h_jpeg; }
                    for (int rep = 0; rep < 2; rep++) { const int st = njd_decode_tiles(njd, wsrc, b.size(), target, s0.net, s0.batch, s0.batch > 1 ? 1 : 0, dec_s ? dec_s : parts.front().s, t_ms, wh, rois); if (lf && rep == 1) std::fprintf(lf, "warm-up: %s %dx%d status %d (%.1f + %.1f + %.1f ms)\n", wf.c_str(), wh[0], wh[1], st, t_ms[0], t_ms[1], t_ms[2]); }
                }
                cudaStreamSynchronize(dec_s ? dec_s : parts.front().s);
            }
            size_t need = 0; for (auto& kv : models) { if (kv.second.loaded && kv.second.in_half) need = std::max(need, kv.second.bytes["images"] / 2); }
            if (need) { if (cudaMalloc(&d_dec, need * sizeof(float)) != cudaSuccess) d_dec = nullptr; d_dec_n = need; }
            unsigned eng = 0, cores = 0; if (njd) njd_info(njd, &eng, &cores);
            if (lf) { std::fprintf(lf, "==== agent: port %d, models", srv_port); for (const std::string& m : model_names) std::fprintf(lf, " %s", m.c_str()); std::fprintf(lf, ", nvjpeg %s (%u engines), freerun %d\n", njd ? "ok" : "MISSING", eng, freerun ? 1 : 0); std::fflush(lf); }
        }
        if (lf) { std::fprintf(lf, "==== tenant ready (engines loaded and validated, decoder warm) %lld\n", static_cast<long long>(now_real_ns() / 1000000000LL)); std::fflush(lf); }
        std::FILE* cf = csv_file.empty() ? nullptr : std::fopen(csv_file.c_str(), "w");
        if (cf) { std::fprintf(cf, "abs_slot,sfn,slot,pos,P,N,part,class,set,segments,paused_us,known,plan_ul_sm,plan_dl_sm\n"); }
        int64_t last_log = now_mono_ns(); int64_t last_w = -1; int segs_in_slot = 0, segs_q = 0, segs_b = 0; int64_t paused_in_slot = 0; int cur_part_slot = -1; bool known_slot = true;
        uint64_t q_inf_at_log = 0, b_inf_at_log = 0, q_req_at_log = 0, b_req_at_log = 0;
        auto flush_slot = [&](int64_t w) {
            if (cf && last_w >= 0) {
                bool pred; const SlotPlanEntry* e = plan_of(last_w, pred);
                std::fprintf(cf, "%lld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%d,%d,%d\n", static_cast<long long>(last_w), static_cast<int>((last_w / kSzSlotsPerFrame) & 1023), static_cast<int>(last_w % kSzSlotsPerFrame), static_cast<int>(last_w % 10),
                             cur_part_slot >= 0 ? parts[cur_part_slot].P : -1, cur_part_slot >= 0 ? parts[cur_part_slot].N : -1, cur_part_slot, cur_part_slot >= 0 ? parts[cur_part_slot].cls : 0,
                             segs_b > segs_q ? busy.batch : (segs_q > 0 ? quiet.batch : 0), segs_in_slot, static_cast<long long>(paused_in_slot / 1000), known_slot ? 1 : 0,
                             (e && !pred) ? (e->ul_any ? static_cast<int>(e->ul_sm) : 0) : -1, (e && !pred) ? (e->dl_any ? static_cast<int>(e->dl_sm) : 0) : -1);
            }
            last_w = w; segs_in_slot = 0; segs_q = 0; segs_b = 0; paused_in_slot = 0; cur_part_slot = -1; known_slot = true;
        };
        auto stats = [&](Pipe& pp, const char* nm, uint64_t& inf_at, uint64_t& req_at, double sec) {
            if (!pp.set || !pp.set->loaded) return;
            std::fprintf(lf, " | %s(b%d): inferences %llu (%.1f/s) requests %llu (%.1f/s) inf p50 %.2f p90 %.2f max %.2f ms req p50 %.2f p90 %.2f max %.2f ms segments %llu switches %llu classes",
                         nm, pp.set->batch, static_cast<unsigned long long>(pp.inferences - inf_at), (pp.inferences - inf_at) / sec, static_cast<unsigned long long>(pp.requests - req_at), (pp.requests - req_at) / sec,
                         pct(pp.inf_lat, 0.5), pct(pp.inf_lat, 0.9), pp.inf_lat.empty() ? 0.0 : *std::max_element(pp.inf_lat.begin(), pp.inf_lat.end()),
                         pct(pp.req_lat, 0.5), pct(pp.req_lat, 0.9), pp.req_lat.empty() ? 0.0 : *std::max_element(pp.req_lat.begin(), pp.req_lat.end()),
                         static_cast<unsigned long long>(pp.segments), static_cast<unsigned long long>(pp.switches));
            for (size_t i = 0; i < pp.set->classes.size(); i++) { std::fprintf(lf, " %d:%llu", pp.set->classes[i], static_cast<unsigned long long>(pp.cls_hist[i])); }
            inf_at = pp.inferences; req_at = pp.requests; pp.inf_lat.clear(); pp.req_lat.clear();
        };
        while (!stop.load(std::memory_order_acquire)) {
            if (prof_on && now_real_ns() - prof_t0 > static_cast<int64_t>(prof_s) * 1000000000LL) { cudaProfilerStop(); prof_on = false; prof_done = true; if (lf) { std::fprintf(lf, "agent: cudaProfilerStop\n"); std::fflush(lf); } }
            const int64_t tnow = now_mono_ns();
            if (tnow - last_log >= static_cast<int64_t>(log_s) * 1000000000LL) {
                const double sec = (tnow - last_log) / 1e9;
                if (lf) {
                    const uint64_t tiles = (qpipe.inferences - q_inf_at_log) * quiet.batch + (bpipe.inferences - b_inf_at_log) * (busy.loaded ? busy.batch : 0);
                    std::fprintf(lf, "[%lld] tiles %llu (%.1f/s = %.1f 4K/s) pauses %llu (%.1f ms)", static_cast<long long>(now_real_ns() / 1000000000LL), static_cast<unsigned long long>(tiles), tiles / sec, tiles / sec / tiles_per_request,
                                 static_cast<unsigned long long>(pauses), pause_ns / 1e6);
                    stats(qpipe, "quiet", q_inf_at_log, q_req_at_log, sec); stats(bpipe, "busy", b_inf_at_log, b_req_at_log, sec);
                    if (agent_on) { std::fprintf(lf, " | agent ok %llu fail %llu queue %zu received %llu rejected %llu dec_ema %.1f ms", static_cast<unsigned long long>(a_ok), static_cast<unsigned long long>(a_fail), srv.queue_depth(), static_cast<unsigned long long>(srv.requests_received()), static_cast<unsigned long long>(srv.rejected()), dec_ema); }
                    std::fprintf(lf, " | sizer plans %llu incomplete %llu wait max %.1f us plan max %.1f us | PUSCH slots %llu infeasible %llu prof",
                                 static_cast<unsigned long long>(g_sizer_stats.plans.load()), static_cast<unsigned long long>(g_sizer_stats.incomplete.load()),
                                 g_sizer_stats.wait_max_ns.load() / 1e3, g_sizer_stats.plan_max_ns.load() / 1e3,
                                 static_cast<unsigned long long>(g_sizer_stats.pusch_slots.load()), static_cast<unsigned long long>(g_sizer_stats.pusch_infeasible.load()));
                    for (size_t i = 0; i < pusch_sm.size(); i++) { std::fprintf(lf, " %d:%llu/%llu", pusch_sm[i], static_cast<unsigned long long>(g_sizer_stats.pusch_prof[i].load()), static_cast<unsigned long long>(cuphydriver_dapp_profile_count(0, static_cast<int>(i)))); }
                    std::fprintf(lf, " fallback %llu | PDSCH slots %llu infeasible %llu prof", static_cast<unsigned long long>(cuphydriver_dapp_fallback_count(0)),
                                 static_cast<unsigned long long>(g_sizer_stats.pdsch_slots.load()), static_cast<unsigned long long>(g_sizer_stats.pdsch_infeasible.load()));
                    for (size_t i = 0; i < pdsch_sm.size(); i++) { std::fprintf(lf, " %d:%llu/%llu", pdsch_sm[i], static_cast<unsigned long long>(g_sizer_stats.pdsch_prof[i].load()), static_cast<unsigned long long>(cuphydriver_dapp_profile_count(1, static_cast<int>(i)))); }
                    std::fprintf(lf, " fallback %llu\n", static_cast<unsigned long long>(cuphydriver_dapp_fallback_count(1)));
                    std::fflush(lf);
                }
                if (cf) std::fflush(cf);
                last_log = tnow;
            }
            // ---- agent path: requests from the Resource Allocator (sequential), estimates every slot
            if (agent_on) {
                int64_t w = 0, slot_end = 0; const bool clk = current_slot(w, slot_end);
                if (clk && w != last_w) {
                    if (last_w >= 0) { int P, N; bool kn, dla, ula; busy_of(last_w, P, N, kn, dla, ula); std::lock_guard<std::mutex> g(rep_m); yolo_sm_sum += (have_req && a_pidx >= 0) ? parts[a_pidx].cls : 0; ran_sm_sum += (nsm - P) + N; slot_samples++; }
                    flush_slot(w);
                    const std::string e = estimate_json(w, slot_end, true); srv.publish_estimate(e); { std::lock_guard<std::mutex> g(rep_m); last_estimate = e; }
                }
                if (have_req && aset) {
                    EngineSet& s = *aset;
                    if (a_chunking) { if (!agent_pump(lf)) continue; if (a_chunking) { std::this_thread::sleep_for(std::chrono::microseconds(5)); if (!loop_on) continue; } }
                    if (have_req && aset && a_chunking) { }
                    else if (apipe.k < s.nseg) {
                        // next segment once the previous one finished (keeps the class choice current for segmented sets)
                        if (apipe.k == 0 || cudaEventQuery(s.ev_end[apipe.k - 1]) == cudaSuccess) {
                            bool kn = true; int pidx = clk ? choose_part(s, apipe.k, w, slot_end, kn, false) : a_pidx;
                            if (pidx >= 0) { a_pidx = pidx; if (cur_part_slot < 0) cur_part_slot = pidx; agent_enqueue(pidx, lf); segs_in_slot++; }
                            else { pauses++; std::this_thread::sleep_for(std::chrono::microseconds(pause_us)); }
                        } else { std::this_thread::sleep_for(std::chrono::microseconds(50)); }
                    } else if (cudaEventQuery(s.ev_end[s.nseg - 1]) == cudaSuccess) { finish_request(lf); }
                    else { std::this_thread::sleep_for(std::chrono::microseconds(50)); }
                    if (!loop_on) continue;
                } else {
                    if (!have_req && srv.pop_request(cur)) { have_req = true; }
                    if (have_req) { if (!begin_request(w, slot_end, clk, lf) && have_req) { pauses++; std::this_thread::sleep_for(std::chrono::microseconds(pause_us)); } continue; }
                    if (!loop_on) { if (!clk) { std::lock_guard<std::mutex> g(rep_m); } std::this_thread::sleep_for(std::chrono::microseconds(100)); continue; }
                }
            }
            // ---- which pipeline and partition now?
            Pipe* pp = &qpipe; int pidx = -1; bool known = true; int64_t w = 0, slot_end = 0;
            if (fixed > 0 && fixed_gap_part < 0) { pidx = 0; }
            else if (fixed_gap_part >= 0) { pidx = fixed_gap_part; if (current_slot(w, slot_end) && w != last_w) flush_slot(w); }
            else if (!current_slot(w, slot_end)) { std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }
            else {
                if (w != last_w) flush_slot(w);
                int P, N; bool dla, ula; busy_of(w, P, N, known, dla, ula);
                pp = (ula && busy.loaded) ? &bpipe : &qpipe;
                // horizon (2026-11-02): the next segment runs until est_us from now; it must fit the busy set of every slot it
                // spans (full engines of 26m/26l/26x run 15-40 ms = 30-80 slots; unknown slots use the previous frame's plan)
                {
                    double est_us = 0.0;
                    for (size_t ci = 0; ci < pp->set->classes.size(); ci++) { est_us = std::max(est_us, pp->set->seg_ema[ci][pp->k] * 1000.0); }
                    const int64_t left_ns = slot_end - now_real_ns();
                    const int64_t span_ns = static_cast<int64_t>((guard_us + est_us) * 1000.0) - left_ns;   // beyond this slot
                    const int H = span_ns > 0 ? static_cast<int>(span_ns / 500000) + 1 : 0;
                    for (int j = 1; j <= std::min(H, horizon_max); j++) {
                        int P2, N2; bool kn2, dla2, ula2; busy_of(w + j, P2, N2, kn2, dla2, ula2); known = known && kn2;
                        P = std::min(P, P2); N = std::max(N, N2); dla = dla || dla2; ula = ula || ula2;
                    }
                }
                pidx = part_for(P, N, dla, ula, *pp->set, pp == &qpipe);
                known_slot = known_slot && known;
            }
            if (pidx < 0) {
                const int64_t t_p = now_mono_ns(); pauses++;
                std::this_thread::sleep_for(std::chrono::microseconds(pause_us));
                const int64_t d = now_mono_ns() - t_p; pause_ns += d; paused_in_slot += d;
                continue;
            }
            if (cur_part_slot < 0) cur_part_slot = pidx;
            if (!step(*pp, pidx, lf)) { stop.store(true); break; }
            segs_in_slot++; if (pp == &qpipe) segs_q++; else segs_b++;
        }
        if (cf) { flush_slot(-1); std::fclose(cf); }
        if (lf) { std::fprintf(lf, "tenant thread stopped: quiet inferences %llu busy inferences %llu agent ok %llu fail %llu\n", static_cast<unsigned long long>(qpipe.inferences), static_cast<unsigned long long>(bpipe.inferences), static_cast<unsigned long long>(a_ok), static_cast<unsigned long long>(a_fail)); std::fclose(lf); }
        if (njd) { njd_destroy(njd); njd = nullptr; }
        if (dec_s) { cudaStreamDestroy(dec_s); dec_s = nullptr; }
    }

    bool init(std::string& err) {
        quiet.name = "quiet"; quiet.dir = envs("DAPP_TENANT_DIR", "/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/seg26n_fine");
        quiet.input = envs("DAPP_TENANT_INPUT", "/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/seg26n/input_uhd_b8_fp16.bin"); quiet.batch = envi("DAPP_TENANT_BATCH", 8);
        busy.name = "busy"; busy.dir = envs("DAPP_TENANT_BUSY_DIR", ""); busy.input = envs("DAPP_TENANT_BUSY_INPUT", "/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/seg26n_b1/input_uhd_b1_fp16.bin"); busy.batch = envi("DAPP_TENANT_BUSY_BATCH", 1);
        log_file = envs("DAPP_TENANT_LOG", "/tmp/dapp_tenant.log"); csv_file = envs("DAPP_TENANT_CSV", "/tmp/dapp_tenant_slots.csv");
        classes = env_list("DAPP_TENANT_CLASSES", {40, 48, 64, 88}); std::sort(classes.begin(), classes.end());
        cpu = envi("DAPP_TENANT_CPU", 45); prio = envi("DAPP_TENANT_PRIO", 0); lookahead = envi("DAPP_TENANT_LOOKAHEAD", 1); guard_us = envi("DAPP_TENANT_GUARD_US", 60);
        ul_back = envi("DAPP_TENANT_UL_BACK", 4); dl_fwd = envi("DAPP_TENANT_DL_FWD", 2); fixed = envi("DAPP_TENANT_FIXED", 0); pause_us = envi("DAPP_TENANT_PAUSE_US", 20); log_s = envi("DAPP_TENANT_LOG_S", 10);
        dl_pause = envi("DAPP_TENANT_DL_PAUSE", 0); dl_max_class = envi("DAPP_TENANT_DL_MAX_CLASS", 0); ul_pause = envi("DAPP_TENANT_UL_PAUSE", 0); tiles_per_request = envi("DAPP_TENANT_TILES", 8);
        unknown_prev = std::string(envs("DAPP_TENANT_UNKNOWN", "prev")) != "max"; horizon_max = envi("DAPP_TENANT_HORIZON_MAX", 200); pred_frames = envi("DAPP_TENANT_PRED_FRAMES", 8);
        one_per_gap = envi("DAPP_TENANT_ONE_PER_GAP", -1);   // -1: union layout keeps only the largest class per (P, N) (the horizon rule never uses a smaller one); gap layout keeps all
        fb_guard = envi("DAPP_TENANT_FB_GUARD", 0); layout = envs("DAPP_TENANT_LAYOUT", "union"); if (one_per_gap < 0) one_per_gap = (layout == "union") ? 1 : 0;
        qos_on = envi("DAPP_AGENT_QOS", 1) != 0; prof_s = envi("DAPP_AGENT_PROFILE_S", 0);
        chunk_n = envi("DAPP_AGENT_GRAPH_CHUNK", 3); chunk_ahead = std::max(1, envi("DAPP_AGENT_CHUNK_AHEAD", 2));
        // guard 0 (14:30): 80 SM at 1 cell, the PUSCH fallback profile ran 0-2 of 48000 UL slots without a recovery. Pieces 3 nodes / 2 ahead (15:15): 3/1 was clean
        // but idled the GPU between pieces (+6..10 ms per request); 6/2 left one FATAL in ~10 runs (pieces up to 2.9 ms). (A comment appended to this line on 14:30
        // had commented out these assignments: chain21 ran with the member defaults 6/2.)
        agent_on = envi("DAPP_AGENT", 0) != 0; srv_port = envi("DAPP_AGENT_PORT", 9810); freerun = envi("DAPP_TENANT_FREERUN", 0) != 0; freerun_t0 = now_real_ns();
        a_skip_decode = envi("DAPP_AGENT_SKIP_DECODE", 0) != 0; a_skip_infer = envi("DAPP_AGENT_SKIP_INFER", 0) != 0;
        loop_on = !agent_on || envi("DAPP_TENANT_LOOP", 0) != 0;
        conf_thr = static_cast<float>(std::atof(envs("DAPP_AGENT_CONF", "0.25"))); iou_thr = static_cast<float>(std::atof(envs("DAPP_AGENT_IOU", "0.7")));
        if (agent_on) {
            // DAPP_TENANT_MODELS=26n=/dir,26s=/dir,... (default: models/full26<m> single-engine sets)
            std::string spec = envs("DAPP_TENANT_MODELS", "26n=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/full26n,26s=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/full26s,26m=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/full26m,26l=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/full26l,26x=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/full26x");
            std::stringstream ss(spec); std::string item;
            while (std::getline(ss, item, ',')) { const size_t k = item.find('='); if (k == std::string::npos) continue; EngineSet& m = models[item.substr(0, k)]; m.name = m.model = item.substr(0, k); m.dir = item.substr(k + 1); m.input = quiet.input; model_names.push_back(m.model); }
        }
        { std::vector<int> g = env_list("DAPP_TENANT_FIXED_GAP", {}); if (g.size() == 2 && fixed > 0) { fixed_gap_P = g[0]; fixed_gap_N = g[1]; } }
        if (fixed > 0 && std::find(classes.begin(), classes.end(), fixed) == classes.end()) { classes.push_back(fixed); std::sort(classes.begin(), classes.end()); }
        CKD(cuInit(0)); CKD(cuDeviceGet(&dev, 0)); CKD(cuDevicePrimaryCtxRetain(&prim, dev)); CKD(cuCtxSetCurrent(prim));
        CKD(cuDeviceGetAttribute(&nsm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)); CKD(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM));
        if (!build_partitions(err)) return false;
        stress = envs("DAPP_TENANT_STRESS", ""); stress_us = envi("DAPP_TENANT_STRESS_US", 200); stress_blocks = envi("DAPP_TENANT_STRESS_BLOCKS", 0); stress_gap_us = envi("DAPP_TENANT_STRESS_GAP_US", 0);
        if (!stress.empty()) {
            // synthetic set: 30 launches per "inference", one buffer per tensor-less set, calibrated to stress_us per launch on the first class
            quiet.nseg = 30; quiet.batch = 8; quiet.classes = classes; quiet.loaded = true; quiet.ev_start.resize(30); quiet.ev_end.resize(30);
            for (int k = 0; k < 30; k++) { CKR(cudaEventCreate(&quiet.ev_start[k])); CKR(cudaEventCreate(&quiet.ev_end[k])); }
            quiet.seg_ema.assign(classes.size(), std::vector<double>(30, 0.0));
            busy.loaded = false;
            stress_mb = envi("DAPP_TENANT_STRESS_MB", (stress == "l2" || stress == "l2w") ? 24 : 1024);
            if (stress == "l2" || stress == "hbm" || stress == "l2w" || stress == "hbmw") { stress_n = static_cast<size_t>(stress_mb) * 1024 * 1024 / sizeof(float4); CKR(cudaMalloc(&stress_buf, stress_n * sizeof(float4))); CKR(cudaMemset(stress_buf, 0, stress_n * sizeof(float4))); CKR(cudaMalloc(&stress_sink, 64)); }
            // calibrate on the first partition: how long does one launch take?
            const Part& p = parts.front(); cudaEvent_t a, b; CKR(cudaEventCreate(&a)); CKR(cudaEventCreate(&b)); const int cal_blocks = stress_blocks > 0 ? stress_blocks : p.cls * 4;
            if (stress == "tc") { CKR(cudaMalloc(&stress_tc_a, 2 * 256 * sizeof(__half))); CKR(cudaMemset(stress_tc_a, 0, 2 * 256 * sizeof(__half))); if (!stress_sink) CKR(cudaMalloc(&stress_sink, 64)); }
            int clock_khz = 0; cuDeviceGetAttribute(&clock_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
            spin_cycles = static_cast<long long>(clock_khz) * stress_us / 1000; read_iters = 1;
            for (int round = 0; round < 3; round++) {
                CKR(cudaEventRecord(a, p.s));
                stress_launch(p);
                CKR(cudaEventRecord(b, p.s)); CKR(cudaEventSynchronize(b)); float ms = 0; CKR(cudaEventElapsedTime(&ms, a, b));
                if (stress == "spin") { spin_cycles = static_cast<long long>(spin_cycles * (stress_us / 1000.0) / std::max(ms, 1e-3f)); }
                else { read_iters = std::max(1, static_cast<int>(read_iters * (stress_us / 1000.0) / std::max(ms, 1e-3f))); }
            }
            CKR(cudaEventRecord(a, p.s));
            stress_launch(p);
            CKR(cudaEventRecord(b, p.s)); CKR(cudaEventSynchronize(b)); float ms = 0; CKR(cudaEventElapsedTime(&ms, a, b));
            std::FILE* lf0 = log_file.empty() ? nullptr : std::fopen(log_file.c_str(), "a");
            if (lf0) { std::fprintf(lf0, "==== stress mode %s: %d MB buffer, launch %.0f us on class %d (spin cycles %lld, read iters %d, blocks %d, gap %d us)\n", stress.c_str(), stress_mb, ms * 1000.0, p.cls, spin_cycles, read_iters, cal_blocks, stress_gap_us); std::fclose(lf0); }
        } else {
            if (loop_on && !load_set(quiet, err)) return false;
            if (loop_on && !busy.dir.empty() && !load_set(busy, err)) return false;
            for (const std::string& m : model_names) { if (!load_set(models[m], err)) { err = "model " + m + ": " + err; return false; } }
        }
        qpipe.set = &quiet; qpipe.cls_hist.assign(quiet.classes.size(), 0); qpipe.tiles_left = tiles_per_request;
        bpipe.set = &busy; bpipe.cls_hist.assign(busy.classes.size(), 0); bpipe.tiles_left = tiles_per_request;
        std::FILE* lf = log_file.empty() ? nullptr : std::fopen(log_file.c_str(), "a");
        if (lf) {
            std::fprintf(lf, "==== dapp_tenant quiet %s (b%d, %d segments, classes", quiet.dir.c_str(), quiet.batch, quiet.nseg); for (int c : quiet.classes) std::fprintf(lf, " %d", c);
            if (busy.loaded) { std::fprintf(lf, ") busy %s (b%d, %d segments, classes", busy.dir.c_str(), busy.batch, busy.nseg); for (int c : busy.classes) std::fprintf(lf, " %d", c); }
            std::fprintf(lf, ") | PUSCH partitions"); for (int y : pusch_sm) std::fprintf(lf, " %d", y); std::fprintf(lf, " | PDSCH partitions"); for (int n : pdsch_sm) std::fprintf(lf, " %d", n);
            std::fprintf(lf, " | cpu %d prio %d lookahead %d guard %d us ul_back %d dl_fwd %d fixed %d (gap %d,%d) one_per_gap %d dl_pause %d dl_max_class %d ul_pause %d unknown %s tiles/request %d\n", cpu, prio, lookahead, guard_us, ul_back, dl_fwd, fixed, fixed_gap_P, fixed_gap_N, one_per_gap, dl_pause, dl_max_class, ul_pause, unknown_prev ? "prev" : "max", tiles_per_request);
            for (auto& kv : part_of) { std::string d; for (int i : kv.second) d += " part " + std::to_string(i) + " class " + std::to_string(parts[i].cls); std::fprintf(lf, "  (P %d, N %d) ->%s%s\n", kv.first.first, kv.first.second, kv.second.empty() ? " no class fits (pause)" : d.c_str(), kv.second.empty() ? "" : (" of gap " + std::to_string(parts[kv.second.front()].gap)).c_str()); }
        }
        if (lf && !layout_report.empty()) { std::fputs(layout_report.c_str(), lf); std::fflush(lf); }
        // validation: one inference of every set on every partition whose class it has, detections must agree within a set
        std::vector<EngineSet*> vsets = {&quiet, &busy}; for (const std::string& m : model_names) vsets.push_back(&models[m]);
        for (EngineSet* s : vsets) {
            if (!s->loaded || !stress.empty()) continue;
            int ref = -1;
            for (size_t i = 0; i < parts.size(); i++) { if (s->cls_index(parts[i].cls) < 0) continue; int d = 0; double ms = 0; if (!run_once(*s, parts[i], d, ms, err)) { if (lf) std::fclose(lf); return false; }
                if (ref < 0) ref = d; if (lf) std::fprintf(lf, "  %s part %zu (P %d, N %d, class %d): %.2f ms, %d detections%s\n", s->name.c_str(), i, parts[i].P, parts[i].N, parts[i].cls, ms, d, d == ref ? "" : "  <-- MISMATCH");
                // seed the per-segment time estimates (horizon rule) from this measurement: ms / nseg per segment
                { const int ci = s->cls_index(parts[i].cls); for (int k = 0; k < s->nseg; k++) { if (s->seg_ema[ci][k] == 0.0) s->seg_ema[ci][k] = ms / s->nseg; } } }
        }
        // CUDA graphs (2026-11-02 09:00): capture enqueueV3 of every (model, segment, partition) once, after the validation run warmed the
        // execution contexts. A 26x e2e enqueue is hundreds of kernel launches in a few ms; inside cuphycontroller (under MPS) that launch
        // burst delayed cuPHY's own launches (DL C-plane late, AGGR waits) at every request. One graph launch per segment instead.
        a_use_graph = envi("DAPP_AGENT_GRAPH", 1) != 0;
        if (a_use_graph && stress.empty()) {
            int ng = 0, nchunk = 0; double chunk_ms = 0.0;
            for (const std::string& m : model_names) { EngineSet& es = models[m]; if (!es.loaded) continue;
                for (size_t pi = 0; pi < parts.size(); pi++) { const Part& p = parts[pi]; const int ci = es.cls_index(p.cls); if (ci < 0) continue;
                    for (int k = 0; k < es.nseg; k++) {
                        cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
                        if (cudaStreamBeginCapture(p.s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "graph capture begin failed"; if (lf) std::fclose(lf); return false; }
                        bind_aux(es.segs[k], ci, p);
                        const bool okq = es.segs[k].exec[ci]->enqueueV3(p.s);
                        const cudaError_t ce = cudaStreamEndCapture(p.s, &g);
                        if (!okq || ce != cudaSuccess || !g) { err = "graph capture of " + m + " seg " + std::to_string(k) + " part " + std::to_string(pi) + " failed (" + (okq ? cudaGetErrorString(ce) : "enqueue") + ")"; if (lf) std::fclose(lf); return false; }
                        if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) { err = "graph instantiate failed"; if (lf) std::fclose(lf); return false; }
                        if (chunk_n > 0) { std::vector<cudaGraphExec_t> pieces; std::string e2; const int64_t tc0 = now_mono_ns(); if (!chunk_graph(g, chunk_n, pieces, p.s, e2)) { err = "graph chunking of " + m + ": " + e2; if (lf) std::fclose(lf); return false; }
                            es.chunks[{k, static_cast<int>(pi)}] = pieces; nchunk += static_cast<int>(pieces.size()); chunk_ms += (now_mono_ns() - tc0) / 1e6;
                            while (es.chunk_ev.size() < pieces.size()) { cudaEvent_t ce2 = nullptr; cudaEventCreate(&ce2); es.chunk_ev.push_back(ce2); } }
                        cudaGraphDestroy(g); cudaGraphUpload(ge, p.s); es.graphs[{k, static_cast<int>(pi)}] = ge; ng++;
                    }
                }
                cudaDeviceSynchronize();
            }
            if (lf) std::fprintf(lf, "  CUDA graphs captured: %d (DAPP_AGENT_GRAPH=1), cut into %d pieces of %d nodes (DAPP_AGENT_GRAPH_CHUNK) in %.1f s, %d queued ahead (DAPP_AGENT_CHUNK_AHEAD)\n", ng, nchunk, chunk_n, chunk_ms / 1000.0, chunk_ahead);
        }
        if (lf) std::fclose(lf);
        return true;
    }
    bool launch(std::string& err) {
        if (agent_on) {
            agent::Config c; c.port = srv_port; c.udp = envs("DAPP_AGENT_UDP", ""); c.cpu = envi("DAPP_AGENT_CPU", 46);   // an isolated core nobody else uses (tenant 45, export 43); unpinned it inherited the L1 main thread's core -> L1 recovery exit c.max_queue = envi("DAPP_AGENT_MAX_QUEUE", 8); c.cells = envi("DAPP_AGENT_CELLS", 1); c.gpu_sm = nsm;
            std::string mj = "["; for (size_t i = 0; i < model_names.size(); i++) { if (i) mj += ','; mj += json::quote(model_names[i]); } mj += ']'; c.models_json = mj;
            srv.set_report_fn([this](int window_ms) { return report_json(window_ms); });
            srv.set_estimate_fn([this]() { std::lock_guard<std::mutex> g(rep_m); return last_estimate; });
            if (!srv.start(c, err)) return false;
        }
        stop.store(false); th = std::thread([this] { run(); });
        if (cpu >= 0) { cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set); if (pthread_setaffinity_np(th.native_handle(), sizeof(set), &set) != 0) { err = "pthread_setaffinity_np failed"; return false; } }
        if (prio > 0) { struct sched_param sp{}; sp.sched_priority = prio; if (pthread_setschedparam(th.native_handle(), SCHED_FIFO, &sp) != 0) { err = "pthread_setschedparam failed"; return false; } }
        return true;
    }
};

std::unique_ptr<Tenant> g_tenant;

} // namespace

bool start(std::string& err)
{
    err.clear();
    const char* on = std::getenv("DAPP_TENANT");
    if (on == nullptr || std::atoi(on) == 0) return false;
    if (g_tenant) { err = "tenant already started"; return false; }
    auto t = std::make_unique<Tenant>();
    if (!t->init(err)) return false;
    if (!t->launch(err)) return false;
    g_tenant = std::move(t);
    return true;
}
void stop()
{
    if (!g_tenant) return;
    g_tenant->stop.store(true);
    if (g_tenant->th.joinable()) g_tenant->th.join();
    if (g_tenant->agent_on) g_tenant->srv.stop();
    g_tenant.reset();
}

} // namespace tenant
} // namespace dapp
} // namespace nv
