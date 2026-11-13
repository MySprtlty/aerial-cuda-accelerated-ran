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
//   DAPP_TENANT_EMPTY_US=400            2026-11-05: a slot still unplanned this long (us) after its L2 message window opened (tick) is idle - a real L2 (OAI) sends
//                                       DL_TTI/UL_TTI only for slots with content (SSB/SIB1, PRACH, scheduled UEs); testMAC sends them every slot. -1 = off (old behaviour)
//                                       (fallback: worst case), max = worst case (largest profiles)
//   DAPP_TENANT_FIXED=<c>               debug: always run the quiet set at class c on a plain partition split from the full device
//   DAPP_TENANT_FIXED_GAP=P,N           with DAPP_TENANT_FIXED=<c>: the static context is the class-c context of gap (P, N) instead
//                                       (e.g. 88,40 = the Average busy-window gap: disjoint from PUSCH 44 / PDSCH 40 / order, overlaps PUSCH >= 60)
//   DAPP_TENANT_ENGINE_CTX=prim|part   CUDA context the TensorRT engines are deserialized in. prim (default since 2026-11-10): the primary
//                                       context for every engine. part (2026-11-02..11-10 behaviour): the class's own green context - broken with
//                                       TensorRT 10.3: Myelin sizes its segment-K / cluster GEMM grids from cuOccupancyMaxActiveClusters in the
//                                       context current at deserialization and caches that per kernel for the whole process, so the first class
//                                       loaded dictates the grids of every later class (see the note in load_set). Diagnostic use only.
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
//   DAPP_INPUT=4k|fhd                   input profile reported in the agent's welcome (2026-11-09): 4k = models/full26<m> batch-8 (1280 crops 6 + 2 whole
//                                       views), fhd = models/fhd26<m> batch-3 (1280x1080 crops 2 + 1 whole view); the crop grid itself follows the JPEG size
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
#include <cerrno>
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
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

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

// ---- off-thread file writes (2026-11-13) ------------------------------------------------------------------------------------------
// The tenant thread runs on an isolated core and must not sleep in write(2): its per-slot CSV showed stalls of 18-103 ms about once per
// minute (old and new binaries; an idle L1 next to it kept its timing). Log, per-slot CSV and piece trace are therefore FILE* streams
// from fopencookie whose write callback only appends to a memory buffer; one writer thread on the housekeeping cores
// (DAPP_TENANT_IO_CPUS, default 0-3) drains the buffers every 20 ms. DAPP_TENANT_IO_ASYNC=0 = plain fopen (the behaviour before).
struct AsyncSink { int fd = -1; std::mutex m; std::string buf, spare; };
class AsyncIo {
public:
    static AsyncIo& get() { static AsyncIo* io = new AsyncIo; return *io; }   // never destroyed: streams may be closed from static destructors
    static bool async_on() { const char* e = std::getenv("DAPP_TENANT_IO_ASYNC"); return !(e && *e == '0'); }
    std::FILE* open(const char* path, const char* mode) {
        if (!async_on()) return std::fopen(path, mode);
        const int fd = ::open(path, O_WRONLY | O_CREAT | O_CLOEXEC | (mode[0] == 'a' ? O_APPEND : O_TRUNC), 0644);
        if (fd < 0) return nullptr;
        AsyncSink* k = new AsyncSink; k->fd = fd; k->buf.reserve(1 << 20); k->spare.reserve(1 << 20);
        cookie_io_functions_t f = {nullptr, &AsyncIo::wr, nullptr, &AsyncIo::cl};
        std::FILE* fp = fopencookie(k, "w", f);
        if (!fp) { ::close(fd); delete k; return std::fopen(path, mode); }
        std::setvbuf(fp, nullptr, _IOLBF, 1 << 16);
        std::lock_guard<std::mutex> g(lm_); sinks_.push_back(k);
        if (!started_) { started_ = true; std::thread([this] { run(); }).detach(); }
        return fp;
    }
private:
    static ssize_t wr(void* c, const char* b, size_t n) { AsyncSink* k = static_cast<AsyncSink*>(c); std::lock_guard<std::mutex> g(k->m); k->buf.append(b, n); return static_cast<ssize_t>(n); }
    static int cl(void* c) {   // fclose: drain on the caller's thread (stop paths only), then forget the sink
        AsyncSink* k = static_cast<AsyncSink*>(c); AsyncIo& io = get(); std::lock_guard<std::mutex> g(io.lm_);
        drain(k); ::close(k->fd); io.sinks_.erase(std::remove(io.sinks_.begin(), io.sinks_.end(), k), io.sinks_.end()); delete k; return 0;
    }
    static void drain(AsyncSink* k) {
        { std::lock_guard<std::mutex> g(k->m); k->spare.swap(k->buf); }
        const char* b = k->spare.data(); size_t left = k->spare.size();
        while (left > 0) { const ssize_t w = ::write(k->fd, b, left); if (w <= 0) { if (w < 0 && errno == EINTR) continue; break; } b += w; left -= static_cast<size_t>(w); }
        k->spare.clear();
    }
    static bool pin(const char* spec) {   // "a-b" or "a"
        int a = -1, b = -1; if (std::sscanf(spec, "%d-%d", &a, &b) < 1 || a < 0) return false; if (b < a) b = a;
        cpu_set_t set; CPU_ZERO(&set); for (int c = a; c <= b && c < CPU_SETSIZE; c++) CPU_SET(c, &set);
        return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
    }
    void run() {
        pthread_setname_np(pthread_self(), "dapp_tenant_io");
        struct sched_param sp{}; pthread_setschedparam(pthread_self(), SCHED_OTHER, &sp);
        const char* e = std::getenv("DAPP_TENANT_IO_CPUS");
        if (!pin((e && *e) ? e : "0-3") && !pin("0-3")) std::fprintf(stderr, "dapp_tenant_io: cannot leave the creating thread's cores (DAPP_TENANT_IO_CPUS)\n");
        for (;;) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); std::lock_guard<std::mutex> g(lm_); for (AsyncSink* k : sinks_) drain(k); }
    }
    std::mutex lm_; std::vector<AsyncSink*> sinks_; bool started_ = false;
};
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
    // 2026-11-12 (feature/slot_level): one_engine = <dir>/seg<k>_sm<DAPP_TENANT_ENGINE_SM>.engine alone serves every class (one engine + one
    // execution context, graphs captured per partition stream); piece_us = table time (us) of every graph piece per (segment, partition),
    // measured at init; piece_ok = every partition has the same pieces, so the class can change between two pieces (DAPP_AGENT_PIECE=1)
    bool one_engine = false, piece_ok = false; std::map<std::pair<int, int>, std::vector<float>> piece_us;
    std::map<int, std::vector<int>> bounds;       // 2026-11-13 DAPP_AGENT_PIECE_US: segment -> first node (position in the topological order) of every piece, shared by all partitions
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
    int pred_frames = 12;             // DAPP_TENANT_PRED_FRAMES: frames to look back for an unplanned slot's prediction (>= horizon_max / 20 + lead: 2026-11-05, a 26x estimate of 110 ms = 220 slots left the far horizon unknown with 8)
    int empty_us = 400;               // DAPP_TENANT_EMPTY_US: unplanned slot = idle once its message window has been open this long (-1: never)
    int srv_port = 9810;
    // ---- xApp agent (2026-11-02, docs/XAPP_INTERFACE.md): requests from the Resource Allocator, per-slot estimates, SM reports
    bool agent_on = false, loop_on = true, freerun = false; int64_t freerun_t0 = 0;
    agent::Server srv; std::map<std::string, EngineSet> models; std::vector<std::string> model_names;
    void* njd = nullptr; float* d_dec = nullptr; size_t d_dec_n = 0; double dec_ema = 6.0; cudaStream_t dec_s = nullptr; float conf_thr = 0.25f, iou_thr = 0.7f;   // dec_s: decode on a primary-context stream (nvjpeg + green-context streams hung the thread, k1_26x_av4)
    agent::Request cur; bool have_req = false; EngineSet* aset = nullptr; Pipe apipe; int arois[64] = {0}; int awh[2] = {0, 0};
    std::string input_profile = "4k"; std::set<std::string> grid_warned;   // DAPP_INPUT (welcome "input"); models already warned about an image that got whole-image views only
    int64_t a_t_start = 0; double a_dec_ms = 0.0; std::vector<int> a_cls; int a_pidx = -1; int64_t a_est_end_ns = 0;
    std::mutex rep_m; struct DoneRec { int64_t t_end_ns; std::string model; double sm_mean; }; std::deque<DoneRec> done;
    // the most recently finished request (sm_report "last": model + SMs it ran on), for the slicing xApp (2026-11-05)
    struct LastRec { bool valid = false; int64_t t_end_ns = 0; std::string model, status; int sm = 0; double sm_mean = 0.0, infer_ms = 0.0, latency_ms = 0.0; } last;
    double yolo_sm_sum = 0.0, ran_sm_sum = 0.0; uint64_t slot_samples = 0; std::string last_estimate; uint64_t a_ok = 0, a_fail = 0;
    bool a_skip_decode = false, a_skip_infer = false, a_use_graph = true;
    unsigned char* h_jpeg = nullptr; size_t h_jpeg_cap = 0; void* h_out = nullptr; size_t h_out_cap = 0;   // pinned staging (GH200: a GPU copy engine touching pageable memory goes through ATS; keep cuPHY out of that)   // bisection knobs (DAPP_AGENT_SKIP_DECODE / _SKIP_INFER): request path without the decode / without the TRT enqueue
    int fixed_gap_P = 0, fixed_gap_N = 0, fixed_gap_part = -1, one_per_gap = 0;
    unsigned split_flags = 0;   // 2026-11-13: cuDevSmResourceSplitByCount flags of split(); 0 = 8-SM steps (default), the range layout sets IGNORE_SM_COSCHEDULING (2-SM steps)
    // union layout (2026-11-02, DAPP_TENANT_LAYOUT=union|gap): class contexts avoid the UL order / UL control contexts and the next
    // DAPP_TENANT_FB_GUARD larger PUSCH profiles (pool-busy fallbacks); DAPP_AGENT_QOS=1: refuse a request whose estimate misses its deadline
    int fb_guard = 1; std::string layout = "union"; bool qos_on = true; std::string layout_report; bool engine_ctx_per_class = false;
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
    // ---- piece mode (2026-11-12, DAPP_AGENT_PIECE=1): the partition (= SM class) is chosen per graph piece when the piece is launched, from the
    // slot plans of the slots that piece will touch, instead of once per request from the minimum over the whole request. A piece that moves
    // to another partition waits on the previous piece's event (cudaStreamWaitEvent); measured cost of a change: a few us (prof/tenant/slot_level).
    bool piece_on = false, a_piece = false, piece_window = true; int piece_guard_us = 30, engine_sm = 0, a_chunk_pidx = -1, a_free_for = -1; double piece_scale = 1.0;
    std::vector<int> a_ppart; std::vector<float> a_pt; std::vector<uint8_t> a_pclean; int64_t a_gpu_free_ns = 0, a_wait_t0 = 0, pc_wait_ns = 0;
    // 2026-11-13: DAPP_AGENT_PIECE_US > 0: pieces of about that many us on the smallest class instead of chunk_n nodes (boundaries from per-node times
    // measured at init; a node longer than the target is a piece of its own - a kernel cannot be cut). DAPP_AGENT_AHEAD_US > 0: a piece is queued when the
    // work in flight ends within that many us (at most DAPP_AGENT_AHEAD_MAX pieces in flight) instead of always chunk_ahead pieces: the short pieces of the
    // large classes no longer leave the GPU idle between two pump passes (~70 us), and nothing is committed early behind a long piece of a small class.
    int piece_us_target = 0, ahead_us = 0, ahead_max = 8; std::vector<int64_t> a_pend;   // a_pend: estimated end of every launched piece
    // DAPP_AGENT_PIECE_DEFER = n > 0: waiting rule. The greedy rule starts a piece at once on whatever class the current window allows; a kernel that cannot be cut
    // (26x: up to 1.8 ms on 16 SMs, 0.34 ms on 96) then runs through the next, larger windows on the small class. With the rule, a piece longer than
    // DAPP_AGENT_PIECE_DEFER_MIN_US is held back when starting it right after one of the next n slot boundaries ends it at least DAPP_AGENT_PIECE_DEFER_GAIN_US earlier.
    int defer_slots = 0, defer_min_us = 150, defer_gain_us = 50; uint64_t pc_defer = 0;
    uint64_t pc_pieces = 0, pc_switch = 0, pc_wait = 0; double pc_obs_us = 0.0, pc_tab_us = 0.0;
    // DAPP_AGENT_PIECE_TRACE=<file>: one csv row per piece (request, piece, host launch time, class, observed and table time) for offline checks
    std::FILE* piece_trace = nullptr; std::vector<int64_t> a_plaunch; std::vector<float> a_plead; int64_t a_p0_ns = 0;
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
    bool split(const CUdevResource& in, unsigned count, CUdevResource* res, CUdevResource* rem, std::string& err) { unsigned ng = 1; CKD(cuDevSmResourceSplitByCount(res, &ng, const_cast<CUdevResource*>(&in), rem, split_flags, count)); return true; }
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
            parts.push_back(p); part_of[{nsm, 0}] = {0}; return true;
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
    bool build_partitions(std::string& err) { return layout == "gap" ? build_partitions_gap(err) : layout == "range" ? build_partitions_range(err) : build_partitions_union(err); }

    // ---- range layout (2026-11-13, DAPP_TENANT_LAYOUT=range; the L1 must run with DAPP_GC_SPLIT_FLAGS=1) ----------------------
    // 2-SM-step green contexts. With CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING every cuphydriver context is a prefix or a suffix
    // of ONE linear split order in 2-SM steps (measured stand-alone with %smid, no MPS: positions 0-15 = smid 0-15, 16-23 = smid
    // 124-131, 24-131 = smid 16-123), so the SMs cuPHY leaves free in a slot are one contiguous position range [a, b): a = the PDSCH
    // prefix N (moved past the UL order slice when that touches the range), b = 132 - PUSCH profile (132 in slots without PUSCH).
    // The class context of (P, N) is that WHOLE range - one class per (P, N); `classes` becomes the list of the distinct range sizes
    // (needs the one-engine sets, DAPP_TENANT_ENGINE_SM). It is built as the REMAINDER of a nested split (result [0, b) -> green
    // context -> remainder after a): unions of 2-SM groups and result-side cuts only support thread-block clusters of 2..6, and
    // TensorRT's kernels of 26s..x need clusters of 8 (26n: 4). A range under DAPP_TENANT_RANGE_MIN SMs (56) or whose context reports
    // a smaller maximum cluster than DAPP_TENANT_RANGE_CLUSTER (8) gets no class (pause). Self-check as in the union layout: %smid of
    // every shadow set and class context when MPS is off; the position -> smid map comes from the nested prefixes [0, 2 i).
    bool build_partitions_range(std::string& err) {
        if (envi("DAPP_GC_SPLIT_FLAGS", 0) != 1) { err = "DAPP_TENANT_LAYOUT=range needs the L1 on 2-SM-step green contexts (DAPP_GC_SPLIT_FLAGS=1)"; return false; }
        if (fixed > 0) { err = "DAPP_TENANT_LAYOUT=range does not support DAPP_TENANT_FIXED"; return false; }
        if (engine_sm <= 0) { err = "DAPP_TENANT_LAYOUT=range needs the one-engine sets (DAPP_TENANT_ENGINE_SM)"; return false; }
        split_flags = CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING;
        const int range_min = std::max(2, envi("DAPP_TENANT_RANGE_MIN", 56)), range_cluster = envi("DAPP_TENANT_RANGE_CLUSTER", 8);
        int tmp[16]; int n = cuphydriver_dapp_profile_sms(0, tmp, 16); pusch_sm.assign(tmp, tmp + n);
        n = cuphydriver_dapp_profile_sms(1, tmp, 16); pdsch_sm.assign(tmp, tmp + n);
        if (pusch_sm.empty() || pdsch_sm.empty()) { err = "cuphydriver reports no PUSCH/PDSCH profiles (DAPP_PUSCH_PROFILES / DAPP_PDSCH_PROFILES unset?)"; return false; }
        int ord[2] = {0, 0}; const int no = cuphydriver_dapp_profile_sms(2, ord, 2); const int order_sm = no > 0 ? ord[0] : 0, order_mode = no > 1 ? ord[1] : 0;
        int ulc[2] = {0, 0}; const int nu = cuphydriver_dapp_profile_sms(3, ulc, 2); const int ulctrl_sm = nu > 1 ? ulc[0] + ulc[1] : 0;
        if (order_sm <= 0 || ulctrl_sm <= 0) { err = "cuphydriver does not report the UL order / UL control contexts (profile_sms channels 2, 3): old cuphydriver build?"; return false; }
        int mps = 0; cuDeviceGetAttribute(&mps, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, dev); const bool verify = (mps == 0) && envi("DAPP_TENANT_PROBE", 1) != 0;
        unsigned* hist = nullptr; if (verify) CKR(cudaMalloc(&hist, nsm * sizeof(unsigned)));
        auto interval = [](int a, int b) { std::set<int> o; for (int i = a; i < b; i++) o.insert(i); return o; };
        // position -> physical SM map in 2-SM steps: pair i = positions [2 i, 2 i + 2) = smid of prefix [0, 2 i + 2) minus prefix [0, 2 i)
        const int np = nsm / 2; std::vector<std::set<int>> gphys(np);
        if (verify) {
            std::set<int> prev;
            for (int i = 1; i <= np; i++) {
                std::set<int> cur;
                if (2 * i < nsm) { CUdevResource r{}, rem{}; if (!split(all, static_cast<unsigned>(2 * i), &r, &rem, err)) return false; if (!probe_res(r, cur, hist, err)) return false; }
                else { if (!probe_res(all, cur, hist, err)) return false; }
                if (static_cast<int>(cur.size()) != 2 * i || sm_inter(prev, cur) != static_cast<int>(prev.size())) { err = "layout self-check: prefix of " + std::to_string(2 * i) + " SMs measured smid [" + sm_ranges(cur) + "], not nested in 2-SM steps"; return false; }
                for (int x : cur) { if (!prev.count(x)) gphys[i - 1].insert(x); }
                prev = cur;
            }
        }
        auto phys_of = [&](const std::set<int>& pos, std::set<int>& out) -> bool {   // the position set must consist of whole pairs
            out.clear();
            for (int i = 0; i < np; i++) { const bool lo = pos.count(2 * i) != 0, hi = pos.count(2 * i + 1) != 0; if (lo != hi) return false; if (lo) out.insert(gphys[i].begin(), gphys[i].end()); }
            return true; };
        std::vector<SmSet> sets;
        auto add_set = [&](const std::string& name, int a, int b, const CUdevResource* r) -> bool {   // positions [a, b); r = the same resource for the optional %smid check
            SmSet x; x.name = name; x.ids = interval(a, b); x.sm = b - a;
            if (r && static_cast<int>(r->sm.smCount) != b - a) { err = "range layout: shadow split of " + name + " gives " + std::to_string(r->sm.smCount) + " SMs, cuphydriver reports " + std::to_string(b - a); return false; }
            if (verify && r) { std::set<int> m, e; if (!probe_res(*r, m, hist, err)) return false; if (!phys_of(x.ids, e) || m != e) { err = "layout self-check: " + name + " positions [" + sm_ranges(x.ids) + "] measured smid [" + sm_ranges(m) + "] expected [" + sm_ranges(e) + "]"; return false; } }
            sets.push_back(x); return true; };
        const int C = nsm - pusch_sm.back(); CUdevResource yaml_result{};
        for (int y : pusch_sm) { CUdevResource r{}, rem{}; if (!split(all, static_cast<unsigned>(nsm - y), &r, &rem, err)) return false; if (!add_set("PUSCH" + std::to_string(y), nsm - y, nsm, &rem)) return false; if (y == pusch_sm.back()) yaml_result = r; }
        for (int N : pdsch_sm) { CUdevResource r{}, rem{}; if (!split(all, static_cast<unsigned>(N), &r, &rem, err)) return false; if (!add_set("PDSCH" + std::to_string(N), 0, N, &r)) return false; }
        { CUdevResource r{}, rem{}; if (!split(all, static_cast<unsigned>(nsm - ulctrl_sm), &r, &rem, err)) return false; if (!add_set("ULCTRL", nsm - ulctrl_sm, nsm, &rem)) return false; }   // PUCCH + PRACH = remainder of cuphydriver's prep split
        if (order_mode == 1) { CUdevResource r{}, rem{}; if (!split(all, static_cast<unsigned>(order_sm), &r, &rem, err)) return false; if (!add_set("ORDER", 0, order_sm, &r)) return false; }
        else { CUdevResource comp{}, r{}, rem{}; if (!via_ctx(yaml_result, comp, err)) return false; if (!split(comp, static_cast<unsigned>(C - order_sm), &r, &rem, err)) return false; if (!add_set("ORDER", C - order_sm, C, &rem)) return false; }
        auto find = [&](const std::string& nm) -> const SmSet& { for (const SmSet& s : sets) { if (s.name == nm) return s; } return sets.front(); };
        char line[2048];
        layout_report += verify ? "==== range layout (2-SM steps; positions, %smid-verified):" : "==== range layout (2-SM steps; split-order positions; MPS client, %smid virtual):";
        for (const SmSet& s : sets) { std::snprintf(line, sizeof line, " %s %d [%s]", s.name.c_str(), s.sm, sm_ranges(s.ids).c_str()); layout_report += line; }
        std::snprintf(line, sizeof line, " | fb_guard %d, min range %d SMs, min cluster %d\n", fb_guard, range_min, range_cluster); layout_report += line;
        const SmSet& so = find("ORDER"); const int oa = *so.ids.begin(), ob = *so.ids.rbegin() + 1;
        std::set<int> sizes;
        std::vector<int> Ps; for (int y : pusch_sm) { Ps.push_back(nsm - y); } Ps.push_back(nsm);
        for (size_t pi = 0; pi < Ps.size(); pi++) {
            const int P = Ps[pi];
            for (int N : pdsch_sm) {
                std::set<int> busy = so.ids; { const SmSet& d = find("PDSCH" + std::to_string(N)); busy.insert(d.ids.begin(), d.ids.end()); }
                std::string guarded; int a = N, b = nsm;
                if (P < nsm) {
                    const size_t jg = std::min(pi + static_cast<size_t>(std::max(fb_guard, 0)), pusch_sm.size() - 1);
                    for (size_t j = pi; j <= jg; j++) { const SmSet& u = find("PUSCH" + std::to_string(pusch_sm[j])); busy.insert(u.ids.begin(), u.ids.end()); if (j > pi) guarded += " PUSCH" + std::to_string(pusch_sm[j]); }
                    const SmSet& u = find("ULCTRL"); busy.insert(u.ids.begin(), u.ids.end());
                    b = std::min(nsm - pusch_sm[jg], nsm - ulctrl_sm);
                }
                // the UL order slice [oa, ob): cut off the edge of the range it touches; when it lies inside, keep the larger side
                if (ob > a && oa < b) { if (oa <= a) a = ob; else if (ob >= b) b = oa; else if (oa - a >= b - ob) b = oa; else a = ob; }
                const int gsz = std::max(0, b - a);
                std::vector<int> idx; std::string why;
                if (gsz < range_min) why = "under the minimum";
                else {
                    Part p; p.P = P; p.N = N; p.cls = gsz; p.gap = gsz;
                    CUdevResource cur = all, range = all;   // positions [a, b): the remainder side of the last split
                    if (b < nsm) { CUdevResource pre{}, rem{}; if (!split(all, static_cast<unsigned>(b), &pre, &rem, err)) return false; if (a == 0) range = pre; else if (!via_ctx(pre, cur, err)) return false; }
                    if (a > 0) { CUdevResource lo{}; if (!split(cur, static_cast<unsigned>(a), &lo, &range, err)) return false; }
                    if (static_cast<int>(range.sm.smCount) != gsz) { err = "range layout: (P " + std::to_string(P) + ", N " + std::to_string(N) + ") split gives " + std::to_string(range.sm.smCount) + " SMs for positions [" + std::to_string(a) + ", " + std::to_string(b) + ")"; return false; }
                    if (!make_ctx(p, range, err)) return false;
                    int maxcl = 0;
                    { CKD(cuCtxSetCurrent(p.ctx)); cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(64, 1, 1); cfg.blockDim = dim3(64, 1, 1); cfg.stream = p.s;
                      if (cudaOccupancyMaxPotentialClusterSize(&maxcl, k_smid, &cfg) != cudaSuccess) { maxcl = 0; cudaGetLastError(); } CKD(cuCtxSetCurrent(prim)); }
                    const std::set<int> ids = interval(a, b);
                    std::string phys;
                    if (verify) { std::set<int> m, e; CKD(cuCtxSetCurrent(p.ctx)); const bool okp = probe_stream(p.s, m, hist, err); cuCtxSetCurrent(prim); if (!okp) return false; if (!phys_of(ids, e) || m != e) { err = "layout self-check: range [" + sm_ranges(ids) + "] context measured smid [" + sm_ranges(m) + "] expected [" + sm_ranges(e) + "]"; return false; } phys = " smid [" + sm_ranges(e) + "]"; }
                    std::string ov; for (const SmSet& s : sets) { const int k = sm_inter(ids, s.ids); if (k && busy.count(*s.ids.begin())) ov += " " + s.name + "(" + std::to_string(k) + ")"; }
                    if (sm_inter(ids, busy) != 0) { err = "range layout violated at (P " + std::to_string(P) + ", N " + std::to_string(N) + "):" + ov; return false; }
                    if (maxcl < range_cluster) {   // usable for plain kernels only: TensorRT's cluster kernels would fail on it
                        cuCtxSetCurrent(p.ctx); for (cudaStream_t x : p.aux) cudaStreamDestroy(x); cudaStreamDestroy(p.s); cuCtxSetCurrent(prim); cuGreenCtxDestroy(p.g);
                        why = "max cluster " + std::to_string(maxcl);
                    } else {
                        p.ids.assign(ids.begin(), ids.end());
                        std::snprintf(line, sizeof line, "  (P %d, N %d) free range [%d, %d) = %d SMs%s%s -> class %d, max cluster %d%s\n", P, N, a, b, gsz, guarded.empty() ? "" : " guarding", guarded.c_str(), gsz, maxcl, phys.c_str()); layout_report += line;
                        idx.push_back(static_cast<int>(parts.size())); parts.push_back(p); sizes.insert(gsz);
                    }
                }
                if (idx.empty()) { std::snprintf(line, sizeof line, "  (P %d, N %d) free range [%d, %d) = %d SMs%s%s -> no class (%s): pause\n", P, N, a, b, gsz, guarded.empty() ? "" : " guarding", guarded.c_str(), why.c_str()); layout_report += line; }
                part_of[{P, N}] = idx;
            }
        }
        if (hist) cudaFree(hist);
        if (sizes.empty()) { err = "range layout: no (P, N) range is usable"; return false; }
        classes.assign(sizes.begin(), sizes.end());   // ascending; the one-engine sets take every class from this list (load_set)
        return true;
    }

    // ---- gap layout (2026-11-01, DAPP_TENANT_LAYOUT=gap): for every (PUSCH complement P, PDSCH prefix N) one context per class that fits the gap
    bool build_partitions_gap(std::string& err) {
        int tmp[16]; int n = cuphydriver_dapp_profile_sms(0, tmp, 16); pusch_sm.assign(tmp, tmp + n);
        n = cuphydriver_dapp_profile_sms(1, tmp, 16); pdsch_sm.assign(tmp, tmp + n);
        if (fixed > 0 && fixed_gap_P == 0) {
            Part p; p.cls = fixed; p.gap = fixed; CUdevResource r{}, rem{}; if (!split(all, fixed, &r, &rem, err)) return false; if (!make_ctx(p, r, err)) return false;
            parts.push_back(p); part_of[{nsm, 0}] = {0}; return true;
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
        // 2026-11-12: DAPP_TENANT_ENGINE_SM=<sm> (e.g. 132) and <dir>/seg00_sm<sm>.engine present -> one-engine set: that engine, deserialized in the
        // primary context, runs bit-identically on the streams of every class (8..132 SMs measured, +2..7 % vs the native class engines)
        { char p1[512]; std::snprintf(p1, sizeof p1, "%s/seg00_sm%d.engine", s.dir.c_str(), engine_sm); s.one_engine = engine_sm > 0 && file_exists(p1); }
        if (s.one_engine) s.classes = classes;
        else for (int c : classes) { char p[512]; std::snprintf(p, sizeof p, "%s/seg00_sm%d.engine", s.dir.c_str(), c); if (file_exists(p)) s.classes.push_back(c); }
        if (s.classes.empty()) { err = "no seg00_sm<c>.engine in " + s.dir; return false; }
        // engines of one set must share the I/O tensor layout (one buffer per tensor name): keep the classes whose seg00 engine has the same
        // tensor signature as the largest class (an older export left e.g. a 32-SM 26n engine with another input size) (2026-11-02 14:20)
        if (!s.one_engine) { std::map<int, std::string> sig;
          for (int c : s.classes) { char p[512]; std::snprintf(p, sizeof p, "%s/seg00_sm%d.engine", s.dir.c_str(), c); auto blob = slurp(p); if (blob.empty()) continue;
              CKD(cuCtxSetCurrent(prim)); nvinfer1::ICudaEngine* e = rt->deserializeCudaEngine(blob.data(), blob.size()); if (!e) continue;
              std::string g; for (int i = 0; i < e->getNbIOTensors(); i++) { const char* n = e->getIOTensorName(i); const nvinfer1::Dims d = e->getTensorShape(n); g += n; g += ':'; for (int j = 0; j < d.nbDims; j++) { g += std::to_string(d.d[j]); g += 'x'; } g += std::to_string(static_cast<int>(e->getTensorDataType(n))); g += ';'; }
              sig[c] = g; delete e; }
          const std::string ref = sig.count(s.classes.back()) ? sig[s.classes.back()] : std::string();
          std::vector<int> keep; for (int c : s.classes) { if (sig.count(c) && sig[c] == ref) keep.push_back(c); else std::fprintf(stderr, "dapp_tenant: %s: class %d engine skipped (tensor layout differs from class %d)\n", s.dir.c_str(), c, s.classes.back()); }
          s.classes = keep; if (s.classes.empty()) { err = "no usable seg00_sm<c>.engine in " + s.dir; return false; } }
        for (s.nseg = 0;; s.nseg++) { char p[512]; std::snprintf(p, sizeof p, "%s/seg%02d_sm%d.engine", s.dir.c_str(), s.nseg, s.one_engine ? engine_sm : s.classes[0]); if (!file_exists(p)) break; }
        s.segs.resize(s.nseg);
        for (int k = 0; k < s.nseg; k++) {
            for (size_t ci = 0; ci < s.classes.size(); ci++) {
                if (s.one_engine && ci > 0) { s.segs[k].eng.push_back(s.segs[k].eng[0]); s.segs[k].exec.push_back(s.segs[k].exec[0]); continue; }   // every class shares the one engine + execution context
                char p[512]; std::snprintf(p, sizeof p, "%s/seg%02d_sm%d.engine", s.dir.c_str(), k, s.one_engine ? engine_sm : s.classes[ci]);
                auto blob = slurp(p); if (blob.empty()) { err = std::string("missing ") + p; return false; }
                // 2026-11-10: every engine is deserialized with the PRIMARY context current (DAPP_TENANT_ENGINE_CTX=prim, default). TensorRT 10.3 (Myelin)
                // chooses the grid of its segment-K / thread-block-cluster GEMMs from cuOccupancyMaxActiveClusters evaluated in the CUDA context current
                // at deserialization (a green context answers with its own SM count) and caches the result per kernel hash for the whole process: the
                // engines of one network share those hashes across SM classes, so the FIRST class deserialized fixed the grids of every later class. With
                // per-class green contexts (the old 'part' mode) the FHD 72+80 pair made the sm80 engine run 72-sized grids and its workspace memset
                // failed ("Myelin ... cask.cpp:exec ... Cuda Runtime (invalid argument)"), and every 24..72/80 ladder silently ran the shared GEMMs with
                // grids sized for 24 SMs. Deserialized in the primary context all engines get 132-SM grids, valid on every partition (26x sm80 on 80 SMs:
                // 31.3 vs 30.9 ms stand-alone, -1 %). Reproduced and measured with models/trt_green.py, trt_seq.py, trt_perf.py (prof/tenant/fhd).
                // Separate issue (still present, plain TensorRT): the fhd26s/fhd26l seg00_sm80 engines built with the sm80 tactics fail "Cask Gemm
                // execution" next to another class of the same network; they are built with the sm72 timing cache instead (build_yolo_engines.sh).
                CUcontext c0 = prim; if (engine_ctx_per_class) for (const Part& q : parts) { if (q.cls == s.classes[ci]) { c0 = q.ctx; break; } }
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
        // 2026-11-05: a slot still unplanned although its L2 message window has been open for empty_us is IDLE: a real L2 (OAI) sends DL_TTI/UL_TTI
        // only for slots with content (SSB/SIB1, PRACH occasions, scheduled UEs), testMAC for every slot. Window open = t0(n) - plan lead (largest
        // observed t0 - plan time, ~ slot_advance slots), t0(n) from the slot clock reference. idle_max = newest slot whose window has been open that long.
        static const SlotPlanEntry kEmpty;   // ul_any 0, dl_any 0
        const int64_t id = g_plan_ref_id.load(std::memory_order_acquire);   // newest planned slot: no plan can exist beyond it
        int64_t idle_max = INT64_MIN;
        if (empty_us >= 0 && id >= 0) {
            const int64_t t0 = g_plan_ref_t0_ns.load(std::memory_order_relaxed); const int64_t lead = g_plan_lead_ns.load(std::memory_order_relaxed);
            if (t0 != 0 && lead > 0) { const int64_t d = now_real_ns() - (t0 - lead) + static_cast<int64_t>(empty_us) * 1000; idle_max = id + (d >= 0 ? d / 500000 : -((-d + 499999) / 500000)); }
        }
        if (n >= 0 && n <= idle_max) { predicted = true; return &kEmpty; }
        // 2026-11-02: look back up to DAPP_TENANT_PRED_FRAMES frames (a 26x request spans 60+ slots, far beyond the planned horizon); 2026-11-05: start at
        // the first frame back whose slot can have a plan (<= id); a looked-back slot that was never planned although its window closed predicts idle
        if (unknown_prev) {
            int k = 1; if (id >= 0 && n > id) { k = static_cast<int>((n - id + kSzSlotsPerFrame - 1) / kSzSlotsPerFrame); if (k < 1) k = 1; }
            for (; k <= pred_frames; k++) {
                const int64_t m = n - k * kSzSlotsPerFrame; if (m < 0) break;
                const SlotPlanEntry& e = slot_plan_entry(m); if (e.key.load(std::memory_order_acquire) == key_of(m)) { predicted = true; return &e; }
                if (m <= idle_max) { predicted = true; return &kEmpty; }
            }
        }
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
    // 2026-11-05: horizon prefix aggregates of wall slot w, cached per (slot, newest plan) - tenant thread only. estimate_json evaluates every
    // model's horizon each slot and the agent re-chooses per segment: H (up to 200) x models x 6 plan lookups per busy_of kept the tenant
    // thread busy for most of each slot and starved the graph-piece pump (26n at 80 SM took 11 ms instead of 5.6 under the real gNB).
    int64_t hc_w = -1, hc_ref = -1; std::vector<int> hcP, hcN; std::vector<uint8_t> hcK, hcDL, hcUL;
    void horizon_cache(int64_t w) {
        const int64_t ref = g_plan_ref_id.load(std::memory_order_acquire);
        if (hc_w == w && hc_ref == ref && !hcP.empty()) return;
        hc_w = w; hc_ref = ref; const int n = horizon_max + 1;
        hcP.assign(n, nsm); hcN.assign(n, 0); hcK.assign(n, 1); hcDL.assign(n, 0); hcUL.assign(n, 0);
        int P = nsm, N = 0; bool known = true, dla = false, ula = false;
        for (int j = 0; j < n; j++) {
            int P2, N2; bool kn2, dla2, ula2; busy_of(w + j, P2, N2, kn2, dla2, ula2);
            known = known && kn2; P = std::min(P, P2); N = std::max(N, N2); dla = dla || dla2; ula = ula || ula2;
            hcP[j] = P; hcN[j] = N; hcK[j] = known ? 1 : 0; hcDL[j] = dla ? 1 : 0; hcUL[j] = ula ? 1 : 0;
        }
    }
    int choose_part(EngineSet& s, int k, int64_t w, int64_t slot_end, bool& known, bool apply_ul_pause, int* gap_now = nullptr) {
        horizon_cache(w);
        if (gap_now) *gap_now = hcP[0] - hcN[0];
        double est_us = 0.0; for (size_t ci = 0; ci < s.classes.size(); ci++) { est_us = std::max(est_us, s.seg_ema[ci][k] * 1000.0); }
        const int64_t span_ns = static_cast<int64_t>((guard_us + est_us) * 1000.0) - (slot_end - now_real_ns());
        const int H = span_ns > 0 ? static_cast<int>(span_ns / 500000) + 1 : 0;
        const int J = std::min(H, horizon_max);
        known = hcK[J] != 0;
        return part_for(hcP[J], hcN[J], hcDL[J] != 0, hcUL[J] != 0, s, apply_ul_pause);
    }
    // piece mode (2026-11-12): partition for piece i of segment k. The piece starts when the pieces queued before it end (lead_us from now) and
    // must end, with the guard, inside slots whose free SMs cover its class: the aggregate of the slots from its start (the current slot with
    // DAPP_AGENT_PIECE_WINDOW=0) to its end. A smaller class makes the piece longer, so the choice is iterated to its fixed point.
    // -1 = no class fits now (the caller retries).
    int piece_fit(EngineSet& s, int k, int i, int64_t w, int64_t left_ns, double lead_us, double* t_us) {
        // slot (relative to w) that holds the instant span_us from now
        auto slot_at = [&](double span_us) { const int64_t d = static_cast<int64_t>(span_us * 1000.0) - left_ns; return d > 0 ? std::min(static_cast<int>(d / 500000) + 1, horizon_max) : 0; };
        const int J0 = piece_window ? slot_at(lead_us - piece_guard_us) : 0;
        int J = std::max(J0, slot_at(lead_us + piece_guard_us)), jdone = J0 - 1, P = nsm, N = 0; bool dl = false, ul = false;
        for (int it = 0; it < 8; it++) {
            for (int j = jdone + 1; j <= J; j++) { int p = nsm, n = 0; bool kn, d, u; busy_of(w + j, p, n, kn, d, u); P = std::min(P, p); N = std::max(N, n); dl = dl || d; ul = ul || u; }
            jdone = J;
            const int pidx = part_for(P, N, dl, ul, s, false); if (pidx < 0) return -1;
            auto pt = s.piece_us.find({k, pidx}); if (pt == s.piece_us.end() || i >= static_cast<int>(pt->second.size())) return -1;
            const double t = pt->second[i] * piece_scale; const int J2 = slot_at(lead_us + t + piece_guard_us);
            if (J2 <= J) { if (t_us) *t_us = t; return pidx; }
            J = J2;
        }
        return -1;
    }
    int choose_piece(EngineSet& s, int k, int i, int64_t w, int64_t slot_end, double lead_us, double* t_us = nullptr) {
        const int64_t left_ns = slot_end - now_real_ns(); double t = 0.0;
        const int pidx = piece_fit(s, k, i, w, left_ns, lead_us, &t); if (pidx < 0) return -1;
        if (defer_slots > 0 && t > defer_min_us) {
            const double fin_now = lead_us + t;
            for (int j = 1; j <= defer_slots; j++) {
                const double lead_j = left_ns / 1e3 + (j - 1) * 500.0 + piece_guard_us + 1.0;   // first instant the piece may start in slot w + j
                if (lead_j <= lead_us) continue;                 // it starts that late anyway
                if (lead_j + defer_gain_us >= fin_now) break;    // waiting that long cannot pay any more
                double tj = 0.0;
                if (piece_fit(s, k, i, w, left_ns, lead_j, &tj) >= 0 && lead_j + tj + defer_gain_us < fin_now) { pc_defer++; return -1; }
            }
        }
        if (t_us) *t_us = t; return pidx;
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
        if (last.valid) { json::Obj l; l.kv("model", last.model).kv("status", last.status).kv("sm", last.sm).kv("sm_mean", last.sm_mean).kv("infer_ms", last.infer_ms).kv("latency_ms", last.latency_ms)
                           .kv("t_ns", static_cast<long long>(last.t_end_ns)).kv("age_ms", (now - last.t_end_ns) / 1e6); o.raw("last", l.str()); } else { o.null("last"); }
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
        if (clock && piece_on && s.piece_ok) {   // piece mode: the decode (one synchronous launch burst) gets the class that fits every slot it will touch
            horizon_cache(w); const int64_t span_ns = static_cast<int64_t>((piece_guard_us + dec_ema * 1250.0) * 1000.0) - (slot_end - now_real_ns());
            const int J = span_ns > 0 ? std::min(static_cast<int>(span_ns / 500000) + 1, horizon_max) : 0;
            a_pidx = part_for(hcP[J], hcN[J], hcDL[J] != 0, hcUL[J] != 0, s, false);
        }
        else if (clock) a_pidx = choose_part(s, 0, w, slot_end, known, false);
        else { a_pidx = -1; auto pt = part_of.find({nsm - pusch_sm.back(), pdsch_sm.back()}); if (pt != part_of.end()) for (int i : pt->second) { if (s.cls_index(parts[i].cls) >= 0) { a_pidx = i; break; } } }
        if (a_pidx < 0) { aset = nullptr; return false; }   // paused now: keep the request, retry next loop (2026-11-13: with aset left set the loop read the previous request's pipe state and answered this one at once with stale output)
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
        if (!a_skip_decode && s.batch > 1 && arois[2] == awh[0] && arois[3] == awh[1] && lf && grid_warned.insert(s.model).second) {   // slot 0 = whole image: the crop grid did not fit this batch
            std::fprintf(lf, "agent: %s: a %dx%d image does not fit the crop grid of this b%d engine set (DAPP_INPUT %s) - whole-image views only\n", s.model.c_str(), awh[0], awh[1], s.batch, input_profile.c_str()); std::fflush(lf); }
        if (s.in_half && !a_skip_decode) { const size_t n = s.bytes["images"] / 2; k_f2h<<<static_cast<unsigned>((n + 255) / 256), 256, 0, ds>>>(d_dec, static_cast<__half*>(s.buf["images"]), n); }
        if (!a_skip_decode) cudaStreamSynchronize(ds);
        a_dec_ms = (now_real_ns() - a_t_start) / 1e6; dec_ema = 0.9 * dec_ema + 0.1 * a_dec_ms;
        apipe = Pipe(); apipe.set = &s; apipe.cls_hist.assign(s.classes.size(), 0); apipe.tiles_left = s.batch; a_cls.clear(); a_piece = false; a_ppart.clear(); a_pt.clear(); a_pclean.clear(); a_wait_t0 = 0;
        a_est_end_ns = now_real_ns() + static_cast<int64_t>(set_time_ms(s, s.cls_index(p.cls)) * 1e6);
        return true;
    }
    // enqueue segment k of the agent request on partition pidx (no blocking sync; completion is polled)
    // cut a captured graph into pieces of chunk_n nodes (or at the given boundaries) in topological order; each piece is a clone with the other nodes removed
    // (edges to earlier pieces are implied by stream order, edges to later pieces cannot exist in topological order)
    bool chunk_graph(cudaGraph_t g, int chunk, const std::vector<int>* bounds, std::vector<cudaGraphExec_t>& out, cudaStream_t st, std::string& err) {
        size_t n = 0; CKR(cudaGraphGetNodes(g, nullptr, &n)); if (n == 0) return true;
        if (chunk < 1) chunk = 1;
        std::vector<cudaGraphNode_t> nodes(n); CKR(cudaGraphGetNodes(g, nodes.data(), &n));
        // TensorRT graphs carry edge data (programmatic dependent launch), which the legacy cudaGraphGetEdges refuses ("semantically lossy"): read the topology with _v2
        size_t ne = 0; CKR(cudaGraphGetEdges_v2(g, nullptr, nullptr, nullptr, &ne)); std::vector<cudaGraphNode_t> from(ne), to(ne); std::vector<cudaGraphEdgeData> ed(ne); if (ne) CKR(cudaGraphGetEdges_v2(g, from.data(), to.data(), ed.data(), &ne));
        std::map<cudaGraphNode_t, int> idx; for (size_t i = 0; i < n; i++) idx[nodes[i]] = static_cast<int>(i);
        std::vector<int> indeg(n, 0); std::vector<std::vector<int>> adj(n);
        for (size_t e = 0; e < ne; e++) { adj[idx[from[e]]].push_back(idx[to[e]]); indeg[idx[to[e]]]++; }
        std::vector<int> order; std::deque<int> q; for (size_t i = 0; i < n; i++) { if (indeg[i] == 0) q.push_back(static_cast<int>(i)); }
        while (!q.empty()) { const int u = q.front(); q.pop_front(); order.push_back(u); for (int v : adj[u]) { if (--indeg[v] == 0) q.push_back(v); } }
        if (order.size() != n) { err = "graph is not a DAG"; return false; }
        std::vector<size_t> b; if (bounds && !bounds->empty()) { for (int x : *bounds) b.push_back(static_cast<size_t>(x)); } else { for (size_t c0 = 0; c0 < n; c0 += chunk) b.push_back(c0); }
        if (b.front() != 0 || b.back() >= n) { err = "piece boundaries do not match the graph (" + std::to_string(n) + " nodes)"; return false; }
        for (size_t bi = 0; bi < b.size(); bi++) { const size_t c0 = b[bi], c1 = bi + 1 < b.size() ? b[bi + 1] : n;
            std::set<int> keep(order.begin() + c0, order.begin() + c1);
            cudaGraph_t cg = nullptr; CKR(cudaGraphClone(&cg, g));
            for (size_t i = 0; i < n; i++) { if (keep.count(static_cast<int>(i))) continue; cudaGraphNode_t cn = nullptr; CKR(cudaGraphNodeFindInClone(&cn, nodes[i], cg)); CKR(cudaGraphDestroyNode(cn)); }
            cudaGraphExec_t ge = nullptr; CKR(cudaGraphInstantiate(&ge, cg, 0)); cudaGraphDestroy(cg); cudaGraphUpload(ge, st); out.push_back(ge);
        }
        return true;
    }
    // 2026-11-13 balanced cut (DAPP_AGENT_PIECE_US): every node of segment k is timed alone (1-node pieces launched back to back, median of 3 passes after a
    // warm-up pass) on the partition with the smallest class; a piece is closed before the node that would take it over the target. The boundaries are
    // shared by all partitions, so a piece takes about the target on the smallest class and less on the others; a node above the target stays one piece.
    bool balance_cut(EngineSet& es, const std::string& m, int k, std::FILE* lf, std::string& err) {
        int ref = -1; for (size_t pi = 0; pi < parts.size(); pi++) { if (es.cls_index(parts[pi].cls) < 0) continue; if (ref < 0 || parts[pi].cls < parts[ref].cls) ref = static_cast<int>(pi); }
        if (ref < 0) return true;
        const Part& p = parts[ref]; const int ci = es.cls_index(p.cls); cudaGraph_t g = nullptr;
        if (cudaStreamBeginCapture(p.s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "graph capture begin failed"; return false; }
        bind_aux(es.segs[k], ci, p);
        const bool okq = es.segs[k].exec[ci]->enqueueV3(p.s); const cudaError_t ce = cudaStreamEndCapture(p.s, &g);
        if (!okq || ce != cudaSuccess || !g) { err = std::string("graph capture failed (") + (okq ? cudaGetErrorString(ce) : "enqueue") + ")"; return false; }
        std::vector<cudaGraphExec_t> one; const bool okc = chunk_graph(g, 1, nullptr, one, p.s, err); cudaGraphDestroy(g);
        const size_t n = one.size(); std::vector<float> t(n, 0.0f);
        if (okc && n > 0) {
            while (es.chunk_ev.size() < n) { cudaEvent_t e = nullptr; cudaEventCreate(&e); es.chunk_ev.push_back(e); }
            std::vector<std::vector<float>> rep(n);
            for (int r = 0; r < 4; r++) {
                cudaEventRecord(es.ev_start[k], p.s); for (size_t j = 0; j < n; j++) { cudaGraphLaunch(one[j], p.s); cudaEventRecord(es.chunk_ev[j], p.s); } cudaStreamSynchronize(p.s);
                for (size_t j = 0; r > 0 && j < n; j++) { float ms = 0; if (cudaEventElapsedTime(&ms, j ? es.chunk_ev[j - 1] : es.ev_start[k], es.chunk_ev[j]) == cudaSuccess) rep[j].push_back(ms * 1000.0f); }
            }
            for (size_t j = 0; j < n; j++) { if (!rep[j].empty()) { std::sort(rep[j].begin(), rep[j].end()); t[j] = rep[j][rep[j].size() / 2]; } }
        }
        for (cudaGraphExec_t ge : one) cudaGraphExecDestroy(ge);
        if (!okc) return false;
        std::vector<int> b; double cur = 0.0, sum = 0.0, mx = 0.0; int over = 0;
        for (size_t j = 0; j < n; j++) {
            if (b.empty() || cur + t[j] > piece_us_target) { b.push_back(static_cast<int>(j)); cur = 0.0; }
            cur += t[j]; sum += t[j]; mx = std::max(mx, static_cast<double>(t[j])); if (t[j] > piece_us_target) over++;
        }
        es.bounds[k] = b;
        if (lf) std::fprintf(lf, "  %s balanced cut seg %d: %zu nodes -> %zu pieces of about %d us on class %d (DAPP_AGENT_PIECE_US; node sum %.2f ms, longest node %.0f us, %d node(s) above the target)\n", m.c_str(), k, n, b.size(), piece_us_target, p.cls, sum / 1000.0, mx, over);
        return true;
    }
    bool agent_enqueue(int pidx, std::FILE* lf) {
        EngineSet& s = *aset; const Part& p = parts[pidx]; const int ci = s.cls_index(p.cls); const int k = apipe.k;
        if (k == 0) apipe.t_inf_start = now_mono_ns();
        if (apipe.prev_part >= 0 && apipe.prev_part != pidx && k > 0) { cudaStreamWaitEvent(p.s, s.ev_end[k - 1], 0); apipe.switches++; }
        cudaEventRecord(s.ev_start[k], p.s);
        auto git = s.graphs.find({k, pidx}); auto cit = s.chunks.find({k, pidx});
        if (!a_skip_infer && a_use_graph && cit != s.chunks.end() && !cit->second.empty()) {
            a_chunks = &cit->second; a_chunk_part = &p; a_chunk_k = k; a_chunk_next = 0; a_chunking = true; a_chunk_pidx = pidx; a_piece = piece_on && s.piece_ok; a_ppart.clear(); a_pt.clear(); a_pclean.clear(); a_free_for = -1; a_plaunch.clear(); a_plead.clear(); a_pend.clear(); a_p0_ns = now_real_ns();
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
        EngineSet& s = *aset; const int total = static_cast<int>(a_chunks->size());
        while (a_chunk_next < total) {
            const bool time_rule = a_piece && ahead_us > 0;
            if (!time_rule && a_chunk_next >= chunk_ahead && cudaEventQuery(s.chunk_ev[a_chunk_next - chunk_ahead]) != cudaSuccess) return true;
            if (a_piece) {   // piece mode (2026-11-12): piece i goes to the partition whose class fits the slots it will touch; piece 0 keeps the one agent_enqueue was given
                const int i = a_chunk_next; int pidx = a_ppart.empty() ? a_chunk_pidx : a_ppart.back(); double t_us = 0.0, lead_us = 0.0; bool prev_running = false;
                if (i > 0) {
                    // the pieces still queued or running end at most their table time after now: the piece chunk_ahead back has just been seen complete
                    const int64_t now = now_real_ns();
                    if (time_rule) {
                        // work in flight = the incomplete pieces: the oldest one is running and ends at its estimated end, the others add their table time
                        int nin = 0, oldest = -1; double sum = 0.0;
                        for (int j = i - 1; j >= 0 && cudaEventQuery(s.chunk_ev[j]) != cudaSuccess; j--) { sum += a_pt[j]; oldest = j; nin++; }
                        if (nin > 0) {
                            prev_running = true; lead_us = sum - a_pt[oldest] + std::min(static_cast<double>(a_pt[oldest]), std::max(0.0, (a_pend[oldest] - now) / 1e3));
                            if (lead_us > ahead_us || nin >= ahead_max) return true;   // enough work queued: looked at again by the next pass
                        }
                    } else {
                    for (int j = i - 1; j >= 0 && cudaEventQuery(s.chunk_ev[j]) != cudaSuccess; j--) { lead_us += a_pt[j]; prev_running = true; }
                    if (a_free_for != i) { a_free_for = i; a_gpu_free_ns = now + static_cast<int64_t>(lead_us * 1000.0); }
                    lead_us = prev_running ? std::max(0.0, (a_gpu_free_ns - now) / 1e3) : 0.0;
                    }
                    int64_t w = 0, slot_end = 0;
                    if (current_slot(w, slot_end)) {
                        const int c = choose_piece(s, a_chunk_k, i, w, slot_end, lead_us, &t_us);
                        if (c < 0) { if (a_wait_t0 == 0) { a_wait_t0 = now; pc_wait++; } return true; }   // no class fits now: retried by the next loop pass
                        pidx = c;
                    }
                }
                if (a_wait_t0 != 0) { pc_wait_ns += now_real_ns() - a_wait_t0; a_wait_t0 = 0; }
                auto cv = s.chunks.find({a_chunk_k, pidx}); auto tv = s.piece_us.find({a_chunk_k, pidx});
                if (cv == s.chunks.end() || i >= static_cast<int>(cv->second.size())) { pidx = a_ppart.empty() ? a_chunk_pidx : a_ppart.back(); cv = s.chunks.find({a_chunk_k, pidx}); tv = s.piece_us.find({a_chunk_k, pidx}); t_us = 0.0; }
                if (t_us == 0.0 && tv != s.piece_us.end() && i < static_cast<int>(tv->second.size())) t_us = tv->second[i] * piece_scale;
                const Part& q = parts[pidx];
                if (i > 0 && pidx != a_ppart.back()) { cudaStreamWaitEvent(q.s, s.chunk_ev[i - 1], 0); apipe.switches++; pc_switch++; }
                if (cudaGraphLaunch(cv->second[i], q.s) != cudaSuccess) { if (lf) { std::fprintf(lf, "agent: piece %d launch failed seg %d part %d\n", i, a_chunk_k, pidx); std::fflush(lf); } a_chunking = false; fail_request("internal"); return false; }
                cudaEventRecord(s.chunk_ev[i], q.s);
                { const int64_t tl = now_real_ns(); a_pend.push_back(((prev_running && i > 0) ? std::max(tl, a_pend[i - 1]) : tl) + static_cast<int64_t>(t_us * 1000.0)); }
                a_ppart.push_back(pidx); a_pt.push_back(static_cast<float>(t_us)); a_pclean.push_back((i == 0 || prev_running) ? 1 : 0); a_plaunch.push_back(now_real_ns()); a_plead.push_back(static_cast<float>(lead_us)); a_pidx = pidx; a_chunk_part = &q; pc_pieces++; a_chunk_next++;
                continue;
            }
            const Part& p = *a_chunk_part;
            if (cudaGraphLaunch((*a_chunks)[a_chunk_next], p.s) != cudaSuccess) { if (lf) { std::fprintf(lf, "agent: chunk %d launch failed seg %d\n", a_chunk_next, a_chunk_k); std::fflush(lf); } a_chunking = false; fail_request("internal"); return false; }
            cudaEventRecord(s.chunk_ev[a_chunk_next], p.s); a_chunk_next++;
        }
        cudaEventRecord(s.ev_end[a_chunk_k], a_chunk_part->s); a_chunking = false; return true;
    }
    void finish_request(std::FILE* lf) {
        EngineSet& s = *aset; const int64_t t_end = now_real_ns();
        // 2026-11-12: piece mode mixes classes inside a segment, so the per-class segment estimates keep their init values there
        for (int k = 0; !a_piece && k < s.nseg; k++) { float ms = 0; const int ci = s.cls_index(a_cls[k]); if (cudaEventElapsedTime(&ms, s.ev_start[k], s.ev_end[k]) == cudaSuccess) { const double al = (ci < static_cast<int>(s.ema_n.size()) && s.ema_n[ci] < 5) ? 0.5 : 0.1; s.seg_ema[ci][k] = s.seg_ema[ci][k] == 0.0 ? ms : (1.0 - al) * s.seg_ema[ci][k] + al * ms; } }
        if (!a_cls.empty()) { const int ci0 = s.cls_index(a_cls.front()); if (ci0 >= 0 && ci0 < static_cast<int>(s.ema_n.size())) s.ema_n[ci0]++; }
        if (lf && s.chunk_log_n < 3 && a_pidx >= 0) { auto cit = s.chunks.find({0, a_pidx}); if (cit != s.chunks.end() && !cit->second.empty()) {
            std::vector<double> d; float ms = 0; if (cudaEventElapsedTime(&ms, s.ev_start[0], s.chunk_ev[0]) == cudaSuccess) d.push_back(ms);
            for (size_t j = 1; j < cit->second.size(); j++) { if (cudaEventElapsedTime(&ms, s.chunk_ev[j - 1], s.chunk_ev[j]) == cudaSuccess) d.push_back(ms); }
            if (!d.empty()) { std::fprintf(lf, "agent: %s class %d: %zu graph pieces of %d nodes, piece ms p50 %.2f p90 %.2f max %.2f (queued at most %d)\n", s.model.c_str(), parts[a_pidx].cls, cit->second.size(), chunk_n, pct(d, 0.5), pct(d, 0.9), *std::max_element(d.begin(), d.end()), chunk_ahead); std::fflush(lf); s.chunk_log_n++; } } }
        agent::Result r; r.id = cur.id; r.client = cur.client; r.model = cur.model; r.t_recv_ns = cur.t_recv_ns; r.t_start_ns = a_t_start; r.t_end_ns = t_end;
        r.wait_ms = (a_t_start - cur.t_recv_ns) / 1e6; r.decode_ms = a_dec_ms; r.infer_ms = (now_mono_ns() - apipe.t_inf_start) / 1e6; r.latency_ms = (t_end - cur.t_recv_ns) / 1e6;
        r.sm = a_cls.empty() ? 0 : a_cls.front(); double sm_sum = 0; for (int c : a_cls) sm_sum += c; r.sm_mean = a_cls.empty() ? 0.0 : sm_sum / a_cls.size(); r.tiles = s.batch;
        if (a_piece && !a_ppart.empty()) {   // piece mode: time-weighted mean class; observed/table time of the pieces that were queued behind a running one corrects the table (piece_scale)
            double tw = 0.0, tc = 0.0, obs = 0.0, tab = 0.0;
            for (size_t i = 0; i < a_ppart.size(); i++) { auto tv = s.piece_us.find({a_chunk_k, a_ppart[i]}); if (tv == s.piece_us.end() || i >= tv->second.size()) continue;
                const double tt = tv->second[i]; tw += tt; tc += tt * parts[a_ppart[i]].cls; float ms = 0;
                if (a_pclean[i] && cudaEventElapsedTime(&ms, i ? s.chunk_ev[i - 1] : s.ev_start[a_chunk_k], s.chunk_ev[i]) == cudaSuccess) { obs += ms * 1000.0; tab += tt; } }
            if (tw > 0.0) { r.sm = parts[a_ppart.front()].cls; r.sm_mean = tc / tw; }
            if (tab > 0.5 * tw) { piece_scale = std::min(2.0, std::max(1.0, 0.8 * piece_scale + 0.2 * obs / tab)); } pc_obs_us += obs; pc_tab_us += tab;
            if (piece_trace) {   // req,piece,launch_ns,class,observed_us (end of the previous piece -> end of this one),table_us,lead_us,P,N
                for (size_t i = 0; i < a_ppart.size(); i++) { const Part& q = parts[a_ppart[i]]; float ms = 0; cudaEventElapsedTime(&ms, i ? s.chunk_ev[i - 1] : s.ev_start[a_chunk_k], s.chunk_ev[i]);
                    std::fprintf(piece_trace, "%llu,%zu,%lld,%d,%.1f,%.1f,%.1f,%d,%d\n", static_cast<unsigned long long>(cur.id), i, static_cast<long long>(i ? a_plaunch[i] : a_p0_ns), q.cls, ms * 1000.0, a_pt[i], a_plead[i], q.P, q.N); }
            }
        }
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
        { std::lock_guard<std::mutex> g(rep_m); done.push_back(DoneRec{t_end, cur.model, r.sm_mean}); while (done.size() > 10000) done.pop_front();
          last.valid = true; last.t_end_ns = t_end; last.model = cur.model; last.status = r.status; last.sm = r.sm; last.sm_mean = r.sm_mean; last.infer_ms = r.infer_ms; last.latency_ms = r.latency_ms; }
        if (r.status == "success") a_ok++; else a_fail++;
        have_req = false; aset = nullptr; a_pidx = -1; a_piece = false;
    }

    // ---- loop stall monitor (2026-11-13): which part of the agent loop held the tenant thread longer than DAPP_TENANT_STALL_US
    // (default 1000 us, 0 = off; the request start = JPEG decode is synchronous by design and counts above 10 x that)
    enum { ST_LOOP, ST_LOG, ST_CSV, ST_EST, ST_PUMP, ST_ENQ, ST_FIN, ST_BEGIN, ST_NAP, ST_N };
    struct StallEv { int64_t t_ns; int sec; float us; };
    int stall_us = 1000; int64_t st_last = 0; uint64_t st_cnt[ST_N] = {0}; double st_max[ST_N] = {0}; std::vector<StallEv> st_ev; size_t st_printed = 0;
    void st_note(int sec, double us) { if (us > st_max[sec]) st_max[sec] = us; if (us > (sec == ST_BEGIN ? 10.0 * stall_us : 1.0 * stall_us)) { st_cnt[sec]++; if (st_ev.size() < 4096) st_ev.push_back(StallEv{now_real_ns(), sec, static_cast<float>(us)}); } }
    void st_mark(int sec) { if (stall_us <= 0) return; const int64_t t = now_mono_ns(); if (st_last != 0) st_note(sec, (t - st_last) / 1e3); st_last = t; }
    void nap(int us) {   // sleep; an oversleep is its own section and never charged to the caller's
        const int64_t t0 = now_mono_ns(); std::this_thread::sleep_for(std::chrono::microseconds(us)); if (stall_us <= 0) return;
        const int64_t t1 = now_mono_ns(); st_note(ST_NAP, (t1 - t0) / 1e3 - us); if (st_last != 0) st_last += t1 - t0;
    }
    void st_print(std::FILE* lf) {
        if (!lf || stall_us <= 0) return;
        static const char* nm[ST_N] = {"loop", "log", "csv", "est", "pump", "enq", "finish", "begin", "nap"};
        std::fprintf(lf, "tenant loop: longest us (count over %d us; begin over %d us) |", stall_us, 10 * stall_us);
        for (int i = 0; i < ST_N; i++) { std::fprintf(lf, " %s %.0f (%llu)", nm[i], st_max[i], static_cast<unsigned long long>(st_cnt[i])); st_max[i] = 0; }
        std::fprintf(lf, " | io %s | new events:", AsyncIo::async_on() ? "async" : "direct");
        for (; st_printed < st_ev.size(); st_printed++) { const StallEv& e = st_ev[st_printed]; std::fprintf(lf, " %s@%.6f:%.0f", nm[e.sec], e.t_ns / 1e9, e.us); }
        std::fprintf(lf, "\n"); std::fflush(lf);
    }
    // ---- main loop ----------------------------------------------------------------------------------------------------
    void run() {
        pthread_setname_np(pthread_self(), "dapp_tenant");
        cuCtxSetCurrent(prim);
        std::FILE* lf = log_file.empty() ? nullptr : AsyncIo::get().open(log_file.c_str(), "a");
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
            if (lf) { std::fprintf(lf, "==== agent: port %d, models", srv_port); for (const std::string& m : model_names) std::fprintf(lf, " %s", m.c_str()); std::fprintf(lf, ", input %s (b%d), nvjpeg %s (%u engines), freerun %d\n", input_profile.c_str(), model_names.empty() ? 0 : models[model_names.front()].batch, njd ? "ok" : "MISSING", eng, freerun ? 1 : 0); std::fflush(lf); }
        }
        if (lf) { std::fprintf(lf, "==== tenant ready (engines loaded and validated, decoder warm) %lld\n", static_cast<long long>(now_real_ns() / 1000000000LL)); std::fflush(lf); }
        std::FILE* cf = csv_file.empty() ? nullptr : AsyncIo::get().open(csv_file.c_str(), "w");
        if (cf) { std::fprintf(cf, "abs_slot,sfn,slot,pos,P,N,part,class,set,segments,paused_us,known,plan_ul_sm,plan_dl_sm,t_ns,gap_P,gap_N,run_cls,queue,model,plan_ul_us,plan_dl_us\n"); }
        int64_t last_log = now_mono_ns(); int64_t last_w = -1; int segs_in_slot = 0, segs_q = 0, segs_b = 0; int64_t paused_in_slot = 0; int cur_part_slot = -1; bool known_slot = true;
        uint64_t q_inf_at_log = 0, b_inf_at_log = 0, q_req_at_log = 0, b_req_at_log = 0;
        auto flush_slot = [&](int64_t w) {
            if (cf && last_w >= 0) {
                bool pred; const SlotPlanEntry* e = plan_of(last_w, pred);
                // 2026-11-11 (slot-map video, tools/slotmap_render.py): t_ns = wall clock at the slot change; gap_P/gap_N = the (P, N) gap the sizer
                // offered YOLO in this slot (free SMs = P - N - 8, RAN reserved = 132 - free), request or not; run_cls = SMs the in-flight inference
                // occupies (chunked graphs included; the older `class` column only marks slots with a segment enqueue); queue = waiting requests
                // 2026-11-13: plan_ul_us / plan_dl_us = the sizer's predicted PUSCH completion after t0 / PDSCH duration at the chosen profile (estimator check)
                int gP = -1, gN = -1; { bool gk, gd, gu; busy_of(last_w, gP, gN, gk, gd, gu); }
                const int run_cls = (have_req && a_pidx >= 0) ? parts[a_pidx].cls : 0;
                std::fprintf(cf, "%lld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%d,%d,%d,%lld,%d,%d,%d,%zu,%s,%.0f,%.0f\n", static_cast<long long>(last_w), static_cast<int>((last_w / kSzSlotsPerFrame) & 1023), static_cast<int>(last_w % kSzSlotsPerFrame), static_cast<int>(last_w % 10),
                             cur_part_slot >= 0 ? parts[cur_part_slot].P : -1, cur_part_slot >= 0 ? parts[cur_part_slot].N : -1, cur_part_slot, cur_part_slot >= 0 ? parts[cur_part_slot].cls : 0,
                             segs_b > segs_q ? busy.batch : (segs_q > 0 ? quiet.batch : 0), segs_in_slot, static_cast<long long>(paused_in_slot / 1000), known_slot ? 1 : 0,
                             (e && !pred) ? (e->ul_any ? static_cast<int>(e->ul_sm) : 0) : -1, (e && !pred) ? (e->dl_any ? static_cast<int>(e->dl_sm) : 0) : -1,
                             static_cast<long long>(now_real_ns()), gP, gN, run_cls, agent_on ? srv.queue_depth() : static_cast<size_t>(0), have_req ? cur.model.c_str() : "",
                             (e && !pred && e->ul_prof >= 0) ? static_cast<double>(e->pusch_t) : -1.0, (e && !pred && e->dl_prof >= 0) ? static_cast<double>(e->pdsch_t) : -1.0);
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
            st_mark(ST_LOOP);   // the rest of the previous pass
            if (prof_on && now_real_ns() - prof_t0 > static_cast<int64_t>(prof_s) * 1000000000LL) { cudaProfilerStop(); prof_on = false; prof_done = true; if (lf) { std::fprintf(lf, "agent: cudaProfilerStop\n"); std::fflush(lf); } }
            const int64_t tnow = now_mono_ns();
            if (tnow - last_log >= static_cast<int64_t>(log_s) * 1000000000LL) {
                st_mark(ST_LOOP);
                const double sec = (tnow - last_log) / 1e9;
                if (lf) {
                    const uint64_t tiles = (qpipe.inferences - q_inf_at_log) * quiet.batch + (bpipe.inferences - b_inf_at_log) * (busy.loaded ? busy.batch : 0);
                    std::fprintf(lf, "[%lld] tiles %llu (%.1f/s = %.1f 4K/s) pauses %llu (%.1f ms)", static_cast<long long>(now_real_ns() / 1000000000LL), static_cast<unsigned long long>(tiles), tiles / sec, tiles / sec / tiles_per_request,
                                 static_cast<unsigned long long>(pauses), pause_ns / 1e6);
                    stats(qpipe, "quiet", q_inf_at_log, q_req_at_log, sec); stats(bpipe, "busy", b_inf_at_log, b_req_at_log, sec);
                    if (agent_on) { std::fprintf(lf, " | agent ok %llu fail %llu queue %zu received %llu rejected %llu dec_ema %.1f ms", static_cast<unsigned long long>(a_ok), static_cast<unsigned long long>(a_fail), srv.queue_depth(), static_cast<unsigned long long>(srv.requests_received()), static_cast<unsigned long long>(srv.rejected()), dec_ema); }
                    if (agent_on && piece_on) { std::fprintf(lf, " | piece %llu pieces %llu switches %llu waits (%.1f ms, %llu deferrals) scale %.3f observed/table %.3f plan lead %.0f us", static_cast<unsigned long long>(pc_pieces), static_cast<unsigned long long>(pc_switch), static_cast<unsigned long long>(pc_wait), pc_wait_ns / 1e6, static_cast<unsigned long long>(pc_defer), piece_scale, pc_tab_us > 0.0 ? pc_obs_us / pc_tab_us : 0.0, g_plan_lead_ns.load(std::memory_order_relaxed) / 1e3); }
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
                st_mark(ST_LOG); st_print(lf); st_last = now_mono_ns();
                last_log = tnow;
            }
            // ---- agent path: requests from the Resource Allocator (sequential), estimates every slot
            if (agent_on) {
                int64_t w = 0, slot_end = 0; const bool clk = current_slot(w, slot_end);
                if (clk && w != last_w) {
                    if (last_w >= 0) { int P, N; bool kn, dla, ula; busy_of(last_w, P, N, kn, dla, ula); std::lock_guard<std::mutex> g(rep_m); yolo_sm_sum += (have_req && a_pidx >= 0) ? parts[a_pidx].cls : 0; ran_sm_sum += (nsm - P) + N; slot_samples++; }
                    st_mark(ST_LOOP); flush_slot(w); st_mark(ST_CSV);
                    const std::string e = estimate_json(w, slot_end, true); srv.publish_estimate(e); { std::lock_guard<std::mutex> g(rep_m); last_estimate = e; }
                    st_mark(ST_EST);
                }
                if (have_req && aset) {
                    EngineSet& s = *aset;
                    if (a_chunking) { st_mark(ST_LOOP); const bool pumped = agent_pump(lf); st_mark(ST_PUMP); if (!pumped) continue; if (a_chunking) { nap(5); if (!loop_on) continue; } }
                    if (have_req && aset && a_chunking) { }
                    else if (apipe.k < s.nseg) {
                        // next segment once the previous one finished (keeps the class choice current for segmented sets)
                        if (apipe.k == 0 || cudaEventQuery(s.ev_end[apipe.k - 1]) == cudaSuccess) {
                            st_mark(ST_LOOP);
                            bool kn = true; int pidx = !clk ? a_pidx : (piece_on && s.piece_ok && chunk_n > 0) ? choose_piece(s, apipe.k, 0, w, slot_end, 0.0) : choose_part(s, apipe.k, w, slot_end, kn, false);
                            if (pidx >= 0) { a_pidx = pidx; if (cur_part_slot < 0) cur_part_slot = pidx; agent_enqueue(pidx, lf); segs_in_slot++; st_mark(ST_ENQ); }
                            else { st_mark(ST_ENQ); pauses++; nap(pause_us); }
                        } else { nap(50); }
                    } else if (cudaEventQuery(s.ev_end[s.nseg - 1]) == cudaSuccess) { st_mark(ST_LOOP); finish_request(lf); st_mark(ST_FIN); }
                    else { nap(50); }
                    if (!loop_on) continue;
                } else {
                    if (!have_req && srv.pop_request(cur)) { have_req = true; }
                    if (have_req) { st_mark(ST_LOOP); const bool begun = begin_request(w, slot_end, clk, lf); st_mark(ST_BEGIN); if (!begun && have_req) { pauses++; nap(pause_us); } continue; }
                    if (!loop_on) { if (!clk) { std::lock_guard<std::mutex> g(rep_m); } nap(100); continue; }
                }
            }
            // ---- which pipeline and partition now?
            Pipe* pp = &qpipe; int pidx = -1; bool known = true; int64_t w = 0, slot_end = 0;
            if (fixed > 0 && fixed_gap_part < 0) { pidx = 0; }
            else if (fixed_gap_part >= 0) { pidx = fixed_gap_part; if (current_slot(w, slot_end) && w != last_w) flush_slot(w); }
            else if (!current_slot(w, slot_end)) { nap(200); continue; }
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
                nap(pause_us);
                const int64_t d = now_mono_ns() - t_p; pause_ns += d; paused_in_slot += d;
                continue;
            }
            if (cur_part_slot < 0) cur_part_slot = pidx;
            if (!step(*pp, pidx, lf)) { stop.store(true); break; }
            segs_in_slot++; if (pp == &qpipe) segs_q++; else segs_b++;
        }
        if (cf) { flush_slot(-1); std::fclose(cf); }
        if (piece_trace) { std::fclose(piece_trace); piece_trace = nullptr; }
        st_print(lf);
        if (lf) { std::fprintf(lf, "tenant thread stopped: quiet inferences %llu busy inferences %llu agent ok %llu fail %llu\n", static_cast<unsigned long long>(qpipe.inferences), static_cast<unsigned long long>(bpipe.inferences), static_cast<unsigned long long>(a_ok), static_cast<unsigned long long>(a_fail)); std::fclose(lf); }
        if (njd) { njd_destroy(njd); njd = nullptr; }
        if (dec_s) { cudaStreamDestroy(dec_s); dec_s = nullptr; }
    }

    bool init(std::string& err) {
        quiet.name = "quiet"; quiet.dir = envs("DAPP_TENANT_DIR", "/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/seg26n_fine");
        quiet.input = envs("DAPP_TENANT_INPUT", "/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/seg26n/input_uhd_b8_fp16.bin"); quiet.batch = envi("DAPP_TENANT_BATCH", 8);
        busy.name = "busy"; busy.dir = envs("DAPP_TENANT_BUSY_DIR", ""); busy.input = envs("DAPP_TENANT_BUSY_INPUT", "/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/seg26n_b1/input_uhd_b1_fp16.bin"); busy.batch = envi("DAPP_TENANT_BUSY_BATCH", 1);
        log_file = envs("DAPP_TENANT_LOG", "/tmp/dapp_tenant.log"); csv_file = envs("DAPP_TENANT_CSV", "/tmp/dapp_tenant_slots.csv");
        { const char* ev = std::getenv("DAPP_TENANT_CSV"); if (ev && !*ev) csv_file.clear(); }   // 2026-11-11: DAPP_TENANT_CSV= (empty) switches the per-slot CSV off (.env DAPP_SLOT_CSV=0)
        classes = env_list("DAPP_TENANT_CLASSES", {40, 48, 64, 88}); std::sort(classes.begin(), classes.end());
        cpu = envi("DAPP_TENANT_CPU", 45); prio = envi("DAPP_TENANT_PRIO", 0); lookahead = envi("DAPP_TENANT_LOOKAHEAD", 1); guard_us = envi("DAPP_TENANT_GUARD_US", 60);
        ul_back = envi("DAPP_TENANT_UL_BACK", 4); dl_fwd = envi("DAPP_TENANT_DL_FWD", 2); fixed = envi("DAPP_TENANT_FIXED", 0); pause_us = envi("DAPP_TENANT_PAUSE_US", 20); log_s = envi("DAPP_TENANT_LOG_S", 10);
        dl_pause = envi("DAPP_TENANT_DL_PAUSE", 0); dl_max_class = envi("DAPP_TENANT_DL_MAX_CLASS", 0); ul_pause = envi("DAPP_TENANT_UL_PAUSE", 0); tiles_per_request = envi("DAPP_TENANT_TILES", 8);
        unknown_prev = std::string(envs("DAPP_TENANT_UNKNOWN", "prev")) != "max"; horizon_max = envi("DAPP_TENANT_HORIZON_MAX", 200); pred_frames = envi("DAPP_TENANT_PRED_FRAMES", 12); empty_us = envi("DAPP_TENANT_EMPTY_US", 400);
        one_per_gap = envi("DAPP_TENANT_ONE_PER_GAP", -1);   // -1: union layout keeps only the largest class per (P, N) (the horizon rule never uses a smaller one); gap layout keeps all
        fb_guard = envi("DAPP_TENANT_FB_GUARD", 0); layout = envs("DAPP_TENANT_LAYOUT", "union"); engine_ctx_per_class = std::string(envs("DAPP_TENANT_ENGINE_CTX", "prim")) == "part";   // 2026-11-10: was a const char* pointer comparison (prim never took effect)
        if (one_per_gap < 0) one_per_gap = (layout == "gap") ? 0 : 1;   // 2026-11-13: back on its own line (the 11-10 comment had swallowed it; -1 acted as 1)
        // 2026-11-12 (feature/slot_level): DAPP_TENANT_ENGINE_SM=132 one-engine sets, DAPP_AGENT_PIECE=1 SM class per graph piece (needs DAPP_AGENT_GRAPH_CHUNK > 0), guard per piece
        engine_sm = envi("DAPP_TENANT_ENGINE_SM", 0); piece_on = envi("DAPP_AGENT_PIECE", 0) != 0; piece_guard_us = envi("DAPP_AGENT_PIECE_GUARD_US", 30); piece_window = envi("DAPP_AGENT_PIECE_WINDOW", 1) != 0;
        stall_us = envi("DAPP_TENANT_STALL_US", 1000);
        { const char* pt = std::getenv("DAPP_AGENT_PIECE_TRACE"); if (piece_on && pt && *pt) { piece_trace = AsyncIo::get().open(pt, "w"); if (piece_trace) std::fprintf(piece_trace, "req,piece,launch_ns,cls,obs_us,tab_us,lead_us,P,N\n"); } }
        qos_on = envi("DAPP_AGENT_QOS", 1) != 0; prof_s = envi("DAPP_AGENT_PROFILE_S", 0);
        chunk_n = envi("DAPP_AGENT_GRAPH_CHUNK", 3); chunk_ahead = std::max(1, envi("DAPP_AGENT_CHUNK_AHEAD", 2));
        piece_us_target = envi("DAPP_AGENT_PIECE_US", 0); ahead_us = envi("DAPP_AGENT_AHEAD_US", 0); ahead_max = std::max(1, envi("DAPP_AGENT_AHEAD_MAX", 8));
        defer_slots = std::max(0, envi("DAPP_AGENT_PIECE_DEFER", 0)); defer_min_us = envi("DAPP_AGENT_PIECE_DEFER_MIN_US", 150); defer_gain_us = envi("DAPP_AGENT_PIECE_DEFER_GAIN_US", 50);
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
            std::FILE* lf0 = log_file.empty() ? nullptr : AsyncIo::get().open(log_file.c_str(), "a");
            if (lf0) { std::fprintf(lf0, "==== stress mode %s: %d MB buffer, launch %.0f us on class %d (spin cycles %lld, read iters %d, blocks %d, gap %d us)\n", stress.c_str(), stress_mb, ms * 1000.0, p.cls, spin_cycles, read_iters, cal_blocks, stress_gap_us); std::fclose(lf0); }
        } else {
            if (loop_on && !load_set(quiet, err)) return false;
            if (loop_on && !busy.dir.empty() && !load_set(busy, err)) return false;
            for (const std::string& m : model_names) { if (!load_set(models[m], err)) { err = "model " + m + ": " + err; return false; } }
        }
        qpipe.set = &quiet; qpipe.cls_hist.assign(quiet.classes.size(), 0); qpipe.tiles_left = tiles_per_request;
        bpipe.set = &busy; bpipe.cls_hist.assign(busy.classes.size(), 0); bpipe.tiles_left = tiles_per_request;
        std::FILE* lf = log_file.empty() ? nullptr : AsyncIo::get().open(log_file.c_str(), "a");
        if (lf) {
            std::fprintf(lf, "==== dapp_tenant quiet %s (b%d, %d segments, classes", quiet.dir.c_str(), quiet.batch, quiet.nseg); for (int c : quiet.classes) std::fprintf(lf, " %d", c);
            if (busy.loaded) { std::fprintf(lf, ") busy %s (b%d, %d segments, classes", busy.dir.c_str(), busy.batch, busy.nseg); for (int c : busy.classes) std::fprintf(lf, " %d", c); }
            std::fprintf(lf, ") | PUSCH partitions"); for (int y : pusch_sm) std::fprintf(lf, " %d", y); std::fprintf(lf, " | PDSCH partitions"); for (int n : pdsch_sm) std::fprintf(lf, " %d", n);
            std::fprintf(lf, " | cpu %d prio %d lookahead %d guard %d us ul_back %d dl_fwd %d fixed %d (gap %d,%d) one_per_gap %d dl_pause %d dl_max_class %d ul_pause %d unknown %s empty_us %d tiles/request %d\n", cpu, prio, lookahead, guard_us, ul_back, dl_fwd, fixed, fixed_gap_P, fixed_gap_N, one_per_gap, dl_pause, dl_max_class, ul_pause, unknown_prev ? "prev" : "max", empty_us, tiles_per_request);
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
                if (chunk_n > 0 && piece_on && piece_us_target > 0) { for (int k = 0; k < es.nseg; k++) { std::string e2; if (!balance_cut(es, m, k, lf, e2)) { err = "balanced cut of " + m + ": " + e2; if (lf) std::fclose(lf); return false; } } }
                for (size_t pi = 0; pi < parts.size(); pi++) { const Part& p = parts[pi]; const int ci = es.cls_index(p.cls); if (ci < 0) continue;
                    for (int k = 0; k < es.nseg; k++) {
                        cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
                        if (cudaStreamBeginCapture(p.s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "graph capture begin failed"; if (lf) std::fclose(lf); return false; }
                        bind_aux(es.segs[k], ci, p);
                        const bool okq = es.segs[k].exec[ci]->enqueueV3(p.s);
                        const cudaError_t ce = cudaStreamEndCapture(p.s, &g);
                        if (!okq || ce != cudaSuccess || !g) { err = "graph capture of " + m + " seg " + std::to_string(k) + " part " + std::to_string(pi) + " failed (" + (okq ? cudaGetErrorString(ce) : "enqueue") + ")"; if (lf) std::fclose(lf); return false; }
                        if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) { err = "graph instantiate failed"; if (lf) std::fclose(lf); return false; }
                        if (chunk_n > 0) { std::vector<cudaGraphExec_t> pieces; std::string e2; const int64_t tc0 = now_mono_ns(); { auto bt = es.bounds.find(k); if (!chunk_graph(g, chunk_n, bt != es.bounds.end() ? &bt->second : nullptr, pieces, p.s, e2)) { err = "graph chunking of " + m + ": " + e2; if (lf) std::fclose(lf); return false; }
                            } es.chunks[{k, static_cast<int>(pi)}] = pieces; nchunk += static_cast<int>(pieces.size()); chunk_ms += (now_mono_ns() - tc0) / 1e6;
                            while (es.chunk_ev.size() < pieces.size()) { cudaEvent_t ce2 = nullptr; cudaEventCreate(&ce2); es.chunk_ev.push_back(ce2); } }
                        cudaGraphDestroy(g); cudaGraphUpload(ge, p.s); es.graphs[{k, static_cast<int>(pi)}] = ge; ng++;
                    }
                }
                cudaDeviceSynchronize();
            }
            // 2026-11-12 piece mode: table time of every piece on every partition - the pieces launched back to back on the partition's stream,
            // median of 3 passes after one warm-up pass (the launch-time choice needs the time of ONE piece on a given class)
            if (piece_on && chunk_n > 0) {
                for (const std::string& m : model_names) { EngineSet& es = models[m]; if (!es.loaded || es.nseg != 1) continue;
                    size_t np = 0; bool same = true; int nparts = 0;
                    for (auto& kv : es.chunks) { const Part& p = parts[kv.first.second]; const std::vector<cudaGraphExec_t>& pv = kv.second; const size_t n = pv.size();
                        if (np == 0) np = n; if (n != np || n == 0) { same = false; continue; }
                        std::vector<std::vector<float>> rep(n);
                        for (int r = 0; r < 4; r++) {
                            cudaEventRecord(es.ev_start[0], p.s); for (size_t j = 0; j < n; j++) { cudaGraphLaunch(pv[j], p.s); cudaEventRecord(es.chunk_ev[j], p.s); } cudaStreamSynchronize(p.s);
                            for (size_t j = 0; r > 0 && j < n; j++) { float ms = 0; if (cudaEventElapsedTime(&ms, j ? es.chunk_ev[j - 1] : es.ev_start[0], es.chunk_ev[j]) == cudaSuccess) rep[j].push_back(ms * 1000.0f); }
                        }
                        std::vector<float> t(n, 0.0f); double sum = 0.0, mx = 0.0;
                        for (size_t j = 0; j < n; j++) { if (rep[j].empty()) { same = false; break; } std::sort(rep[j].begin(), rep[j].end()); t[j] = rep[j][rep[j].size() / 2]; sum += t[j]; mx = std::max(mx, static_cast<double>(t[j])); }
                        es.piece_us[kv.first] = t; nparts++;
                        if (lf) std::fprintf(lf, "  %s piece table part %d (P %d, N %d, class %d): %zu pieces, sum %.2f ms, longest piece %.0f us\n", m.c_str(), kv.first.second, p.P, p.N, p.cls, n, sum / 1000.0, mx);
                    }
                    es.piece_ok = same && nparts > 0;
                    if (lf) std::fprintf(lf, "  %s piece mode %s (DAPP_AGENT_PIECE=1, guard %d us, slots %s, unknown slots = %s)\n", m.c_str(), es.piece_ok ? "ON: SM class chosen per graph piece" : "OFF (pieces differ between partitions): request-level rule", piece_guard_us, piece_window ? "from the piece start" : "from now", unknown_prev ? "previous frame" : "largest profile");
                }
            }
            if (lf) std::fprintf(lf, "  CUDA graphs captured: %d (DAPP_AGENT_GRAPH=1), cut into %d pieces of %d nodes (DAPP_AGENT_GRAPH_CHUNK) in %.1f s, %d queued ahead (DAPP_AGENT_CHUNK_AHEAD)\n", ng, nchunk, chunk_n, chunk_ms / 1000.0, chunk_ahead);
            if (lf && (piece_us_target > 0 || ahead_us > 0 || defer_slots > 0)) std::fprintf(lf, "  piece cut: %s | piece queue: %s | waiting rule: %s (DAPP_AGENT_PIECE_US %d, DAPP_AGENT_AHEAD_US %d, DAPP_AGENT_AHEAD_MAX %d, DAPP_AGENT_PIECE_DEFER %d slots / pieces over %d us / gain %d us)\n", piece_us_target > 0 && piece_on ? "balanced by time" : "fixed node count", ahead_us > 0 && piece_on ? "by time" : "by count", defer_slots > 0 && piece_on ? "on" : "off", piece_us_target, ahead_us, ahead_max, defer_slots, defer_min_us, defer_gain_us);
        }
        if (lf) std::fclose(lf);
        return true;
    }
    bool launch(std::string& err) {
        if (agent_on) {
            agent::Config c; c.port = srv_port; c.input = input_profile = envs("DAPP_INPUT", "4k"); c.udp = envs("DAPP_AGENT_UDP", ""); c.cpu = envi("DAPP_AGENT_CPU", 46);   // an isolated core nobody else uses (tenant 45, export 43); unpinned it inherited the L1 main thread's core -> L1 recovery exit c.max_queue = envi("DAPP_AGENT_MAX_QUEUE", 8); c.cells = envi("DAPP_AGENT_CELLS", 1); c.gpu_sm = nsm;
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
