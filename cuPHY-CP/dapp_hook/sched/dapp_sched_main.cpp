/*
 * dapp_sched: prototype scheduler component.
 *
 * Watches the FAPI hook ring, keeps a windowed estimate of how loaded cuPHY
 * is, and when an inference request arrives decides how many SMs that
 * inference may use, then runs it on a CUDA context capped to that budget.
 *
 * No model. The policy is the linear rule in dapp_sched.hpp; swapping in a
 * trained predictor later means replacing one function.
 *
 * The inference is a real YOLO network through TensorRT when the binary is
 * built with TensorRT and started with --engine (dapp_yolo.hpp). One engine
 * instance lives in every SM-capped context, so switching budgets costs
 * nothing at request time. Without --engine the old placeholder kernel runs.
 *
 * Request interface: a Unix domain socket. A client writes a request line and
 * reads back one reply line:
 *
 *     -> RUN <image.jpg>                (YOLO; "RUN" alone uses --image)
 *     <- OK sm=48 granted=48 pct=36.4 load=0.31 ms=1.9 pre_ms=4.1 dets=5 top=bus:0.87,person:0.85 capped=1 reason=moderate uplink
 *     -> RUN <milliseconds_of_work>     (placeholder kernel)
 *     <- OK sm=48 granted=48 pct=36.4 load=0.31 ms=12.4 capped=1 reason=moderate uplink
 *
 *   dapp_sched [-n /aerial_dapp_ring] [-s /tmp/dapp_sched.sock] [--cells N]
 *              [--classes 16,32,48,64,96] [--engine yolo.engine] [--image img.jpg]
 *              [--conf 0.25] [--iou 0.7] [--sweep N] [--self-test N] [--verbose]
 */
#include "dapp_sched.hpp"
#include "dapp_sm_pool.hpp"
#ifdef DAPP_SCHED_TENSORRT
#include "dapp_yolo.hpp"
#endif

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

using namespace nv::dapp;
using namespace nv::dapp::sched;

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

// ---------------------------------------------------------------------------
// Placeholder inference: keeps the GPU busy for roughly the requested time so
// the SM cap has a visible effect. Used when no engine is loaded.
// ---------------------------------------------------------------------------
static double run_placeholder(const SmPool& pool, const SmPool::Slot* slot, int work_ms)
{
    const auto t0 = std::chrono::steady_clock::now();
    if (!pool.bind(slot)) { return -1.0; }
    CUdeviceptr d = 0;
    if (cuMemAlloc(&d, 1024 * 1024) == CUDA_SUCCESS) {
        const auto deadline = t0 + std::chrono::milliseconds(work_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            cuMemsetD8(d, 0xA5, 1024 * 1024);
        }
        cuCtxSynchronize();
        cuMemFree(d);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// YOLO: one engine per SM class (plus one for the uncapped fallback context)
// ---------------------------------------------------------------------------
struct InferResult {
    bool        ok = false;
    double      gpu_ms = -1.0;   // H2D + network + D2H on the GPU
    double      pre_ms = 0.0;    // decode + letterbox on the CPU (0 when cached)
    double      total_ms = 0.0;  // request in, reply out
    int         dets = 0;
    std::string top;             // "bus:0.87,person:0.85,..."
    std::string err;
};

struct YoloRig {
    bool  enabled = false;
    float conf = 0.25f, iou = 0.7f;
    int   net_w = 0, net_h = 0, nc = 0, na = 0;
#ifdef DAPP_SCHED_TENSORRT
    std::vector<std::unique_ptr<YoloEngine>> per_slot;   // index == pool.slots() index
    std::unique_ptr<YoloEngine>              fallback;   // uncapped context
    std::map<std::string, YoloInput>         cache;      // preprocessed images
    std::vector<Detection>                   last;       // detections of the last run
#endif
};

#ifdef DAPP_SCHED_TENSORRT
static bool yolo_init(YoloRig& rig, const SmPool& pool, const std::string& path, std::string& err)
{
    std::vector<char> blob;
    if (!YoloEngine::read_file(path, blob, err)) { return false; }

    rig.per_slot.resize(pool.slots().size());
    for (size_t i = 0; i < pool.slots().size(); ++i) {
        if (!pool.bind(&pool.slots()[i])) { err = "cannot bind SM class context"; return false; }
        rig.per_slot[i] = std::make_unique<YoloEngine>();
        if (!rig.per_slot[i]->load(blob, err)) {
            err = "class " + std::to_string(pool.slots()[i].sm) + " SM: " + err;
            return false;
        }
    }
    if (!pool.capping_active()) {
        if (!pool.bind(nullptr)) { err = "cannot bind fallback context"; return false; }
        rig.fallback = std::make_unique<YoloEngine>();
        if (!rig.fallback->load(blob, err)) { return false; }
    }
    const YoloEngine* any = rig.fallback ? rig.fallback.get() : rig.per_slot.front().get();
    rig.net_w = any->net_w();
    rig.net_h = any->net_h();
    rig.nc    = any->num_classes();
    rig.na    = any->num_anchors();
    rig.enabled = true;
    return true;
}

// Engines must die while their context is still alive and current.
static void yolo_shutdown(YoloRig& rig, const SmPool& pool)
{
    for (size_t i = 0; i < rig.per_slot.size(); ++i) {
        if (rig.per_slot[i]) { pool.bind(&pool.slots()[i]); rig.per_slot[i].reset(); }
    }
    if (rig.fallback) { pool.bind(nullptr); rig.fallback.reset(); }
}

static InferResult run_yolo(YoloRig& rig, const SmPool& pool, const SmPool::Slot* slot, const std::string& image)
{
    InferResult r;
    const auto t0 = std::chrono::steady_clock::now();

    // CPU side first, outside any GPU context: decode + letterbox, cached per path.
    auto it = rig.cache.find(image);
    if (it == rig.cache.end()) {
        YoloInput in;
        if (!yolo_preprocess(image, rig.net_w, rig.net_h, in, r.err)) { return r; }
        if (rig.cache.size() >= 16) { rig.cache.clear(); }
        it = rig.cache.emplace(image, std::move(in)).first;
        r.pre_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    YoloEngine* eng = nullptr;
    if (slot) {
        const size_t idx = static_cast<size_t>(slot - &pool.slots()[0]);
        if (idx < rig.per_slot.size()) { eng = rig.per_slot[idx].get(); }
    } else {
        eng = rig.fallback.get();
    }
    if (!eng || !pool.bind(slot)) { r.err = "no engine for the selected context"; return r; }

    float gpu_ms = 0.f;
    if (!eng->infer(it->second.chw.data(), gpu_ms)) { r.err = "inference failed"; return r; }
    rig.last = eng->detections(it->second, rig.conf, rig.iou);

    r.ok     = true;
    r.gpu_ms = gpu_ms;
    r.dets   = static_cast<int>(rig.last.size());
    for (size_t i = 0; i < rig.last.size() && i < 3; ++i) {
        char t[48];
        std::snprintf(t, sizeof(t), "%s%s:%.2f", i ? "," : "", coco_name(rig.last[i].cls), rig.last[i].conf);
        r.top += t;
    }
    r.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

static void print_detections(const YoloRig& rig)
{
    for (const Detection& d : rig.last) {
        std::printf("        %-14s %.2f  [%4.0f %4.0f %4.0f %4.0f]\n",
                    coco_name(d.cls), d.conf, d.x1, d.y1, d.x2, d.y2);
    }
}
#endif // DAPP_SCHED_TENSORRT

// ---------------------------------------------------------------------------
struct Shared {
    Config cfg;
    std::atomic<double>   peak{1.0};
    std::atomic<bool>     stale{true};
    std::atomic<uint64_t> slots{0};
    std::atomic<uint64_t> dropped{0};
    std::atomic<uint64_t> requests{0};
};

// Ring consumer thread: folds records into the load tracker.
static void ring_thread(Shared* sh, const std::string& ring_name, bool verbose)
{
    LoadTracker tracker(sh->cfg);
    std::unique_ptr<Consumer> cons;
    std::string err;

    while (!g_stop) {
        if (!cons) {
            cons = Consumer::open(ring_name, err);
            if (!cons) {
                sh->stale.store(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
            cons->seek_to_head();
            std::printf("[ring] attached to %s\n", ring_name.c_str());
        }
        dapp_rec_t r;
        uint64_t lost = 0;
        const int st = cons->next(r, &lost);
        if (st == Consumer::RESTARTED) {
            std::printf("[ring] producer restarted\n");
            continue;
        }
        if (st == Consumer::NONE) {
            sh->stale.store(tracker.stale());
            sh->peak.store(tracker.peak());
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }
        if (tracker.on_record(r)) {
            sh->peak.store(tracker.peak());
            sh->stale.store(tracker.stale());
            sh->slots.store(tracker.slots_seen());
            sh->dropped.store(tracker.slots_dropped());
            if (verbose) {
                const SlotLoad* s = tracker.newest();
                if (s) {
                    std::printf("[slot] %4u.%-2u cells=%u prb*layers=%" PRIu64
                                " tb=%" PRIu64 "B index=%.3f peak=%.3f\n",
                                s->sfn, s->slot, s->cells, s->prb_layers, s->tb_bytes,
                                s->index, tracker.peak());
                }
            }
        }
    }
}

static bool is_number(const std::string& s)
{
    if (s.empty()) { return false; }
    for (char c : s) { if (c < '0' || c > '9') { return false; } }
    return true;
}

// One inference request: decide, run, report.
// arg: image path (YOLO), a number of milliseconds (placeholder), or empty.
static std::string handle_request(Shared* sh, const SmPool& pool, YoloRig& rig,
                                  const std::string& arg, const std::string& default_image,
                                  int work_ms, bool verbose)
{
    const bool   stale = sh->stale.load();
    const double load  = sh->peak.load();
    const Decision d   = decide(sh->cfg, load, stale);
    const SmPool::Slot* slot = pool.pick(d.infer_sm);
    const uint32_t granted = slot ? slot->sm : pool.device_sm();
    sh->requests.fetch_add(1);

    char buf[400];
#ifdef DAPP_SCHED_TENSORRT
    const bool want_yolo = rig.enabled && !is_number(arg);
    if (want_yolo) {
        const std::string image = arg.empty() ? default_image : arg;
        if (image.empty()) { return "ERR no image given and no --image default\n"; }
        const InferResult r = run_yolo(rig, pool, slot, image);
        if (!r.ok) {
            std::snprintf(buf, sizeof(buf), "ERR %s\n", r.err.c_str());
            return std::string(buf);
        }
        std::snprintf(buf, sizeof(buf),
                      "OK sm=%u granted=%u pct=%.1f load=%.3f ms=%.2f pre_ms=%.1f dets=%d top=%s capped=%d reason=%s\n",
                      d.infer_sm, granted, d.infer_pct, d.load, r.gpu_ms, r.pre_ms, r.dets,
                      r.top.empty() ? "-" : r.top.c_str(), pool.capping_active() ? 1 : 0, d.reason);
        std::printf("[run ] budget=%u granted=%u load=%.3f%s -> yolo %.2f ms gpu, %d dets [%s]  (%s)\n",
                    d.infer_sm, granted, d.load, stale ? " STALE" : "", r.gpu_ms, r.dets,
                    r.top.c_str(), d.reason);
        if (verbose) { print_detections(rig); }
        return std::string(buf);
    }
#else
    (void)rig; (void)default_image; (void)verbose;
#endif
    const int ms_req = is_number(arg) ? std::atoi(arg.c_str()) : work_ms;
    const double ms = run_placeholder(pool, slot, ms_req > 0 ? ms_req : work_ms);
    std::snprintf(buf, sizeof(buf),
                  "OK sm=%u granted=%u pct=%.1f load=%.3f ms=%.1f capped=%d reason=%s\n",
                  d.infer_sm, granted, d.infer_pct, d.load, ms,
                  pool.capping_active() ? 1 : 0, d.reason);
    std::printf("[run ] budget=%u granted=%u load=%.3f%s -> placeholder %.1f ms  (%s)\n",
                d.infer_sm, granted, d.load, stale ? " STALE" : "", ms, d.reason);
    return std::string(buf);
}

// --sweep: run the default image N times on every SM class and report the
// median, so the effect of the cap on a real network can be seen directly.
#ifdef DAPP_SCHED_TENSORRT
static int run_sweep(YoloRig& rig, const SmPool& pool, const std::string& image, int n)
{
    if (!rig.enabled) { std::fprintf(stderr, "--sweep needs --engine\n"); return 2; }
    if (image.empty()) { std::fprintf(stderr, "--sweep needs --image\n"); return 2; }
    std::printf("[sweep] %s, %d runs per class\n", image.c_str(), n);
    std::vector<const SmPool::Slot*> targets;
    for (const SmPool::Slot& s : pool.slots()) { targets.push_back(&s); }
    if (!pool.capping_active()) { targets.push_back(nullptr); }
    bool printed = false;
    for (const SmPool::Slot* slot : targets) {
        std::vector<double> ms;
        int dets = 0;
        for (int i = 0; i < n; ++i) {
            const InferResult r = run_yolo(rig, pool, slot, image);
            if (!r.ok) { std::fprintf(stderr, "sweep: %s\n", r.err.c_str()); return 1; }
            ms.push_back(r.gpu_ms);
            dets = r.dets;
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        if (slot) { std::printf("[sweep] SM=%3u  median=%.2f ms  min=%.2f ms  max=%.2f ms  dets=%d\n",
                                slot->sm, med, ms.front(), ms.back(), dets); }
        else      { std::printf("[sweep] uncapped  median=%.2f ms  min=%.2f ms  max=%.2f ms  dets=%d\n",
                                med, ms.front(), ms.back(), dets); }
        if (!printed) { print_detections(rig); printed = true; }
    }
    return 0;
}
#endif

int main(int argc, char** argv)
{
    Shared sh;
    std::string ring_name = DAPP_RING_DEFAULT_NAME;
    std::string sock_path = "/tmp/dapp_sched.sock";
    std::string engine_path, default_image;
    std::vector<uint32_t> classes = {16, 32, 48, 64, 96};
    int self_test = 0, work_ms = 10, sweep = 0;
    bool verbose = false;
    YoloRig rig;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-n" || a == "--ring") && i + 1 < argc) ring_name = argv[++i];
        else if ((a == "-s" || a == "--sock") && i + 1 < argc) sock_path = argv[++i];
        else if (a == "--cells" && i + 1 < argc) sh.cfg.num_cells = (uint32_t)atoi(argv[++i]);
        else if (a == "--window" && i + 1 < argc) sh.cfg.window_slots = (uint32_t)atoi(argv[++i]);
        else if (a == "--max-prb" && i + 1 < argc) sh.cfg.max_prb = (uint32_t)atoi(argv[++i]);
        else if (a == "--max-layers" && i + 1 < argc) sh.cfg.max_layers = (uint32_t)atoi(argv[++i]);
        else if (a == "--max-tb-kb" && i + 1 < argc) sh.cfg.max_tb_kbytes = (uint32_t)atoi(argv[++i]);
        else if (a == "--cuphy-sm" && i + 2 < argc) {
            sh.cfg.cuphy_sm_idle = (uint32_t)atoi(argv[++i]);
            sh.cfg.cuphy_sm_peak = (uint32_t)atoi(argv[++i]);
        }
        else if (a == "--infer-sm" && i + 2 < argc) {
            sh.cfg.infer_sm_min = (uint32_t)atoi(argv[++i]);
            sh.cfg.infer_sm_max = (uint32_t)atoi(argv[++i]);
        }
        else if (a == "--self-test" && i + 1 < argc) self_test = atoi(argv[++i]);
        else if (a == "--sweep" && i + 1 < argc) sweep = atoi(argv[++i]);
        else if (a == "--work-ms" && i + 1 < argc) work_ms = atoi(argv[++i]);
        else if (a == "--engine" && i + 1 < argc) engine_path = argv[++i];
        else if (a == "--image" && i + 1 < argc) default_image = argv[++i];
        else if (a == "--conf" && i + 1 < argc) rig.conf = (float)atof(argv[++i]);
        else if (a == "--iou" && i + 1 < argc) rig.iou = (float)atof(argv[++i]);
        else if (a == "--verbose" || a == "-v") verbose = true;
        else if (a == "--classes" && i + 1 < argc) {
            classes.clear();
            char* s = argv[++i];
            for (char* tok = strtok(s, ","); tok; tok = strtok(nullptr, ",")) {
                classes.push_back((uint32_t)atoi(tok));
            }
        } else {
            std::fprintf(stderr,
                "usage: %s [-n ring] [-s sock] [--cells N] [--window N] [--classes a,b,c]\n"
                "          [--max-prb N] [--max-layers N] [--max-tb-kb N]\n"
                "          [--cuphy-sm IDLE PEAK] [--infer-sm MIN MAX]\n"
                "          [--engine yolo.engine] [--image img.jpg] [--conf 0.25] [--iou 0.7]\n"
                "          [--sweep N] [--self-test N] [--work-ms N] [--verbose]\n", argv[0]);
            return 2;
        }
    }

    SmPool pool;
    std::string err;
    if (!pool.init(0, classes, err)) {
        std::fprintf(stderr, "GPU init failed: %s\n", err.c_str());
        return 1;
    }
    sh.cfg.total_sm = pool.device_sm();

    std::printf("dapp_sched: device SM=%u  MPS=%s  SM capping=%s\n",
                pool.device_sm(), pool.mps_enabled() ? "on" : "off",
                pool.capping_active() ? "active" : "INACTIVE (decisions logged, not enforced)");
    if (!pool.capping_active()) {
        std::printf("            cuCtxCreate_v3 rejected SM affinity (err %d); start MPS to enforce\n",
                    pool.last_affinity_error());
    } else {
        std::printf("            classes:");
        for (const SmPool::Slot& s : pool.slots()) { std::printf(" %u", s.sm); }
        std::printf(" SMs\n");
    }
    std::printf("            policy: cuPHY reserve %u..%u SM, inference %u..%u SM, window %u slots\n",
                sh.cfg.cuphy_sm_idle, sh.cfg.cuphy_sm_peak,
                sh.cfg.infer_sm_min, sh.cfg.infer_sm_max, sh.cfg.window_slots);
    std::printf("            full load = %u cells x %u PRB x %u layers, %u kB TB per cell per slot\n",
                sh.cfg.num_cells, sh.cfg.max_prb, sh.cfg.max_layers, sh.cfg.max_tb_kbytes);

    if (!engine_path.empty()) {
#ifdef DAPP_SCHED_TENSORRT
        const auto t0 = std::chrono::steady_clock::now();
        if (!yolo_init(rig, pool, engine_path, err)) {
            std::fprintf(stderr, "YOLO init failed: %s\n", err.c_str());
            yolo_shutdown(rig, pool);
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("            yolo: %s  input %dx%d  %d classes  %d anchors  %zu engine instance(s), %.0f ms to load\n",
                    engine_path.c_str(), rig.net_w, rig.net_h, rig.nc, rig.na,
                    rig.per_slot.size() + (rig.fallback ? 1 : 0), ms);
#else
        std::fprintf(stderr, "--engine given but this build has no TensorRT support\n");
        return 1;
#endif
    } else {
        std::printf("            inference: placeholder kernel (no --engine)\n");
    }

#ifdef DAPP_SCHED_TENSORRT
    if (sweep > 0) {
        const int rc = run_sweep(rig, pool, default_image, sweep);
        yolo_shutdown(rig, pool);
        return rc;
    }
#endif

    // No SA_RESTART: accept() must return EINTR so the server loop sees g_stop.
    struct sigaction sa{};
    sa.sa_handler = on_sig;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    std::thread rt(ring_thread, &sh, ring_name, verbose);

    if (self_test > 0) {
        std::printf("[self] %d synthetic inference requests\n", self_test);
        std::this_thread::sleep_for(std::chrono::milliseconds(300)); // let the ring attach
        for (int i = 0; i < self_test && !g_stop; ++i) {
            handle_request(&sh, pool, rig, "", default_image, work_ms, verbose);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        g_stop = 1;
        rt.join();
        std::printf("[self] done: %" PRIu64 " slots seen, %" PRIu64 " dropped by L1\n",
                    sh.slots.load(), sh.dropped.load());
#ifdef DAPP_SCHED_TENSORRT
        yolo_shutdown(rig, pool);
#endif
        return 0;
    }

    ::unlink(sock_path.c_str());
    const int srv = ::socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);
    if (srv < 0 || ::bind(srv, (sockaddr*)&addr, sizeof(addr)) != 0 || ::listen(srv, 8) != 0) {
        std::fprintf(stderr, "cannot listen on %s: %s\n", sock_path.c_str(), std::strerror(errno));
        g_stop = 1;
        rt.join();
        return 1;
    }
    ::chmod(sock_path.c_str(), 0666);
    std::printf("[sock] listening on %s   (echo 'RUN %s' | nc -q1 -U %s)\n",
                sock_path.c_str(), rig.enabled ? "image.jpg" : "10", sock_path.c_str());

    while (!g_stop) {
        const int fd = ::accept(srv, nullptr, nullptr);
        if (fd < 0) { if (g_stop) break; continue; }
        char req[512] = {0};
        const ssize_t n = ::read(fd, req, sizeof(req) - 1);
        std::string arg;
        if (n > 0) {
            char verb[32] = {0};
            char rest[480] = {0};
            if (std::sscanf(req, "%31s %479s", verb, rest) >= 2) { arg = rest; }
        }
        const std::string reply = handle_request(&sh, pool, rig, arg, default_image, work_ms, verbose);
        (void)!::write(fd, reply.data(), reply.size());
        ::close(fd);
    }

    ::close(srv);
    ::unlink(sock_path.c_str());
    rt.join();
#ifdef DAPP_SCHED_TENSORRT
    yolo_shutdown(rig, pool);
#endif
    std::printf("dapp_sched: %" PRIu64 " requests served\n", sh.requests.load());
    return 0;
}
