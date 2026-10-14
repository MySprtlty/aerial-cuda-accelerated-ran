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
 * built with TensorRT and started with --engine (dapp_yolo.hpp). Engines are
 * given per (model, SM class): --engine NAME:SM=file, repeatable. A TensorRT
 * engine only runs near the SM count it was built for, so every SM class
 * loads the file of the largest key <= its SM count (a warning when there is
 * no exact match), the uncapped fallback context the largest key; only the
 * pairs given are loaded. Switching budgets then costs nothing at request
 * time. Batch-N engines run the tiled 4K path (nvjpeg decode on the GPU).
 * Without --engine the old placeholder kernel runs.
 *
 * Request interface: a Unix domain socket. A client writes a request line and
 * reads back one reply line:
 *
 *     -> RUN [model=NAME] <image.jpg>    (YOLO; "RUN" alone uses --image, no model= = the first --engine model)
 *     <- OK sm=48 granted=48 pct=36.4 load=0.31 ms=1.9 pre_ms=4.1 dets=5 top=bus:0.87,person:0.85 capped=1 reason=moderate uplink model=26n net=1.5 dec=0.0
 *     -> RUN <milliseconds_of_work>     (placeholder kernel)
 *     <- OK sm=48 granted=48 pct=36.4 load=0.31 ms=12.4 capped=1 reason=moderate uplink
 *
 * With --ctrl the SM budget comes from the estimator's control block
 * (estimator/live.py -> /dev/shm/aerial_dapp_ctrl, dapp_ctrl_abi.h) instead
 * of the built-in heuristic: cap_pct picks the SM class, and a closed gate
 * (gate_slots == 0), a stale block (older than --ctrl-max-age-ms) or a
 * missing block holds the request for up to --gate-wait-ms, then answers
 *
 *     <- HOLD gate=0 cap=40 slot_id=1234 age_us=812 reason=gate closed
 *     <- OK sm=53 granted=48 cap=40 gate=40 slot_id=1234 age_us=812 ms=0.91 pre_ms=4.1 dets=5 top=... capped=1 source=ctrl model=26n net=0.8 dec=0.0
 *
 *   dapp_sched [-n /aerial_dapp_ring] [-s /tmp/dapp_sched.sock] [--cells N]
 *              [--classes 16,32,48,64,96] [--engine NAME:SM=yolo.engine ...] [--image img.jpg]
 *              [--decode auto|cpu|gpu] [--conf 0.25] [--iou 0.7] [--max-det 300]
 *              [--sweep N] [--sweep-skip 2] [--self-test N] [--verbose]
 *              [--ctrl [/aerial_dapp_ctrl]] [--ctrl-max-age-ms 20] [--gate-wait-ms 50]
 */
#include "dapp_sched.hpp"
#include "dapp_sm_pool.hpp"
#include "dapp_ctrl_reader.hpp"
#ifdef DAPP_SCHED_TENSORRT
#include "dapp_yolo.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
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
// YOLO: one engine per (model, SM class), plus one per model for the uncapped
// fallback context when MPS is absent.
// ---------------------------------------------------------------------------
struct InferResult {
    bool        ok = false;
    double      gpu_ms = -1.0;   // h2d + network + d2h on the engine stream
    double      dec_ms = 0.0;    // nvjpeg path: parse + decode + letterbox/tiles
    double      h2d_ms = 0.0, net_ms = 0.0, d2h_ms = 0.0;
    double      pre_ms = 0.0;    // CPU: decode + letterbox (CPU path) or file read (GPU path); 0 when cached
    double      total_ms = 0.0;  // request in, reply out
    int         dets = 0;
    std::string top;             // "bus:0.87,person:0.85,..."
    std::string model;
    std::string err;
};

// --engine NAME:SM=file
struct EngineSpec {
    std::string model;
    uint32_t    sm = 0;
    std::string path;
};

#ifdef DAPP_SCHED_TENSORRT
struct ModelRig {
    std::string name;
    std::map<uint32_t, std::string>          files;      // SM key -> engine file
    std::vector<std::unique_ptr<YoloEngine>> per_slot;   // index == pool.slots() index; null = no engine for that class
    std::unique_ptr<YoloEngine>              fallback;   // uncapped context (no MPS)
    int net_w = 0, net_h = 0, nc = 0, na = 0, batch = 1;
};
#endif

struct YoloRig {
    bool  enabled = false;
    float conf = 0.25f, iou = 0.7f;
    int   max_det = 300;
    std::string default_model;                            // the first --engine model
#ifdef DAPP_SCHED_TENSORRT
    DecodeMode decode = DecodeMode::Auto;
    std::map<std::string, ModelRig>                      models;
    std::map<std::string, std::vector<unsigned char>>    bytes_cache;   // path -> JPEG bytes (nvjpeg path)
    std::map<std::string, YoloInput>                     cpu_cache;     // path@WxH -> letterboxed input (CPU path)
    std::vector<Detection>                               last;          // detections of the last run
#endif
};

static std::string base_name(const std::string& p)
{
    const size_t k = p.find_last_of('/');
    return k == std::string::npos ? p : p.substr(k + 1);
}

#ifdef DAPP_SCHED_TENSORRT
static bool yolo_init(YoloRig& rig, const SmPool& pool, const std::vector<EngineSpec>& specs, std::string& err)
{
    for (const EngineSpec& e : specs) {
        ModelRig& m = rig.models[e.model];
        m.name = e.model;
        if (m.files.count(e.sm)) { err = "duplicate --engine for model " + e.model + " SM " + std::to_string(e.sm); return false; }
        m.files[e.sm] = e.path;
    }
    rig.default_model = specs.front().model;

    std::map<std::string, std::vector<char>> blobs;   // a file given for several classes is read once
    auto blob_of = [&](const std::string& path) -> const std::vector<char>* {
        auto it = blobs.find(path);
        if (it == blobs.end()) {
            std::vector<char> b;
            if (!YoloEngine::read_file(path, b, err)) { return nullptr; }
            it = blobs.emplace(path, std::move(b)).first;
        }
        return &it->second;
    };

    for (auto& kv : rig.models) {
        ModelRig& m = kv.second;
        m.per_slot.resize(pool.slots().size());

        auto load_into = [&](std::unique_ptr<YoloEngine>& eng, uint32_t slot_sm, uint32_t key, const char* ctx_name) -> bool {
            const std::string& path = m.files[key];
            const std::vector<char>* blob = blob_of(path);
            if (!blob) { return false; }
            eng = std::make_unique<YoloEngine>();
            if (!eng->load(*blob, rig.decode, err)) {
                err = "model " + m.name + " SM " + std::to_string(slot_sm) + " (" + path + "): " + err;
                return false;
            }
            std::printf("            yolo: model %-6s %s SM %3u <- %s (%s, batch %d, %dx%d%s)\n",
                        m.name.c_str(), ctx_name, slot_sm, base_name(path).c_str(),
                        key == slot_sm ? "exact" : ("built for " + std::to_string(key) + " SM").c_str(),
                        eng->batch(), eng->net_w(), eng->net_h(), eng->has_decoder() ? ", nvjpeg tiled" : ", cpu letterbox");
            if (key != slot_sm) {
                std::fprintf(stderr, "warning: model %s has no engine built for the %u SM class; using the SM %u engine %s\n",
                             m.name.c_str(), slot_sm, key, base_name(path).c_str());
            }
            return true;
        };

        for (size_t i = 0; i < pool.slots().size(); ++i) {
            const uint32_t sm = pool.slots()[i].sm;
            uint32_t key = 0;
            bool have = false;
            for (const auto& f : m.files) { if (f.first <= sm) { key = f.first; have = true; } }   // largest key <= sm
            if (!have) {
                std::fprintf(stderr, "warning: model %s has no engine for the %u SM class (smallest key %u); requests there fail\n",
                             m.name.c_str(), sm, m.files.begin()->first);
                continue;
            }
            if (!pool.bind(&pool.slots()[i])) { err = "cannot bind SM class context"; return false; }
            if (!load_into(m.per_slot[i], sm, key, "class")) { return false; }
        }
        if (!pool.capping_active()) {
            if (!pool.bind(nullptr)) { err = "cannot bind fallback context"; return false; }
            if (!load_into(m.fallback, pool.device_sm(), m.files.rbegin()->first, "uncapped")) { return false; }
        }
        const YoloEngine* any = m.fallback.get();
        for (const auto& e : m.per_slot) { if (!any && e) { any = e.get(); } }
        if (!any) { err = "model " + m.name + ": no engine loaded in any SM class"; return false; }
        m.net_w = any->net_w();
        m.net_h = any->net_h();
        m.nc    = any->num_classes();
        m.na    = any->num_anchors();
        m.batch = any->batch();
    }
    rig.enabled = true;
    return true;
}

// Engines must die while their context is still alive and current.
static void yolo_shutdown(YoloRig& rig, const SmPool& pool)
{
    for (auto& kv : rig.models) {
        ModelRig& m = kv.second;
        for (size_t i = 0; i < m.per_slot.size(); ++i) {
            if (m.per_slot[i]) { pool.bind(&pool.slots()[i]); m.per_slot[i].reset(); }
        }
        if (m.fallback) { pool.bind(nullptr); m.fallback.reset(); }
    }
}

static YoloEngine* engine_for(ModelRig& m, const SmPool& pool, const SmPool::Slot* slot)
{
    if (!slot) { return m.fallback.get(); }
    const size_t idx = static_cast<size_t>(slot - &pool.slots()[0]);
    return idx < m.per_slot.size() ? m.per_slot[idx].get() : nullptr;
}

static InferResult run_yolo(YoloRig& rig, const SmPool& pool, const SmPool::Slot* slot,
                            const std::string& model, const std::string& image)
{
    InferResult r;
    r.model = model;
    const auto t0 = std::chrono::steady_clock::now();

    auto mi = rig.models.find(model);
    if (mi == rig.models.end()) { r.err = "unknown model " + model; return r; }
    ModelRig& m = mi->second;
    YoloEngine* eng = engine_for(m, pool, slot);
    if (!eng) {
        r.err = "no engine for model " + model + " at SM " + std::to_string(slot ? slot->sm : pool.device_sm());
        return r;
    }

    InferTiming t;
    const YoloInput* in = nullptr;
    YoloInput gpu_in;
    if (eng->has_decoder()) {
        // nvjpeg path: JPEG bytes (cached per path) -> NVJPG decode + letterbox/tiles straight into the engine input
        auto it = rig.bytes_cache.find(image);
        if (it == rig.bytes_cache.end()) {
            std::ifstream f(image, std::ios::binary);
            if (!f) { r.err = "cannot open " + image; return r; }
            std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            if (bytes.empty()) { r.err = image + " is empty"; return r; }
            if (rig.bytes_cache.size() >= 16) { rig.bytes_cache.clear(); }
            it = rig.bytes_cache.emplace(image, std::move(bytes)).first;
            r.pre_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        if (!pool.bind(slot)) { r.err = "cannot bind the selected context"; return r; }
        std::string derr;
        if (!eng->decode(it->second.data(), it->second.size(), gpu_in, t.dec_ms, derr)) { r.err = derr; return r; }
        if (!eng->infer(nullptr, t)) { r.err = "inference failed"; return r; }
        in = &gpu_in;
    } else {
        // CPU path (batch 1): decode + letterbox, cached per path and input size, pinned H2D on the engine stream
        const std::string key = image + "@" + std::to_string(eng->net_w()) + "x" + std::to_string(eng->net_h());
        auto it = rig.cpu_cache.find(key);
        if (it == rig.cpu_cache.end()) {
            YoloInput cin;
            if (!yolo_preprocess(image, eng->net_w(), eng->net_h(), cin, r.err)) { return r; }
            if (rig.cpu_cache.size() >= 16) { rig.cpu_cache.clear(); }
            it = rig.cpu_cache.emplace(key, std::move(cin)).first;
            r.pre_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        if (!pool.bind(slot)) { r.err = "cannot bind the selected context"; return r; }
        if (!eng->infer(it->second.chw.data(), t)) { r.err = "inference failed"; return r; }
        in = &it->second;
    }
    rig.last = eng->detections(*in, rig.conf, rig.iou, rig.max_det);

    r.ok     = true;
    r.gpu_ms = t.gpu_ms();
    r.dec_ms = t.dec_ms;
    r.h2d_ms = t.h2d_ms;
    r.net_ms = t.net_ms;
    r.d2h_ms = t.d2h_ms;
    r.dets   = static_cast<int>(rig.last.size());
    for (size_t i = 0; i < rig.last.size() && i < 3; ++i) {
        char tb[48];
        std::snprintf(tb, sizeof(tb), "%s%s:%.2f", i ? "," : "", coco_name(rig.last[i].cls), rig.last[i].conf);
        r.top += tb;
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
    std::atomic<uint64_t> held{0};       // requests answered with HOLD (estimator mode)
};

// Estimator (control block) mode. Values are command-line options; the
// defaults match rules.yaml safety.stale_after_ms.
struct CtrlCfg {
    bool        enabled = false;
    std::string name = DAPP_CTRL_SHM_NAME;
    uint32_t    max_age_ms = 20;     // block older than this = estimator dead or ring stalled
    uint32_t    gate_wait_ms = 50;   // how long a request waits for the gate to open
};

// Reads the control block, waiting up to gate_wait_ms for a fresh, open gate.
// Returns true with the block copied into cb when the tenant may run now.
static bool ctrl_wait_for_gate(CtrlReader& rd, const CtrlCfg& cfg, dapp_ctrl_block& cb,
                               int64_t& age_us, const char*& why)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(cfg.gate_wait_ms);
    why = "ctrl block missing";
    age_us = -1;
    for (;;) {
        if (!rd.is_open()) {
            std::string err;
            rd.open(cfg.name, err);
        }
        if (rd.is_open() && rd.read(cb)) {
            age_us = (CtrlReader::now_ns() - static_cast<int64_t>(cb.ts_ns)) / 1000;
            if (cb.n_decisions == 0)                                   { why = "no decision yet"; }
            else if (age_us > static_cast<int64_t>(cfg.max_age_ms) * 1000) { why = "ctrl stale"; }
            else if (cb.cap_pct == 0)                                  { why = "cap 0"; }
            else if (cb.gate_slots == 0)                               { why = "gate closed"; }
            else                                                       { return true; }
        }
        if (std::chrono::steady_clock::now() >= deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

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
// model_req: model name from the request ("" = the default model).
static std::string handle_request(Shared* sh, const SmPool& pool, YoloRig& rig,
                                  const std::string& arg, const std::string& model_req,
                                  const std::string& default_image,
                                  int work_ms, bool verbose, CtrlReader* ctrl, const CtrlCfg& ccfg)
{
    const bool   stale = sh->stale.load();
    const double load  = sh->peak.load();
    const Decision d   = decide(sh->cfg, load, stale);
    sh->requests.fetch_add(1);
    char buf[512];

    // Budget: estimator control block (--ctrl) or the built-in heuristic.
    uint32_t budget_sm = d.infer_sm;
    double   budget_pct = d.infer_pct;
    dapp_ctrl_block cb{};
    int64_t  age_us = -1;
    if (ccfg.enabled) {
        const char* why = "";
        if (!ctrl_wait_for_gate(*ctrl, ccfg, cb, age_us, why)) {
            sh->held.fetch_add(1);
            std::snprintf(buf, sizeof(buf), "HOLD gate=%u cap=%u slot_id=%u age_us=%" PRId64 " reason=%s\n",
                          cb.gate_slots, cb.cap_pct, cb.slot_id, age_us, why);
            std::printf("[hold] gate=%u cap=%u slot_id=%u age_us=%" PRId64 " (%s)\n",
                        cb.gate_slots, cb.cap_pct, cb.slot_id, age_us, why);
            return std::string(buf);
        }
        budget_sm  = (cb.cap_pct * pool.device_sm() + 50) / 100;
        budget_pct = static_cast<double>(cb.cap_pct);
    }
    const SmPool::Slot* slot = pool.pick(budget_sm);
    const uint32_t granted = slot ? slot->sm : pool.device_sm();

#ifdef DAPP_SCHED_TENSORRT
    const bool want_yolo = rig.enabled && !is_number(arg);
    if (want_yolo) {
        const std::string image = arg.empty() ? default_image : arg;
        const std::string model = model_req.empty() ? rig.default_model : model_req;
        if (image.empty()) { return "ERR no image given and no --image default\n"; }
        const InferResult r = run_yolo(rig, pool, slot, model, image);
        if (!r.ok) {
            std::snprintf(buf, sizeof(buf), "ERR %s\n", r.err.c_str());
            return std::string(buf);
        }
        if (ccfg.enabled) {
            std::snprintf(buf, sizeof(buf),
                          "OK sm=%u granted=%u cap=%u gate=%u slot_id=%u age_us=%" PRId64
                          " ms=%.2f pre_ms=%.1f dets=%d top=%s capped=%d source=ctrl model=%s net=%.2f dec=%.2f\n",
                          budget_sm, granted, cb.cap_pct, cb.gate_slots, cb.slot_id, age_us, r.gpu_ms, r.pre_ms,
                          r.dets, r.top.empty() ? "-" : r.top.c_str(), pool.capping_active() ? 1 : 0,
                          r.model.c_str(), r.net_ms, r.dec_ms);
            std::printf("[run ] ctrl cap=%u%% gate=%u slot_id=%u age=%" PRId64 "us -> budget=%u granted=%u"
                        " -> %s %.2f ms gpu (net %.2f, dec %.2f), %d dets [%s]\n",
                        cb.cap_pct, cb.gate_slots, cb.slot_id, age_us, budget_sm, granted, r.model.c_str(),
                        r.gpu_ms, r.net_ms, r.dec_ms, r.dets, r.top.c_str());
        } else {
            std::snprintf(buf, sizeof(buf),
                          "OK sm=%u granted=%u pct=%.1f load=%.3f ms=%.2f pre_ms=%.1f dets=%d top=%s capped=%d reason=%s"
                          " model=%s net=%.2f dec=%.2f\n",
                          d.infer_sm, granted, d.infer_pct, d.load, r.gpu_ms, r.pre_ms, r.dets,
                          r.top.empty() ? "-" : r.top.c_str(), pool.capping_active() ? 1 : 0, d.reason,
                          r.model.c_str(), r.net_ms, r.dec_ms);
            std::printf("[run ] budget=%u granted=%u load=%.3f%s -> %s %.2f ms gpu (net %.2f, dec %.2f), %d dets [%s]  (%s)\n",
                        d.infer_sm, granted, d.load, stale ? " STALE" : "", r.model.c_str(), r.gpu_ms, r.net_ms, r.dec_ms,
                        r.dets, r.top.c_str(), d.reason);
        }
        if (verbose) { print_detections(rig); }
        return std::string(buf);
    }
#else
    (void)rig; (void)model_req; (void)default_image; (void)verbose;
#endif
    const int ms_req = is_number(arg) ? std::atoi(arg.c_str()) : work_ms;
    const double ms = run_placeholder(pool, slot, ms_req > 0 ? ms_req : work_ms);
    if (ccfg.enabled) {
        std::snprintf(buf, sizeof(buf),
                      "OK sm=%u granted=%u cap=%u gate=%u slot_id=%u age_us=%" PRId64 " ms=%.1f capped=%d source=ctrl\n",
                      budget_sm, granted, cb.cap_pct, cb.gate_slots, cb.slot_id, age_us, ms,
                      pool.capping_active() ? 1 : 0);
        std::printf("[run ] ctrl cap=%u%% gate=%u -> budget=%u granted=%u -> placeholder %.1f ms\n",
                    cb.cap_pct, cb.gate_slots, budget_sm, granted, ms);
        return std::string(buf);
    }
    std::snprintf(buf, sizeof(buf),
                  "OK sm=%u granted=%u pct=%.1f load=%.3f ms=%.1f capped=%d reason=%s\n",
                  d.infer_sm, granted, budget_pct, d.load, ms,
                  pool.capping_active() ? 1 : 0, d.reason);
    std::printf("[run ] budget=%u granted=%u load=%.3f%s -> placeholder %.1f ms  (%s)\n",
                d.infer_sm, granted, d.load, stale ? " STALE" : "", ms, d.reason);
    return std::string(buf);
}

// --sweep: run the default image N times on every (model, SM class) and
// report median/mean/p99 of the GPU time plus the per-stage means, so the
// effect of the cap on a real network can be seen directly. The first
// `skip` runs of each class are left out of the statistics.
#ifdef DAPP_SCHED_TENSORRT
static double pct_of(std::vector<double> v, double p)
{
    if (v.empty()) { return 0.0; }
    std::sort(v.begin(), v.end());
    const size_t k = std::min(v.size() - 1, static_cast<size_t>(std::max(0.0, std::ceil(p / 100.0 * v.size()) - 1)));
    return v[k];
}

static double mean_of(const std::vector<double>& v)
{
    if (v.empty()) { return 0.0; }
    double s = 0.0;
    for (double x : v) { s += x; }
    return s / static_cast<double>(v.size());
}

static int run_sweep(YoloRig& rig, const SmPool& pool, const std::string& image, int n, int skip)
{
    if (!rig.enabled) { std::fprintf(stderr, "--sweep needs --engine\n"); return 2; }
    if (image.empty()) { std::fprintf(stderr, "--sweep needs --image\n"); return 2; }
    if (skip >= n) { skip = 0; }
    std::printf("[sweep] %s, %d runs per class (first %d excluded from the statistics)\n", image.c_str(), n, skip);
    std::vector<const SmPool::Slot*> targets;
    for (const SmPool::Slot& s : pool.slots()) { targets.push_back(&s); }
    if (!pool.capping_active()) { targets.push_back(nullptr); }
    for (auto& kv : rig.models) {
        ModelRig& m = kv.second;
        std::printf("[sweep] model=%s batch=%d input=%dx%d\n", m.name.c_str(), m.batch, m.net_w, m.net_h);
        bool printed = false;
        for (const SmPool::Slot* slot : targets) {
            if (!engine_for(m, pool, slot)) { continue; }
            std::vector<double> ms, net, h2d, d2h, dec;
            int dets = 0;
            for (int i = 0; i < n; ++i) {
                const InferResult r = run_yolo(rig, pool, slot, m.name, image);
                if (!r.ok) { std::fprintf(stderr, "sweep: %s\n", r.err.c_str()); return 1; }
                dets = r.dets;
                if (i < skip) { continue; }
                ms.push_back(r.gpu_ms); net.push_back(r.net_ms); h2d.push_back(r.h2d_ms);
                d2h.push_back(r.d2h_ms); dec.push_back(r.dec_ms);
            }
            std::vector<double> sorted = ms;
            std::sort(sorted.begin(), sorted.end());
            const double med = sorted[sorted.size() / 2];
            char head[32];
            if (slot) { std::snprintf(head, sizeof(head), "SM=%3u", slot->sm); }
            else      { std::snprintf(head, sizeof(head), "uncapped"); }
            std::printf("[sweep] %s  median=%.2f ms  min=%.2f ms  max=%.2f ms  dets=%d  mean=%.2f  p99=%.2f"
                        "  net=%.2f/%.2f  h2d=%.2f  d2h=%.2f  dec=%.2f  n=%zu  model=%s\n",
                        head, med, sorted.front(), sorted.back(), dets, mean_of(ms), pct_of(ms, 99.0),
                        mean_of(net), pct_of(net, 99.0), mean_of(h2d), mean_of(d2h), mean_of(dec), ms.size(), m.name.c_str());
            if (!printed) { print_detections(rig); printed = true; }
        }
    }
    return 0;
}
#endif

// NAME:SM=file
static bool parse_engine_spec(const std::string& s, EngineSpec& e)
{
    const size_t colon = s.find(':');
    const size_t eq    = s.find('=');
    if (colon == std::string::npos || eq == std::string::npos || eq < colon + 2 || colon == 0 || eq + 1 >= s.size()) { return false; }
    e.model = s.substr(0, colon);
    const std::string sm = s.substr(colon + 1, eq - colon - 1);
    if (!is_number(sm)) { return false; }
    e.sm   = static_cast<uint32_t>(std::atoi(sm.c_str()));
    e.path = s.substr(eq + 1);
    return e.sm > 0;
}

int main(int argc, char** argv)
{
    Shared sh;
    std::string ring_name = DAPP_RING_DEFAULT_NAME;
    std::string sock_path = "/tmp/dapp_sched.sock";
    std::string default_image;
    std::vector<EngineSpec> engines;
    std::vector<uint32_t> classes = {16, 32, 48, 64, 96};
    int self_test = 0, work_ms = 10, sweep = 0, sweep_skip = 2;
    bool verbose = false;
    YoloRig rig;
    CtrlCfg ccfg;
    CtrlReader ctrl;

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
        else if (a == "--sweep-skip" && i + 1 < argc) sweep_skip = atoi(argv[++i]);
        else if (a == "--work-ms" && i + 1 < argc) work_ms = atoi(argv[++i]);
        else if (a == "--engine" && i + 1 < argc) {
            EngineSpec e;
            if (!parse_engine_spec(argv[++i], e)) {
                std::fprintf(stderr, "--engine wants NAME:SM=file (e.g. 26x:132=/path/yolo26x_1280_b8_fp16_sm132.engine), got '%s'\n", argv[i]);
                return 2;
            }
            engines.push_back(e);
        }
        else if (a == "--image" && i + 1 < argc) default_image = argv[++i];
        else if (a == "--conf" && i + 1 < argc) rig.conf = (float)atof(argv[++i]);
        else if (a == "--iou" && i + 1 < argc) rig.iou = (float)atof(argv[++i]);
        else if (a == "--max-det" && i + 1 < argc) rig.max_det = atoi(argv[++i]);
        else if (a == "--decode" && i + 1 < argc) {
            const std::string v = argv[++i];
#ifdef DAPP_SCHED_TENSORRT
            if (v == "auto") rig.decode = DecodeMode::Auto;
            else if (v == "cpu") rig.decode = DecodeMode::Cpu;
            else if (v == "gpu") rig.decode = DecodeMode::Gpu;
            else { std::fprintf(stderr, "--decode wants auto|cpu|gpu\n"); return 2; }
#endif
        }
        else if (a == "--verbose" || a == "-v") verbose = true;
        else if (a == "--ctrl") {
            ccfg.enabled = true;
            if (i + 1 < argc && argv[i + 1][0] == '/') { ccfg.name = argv[++i]; }
        }
        else if (a == "--ctrl-max-age-ms" && i + 1 < argc) ccfg.max_age_ms = (uint32_t)atoi(argv[++i]);
        else if (a == "--gate-wait-ms" && i + 1 < argc) ccfg.gate_wait_ms = (uint32_t)atoi(argv[++i]);
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
                "          [--engine NAME:SM=yolo.engine ...] [--image img.jpg] [--decode auto|cpu|gpu]\n"
                "          [--conf 0.25] [--iou 0.7] [--max-det 300]\n"
                "          [--sweep N] [--sweep-skip 2] [--self-test N] [--work-ms N] [--verbose]\n"
                "          [--ctrl [/aerial_dapp_ctrl]] [--ctrl-max-age-ms 20] [--gate-wait-ms 50]\n", argv[0]);
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
    if (ccfg.enabled) {
        std::string cerr_;
        const bool ok = ctrl.open(ccfg.name, cerr_);
        std::printf("            budget source: estimator control block %s (%s; max age %u ms, gate wait %u ms)\n",
                    CtrlReader::shm_path(ccfg.name).c_str(), ok ? "attached" : "not there yet, will retry per request",
                    ccfg.max_age_ms, ccfg.gate_wait_ms);
    }
    std::fflush(stdout);

    if (!engines.empty()) {
#ifdef DAPP_SCHED_TENSORRT
        const auto t0 = std::chrono::steady_clock::now();
        if (!yolo_init(rig, pool, engines, err)) {
            std::fprintf(stderr, "YOLO init failed: %s\n", err.c_str());
            yolo_shutdown(rig, pool);
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        size_t instances = 0;
        for (const auto& kv : rig.models) {
            for (const auto& e : kv.second.per_slot) { if (e) { ++instances; } }
            if (kv.second.fallback) { ++instances; }
        }
        std::printf("            yolo: %zu model(s), %zu engine instance(s), default model %s, %.0f ms to load\n",
                    rig.models.size(), instances, rig.default_model.c_str(), ms);
#else
        std::fprintf(stderr, "--engine given but this build has no TensorRT support\n");
        return 1;
#endif
    } else {
        std::printf("            inference: placeholder kernel (no --engine)\n");
    }
    std::fflush(stdout);

#ifdef DAPP_SCHED_TENSORRT
    if (sweep > 0) {
        const int rc = run_sweep(rig, pool, default_image, sweep, sweep_skip);
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
            handle_request(&sh, pool, rig, "", "", default_image, work_ms, verbose, &ctrl, ccfg);
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
                sock_path.c_str(), rig.enabled ? "[model=NAME] image.jpg" : "10", sock_path.c_str());
    std::fflush(stdout);

    while (!g_stop) {
        const int fd = ::accept(srv, nullptr, nullptr);
        if (fd < 0) { if (g_stop) break; continue; }
        char req[512] = {0};
        const ssize_t n = ::read(fd, req, sizeof(req) - 1);
        std::string arg, model;
        if (n > 0) {
            // "RUN [model=NAME] [image | ms]"
            std::istringstream is(req);
            std::string verb, tok;
            is >> verb;
            while (is >> tok) {
                if (tok.rfind("model=", 0) == 0) { model = tok.substr(6); }
                else if (arg.empty())             { arg = tok; }
            }
        }
        const std::string reply = handle_request(&sh, pool, rig, arg, model, default_image, work_ms, verbose, &ctrl, ccfg);
        (void)!::write(fd, reply.data(), reply.size());
        ::close(fd);
    }

    ::close(srv);
    ::unlink(sock_path.c_str());
    rt.join();
#ifdef DAPP_SCHED_TENSORRT
    yolo_shutdown(rig, pool);
#endif
    std::printf("dapp_sched: %" PRIu64 " requests served, %" PRIu64 " held by the estimator gate\n",
                sh.requests.load(), sh.held.load());
    return 0;
}
