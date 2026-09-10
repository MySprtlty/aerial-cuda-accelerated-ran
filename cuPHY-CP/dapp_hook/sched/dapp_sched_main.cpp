/*
 * dapp_sched: prototype scheduler component.
 *
 * Watches the FAPI hook ring, keeps a windowed estimate of how loaded cuPHY
 * is, and when an inference request arrives decides how many SMs that
 * inference may use. Runs the inference on a context capped to that budget.
 *
 * No model. The policy is the linear rule in dapp_sched.hpp; swapping in a
 * trained predictor later means replacing one function.
 *
 * Request interface: a Unix domain socket. A client writes a request line and
 * reads back one reply line:
 *
 *     -> RUN <milliseconds_of_work>
 *     <- OK sm=48 pct=36.4 load=0.31 ms=12.4 reason=moderate uplink
 *
 * The inference itself is a placeholder GPU workload. Replace run_inference()
 * with the real TensorRT/YOLO call; everything around it stays.
 *
 *   dapp_sched [-n /aerial_dapp_ring] [-s /tmp/dapp_sched.sock] [--cells N]
 *              [--classes 16,32,48,64,96] [--self-test N] [--verbose]
 */
#include "dapp_sched.hpp"
#include "dapp_sm_pool.hpp"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace nv::dapp;
using namespace nv::dapp::sched;

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

// ---------------------------------------------------------------------------
// Placeholder for the real inference. Keeps the GPU busy for roughly the
// requested time so the SM cap has a visible effect.
// ---------------------------------------------------------------------------
static double run_inference(const SmPool& pool, const SmPool::Slot* slot, int work_ms)
{
    const auto t0 = std::chrono::steady_clock::now();
    if (!pool.bind(slot)) { return -1.0; }

    // TODO: replace with the TensorRT execution context for the YOLO engine.
    // Until then, a device-side spin gives the cap something to throttle.
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

// One inference request: decide, run, report.
static std::string handle_request(Shared* sh, const SmPool& pool, int work_ms)
{
    const bool   stale = sh->stale.load();
    const double load  = sh->peak.load();
    const Decision d   = decide(sh->cfg, load, stale);
    const SmPool::Slot* slot = pool.pick(d.infer_sm);
    const uint32_t granted = slot ? slot->sm : pool.device_sm();

    const double ms = run_inference(pool, slot, work_ms);
    sh->requests.fetch_add(1);

    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "OK sm=%u granted=%u pct=%.1f load=%.3f ms=%.1f capped=%d reason=%s\n",
                  d.infer_sm, granted, d.infer_pct, d.load, ms,
                  pool.capping_active() ? 1 : 0, d.reason);
    std::printf("[run ] budget=%u granted=%u load=%.3f%s -> %.1f ms  (%s)\n",
                d.infer_sm, granted, d.load, stale ? " STALE" : "", ms, d.reason);
    return std::string(buf);
}

int main(int argc, char** argv)
{
    Shared sh;
    std::string ring_name = DAPP_RING_DEFAULT_NAME;
    std::string sock_path = "/tmp/dapp_sched.sock";
    std::vector<uint32_t> classes = {16, 32, 48, 64, 96};
    int self_test = 0, work_ms = 10;
    bool verbose = false;

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
        else if (a == "--work-ms" && i + 1 < argc) work_ms = atoi(argv[++i]);
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
                "          [--self-test N] [--work-ms N] [--verbose]\n", argv[0]);
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

    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);
    std::thread rt(ring_thread, &sh, ring_name, verbose);

    if (self_test > 0) {
        std::printf("[self] %d synthetic inference requests\n", self_test);
        std::this_thread::sleep_for(std::chrono::milliseconds(300)); // let the ring attach
        for (int i = 0; i < self_test && !g_stop; ++i) {
            handle_request(&sh, pool, work_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        g_stop = 1;
        rt.join();
        std::printf("[self] done: %" PRIu64 " slots seen, %" PRIu64 " dropped by L1\n",
                    sh.slots.load(), sh.dropped.load());
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
    std::printf("[sock] listening on %s   (echo 'RUN 10' | nc -U %s)\n",
                sock_path.c_str(), sock_path.c_str());

    while (!g_stop) {
        const int fd = ::accept(srv, nullptr, nullptr);
        if (fd < 0) { if (g_stop) break; continue; }
        char req[128] = {0};
        const ssize_t n = ::read(fd, req, sizeof(req) - 1);
        int ms = work_ms;
        if (n > 0) {
            char verb[32] = {0};
            int v = 0;
            if (std::sscanf(req, "%31s %d", verb, &v) >= 1 && v > 0) { ms = v; }
        }
        const std::string reply = handle_request(&sh, pool, ms);
        (void)!::write(fd, reply.data(), reply.size());
        ::close(fd);
    }

    ::close(srv);
    ::unlink(sock_path.c_str());
    rt.join();
    std::printf("dapp_sched: %" PRIu64 " requests served\n", sh.requests.load());
    return 0;
}
