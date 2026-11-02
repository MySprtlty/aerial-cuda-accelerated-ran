// dapp_agent_test.cpp (2026-11-02): host-only exerciser of the xApp protocol: runs the agent server with a fake tenant
// that "infers" by sleeping (--ms per request), publishes synthetic estimates every 500 us and reports. No CUDA.
//   dapp_agent_test [--port 9810] [--ms 20] [--seconds 60]
#include "dapp_agent.hpp"
#include "dapp_json.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <thread>
#include <time.h>

using namespace nv::dapp;
static int64_t now_ns() { timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec; }

int main(int argc, char** argv) {
    int port = 9810, ms = 20, seconds = 60;
    for (int i = 1; i < argc; i++) { if (!std::strcmp(argv[i], "--port") && i + 1 < argc) port = std::atoi(argv[++i]); else if (!std::strcmp(argv[i], "--ms") && i + 1 < argc) ms = std::atoi(argv[++i]); else if (!std::strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = std::atoi(argv[++i]); }
    agent::Server srv; agent::Config cfg; cfg.port = port; cfg.models_json = "[\"26n\",\"26s\",\"26m\",\"26l\",\"26x\"]"; cfg.cells = 1; std::string err;
    std::deque<double> done_sm;
    srv.set_report_fn([&](int window_ms) { json::Obj o; o.kv("type", "sm_report").kv("t_ns", static_cast<long long>(now_ns())).kv("window_ms", window_ms).kv("requests", static_cast<int>(done_sm.size())).kv("sm_mean", done_sm.empty() ? 0.0 : 80.0).kv("yolo_sm_time_mean", 92.0).kv("ran_sm_time_mean", 40.0); done_sm.clear(); return o.str(); });
    if (!srv.start(cfg, err)) { std::fprintf(stderr, "start failed: %s\n", err.c_str()); return 1; }
    std::printf("agent test on port %d, fake inference %d ms\n", port, ms);
    const auto t_end = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    uint64_t slot = 0; uint64_t n = 0;
    while (std::chrono::steady_clock::now() < t_end) {
        // estimate every 500 us
        json::Obj e; e.kv("type", "estimate").kv("t_ns", static_cast<long long>(now_ns())).kv("sfn", static_cast<int>((slot / 20) % 1024)).kv("slot", static_cast<int>(slot % 20)).kv("ul_sm", 20).kv("dl_sm", 24).kv("yolo_sm_now", 92).kv("yolo_sm_min", 80).kv("queue", static_cast<int>(srv.queue_depth())).kv("inflight", 0)
         .raw("est_ms", "{\"26n\":6.4,\"26s\":9.9,\"26m\":15.8,\"26l\":21.9,\"26x\":34.7}").raw("est_sm", "{\"26n\":104,\"26s\":104,\"26m\":80,\"26l\":80,\"26x\":80}");
        srv.publish_estimate(e.str()); slot++;
        agent::Request r;
        if (srv.pop_request(r)) {
            agent::Result res; res.id = r.id; res.client = r.client; res.model = r.model; res.t_recv_ns = r.t_recv_ns; res.t_start_ns = now_ns();
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            res.t_end_ns = now_ns(); res.status = r.jpeg.size() > 100 ? "success" : "fail"; if (res.status == "fail") res.reason = "decode_error";
            res.sm = 80; res.sm_mean = 80; res.decode_ms = 5.0; res.infer_ms = ms - 5.0; res.wait_ms = (res.t_start_ns - r.t_recv_ns) / 1e6; res.latency_ms = (res.t_end_ns - r.t_recv_ns) / 1e6; res.detections = 42; res.tiles = 8;
            if (r.want_dets) res.dets_json = "[{\"cls\":0,\"name\":\"person\",\"conf\":0.9,\"xyxy\":[1,2,3,4]}]";
            srv.push_result(res); done_sm.push_back(80); n++;
        } else { std::this_thread::sleep_for(std::chrono::microseconds(500)); }
    }
    std::printf("served %llu requests, received %llu, rejected %llu\n", static_cast<unsigned long long>(n), static_cast<unsigned long long>(srv.requests_received()), static_cast<unsigned long long>(srv.rejected()));
    srv.stop(); return 0;
}
