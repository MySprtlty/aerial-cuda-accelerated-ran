// dapp_tenant_main.cpp (2026-11-02): the in-process tenant as a standalone program (no cuphycontroller, no L1) for tests of
// the xApp agent and the request path:  DAPP_TENANT=1 DAPP_AGENT=1 DAPP_TENANT_FREERUN=1 [DAPP_TENANT_CLASSES=56,80,104 ...] dapp_tenant_main [seconds]
// The cuphydriver profile functions the tenant links against are stubbed here (profile ladders from DAPP_PUSCH_PROFILES /
// DAPP_PDSCH_PROFILES, default 16,24,32,40,56 / 24,40 as SM counts); with FREERUN the slot plans stay unknown, so every
// request runs on the most conservative gap class (P = 132 - max PUSCH, N = max PDSCH).
#include "dapp_tenant.hpp"
#include "dapp_hook/dapp_sizer.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static std::vector<int> ladder(const char* env, const char* def) {
    std::vector<int> v; const char* s = std::getenv(env); std::stringstream ss(s ? s : def); std::string it;
    while (std::getline(ss, it, ',')) { if (!it.empty()) v.push_back(std::atoi(it.c_str())); }
    return v;
}
void cuphydriver_dapp_set_profile(uint16_t, uint16_t, uint8_t, uint8_t) {}
int cuphydriver_dapp_profile_sms(int channel, int* sm_counts, int max_count) {
    if (channel == 2) {   // UL order {SM count, mode} as cuphydriver would derive it from the yaml (DAPP_ORDER_SMS, default 8): common case = remainder of the complement
        std::vector<int> y = ladder("DAPP_PUSCH_PROFILES", "16,24,32,40,56"); const int yaml = 132 - ((132 - y.back()) / 8) * 8; const int C = 132 - yaml;
        const int want = std::getenv("DAPP_ORDER_SMS") ? std::atoi(std::getenv("DAPP_ORDER_SMS")) : 8;
        int n = 0; if (C <= want) { if (n < max_count) sm_counts[n++] = ((want + 7) / 8) * 8; if (n < max_count) sm_counts[n++] = 1; }
        else { int cnt = C - want; if (cnt > 8 && cnt % 8) cnt -= cnt % 8; if (n < max_count) sm_counts[n++] = C - cnt; if (n < max_count) sm_counts[n++] = 0; }
        return n;
    }
    if (channel == 3) { int n = 0; if (n < max_count) sm_counts[n++] = 8; if (n < max_count) sm_counts[n++] = 12; return n; }   // PUCCH 8 + PRACH 12 (lab yaml: pucch 2 -> 8, prach 2 + 6)
    // the cuphydriver rounds the PUSCH remainder to 8: Y -> 132 - rounddown8(132 - Y)
    std::vector<int> v = ladder(channel == 0 ? "DAPP_PUSCH_PROFILES" : "DAPP_PDSCH_PROFILES", channel == 0 ? "16,24,32,40,56" : "24,40");
    int n = 0; for (int y : v) { if (n >= max_count) break; sm_counts[n++] = channel == 0 ? 132 - ((132 - y) / 8) * 8 : y; }
    return n;
}
uint64_t cuphydriver_dapp_profile_count(int, int) { return 0; }
uint64_t cuphydriver_dapp_fallback_count(int) { return 0; }


// ---- victim thread (2026-11-02 08:40): stands in for a cuPHY host thread. Every ~400 us it issues a 4-byte cudaMemsetAsync +
// event record + event wait on (a) a non-blocking stream of the primary context and (b) a stream of an 8-SM green context (as a
// cuPHY channel would) and times the round trip; any round trip above DAPP_VICTIM_WARN_US (2000) is a stall caused by whatever
// the tenant / agent did at that moment (context lock, device sync, module load). DAPP_VICTIM=1 enables it, DAPP_VICTIM_CPU=47.
#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <pthread.h>
#include <sched.h>
static std::atomic<bool> g_victim_stop{false};
static void victim_main() {
    const int cpu = std::getenv("DAPP_VICTIM_CPU") ? std::atoi(std::getenv("DAPP_VICTIM_CPU")) : 47;
    const double warn_ms = (std::getenv("DAPP_VICTIM_WARN_US") ? std::atoi(std::getenv("DAPP_VICTIM_WARN_US")) : 2000) / 1000.0;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    void* d = nullptr; if (cudaMalloc(&d, 256) != cudaSuccess) { std::fprintf(stderr, "victim: cudaMalloc failed\n"); return; }
    const bool blocking = std::getenv("DAPP_VICTIM_BLOCKING") && std::atoi(std::getenv("DAPP_VICTIM_BLOCKING")) != 0;   // blocking streams see legacy-default-stream work of other threads
    cudaStream_t ps = nullptr; cudaStreamCreateWithFlags(&ps, blocking ? cudaStreamDefault : cudaStreamNonBlocking);
    cudaStream_t gs = nullptr;
    {   CUdevice dev = 0; CUdevResource all{}, res{}, rem{}; unsigned ng = 1; CUdevResourceDesc desc = nullptr; CUgreenCtx g = nullptr; CUstream s = nullptr;
        if (cuDeviceGet(&dev, 0) == CUDA_SUCCESS && cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM) == CUDA_SUCCESS &&
            cuDevSmResourceSplitByCount(&res, &ng, &all, &rem, 0, 8) == CUDA_SUCCESS && cuDevResourceGenerateDesc(&desc, &res, 1) == CUDA_SUCCESS &&
            cuGreenCtxCreate(&g, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM) == CUDA_SUCCESS && cuGreenCtxStreamCreate(&s, g, blocking ? CU_STREAM_DEFAULT : CU_STREAM_NON_BLOCKING, 0) == CUDA_SUCCESS) gs = (cudaStream_t)s;
        else std::fprintf(stderr, "victim: green context creation failed\n"); }
    cudaEvent_t ev[2]; cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming); cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming);
    double mx[2] = {0, 0}; long stalls[2] = {0, 0}; long iters = 0; cudaStream_t ss[2] = {ps, gs}; const char* nm[2] = {"primary", "green8"};
    std::fprintf(stderr, "victim: running on cpu %d (primary stream %p, green stream %p, %s), warn %.1f ms\n", cpu, (void*)ps, (void*)gs, blocking ? "blocking" : "non-blocking", warn_ms);
    while (!g_victim_stop.load()) {
        for (int i = 0; i < 2; i++) {
            if (!ss[i]) continue;
            const auto t0 = std::chrono::steady_clock::now();
            cudaMemsetAsync(d, 0, 4, ss[i]); cudaEventRecord(ev[i], ss[i]); cudaEventSynchronize(ev[i]);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ms > mx[i]) mx[i] = ms;
            if (ms > warn_ms) { stalls[i]++; if (stalls[i] > 40) continue; const auto now = std::chrono::system_clock::now().time_since_epoch(); const long long us = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
                std::fprintf(stderr, "victim: %s stall %.2f ms at %lld.%06lld\n", nm[i], ms, us / 1000000, us % 1000000); }
        }
        iters++; std::this_thread::sleep_for(std::chrono::microseconds(400));
    }
    std::fprintf(stderr, "victim: %ld iterations | primary max %.2f ms stalls %ld | green8 max %.2f ms stalls %ld\n", iters, mx[0], stalls[0], mx[1], stalls[1]);
}

int main(int argc, char** argv) {
    const int seconds = argc > 1 ? std::atoi(argv[1]) : 60;
    std::string err;
    if (!nv::dapp::tenant::start(err)) { std::fprintf(stderr, "tenant not started: %s\n", err.empty() ? "DAPP_TENANT=1 not set" : err.c_str()); return 1; }
    std::printf("tenant running for %d s (log %s)\n", seconds, std::getenv("DAPP_TENANT_LOG") ? std::getenv("DAPP_TENANT_LOG") : "/tmp/dapp_tenant.log"); std::fflush(stdout);
    std::thread victim; const bool want_victim = std::getenv("DAPP_VICTIM") && std::atoi(std::getenv("DAPP_VICTIM")) != 0;
    if (want_victim) victim = std::thread(victim_main);
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    if (want_victim) { g_victim_stop = true; victim.join(); }
    nv::dapp::tenant::stop();
    std::printf("stopped\n");
    return 0;
}
