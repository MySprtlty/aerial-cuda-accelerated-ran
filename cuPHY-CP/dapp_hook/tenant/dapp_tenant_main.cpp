// dapp_tenant_main.cpp (2026-11-02): the in-process tenant as a standalone program (no cuphycontroller, no L1) for tests of
// the xApp agent and the request path:  DAPP_TENANT=1 DAPP_AGENT=1 DAPP_TENANT_FREERUN=1 [DAPP_TENANT_CLASSES=56,80,104 ...] dapp_tenant_main [seconds]
// The cuphydriver profile functions the tenant links against are stubbed here (profile ladders from DAPP_PUSCH_PROFILES /
// DAPP_PDSCH_PROFILES, default 16,24,32,40,56 / 24,40 as SM counts); with FREERUN the slot plans stay unknown, so every
// request runs on the most conservative gap class (P = 132 - max PUSCH, N = max PDSCH).
// 2026-11-12: DAPP_SYNTH_PLAN=<seq.csv>[:seconds][,...] (without FREERUN) replays recorded sizer slot plans in real time instead - see synth_main.
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
        const int gr = (std::getenv("DAPP_GC_SPLIT_FLAGS") && std::atoi(std::getenv("DAPP_GC_SPLIT_FLAGS")) == 1) ? 2 : 8;   // 2026-11-13: 2-SM steps as cuphydriver with DAPP_GC_SPLIT_FLAGS=1
        std::vector<int> y = ladder("DAPP_PUSCH_PROFILES", "16,24,32,40,56"); const int yaml = 132 - ((132 - y.back()) / gr) * gr; const int C = 132 - yaml;
        const int want = std::getenv("DAPP_ORDER_SMS") ? std::atoi(std::getenv("DAPP_ORDER_SMS")) : 8;
        int n = 0; if (C <= want) { if (n < max_count) sm_counts[n++] = ((want + gr - 1) / gr) * gr; if (n < max_count) sm_counts[n++] = 1; }
        else { int cnt = C - want; if (cnt > gr && cnt % gr) cnt -= cnt % gr; if (n < max_count) sm_counts[n++] = C - cnt; if (n < max_count) sm_counts[n++] = 0; }
        return n;
    }
    if (channel == 3) { const bool two = std::getenv("DAPP_GC_SPLIT_FLAGS") && std::atoi(std::getenv("DAPP_GC_SPLIT_FLAGS")) == 1; int n = 0; if (n < max_count) sm_counts[n++] = 8; if (n < max_count) sm_counts[n++] = two ? 8 : 12; return n; }   // PUCCH 8 + PRACH 12 (lab yaml: pucch 2 -> 8, prach 2 + 6)
    // the cuphydriver rounds the PUSCH remainder to 8: Y -> 132 - rounddown8(132 - Y)
    std::vector<int> v = ladder(channel == 0 ? "DAPP_PUSCH_PROFILES" : "DAPP_PDSCH_PROFILES", channel == 0 ? "16,24,32,40,56" : "24,40");
    const int gr2 = (std::getenv("DAPP_GC_SPLIT_FLAGS") && std::atoi(std::getenv("DAPP_GC_SPLIT_FLAGS")) == 1) ? 2 : 8;
    int n = 0; for (int y : v) { if (n >= max_count) break; sm_counts[n++] = channel == 0 ? 132 - ((132 - y) / gr2) * gr2 : y; }
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

// ---- synthetic slot plans (2026-11-12, feature/slot_level): DAPP_SYNTH_PLAN=<seq.csv>[:seconds][,<seq.csv>[:seconds]...] replays sizer slot
// plans (tools/slot_level_seq.py: one row per slot with ul_any, dl_any, ul_sm, dl_sm) into g_slot_plan in real time, each written
// DAPP_SYNTH_LEAD_US (1100) before its slot starts, as the in-process sizer of an L1 does on the L2 messages. The tenant then schedules
// against these plans exactly as it does inside cuphycontroller, with no L1 on the GPU ("tenant-only replay"). The parts are concatenated
// (<seconds> of each, whole frames; without it the whole file) and repeated. SM counts that are not in DAPP_PUSCH_PROFILES /
// DAPP_PDSCH_PROFILES go to the next larger profile. Do not set DAPP_TENANT_FREERUN with it. DAPP_SYNTH_CPU pins the thread.
#include <fstream>
#include <time.h>
struct SynthSlot { uint8_t ul_any, dl_any; int16_t ul_sm, dl_sm; };
static std::atomic<bool> g_synth_stop{false};
static std::vector<SynthSlot> synth_load(const std::string& spec) {
    std::vector<SynthSlot> out; std::stringstream parts(spec); std::string part;
    while (std::getline(parts, part, ',')) {
        if (part.empty()) continue;
        double sec = 0.0; std::string path = part; const size_t c = part.rfind(':'); if (c != std::string::npos) { path = part.substr(0, c); sec = std::atof(part.c_str() + c + 1); }
        std::ifstream f(path); std::string line; std::vector<SynthSlot> rows; std::vector<long> ids;
        std::getline(f, line);   // header i,sfn,slot,pos,ul_any,dl_any,ul_sm,dl_sm,...
        while (std::getline(f, line)) {
            long v[8] = {0, 0, 0, 0, 0, 0, 0, 0}; int n = 0; std::stringstream ls(line); std::string it; while (n < 8 && std::getline(ls, it, ',')) v[n++] = std::atol(it.c_str());
            if (n < 8) continue;
            ids.push_back(v[1] * 20 + v[2]); rows.push_back({static_cast<uint8_t>(v[4]), static_cast<uint8_t>(v[5]), static_cast<int16_t>(v[6]), static_cast<int16_t>(v[7])});
        }
        // longest run of consecutive slots, from a slot 0, whole frames (the rule of tools/slot_level_sim.py load_seq)
        size_t bs = 0, be = 0, s = 0;
        for (size_t i = 1; i <= ids.size(); i++) { if (i == ids.size() || ((ids[i] - ids[i - 1]) % 20480 + 20480) % 20480 != 1) { if (i - s > be - bs) { bs = s; be = i; } s = i; } }
        while (bs < be && ids[bs] % 20 != 0) bs++;
        const size_t n = ((be - bs) / 20) * 20;
        if (n == 0) { std::fprintf(stderr, "synth: no whole frame of consecutive slots in %s\n", path.c_str()); continue; }
        const size_t want = sec > 0.0 ? (static_cast<size_t>(sec * 2000.0) / 20) * 20 : n;
        for (size_t i = 0; i < want; i++) out.push_back(rows[bs + i % n]);
        std::fprintf(stderr, "synth: %s: %zu consecutive slots, %zu replayed\n", path.c_str(), n, want);
    }
    return out;
}
static void synth_main(std::vector<SynthSlot> seq) {
    const int cpu = std::getenv("DAPP_SYNTH_CPU") ? std::atoi(std::getenv("DAPP_SYNTH_CPU")) : -1;
    if (cpu >= 0) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs); }
    const int64_t lead_ns = (std::getenv("DAPP_SYNTH_LEAD_US") ? std::atoi(std::getenv("DAPP_SYNTH_LEAD_US")) : 1100) * 1000LL;
    std::vector<int> pu = ladder("DAPP_PUSCH_PROFILES", "16,24,32,40,56"), pd = ladder("DAPP_PDSCH_PROFILES", "24,40");
    for (int& y : pu) y = 132 - ((132 - y) / 8) * 8;
    long over[2] = {0, 0};
    auto prof = [&](const std::vector<int>& v, int sm, int ch) { for (size_t i = 0; i < v.size(); i++) { if (v[i] >= sm) return static_cast<int>(i); } over[ch]++; return static_cast<int>(v.size()) - 1; };
    auto now_ns = []() { timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec; };
    const int64_t n0 = 2000, T0 = now_ns() + 50000000LL; long late = 0; int64_t n = n0;
    for (; !g_synth_stop.load(); n++) {
        const int64_t t0 = T0 + (n - n0) * 500000LL, wake = t0 - lead_ns;
        timespec w; w.tv_sec = wake / 1000000000LL; w.tv_nsec = wake % 1000000000LL; clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &w, nullptr);
        const SynthSlot& r = seq[static_cast<size_t>(n - n0) % seq.size()];
        nv::dapp::SlotPlanEntry& e = nv::dapp::slot_plan_entry(n);
        e.key.store(0xFFFFFFFFu, std::memory_order_release); e.t0_ns = t0;
        e.ul_any = r.ul_any; e.dl_any = r.dl_any; e.ul_prof = -1; e.dl_prof = -1; e.ul_sm = 0; e.dl_sm = 0; e.pusch_t = 0; e.pdsch_t = 0; e.pusch_ok = 1; e.pdsch_ok = 1;
        if (r.ul_any && r.ul_sm > 0 && !pu.empty()) { const int p = prof(pu, r.ul_sm, 0); e.ul_prof = static_cast<int8_t>(p); e.ul_sm = static_cast<int16_t>(pu[p]); }
        if (r.dl_any && r.dl_sm > 0 && !pd.empty()) { const int p = prof(pd, r.dl_sm, 1); e.dl_prof = static_cast<int8_t>(p); e.dl_sm = static_cast<int16_t>(pd[p]); }
        e.key.store(nv::dapp::sizer_key(static_cast<uint16_t>((n / 20) & 1023), static_cast<uint16_t>(n % 20)), std::memory_order_release);
        nv::dapp::g_plan_ref_t0_ns.store(t0, std::memory_order_relaxed); nv::dapp::g_plan_ref_id.store(n, std::memory_order_release);
        const int64_t lead = t0 - now_ns(); if (lead > nv::dapp::g_plan_lead_ns.load(std::memory_order_relaxed)) nv::dapp::g_plan_lead_ns.store(lead, std::memory_order_relaxed);
        if (lead < lead_ns - 200000) late++;
    }
    std::fprintf(stderr, "synth: %lld slot plans written, %ld more than 200 us late, SM counts above the largest profile: PUSCH %ld PDSCH %ld\n", static_cast<long long>(n - n0), late, over[0], over[1]);
}

int main(int argc, char** argv) {
    const int seconds = argc > 1 ? std::atoi(argv[1]) : 60;
    std::string err;
    if (!nv::dapp::tenant::start(err)) { std::fprintf(stderr, "tenant not started: %s\n", err.empty() ? "DAPP_TENANT=1 not set" : err.c_str()); return 1; }
    std::printf("tenant running for %d s (log %s)\n", seconds, std::getenv("DAPP_TENANT_LOG") ? std::getenv("DAPP_TENANT_LOG") : "/tmp/dapp_tenant.log"); std::fflush(stdout);
    std::thread victim; const bool want_victim = std::getenv("DAPP_VICTIM") && std::atoi(std::getenv("DAPP_VICTIM")) != 0;
    if (want_victim) victim = std::thread(victim_main);
    std::thread synth; const char* sp = std::getenv("DAPP_SYNTH_PLAN");
    if (sp && *sp) { std::vector<SynthSlot> seq = synth_load(sp); if (seq.empty()) { std::fprintf(stderr, "synth: nothing to replay (DAPP_SYNTH_PLAN=%s)\n", sp); } else { std::printf("synthetic slot plans: %zu slots per cycle (%.1f s)\n", seq.size(), seq.size() / 2000.0); std::fflush(stdout); synth = std::thread(synth_main, std::move(seq)); } }
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    if (synth.joinable()) { g_synth_stop = true; synth.join(); }
    if (want_victim) { g_victim_stop = true; victim.join(); }
    nv::dapp::tenant::stop();
    std::printf("stopped\n");
    return 0;
}
