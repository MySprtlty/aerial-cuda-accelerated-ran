// gc_nest: verifies the SM layout assumed by the per-slot PUSCH : PDSCH : YOLO partitioning (2026-11-01).
// cuphydriver (context.cpp, use_green_contexts 1) derives every channel partition from cuDeviceGetDevResource() with
// cuDevSmResourceSplitByCount(result, ..., input, remainder, 0, count): PUSCH = remainder of split(all, 132 - Y) ("upper" Y),
// PDSCH = result of split(all, N) ("lower" N), DL ctrl = result of split(all, 24), comms = result of split(all, 16), order =
// remainder of a re-split of PUSCH's complement. The dynamic design keeps several PUSCH profiles Y_j and PDSCH profiles N_i
// alive at once and gives the YOLO tenant the gap between them. This program creates all of them and records the physical
// %smid set each one runs on (no MPS: smid is physical), then checks nesting (PUSCH profiles are suffixes, PDSCH profiles
// prefixes) and tests three constructions of the gap partition for disjointness from the channel partitions:
//   V1: first c of the remainder of split(all, N)                                   "YOLO<c>@<N>"
//   V2: remainder of split(prefix(P), N), prefix(P) = result of the PUSCH split       "G2_P<P>_N<N>"  (|G| = P - N)
//       (its result "A2_P<P>_N<N>" is compared with the global prefix N = PDSCH_N)
//   V4: first (|R1| - Y') of R1 = remainder of split(all, N)                         "G4_N<N>_Y<Y>"
// Run inside the container WITHOUT CUDA_MPS_PIPE_DIRECTORY.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <iterator>
#define CK(x) do { CUresult rc_ = (x); if (rc_ != CUDA_SUCCESS) { const char* s; cuGetErrorString(rc_, &s); printf("%s failed: %s (%d)\n", #x, s, (int)rc_); exit(1); } } while (0)
__global__ void smid_kernel(unsigned* hist) {
    unsigned smid; asm("mov.u32 %0, %%smid;" : "=r"(smid));
    if (threadIdx.x == 0) atomicAdd(&hist[smid], 1u);
    long long t0 = clock64(); while (clock64() - t0 < 200000) {}
}
struct Part { std::string name; CUgreenCtx g = nullptr; CUcontext ctx = nullptr; cudaStream_t s = nullptr; unsigned sm = 0; std::set<int> ids; };
static CUdevice dev; static CUdevResource all_res;
static Part make(const std::string& name, CUdevResource r) {
    Part p; p.name = name; p.sm = r.sm.smCount; CUdevResourceDesc d; CK(cuDevResourceGenerateDesc(&d, &r, 1));
    CK(cuGreenCtxCreate(&p.g, d, dev, CU_GREEN_CTX_DEFAULT_STREAM)); CK(cuCtxFromGreenCtx(&p.ctx, p.g));
    CK(cuCtxSetCurrent(p.ctx)); cudaStreamCreateWithFlags(&p.s, cudaStreamNonBlocking); return p; }
static void split(const CUdevResource& in, unsigned count, CUdevResource* res, CUdevResource* rem) { unsigned ng = 1; CK(cuDevSmResourceSplitByCount(res, &ng, &in, rem, 0, count)); }
// remainder resources cannot be re-split directly (error 915): go through a temporary green context (as context.cpp does)
static CUdevResource via_ctx(CUdevResource r) { CUdevResourceDesc d; CK(cuDevResourceGenerateDesc(&d, &r, 1)); CUgreenCtx g; CK(cuGreenCtxCreate(&g, d, dev, CU_GREEN_CTX_DEFAULT_STREAM)); CUdevResource out = {}; CK(cuGreenCtxGetDevResource(g, &out, CU_DEV_RESOURCE_TYPE_SM)); return out; }
static void probe(Part& p, unsigned* hist_d, int nsm) {
    std::vector<unsigned> h(nsm);
    for (int rep = 0; rep < 3; rep++) { cudaMemsetAsync(hist_d, 0, nsm * 4, p.s); smid_kernel<<<4 * nsm, 64, 0, p.s>>>(hist_d); cudaMemcpyAsync(h.data(), hist_d, nsm * 4, cudaMemcpyDeviceToHost, p.s); cudaStreamSynchronize(p.s);
        for (int i = 0; i < nsm; i++) if (h[i]) p.ids.insert(i); }
}
static std::string rng(const std::set<int>& s) { if (s.empty()) return "{}"; std::string o; int a = *s.begin(), prev = a; for (int x : s) { if (x == a || x == prev + 1) { prev = x; continue; } o += std::to_string(a) + "-" + std::to_string(prev) + ","; a = prev = x; } o += std::to_string(a) + "-" + std::to_string(prev); return o; }
static bool subset(const std::set<int>& a, const std::set<int>& b) { for (int x : a) if (!b.count(x)) return false; return true; }
static bool disjoint(const std::set<int>& a, const std::set<int>& b) { for (int x : a) if (b.count(x)) return false; return true; }
int main(int argc, char** argv) {
    std::vector<int> pusch = {40, 56, 72, 96}, pdsch = {40, 56}; int dlctrl = 24, comms = 16, order = 8;
    // 2026-11-02: ladders from the environment (GC_PUSCH="16,24,32,40,56" GC_PDSCH="24,40" GC_ORDER=24 for the 1-cell configuration)
    auto envl = [](const char* n, std::vector<int>& v) { const char* e = getenv(n); if (!e) return; v.clear(); std::string t; for (const char* c = e;; c++) { if (*c == ',' || !*c) { if (!t.empty()) v.push_back(atoi(t.c_str())); t.clear(); if (!*c) break; } else t += *c; } };
    envl("GC_PUSCH", pusch); envl("GC_PDSCH", pdsch); if (getenv("GC_ORDER")) order = atoi(getenv("GC_ORDER"));
    std::vector<int> classes = {56, 80}; envl("GC_CLASSES", classes);
    CK(cuInit(0)); CK(cuDeviceGet(&dev, 0)); int nsm = 0; CK(cuDeviceGetAttribute(&nsm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    int mps = 0; cuDeviceGetAttribute(&mps, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, dev); printf("GPU %d SMs, MPS %d (must be 0 for physical smid)\n", nsm, mps);
    CUcontext prim; CK(cuDevicePrimaryCtxRetain(&prim, dev)); CK(cuCtxSetCurrent(prim)); unsigned* hist; cudaMalloc(&hist, nsm * 4);
    CK(cuDeviceGetDevResource(dev, &all_res, CU_DEV_RESOURCE_TYPE_SM));
    std::vector<Part> parts;
    // PUSCH profiles: remainder of split(all, 132 - Y) (first-split rounding as in context.cpp: count rounded down to the 8 granularity)
    CUdevResource largest_pusch_result = {}; std::map<int, int> Yp;   // requested Y -> actual Y'
    for (int y : pusch) { int cnt = nsm - y; cnt -= cnt % 8; CUdevResource r = {}, rem = {}; split(all_res, cnt, &r, &rem); parts.push_back(make("PUSCH" + std::to_string(y), rem)); Yp[y] = rem.sm.smCount; if (y == pusch.back()) largest_pusch_result = r; }
    for (int n : pdsch) { CUdevResource r = {}, rem = {}; split(all_res, n, &r, &rem); parts.push_back(make("PDSCH" + std::to_string(n), r)); }
    { CUdevResource r = {}, rem = {}; split(all_res, dlctrl, &r, &rem); parts.push_back(make("DLCTRL", r)); }
    { CUdevResource r = {}, rem = {}; split(all_res, comms, &r, &rem); parts.push_back(make("COMMS", r)); }
    { CUdevResource r = {}, rem = {}; split(all_res, nsm - 24, &r, &rem); CUdevResource top = via_ctx(rem); CUdevResource pu = {}, pr = {}; split(top, 8, &pu, &pr); parts.push_back(make("PUCCH", pu)); parts.push_back(make("PRACH", pr)); }
    { CUdevResource comp = via_ctx(largest_pusch_result); int cnt = (int)comp.sm.smCount - order; cnt -= cnt % 8; CUdevResource r = {}, rem = {}; split(comp, cnt, &r, &rem); parts.push_back(make("ORDER", rem)); }
    // V1
    struct T { int c, a; }; std::vector<T> ten = {{40, 40}, {88, 40}, {64, 56}};
    for (T t : ten) { CUdevResource r = {}, rem = {}; split(all_res, t.a, &r, &rem); CUdevResource mid = via_ctx(rem); CUdevResource c = {}, crem = {}; split(mid, t.c, &c, &crem); parts.push_back(make("YOLO" + std::to_string(t.c) + "@" + std::to_string(t.a), c)); }
    // V2
    std::map<int, int> P2Y; for (int y : pusch) { int cnt = nsm - y; cnt -= cnt % 8; P2Y[cnt] = y; }
    for (auto& kv : P2Y) { int P = kv.first; for (int n : pdsch) { if (P - n < 8) continue; CUdevResource r = {}, rem = {}; split(all_res, P, &r, &rem); CUdevResource pre = via_ctx(r); CUdevResource a = {}, g = {}; split(pre, n, &a, &g);
        parts.push_back(make("A2_P" + std::to_string(P) + "_N" + std::to_string(n), a)); parts.push_back(make("G2_P" + std::to_string(P) + "_N" + std::to_string(n), g)); } }
    // C2: the tenant's class contexts = first c of the V2 gap (split(via_ctx(G2), c).result)
    for (auto& kv : P2Y) { int P = kv.first; for (int n : pdsch) { if (P - n < 8) continue; CUdevResource r = {}, rem = {}; split(all_res, P, &r, &rem); CUdevResource pre = via_ctx(r); CUdevResource a = {}, g = {}; split(pre, n, &a, &g);
        CUdevResource gap = via_ctx(g); for (int c : classes) { if ((int)gap.sm.smCount < c) continue; CUdevResource cc = {}, crem = {}; split(gap, c, &cc, &crem); parts.push_back(make("C2_P" + std::to_string(P) + "_N" + std::to_string(n) + "_" + std::to_string(c), cc)); } } }
    // V4
    for (int n : pdsch) for (int y : pusch) { int yy = Yp[y]; CUdevResource r = {}, rem = {}; split(all_res, n, &r, &rem); CUdevResource R1 = via_ctx(rem); int cnt = (int)R1.sm.smCount - yy; if (cnt < 8) continue; CUdevResource g = {}, tail = {}; split(R1, cnt, &g, &tail);
        parts.push_back(make("G4_N" + std::to_string(n) + "_Y" + std::to_string(y), g)); }
    for (Part& p : parts) { probe(p, hist, nsm); printf("%-14s got %3u SMs  smid %s (%zu)\n", p.name.c_str(), p.sm, rng(p.ids).c_str(), p.ids.size()); }
    // overlap matrix: every gap / class context against every channel context (2026-11-02)
    for (Part& t : parts) { if (t.name.rfind("G2_", 0) != 0 && t.name.rfind("C2_", 0) != 0) continue; std::string line = t.name + " overlaps:";
        for (Part& c : parts) { if (c.name.rfind("G2_", 0) == 0 || c.name.rfind("C2_", 0) == 0 || c.name.rfind("A2_", 0) == 0 || c.name.rfind("G4_", 0) == 0 || c.name.rfind("YOLO", 0) == 0) continue;
            std::vector<int> inter; std::set_intersection(t.ids.begin(), t.ids.end(), c.ids.begin(), c.ids.end(), std::back_inserter(inter)); if (!inter.empty()) line += " " + c.name + "(" + std::to_string(inter.size()) + ")"; }
        printf("%s\n", line.c_str()); }
    if (getenv("GC_MATRIX_ONLY")) return 0;
    auto find = [&](const std::string& n) -> Part& { for (Part& p : parts) if (p.name == n) return p; printf("no %s\n", n.c_str()); exit(1); };
    int bad = 0;
    auto chk = [&](bool ok, const std::string& what) { printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str()); if (!ok) bad++; };
    auto yname = [](int y) { return std::string("PUSCH") + std::to_string(y); };
    for (size_t i = 1; i < pusch.size(); i++) chk(subset(find(yname(pusch[i - 1])).ids, find(yname(pusch[i])).ids), yname(pusch[i - 1]) + " subset of " + yname(pusch[i]));
    for (size_t i = 1; i < pdsch.size(); i++) chk(subset(find("PDSCH" + std::to_string(pdsch[i - 1])).ids, find("PDSCH" + std::to_string(pdsch[i])).ids), "PDSCH" + std::to_string(pdsch[i - 1]) + " subset of PDSCH" + std::to_string(pdsch[i]));
    chk(subset(find("DLCTRL").ids, find("PDSCH40").ids), "DLCTRL inside PDSCH40"); chk(subset(find("COMMS").ids, find("PDSCH40").ids), "COMMS inside PDSCH40");
    chk(subset(find("ORDER").ids, find("PDSCH40").ids), "ORDER inside PDSCH40");
    for (int y : pusch) chk(disjoint(find("ORDER").ids, find(yname(y)).ids), "ORDER disjoint from " + yname(y));
    chk(subset(find("PUCCH").ids, find("PUSCH40").ids) && subset(find("PRACH").ids, find("PUSCH40").ids), "PUCCH+PRACH inside PUSCH40");
    for (Part& p : parts) {
        if (p.name.rfind("YOLO", 0) == 0) { int c, a; sscanf(p.name.c_str(), "YOLO%d@%d", &c, &a); chk(disjoint(p.ids, find("PDSCH" + std::to_string(a)).ids), p.name + " disjoint from PDSCH" + std::to_string(a)); for (int y : pusch) if (a + c <= nsm - Yp[y]) chk(disjoint(p.ids, find(yname(y)).ids), p.name + " disjoint from " + yname(y)); }
        if (p.name.rfind("A2_", 0) == 0) { int P, n; sscanf(p.name.c_str(), "A2_P%d_N%d", &P, &n); chk(p.ids == find("PDSCH" + std::to_string(n)).ids, p.name + " == PDSCH" + std::to_string(n)); }
        if (p.name.rfind("G2_", 0) == 0) { int P, n; sscanf(p.name.c_str(), "G2_P%d_N%d", &P, &n); chk(disjoint(p.ids, find("PDSCH" + std::to_string(n)).ids), p.name + " disjoint from PDSCH" + std::to_string(n)); chk(disjoint(p.ids, find(yname(P2Y[P])).ids), p.name + " disjoint from " + yname(P2Y[P])); chk(disjoint(p.ids, find("ORDER").ids), p.name + " disjoint from ORDER"); }
        if (p.name.rfind("G4_", 0) == 0) { int n, y; sscanf(p.name.c_str(), "G4_N%d_Y%d", &n, &y); chk(disjoint(p.ids, find("PDSCH" + std::to_string(n)).ids), p.name + " disjoint from PDSCH" + std::to_string(n)); chk(disjoint(p.ids, find(yname(y)).ids), p.name + " disjoint from " + yname(y)); }
    }
    printf("%s (%d failures)\n", bad ? "SOME CONSTRUCTIONS VIOLATE THE LAYOUT" : "layout verified", bad);
    return bad ? 1 : 0;
}
