// gc_union (2026-11-02): can the YOLO tenant get a green context that is the UNION of two SM blocks, leaving cuPHY's UL
// order slice (and optionally the PUSCH fallback profiles) empty?  cuDevResourceGenerateDesc() accepts several SM resources
// only when they come from the same cuDevSmResourceSplitByCount() call, so the device is split once into 8-SM groups
// (groups of the granularity cuphydriver uses) and a class context is the union of the groups that are disjoint from every
// cuPHY set that must stay free.  Every set is created with the same split sequence as cuphydriver (context.cpp) and its
// physical %smid set is measured (no MPS), so the overlap table below is ground truth, not arithmetic.
//   env GC_PUSCH="16,24,32,40,56" GC_PDSCH="24,40" GC_ORDER=8 GC_CLASSES="56,80" GC_GUARD=1
//   GC_GUARD = number of PUSCH fallback levels above the chosen profile kept free (0 = only the chosen profile)
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
static CUdevice dev; static CUdevResource all_res; static int nsm = 0; static unsigned* hist = nullptr;
static Part make_desc(const std::string& name, CUdevResourceDesc d, unsigned sm) {
    Part p; p.name = name; p.sm = sm; CK(cuGreenCtxCreate(&p.g, d, dev, CU_GREEN_CTX_DEFAULT_STREAM)); CK(cuCtxFromGreenCtx(&p.ctx, p.g));
    CK(cuCtxSetCurrent(p.ctx)); cudaStreamCreateWithFlags(&p.s, cudaStreamNonBlocking); return p; }
static Part make(const std::string& name, CUdevResource r) { CUdevResourceDesc d; CK(cuDevResourceGenerateDesc(&d, &r, 1)); return make_desc(name, d, r.sm.smCount); }
static void split(const CUdevResource& in, unsigned count, CUdevResource* res, CUdevResource* rem) { unsigned ng = 1; CK(cuDevSmResourceSplitByCount(res, &ng, &in, rem, 0, count)); }
static CUdevResource via_ctx(CUdevResource r) { CUdevResourceDesc d; CK(cuDevResourceGenerateDesc(&d, &r, 1)); CUgreenCtx g; CK(cuGreenCtxCreate(&g, d, dev, CU_GREEN_CTX_DEFAULT_STREAM)); CUdevResource out = {}; CK(cuGreenCtxGetDevResource(g, &out, CU_DEV_RESOURCE_TYPE_SM)); return out; }
static void probe(Part& p) {
    std::vector<unsigned> h(nsm);
    for (int rep = 0; rep < 3; rep++) { cudaMemsetAsync(hist, 0, nsm * 4, p.s); smid_kernel<<<4 * nsm, 64, 0, p.s>>>(hist); cudaMemcpyAsync(h.data(), hist, nsm * 4, cudaMemcpyDeviceToHost, p.s); cudaStreamSynchronize(p.s);
        for (int i = 0; i < nsm; i++) if (h[i]) p.ids.insert(i); }
}
static std::string rng(const std::set<int>& s) { if (s.empty()) return "{}"; std::string o; int a = *s.begin(), prev = a; for (int x : s) { if (x == a || x == prev + 1) { prev = x; continue; } o += std::to_string(a) + "-" + std::to_string(prev) + ","; a = prev = x; } o += std::to_string(a) + "-" + std::to_string(prev); return o; }
static int inter(const std::set<int>& a, const std::set<int>& b) { int n = 0; for (int x : a) if (b.count(x)) n++; return n; }
int main() {
    std::vector<int> pusch = {16, 24, 32, 40, 56}, pdsch = {24, 40}; int order = 8, guard = 1; std::vector<int> classes = {56, 80};
    auto envl = [](const char* n, std::vector<int>& v) { const char* e = getenv(n); if (!e) return; v.clear(); std::string t; for (const char* c = e;; c++) { if (*c == ',' || !*c) { if (!t.empty()) v.push_back(atoi(t.c_str())); t.clear(); if (!*c) break; } else t += *c; } };
    envl("GC_PUSCH", pusch); envl("GC_PDSCH", pdsch); envl("GC_CLASSES", classes); if (getenv("GC_ORDER")) order = atoi(getenv("GC_ORDER")); if (getenv("GC_GUARD")) guard = atoi(getenv("GC_GUARD"));
    CK(cuInit(0)); CK(cuDeviceGet(&dev, 0)); CK(cuDeviceGetAttribute(&nsm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    int mps = 0; cuDeviceGetAttribute(&mps, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, dev); printf("GPU %d SMs, MPS %d (must be 0), order %d, guard %d\n", nsm, mps, order, guard);
    CUcontext prim; CK(cuDevicePrimaryCtxRetain(&prim, dev)); CK(cuCtxSetCurrent(prim)); cudaMalloc(&hist, nsm * 4);
    CK(cuDeviceGetDevResource(dev, &all_res, CU_DEV_RESOURCE_TYPE_SM));
    // ---- cuPHY's sets, same splits as context.cpp
    std::vector<Part> ch; std::vector<int> yact;   // actual PUSCH sizes, ascending
    CUdevResource largest_pusch_result = {};
    for (int y : pusch) { int cnt = nsm - y; cnt -= cnt % 8; CUdevResource r = {}, rem = {}; split(all_res, cnt, &r, &rem); ch.push_back(make("PUSCH" + std::to_string(rem.sm.smCount), rem)); yact.push_back(rem.sm.smCount); if (y == pusch.back()) largest_pusch_result = r; }
    for (int n : pdsch) { CUdevResource r = {}, rem = {}; split(all_res, n, &r, &rem); ch.push_back(make("PDSCH" + std::to_string(n), r)); }
    { CUdevResource r = {}, rem = {}; split(all_res, nsm - 24, &r, &rem); CUdevResource top = via_ctx(rem); CUdevResource pu = {}, pr = {}; split(top, 8, &pu, &pr); ch.push_back(make("PUCCH", pu)); ch.push_back(make("PRACH", pr)); }
    int oact = 0;
    { CUdevResource comp = via_ctx(largest_pusch_result); if ((int)comp.sm.smCount <= order) { CUdevResource r = {}, rem = {}; split(all_res, order, &r, &rem); ch.push_back(make("ORDER", r)); oact = r.sm.smCount; }
      else { int cnt = (int)comp.sm.smCount - order; if (cnt > 8 && cnt % 8) cnt -= cnt % 8; CUdevResource r = {}, rem = {}; split(comp, cnt, &r, &rem); ch.push_back(make("ORDER", rem)); oact = rem.sm.smCount; } }
    for (Part& p : ch) { probe(p); printf("%-10s %3u SMs  smid %s\n", p.name.c_str(), p.sm, rng(p.ids).c_str()); }
    printf("actual ORDER %d SMs\n", oact);
    auto find = [&](const std::string& n) -> Part& { for (Part& p : ch) if (p.name == n) return p; printf("no %s\n", n.c_str()); exit(1); };
    // ---- the device in 8-SM groups (ONE split call: its outputs may be combined by cuDevResourceGenerateDesc)
    unsigned ng = nsm / 8; std::vector<CUdevResource> groups(ng); CUdevResource grem = {};
    CK(cuDevSmResourceSplitByCount(groups.data(), &ng, &all_res, &grem, 0, 8));
    printf("groups: %u of 8 SMs, remainder %u SMs\n", ng, grem.sm.smCount);
    std::vector<Part> gp;
    for (unsigned i = 0; i < ng; i++) { gp.push_back(make("G" + std::to_string(i), groups[i])); probe(gp.back()); }
    Part gr = make("GREM", grem); probe(gr);
    for (Part& g : gp) { std::string hit; for (Part& c : ch) { int k = inter(g.ids, c.ids); if (k) hit += " " + c.name + "(" + std::to_string(k) + ")"; } printf("  %-5s smid %-12s in:%s\n", g.name.c_str(), rng(g.ids).c_str(), hit.c_str()); }
    { std::string hit; for (Part& c : ch) { int k = inter(gr.ids, c.ids); if (k) hit += " " + c.name + "(" + std::to_string(k) + ")"; } printf("  %-5s smid %-12s in:%s\n", gr.name.c_str(), rng(gr.ids).c_str(), hit.c_str()); }
    // ---- class contexts per (chosen PUSCH profile i, PDSCH N): union of groups disjoint from the must-free set
    int bad = 0;
    for (size_t i = 0; i < yact.size(); i++) for (int n : pdsch) {
        std::set<int> busy = find("ORDER").ids; busy.insert(find("PUCCH").ids.begin(), find("PUCCH").ids.end()); busy.insert(find("PRACH").ids.begin(), find("PRACH").ids.end());
        const std::set<int>& pd = find("PDSCH" + std::to_string(n)).ids; busy.insert(pd.begin(), pd.end());
        for (size_t j = i; j < yact.size() && j <= i + guard; j++) { const std::set<int>& pu = find("PUSCH" + std::to_string(yact[j])).ids; busy.insert(pu.begin(), pu.end()); }
        std::vector<unsigned> free; for (unsigned g = 0; g < ng; g++) if (inter(gp[g].ids, busy) == 0) free.push_back(g);
        // preference: groups that also avoid EVERY larger PUSCH profile first (farthest from the fallback region), then the rest
        std::set<int> anyfb; for (size_t j = i + 1; j < yact.size(); j++) { const std::set<int>& pu = find("PUSCH" + std::to_string(yact[j])).ids; anyfb.insert(pu.begin(), pu.end()); }
        std::stable_sort(free.begin(), free.end(), [&](unsigned a, unsigned b) { return (inter(gp[a].ids, anyfb) == 0) > (inter(gp[b].ids, anyfb) == 0); });
        printf("PUSCH %d (+%d fallback) PDSCH %d: %zu free groups = %zu SMs:", yact[i], guard, n, free.size(), free.size() * 8); for (unsigned g : free) printf(" G%u", g); printf("\n");
        for (int c : classes) {
            if ((int)free.size() * 8 < c) { printf("   class %d: does not fit\n", c); continue; }
            std::vector<CUdevResource> sel; for (int k = 0; k < c / 8; k++) sel.push_back(groups[free[k]]);
            CUdevResourceDesc d; CUresult rc = cuDevResourceGenerateDesc(&d, sel.data(), (unsigned)sel.size());
            if (rc != CUDA_SUCCESS) { const char* s; cuGetErrorString(rc, &s); printf("   class %d: GenerateDesc(%zu groups) failed: %s\n", c, sel.size(), s); bad++; continue; }
            Part u = make_desc("U", d, c); probe(u);
            std::string hit; for (Part& x : ch) { int k = inter(u.ids, x.ids); if (k) hit += " " + x.name + "(" + std::to_string(k) + ")"; }
            const bool ok = (int)u.ids.size() == c && inter(u.ids, busy) == 0;
            printf("   class %d: ctx %u SMs, measured %zu smid %s | overlaps:%s  [%s]\n", c, u.sm, u.ids.size(), rng(u.ids).c_str(), hit.empty() ? " none" : hit.c_str(), ok ? "ok" : "FAIL"); if (!ok) bad++;
            // the context must run a plain kernel too (it did: probe); destroy
            cuCtxSetCurrent(prim); cudaStreamDestroy(u.s); cuGreenCtxDestroy(u.g);
        }
    }
    printf("%s (%d failures)\n", bad ? "UNION CONSTRUCTION FAILS" : "union construction verified", bad);
    return bad ? 1 : 0;
}
