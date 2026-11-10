// green_smcount.cu (2026-11-10): what SM count do cudart / the driver report with a green context current, and after switching back to the primary?
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorString(r_, &s); std::printf("%s -> %s\n", #x, s); return 1; } } while (0)
static void report(const char* tag) {
    int a = -1; cudaDeviceGetAttribute(&a, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    CUcontext c = nullptr; cuCtxGetCurrent(&c); CUdevResource r{}; CUresult rr = cuCtxGetDevResource(c, &r, CU_DEV_RESOURCE_TYPE_SM);
    std::printf("%-58s cudaDeviceGetAttribute(MPCount) %3d | cudaGetDeviceProperties %3d | cuCtxGetDevResource(current) %3u (%s)\n", tag, a, p.multiProcessorCount, rr == CUDA_SUCCESS ? r.sm.smCount : 0u, rr == CUDA_SUCCESS ? "ok" : "err");
}
int main(int argc, char** argv) {
    const bool prim_first = argc > 1;   // "1": query in the primary context before any green context exists (TRT-like first use)
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev, 0)); CUcontext prim; CK(cuDevicePrimaryCtxRetain(&prim, dev)); CK(cuCtxSetCurrent(prim));
    if (prim_first) report("A: primary, before any green context");
    CUdevResource all{}; CK(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM));
    CUdevResource part{}, rem{}; unsigned n = 1; CK(cuDevSmResourceSplitByCount(&part, &n, &all, &rem, 0, 72));
    CUdevResourceDesc desc; CK(cuDevResourceGenerateDesc(&desc, &part, 1));
    CUgreenCtx g; CK(cuGreenCtxCreate(&g, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM)); CUcontext gctx; CK(cuCtxFromGreenCtx(&gctx, g));
    CK(cuCtxSetCurrent(gctx)); cudaStream_t s = nullptr; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);   // the tenant's make_ctx_desc
    report("B: green ctx (72 SMs) current, after cudaStreamCreate in it");
    CK(cuCtxSetCurrent(prim)); report("C: back on the primary (cuCtxSetCurrent)");
    void* d = nullptr; cudaMalloc(&d, 16); report("D: primary, after a cudaMalloc");
    cudaStream_t s2 = nullptr; cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking); report("E: primary, after cudaStreamCreate");
    CK(cuCtxSetCurrent(gctx)); report("F: green again"); CK(cuCtxSetCurrent(prim)); report("G: primary again");
    return 0;
}
