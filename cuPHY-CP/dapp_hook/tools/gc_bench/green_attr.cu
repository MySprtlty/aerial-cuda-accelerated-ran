// green_attr.cu (2026-11-10): do CUDA function attributes / module state carry over from the primary context to a green context?
// A: set MaxDynamicSharedMemorySize (100 KB) with the primary context current, launch on a green-context stream with 100 KB dynamic smem.
// B: no attribute set at all, launch 100 KB on the green stream (expected: invalid argument).  C: attribute set while the green ctx is current.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorString(r_, &s); std::printf("%s -> %s\n", #x, s); return 1; } } while (0)
__global__ void k_a(int* out) { extern __shared__ char s[]; s[threadIdx.x] = threadIdx.x; __syncthreads(); if (threadIdx.x == 0) out[0] = s[1]; }
__global__ void k_b(int* out) { extern __shared__ char s[]; s[threadIdx.x] = threadIdx.x; __syncthreads(); if (threadIdx.x == 0) out[0] = s[1]; }
__global__ void k_c(int* out) { extern __shared__ char s[]; s[threadIdx.x] = threadIdx.x; __syncthreads(); if (threadIdx.x == 0) out[0] = s[1]; }
static const char* es(cudaError_t e) { return cudaGetErrorString(e); }
int main() {
    int* d = nullptr; cudaMalloc(&d, 4); cudaFree(0);   // primary context
    CUdevice dev; CK(cuDeviceGet(&dev, 0)); CUcontext prim; CK(cuCtxGetCurrent(&prim));
    CUdevResource all{}; CK(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM));
    CUdevResource part{}, rem{}; unsigned n = 1; CK(cuDevSmResourceSplitByCount(&part, &n, &all, &rem, 0, 16));
    CUdevResourceDesc desc; CK(cuDevResourceGenerateDesc(&desc, &part, 1));
    CUgreenCtx g; CK(cuGreenCtxCreate(&g, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM)); CUstream gs; CK(cuGreenCtxStreamCreate(&gs, g, CU_STREAM_NON_BLOCKING, 0));
    CUcontext gctx; CK(cuCtxFromGreenCtx(&gctx, g));
    std::printf("green ctx %p, primary %p, same handle: %s, green SMs %u\n", (void*)gctx, (void*)prim, gctx == prim ? "yes" : "no", part.sm.smCount);
    const int smem = 100 * 1024;
    // A
    cudaError_t e = cudaFuncSetAttribute(k_a, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); std::printf("A: set attr in primary: %s\n", es(e));
    CK(cuCtxSetCurrent(gctx)); k_a<<<1, 128, smem, (cudaStream_t)gs>>>(d); e = cudaGetLastError(); std::printf("A: launch on green stream (green ctx current): %s\n", es(e)); cudaStreamSynchronize((cudaStream_t)gs);
    CK(cuCtxSetCurrent(prim)); k_a<<<1, 128, smem, (cudaStream_t)gs>>>(d); e = cudaGetLastError(); std::printf("A: launch on green stream (primary current): %s\n", es(e)); cudaStreamSynchronize((cudaStream_t)gs);
    // B
    k_b<<<1, 128, smem, (cudaStream_t)gs>>>(d); e = cudaGetLastError(); std::printf("B: no attribute, 100 KB on green stream: %s\n", es(e)); cudaStreamSynchronize((cudaStream_t)gs);
    // C
    CK(cuCtxSetCurrent(gctx)); e = cudaFuncSetAttribute(k_c, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); std::printf("C: set attr with green ctx current: %s\n", es(e));
    k_c<<<1, 128, smem, (cudaStream_t)gs>>>(d); e = cudaGetLastError(); std::printf("C: launch on green stream: %s\n", es(e)); cudaStreamSynchronize((cudaStream_t)gs);
    CK(cuCtxSetCurrent(prim)); k_c<<<1, 128, smem, 0>>>(d); e = cudaGetLastError(); std::printf("C: launch on the primary default stream (attr set in green): %s\n", es(e)); cudaDeviceSynchronize();
    // D: cuFuncGetAttribute view from both contexts
    cudaFuncAttributes fa{}; cudaFuncGetAttributes(&fa, k_a); std::printf("D: k_a maxDynamicSharedSizeBytes seen from primary: %d\n", fa.maxDynamicSharedSizeBytes);
    CK(cuCtxSetCurrent(gctx)); cudaFuncGetAttributes(&fa, k_a); std::printf("D: k_a maxDynamicSharedSizeBytes seen from green: %d\n", fa.maxDynamicSharedSizeBytes);
    return 0;
}
