// hbm_flood: tunable HBM-bandwidth stressor with a small SM footprint, to measure cuPHY's sensitivity to memory contention
// separately from SM contention. Streams a working set of W MB (W > L2 -> HBM traffic; W < L2 -> L2-resident traffic) with
// B blocks (B <= resident blocks per SM x SMs: run it under an MPS cap or on a green context to pin the SM footprint) and paces
// the kernel launches to a target bandwidth (GB/s). Prints the achieved GB/s every second.
//   hbm_flood --ws-mb 1024 --blocks 64 --gbps 800 --seconds 60 [--read-only]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
__global__ void spin_kernel(long long cycles, int4* sink) {   // compute-only control: same blocks, no memory traffic
    long long t0 = clock64(); int acc = threadIdx.x;
    while (clock64() - t0 < cycles) { acc = acc * 1664525 + 1013904223; }
    if (acc == 0x7fffffff) sink[0] = make_int4(acc, 0, 0, 0);
}
__global__ void stream_kernel(const int4* __restrict__ src, int4* __restrict__ dst, size_t n, int read_only, int4* sink) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
    int4 acc = make_int4(0, 0, 0, 0);
    for (; i < n; i += stride) { int4 v = src[i]; if (read_only) { acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; } else dst[i] = v; }
    if (read_only && acc.x == 0x7fffffff) sink[0] = acc;   // keep the loads alive
}
int main(int argc, char** argv) {
    size_t ws_mb = 1024; int blocks = 64, threads = 256, read_only = 0; double gbps = 500, seconds = 60; double spin_us = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ws-mb")) ws_mb = atol(argv[++i]); else if (!strcmp(argv[i], "--blocks")) blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gbps")) gbps = atof(argv[++i]); else if (!strcmp(argv[i], "--seconds")) seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--read-only")) read_only = 1; else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--spin-us")) spin_us = atof(argv[++i]);   // compute-only: back-to-back spin kernels of this length, no memory traffic
    }
    size_t bytes = ws_mb << 20, n = bytes / sizeof(int4);
    int4 *src, *dst, *sink; cudaMalloc(&src, bytes); cudaMalloc(&dst, read_only ? sizeof(int4) : bytes); cudaMalloc(&sink, sizeof(int4));
    cudaMemset(src, 1, bytes);
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    double pass_bytes = spin_us > 0 ? 0.0 : (read_only ? (double)bytes : 2.0 * bytes);          // read (+ write) per launch
    double period_s = spin_us > 0 ? 0.0 : pass_bytes / (gbps * 1e9);     // launch period for the target bandwidth (spin: back-to-back)
    printf("hbm_flood: ws %zu MB, %d blocks x %d threads, target %.0f GB/s (%s), period %.3f ms, %.0f s\n", ws_mb, blocks, threads, gbps, read_only ? "read" : "copy", period_s * 1e3, seconds);
    auto t_start = std::chrono::steady_clock::now(); auto t_next = t_start; double done_bytes = 0, win_bytes = 0; auto t_win = t_start; long launches = 0;
    float last_ms = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() < seconds) {
        cudaEventRecord(e0, s);
        if (spin_us > 0) spin_kernel<<<blocks, threads, 0, s>>>((long long)(spin_us * 1980.0), sink);
        else stream_kernel<<<blocks, threads, 0, s>>>(src, read_only ? sink : dst, n, read_only, sink);
        cudaEventRecord(e1, s); cudaEventSynchronize(e1); cudaEventElapsedTime(&last_ms, e0, e1);
        done_bytes += pass_bytes; win_bytes += pass_bytes; launches++;
        t_next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(period_s));
        auto now = std::chrono::steady_clock::now();
        if (t_next > now) std::this_thread::sleep_until(t_next); else t_next = now;
        if (std::chrono::duration<double>(now - t_win).count() >= 1.0) {
            printf("  %6.1f s: %.0f GB/s achieved (last kernel %.3f ms, %ld launches)\n", std::chrono::duration<double>(now - t_start).count(), win_bytes / std::chrono::duration<double>(now - t_win).count() / 1e9, last_ms, launches);
            fflush(stdout); win_bytes = 0; t_win = now;
        }
    }
    printf("hbm_flood done: %.0f GB moved, mean %.0f GB/s\n", done_bytes / 1e9, done_bytes / 1e9 / seconds);
    return 0;
}
