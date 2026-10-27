// njd_bench: nvjpeg decode-only load generator / timer for the N3 interference test (night 3, 2026-10-18).
//   build (container):  nvcc -O2 -o njd_bench njd_bench.cu -ldl -lcudart
//   run:  CUDA_MPS_PIPE_DIRECTORY=/var CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=43 ./njd_bench <libnvjpeg_dec.so> <backend 0|1|2> <rate req/s> <seconds> <S> <batch> <tiled> img1.jpg [img2.jpg ...]
// Each request: one JPEG -> njd_decode_tiles(...) into a device tensor (batch x 3 x S x S floats), stream-synchronised;
// per request wall ms and the decoder's t_ms[0..2] (parse / decode / letterbox). Poisson arrivals at <rate>. Prints p50/p95/max
// per image and the overall images/s; exits non-zero when the backend is unavailable.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

typedef void* (*njd_create_t)(int);
typedef int (*njd_decode_tiles_t)(void*, const unsigned char*, size_t, float*, int, int, int, void*, float*, int*, int*);
typedef void (*njd_destroy_t)(void*);

static double pct(std::vector<double> v, double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(p / 100.0 * (v.size() - 1) + 0.5))]; }

int main(int argc, char** argv) {
    if (argc < 9) { fprintf(stderr, "usage: %s lib backend rate seconds S batch tiled img...\n", argv[0]); return 2; }
    const char* lib = argv[1]; int backend = atoi(argv[2]); double rate = atof(argv[3]); double secs = atof(argv[4]);
    int S = atoi(argv[5]), batch = atoi(argv[6]), tiled = atoi(argv[7]);
    std::vector<std::vector<unsigned char>> imgs;
    for (int i = 8; i < argc; i++) { std::ifstream f(argv[i], std::ios::binary); imgs.emplace_back((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()); }
    void* h = dlopen(lib, RTLD_NOW); if (!h) { fprintf(stderr, "dlopen %s: %s\n", lib, dlerror()); return 3; }
    auto create = (njd_create_t)dlsym(h, "njd_create"); auto dec = (njd_decode_tiles_t)dlsym(h, "njd_decode_tiles"); auto destroy = (njd_destroy_t)dlsym(h, "njd_destroy");
    if (!create || !dec) { fprintf(stderr, "symbols missing\n"); return 3; }
    void* ctx = create(backend); if (!ctx) { fprintf(stderr, "backend %d unavailable\n", backend); return 4; }
    float* d_tensor = nullptr; size_t bytes = (size_t)batch * 3 * S * S * sizeof(float);
    if (cudaMalloc(&d_tensor, bytes) != cudaSuccess) { fprintf(stderr, "cudaMalloc failed\n"); return 5; }
    cudaStream_t st; cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    std::vector<int> rois(4 * batch); int wh[2]; float t_ms[4];
    // warm-up
    for (int i = 0; i < 3; i++) { dec(ctx, imgs[i % imgs.size()].data(), imgs[i % imgs.size()].size(), d_tensor, S, batch, tiled, st, t_ms, wh, rois.data()); cudaStreamSynchronize(st); }
    std::mt19937_64 rng(7); std::exponential_distribution<double> gap(rate);
    std::vector<double> wall, parse, decode, lbox; int fails = 0;
    auto t0 = std::chrono::steady_clock::now(); double next = 0.0; size_t i = 0;
    while (true) {
        double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (now >= secs) break;
        if (now < next) { std::this_thread::sleep_for(std::chrono::microseconds((long)((next - now) * 1e6))); continue; }
        next += gap(rng);
        const auto& b = imgs[i++ % imgs.size()];
        auto a = std::chrono::steady_clock::now();
        int rc = dec(ctx, b.data(), b.size(), d_tensor, S, batch, tiled, st, t_ms, wh, rois.data());
        cudaStreamSynchronize(st);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        if (rc) { fails++; continue; }
        wall.push_back(ms); parse.push_back(t_ms[0]); decode.push_back(t_ms[1]); lbox.push_back(t_ms[2]);
    }
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("njd_bench backend %d S %d batch %d tiled %d rate %.1f: %zu decodes in %.1f s (%.2f/s), fails %d | wall ms p50 %.2f p95 %.2f max %.2f | parse p50 %.2f | decode p50 %.2f p95 %.2f | letterbox p50 %.2f | image %dx%d\n",
           backend, S, batch, tiled, rate, wall.size(), el, wall.size() / el, fails, pct(wall, 50), pct(wall, 95), pct(wall, 100), pct(parse, 50), pct(decode, 50), pct(decode, 95), pct(lbox, 50), wh[0], wh[1]);
    destroy(ctx); cudaFree(d_tensor); return 0;
}
