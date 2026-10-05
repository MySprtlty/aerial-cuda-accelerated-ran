// nvjpeg_probe: is the GH200's hardware JPEG decoder (NVJPG engines, no SMs) usable from where the AI
// tenant runs (full GPU / MPS client with a cap / MIG instance), and how long does one 4K baseline JPEG
// take to decode and letterbox into the detector's input tensor?
//
// Build (inside the Aerial container, CUDA 12.9 + nvjpeg 12.4):
//   nvcc -O2 -arch=sm_90 -o nvjpeg_probe nvjpeg_probe.cu -lnvjpeg
// Run:
//   nvjpeg_probe <file.jpg> [--backend hw|hybrid|gpu] [--iters 200] [--resize 1280] [--fmt rgbi|yuv] [--csv out.csv]
//
// hw     = NVJPEG_BACKEND_HARDWARE  (NVJPG engine; Huffman + IDCT off the SMs; baseline single-scan only)
// hybrid = NVJPEG_BACKEND_HYBRID    (Huffman on the CPU, IDCT/colour on the SMs)
// gpu    = NVJPEG_BACKEND_GPU_HYBRID (Huffman and IDCT on the SMs)
//
// Every iteration: parse -> host stage -> transfer -> device stage -> (optional) bilinear letterbox to
// --resize x --resize float32 CHW /255 on the SMs. Reported per stage and in total (wall clock around the
// synchronised pipeline). Prints the nvjpeg hardware-decoder engine count seen by this process, which is
// the number that matters inside a MIG instance.
#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d cuda %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(2); } } while (0)
#define NJ(x) do { nvjpegStatus_t s = (x); if (s != NVJPEG_STATUS_SUCCESS) { fprintf(stderr, "%s:%d nvjpeg status %d (%s)\n", __FILE__, __LINE__, (int)s, #x); exit(3); } } while (0)

static const char* status_name(nvjpegStatus_t s) {
    switch (s) {
        case NVJPEG_STATUS_SUCCESS: return "SUCCESS";
        case NVJPEG_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case NVJPEG_STATUS_INVALID_PARAMETER: return "INVALID_PARAMETER";
        case NVJPEG_STATUS_BAD_JPEG: return "BAD_JPEG";
        case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "JPEG_NOT_SUPPORTED";
        case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "ALLOCATOR_FAILURE";
        case NVJPEG_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case NVJPEG_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH (no hardware decoder on this device/instance)";
        case NVJPEG_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
        case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED: return "IMPLEMENTATION_NOT_SUPPORTED";
        default: return "?";
    }
}

// interleaved RGB (HxWx3 u8, pitch bytes) -> letterboxed CHW float32 /255, grey 114 padding
__global__ void letterbox_kernel(const unsigned char* src, int sw, int sh, int spitch, float* dst, int S, float scale, int ox, int oy, int nw, int nh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= S || y >= S) return;
    float r, g, b;
    int ix = x - ox, iy = y - oy;
    if (ix < 0 || iy < 0 || ix >= nw || iy >= nh) {
        r = g = b = 114.f;
    } else {
        float fx = (ix + 0.5f) * scale - 0.5f, fy = (iy + 0.5f) * scale - 0.5f;
        int x0 = max(0, (int)floorf(fx)), y0 = max(0, (int)floorf(fy));
        int x1 = min(sw - 1, x0 + 1), y1 = min(sh - 1, y0 + 1);
        float ax = fx - x0, ay = fy - y0;
        ax = fminf(fmaxf(ax, 0.f), 1.f); ay = fminf(fmaxf(ay, 0.f), 1.f);
        const unsigned char* p00 = src + y0 * spitch + x0 * 3;
        const unsigned char* p01 = src + y0 * spitch + x1 * 3;
        const unsigned char* p10 = src + y1 * spitch + x0 * 3;
        const unsigned char* p11 = src + y1 * spitch + x1 * 3;
        float w00 = (1 - ax) * (1 - ay), w01 = ax * (1 - ay), w10 = (1 - ax) * ay, w11 = ax * ay;
        r = p00[0] * w00 + p01[0] * w01 + p10[0] * w10 + p11[0] * w11;
        g = p00[1] * w00 + p01[1] * w01 + p10[1] * w10 + p11[1] * w11;
        b = p00[2] * w00 + p01[2] * w01 + p10[2] * w10 + p11[2] * w11;
    }
    int plane = S * S;
    dst[0 * plane + y * S + x] = r / 255.f;
    dst[1 * plane + y * S + x] = g / 255.f;
    dst[2 * plane + y * S + x] = b / 255.f;
}

static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t)std::min((double)v.size() - 1, std::max(0.0, p / 100.0 * v.size()));
    return v[i];
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.jpg [--backend hw|hybrid|gpu] [--iters N] [--resize S] [--fmt rgbi|yuv] [--csv out.csv] [--tag name]\n", argv[0]); return 1; }
    std::string file = argv[1], backend = "hw", fmt = "rgbi", csv, tag;
    int iters = 200, resize = 0;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--resize" && i + 1 < argc) resize = atoi(argv[++i]);
        else if (a == "--fmt" && i + 1 < argc) fmt = argv[++i];
        else if (a == "--csv" && i + 1 < argc) csv = argv[++i];
        else if (a == "--tag" && i + 1 < argc) tag = argv[++i];
    }
    std::ifstream f(file, std::ios::binary);
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (data.empty()) { fprintf(stderr, "cannot read %s\n", file.c_str()); return 1; }

    int dev = 0; CK(cudaSetDevice(dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    printf("device: %s, %d SMs, cc %d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    nvjpegBackend_t be = NVJPEG_BACKEND_HARDWARE;
    if (backend == "hybrid") be = NVJPEG_BACKEND_HYBRID;
    else if (backend == "gpu") be = NVJPEG_BACKEND_GPU_HYBRID;
    else if (backend != "hw") { fprintf(stderr, "bad backend\n"); return 1; }

    nvjpegHandle_t handle;
    nvjpegStatus_t st = nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, nullptr, nullptr, 0, &handle);
    if (st != NVJPEG_STATUS_SUCCESS) { printf("nvjpegCreateEx: %s\n", status_name(st)); return 3; }
    unsigned int n_eng = 0, n_cores = 0;
    st = nvjpegGetHardwareDecoderInfo(handle, &n_eng, &n_cores);
    printf("hardware decoder info: status=%s engines=%u cores_per_engine=%u\n", status_name(st), n_eng, n_cores);

    nvjpegJpegDecoder_t decoder;
    st = nvjpegDecoderCreate(handle, be, &decoder);
    printf("nvjpegDecoderCreate(%s): %s\n", backend.c_str(), status_name(st));
    if (st != NVJPEG_STATUS_SUCCESS) { if (!csv.empty()) { FILE* c = fopen(csv.c_str(), "a"); if (c) { fprintf(c, "%s,%s,%s,%s,unavailable,%s\n", tag.c_str(), file.c_str(), backend.c_str(), fmt.c_str(), status_name(st)); fclose(c); } } return 4; }
    nvjpegJpegState_t state; NJ(nvjpegDecoderStateCreate(handle, decoder, &state));
    nvjpegBufferPinned_t pinned; NJ(nvjpegBufferPinnedCreate(handle, nullptr, &pinned));
    nvjpegBufferDevice_t devbuf; NJ(nvjpegBufferDeviceCreate(handle, nullptr, &devbuf));
    NJ(nvjpegStateAttachPinnedBuffer(state, pinned));
    NJ(nvjpegStateAttachDeviceBuffer(state, devbuf));
    nvjpegJpegStream_t jstream; NJ(nvjpegJpegStreamCreate(handle, &jstream));
    nvjpegDecodeParams_t params; NJ(nvjpegDecodeParamsCreate(handle, &params));
    nvjpegOutputFormat_t ofmt = fmt == "yuv" ? NVJPEG_OUTPUT_YUV : NVJPEG_OUTPUT_RGBI;
    NJ(nvjpegDecodeParamsSetOutputFormat(params, ofmt));
    cudaStream_t cs; CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));

    // image geometry
    NJ(nvjpegJpegStreamParse(handle, data.data(), data.size(), 0, 0, jstream));
    unsigned int nc = 0, widths[4] = {0}, heights[4] = {0};
    NJ(nvjpegJpegStreamGetComponentsNum(jstream, &nc));
    for (unsigned i = 0; i < nc; i++) NJ(nvjpegJpegStreamGetComponentDimensions(jstream, i, &widths[i], &heights[i]));
    nvjpegChromaSubsampling_t css; NJ(nvjpegJpegStreamGetChromaSubsampling(jstream, &css));
    int enc = -1; { nvjpegJpegEncoding_t e; if (nvjpegJpegStreamGetJpegEncoding(jstream, &e) == NVJPEG_STATUS_SUCCESS) enc = (int)e; }
    int W = widths[0], H = heights[0];
    printf("jpeg: %s %zu bytes, %dx%d, %u comps, subsampling=%d (0=444,2=420), encoding=%d (SOF marker: 192=baseline,193=ext seq,194=progressive)\n", file.c_str(), data.size(), W, H, nc, (int)css, enc);
    int supported = -1;
    st = nvjpegDecoderJpegSupported(decoder, jstream, params, &supported);
    printf("nvjpegDecoderJpegSupported(%s): status=%s supported=%d\n", backend.c_str(), status_name(st), supported);
    if (st != NVJPEG_STATUS_SUCCESS || supported != 0) { if (!csv.empty()) { FILE* c = fopen(csv.c_str(), "a"); if (c) { fprintf(c, "%s,%s,%s,%s,unsupported,%s/%d\n", tag.c_str(), file.c_str(), backend.c_str(), fmt.c_str(), status_name(st), supported); fclose(c); } } return 5; }

    nvjpegImage_t out; memset(&out, 0, sizeof(out));
    if (ofmt == NVJPEG_OUTPUT_RGBI) { out.pitch[0] = W * 3; CK(cudaMalloc((void**)&out.channel[0], (size_t)out.pitch[0] * H)); }
    else { for (unsigned i = 0; i < nc; i++) { out.pitch[i] = widths[i]; CK(cudaMalloc((void**)&out.channel[i], (size_t)widths[i] * heights[i])); } }
    float* d_tensor = nullptr;
    if (resize > 0) CK(cudaMalloc((void**)&d_tensor, (size_t)3 * resize * resize * sizeof(float)));
    float scale = 1.f; int nw = W, nh = H, ox = 0, oy = 0;
    if (resize > 0) { float r = (float)resize / std::max(W, H); nw = (int)(W * r + 0.5f); nh = (int)(H * r + 0.5f); scale = 1.f / r; ox = (resize - nw) / 2; oy = (resize - nh) / 2; }

    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::vector<double> t_parse, t_host, t_xfer, t_dev, t_resize, t_total;
    cudaEvent_t e0, e1, e2, e3; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2)); CK(cudaEventCreate(&e3));
    int warm = std::min(10, std::max(1, iters / 10));
    for (int it = 0; it < iters + warm; it++) {
        auto a0 = clk::now();
        NJ(nvjpegJpegStreamParse(handle, data.data(), data.size(), 0, 0, jstream));
        auto a1 = clk::now();
        NJ(nvjpegDecodeJpegHost(handle, decoder, state, params, jstream));
        auto a2 = clk::now();
        CK(cudaEventRecord(e0, cs));
        NJ(nvjpegDecodeJpegTransferToDevice(handle, decoder, state, jstream, cs));
        CK(cudaEventRecord(e1, cs));
        NJ(nvjpegDecodeJpegDevice(handle, decoder, state, &out, cs));
        CK(cudaEventRecord(e2, cs));
        if (resize > 0 && ofmt == NVJPEG_OUTPUT_RGBI) {
            dim3 blk(16, 16), grd((resize + 15) / 16, (resize + 15) / 16);
            letterbox_kernel<<<grd, blk, 0, cs>>>(out.channel[0], W, H, (int)out.pitch[0], d_tensor, resize, scale, ox, oy, nw, nh);
        }
        CK(cudaEventRecord(e3, cs));
        CK(cudaStreamSynchronize(cs));
        auto a3 = clk::now();
        if (it < warm) continue;
        float f01, f12, f23; CK(cudaEventElapsedTime(&f01, e0, e1)); CK(cudaEventElapsedTime(&f12, e1, e2)); CK(cudaEventElapsedTime(&f23, e2, e3));
        t_parse.push_back(ms(a0, a1)); t_host.push_back(ms(a1, a2)); t_xfer.push_back(f01); t_dev.push_back(f12); t_resize.push_back(f23); t_total.push_back(ms(a0, a3));
    }
    auto stat = [&](const char* n, std::vector<double>& v) {
        double s = 0; for (double x : v) s += x;
        printf("  %-8s mean %7.3f  p50 %7.3f  p99 %7.3f  max %7.3f ms\n", n, s / v.size(), pct(v, 50), pct(v, 99), pct(v, 100));
    };
    printf("timing over %d iterations (%s backend, out=%s%s):\n", iters, backend.c_str(), fmt.c_str(), resize ? (", letterbox " + std::to_string(resize)).c_str() : "");
    stat("parse", t_parse); stat("host", t_host); stat("xfer", t_xfer); stat("device", t_dev); if (resize) stat("resize", t_resize); stat("total", t_total);
    if (!csv.empty()) {
        FILE* c = fopen(csv.c_str(), "a");
        if (c) {
            double s = 0; for (double x : t_total) s += x;
            fprintf(c, "%s,%s,%s,%s,ok,%d,%zu,%dx%d,%u,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", tag.c_str(), file.c_str(), backend.c_str(), fmt.c_str(), resize, data.size(), W, H, n_eng, n_cores,
                    pct(t_parse, 50), pct(t_host, 50), pct(t_xfer, 50), pct(t_dev, 50), pct(t_resize, 50), s / t_total.size(), pct(t_total, 50), pct(t_total, 99), pct(t_total, 100));
            fclose(c);
        }
    }
    // checksum of the output so the decode is not optimised away and the backends can be compared
    if (ofmt == NVJPEG_OUTPUT_RGBI) {
        std::vector<unsigned char> h((size_t)W * 3 * 4);
        CK(cudaMemcpy(h.data(), out.channel[0] + (size_t)(H / 2) * out.pitch[0], h.size(), cudaMemcpyDeviceToHost));
        unsigned long sum = 0; for (unsigned char b : h) sum += b;
        printf("checksum(4 middle rows) = %lu\n", sum);
    }
    return 0;
}
