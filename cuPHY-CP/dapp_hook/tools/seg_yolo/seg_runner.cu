// seg_runner: run a segmented TensorRT network (tools/seg_yolo/seg_split.py) with every segment enqueued just in time on the
// CUDA green-context stream of the SM class chosen for the CURRENT slot (slot clock = wall time / --slot-us, TDD position =
// slot % 10), i.e. per-slot SM repartitioning of a YOLO tenant without touching TensorRT. Reports latency, per-segment GPU time
// per class, the number of class switches, and the output difference against the unsegmented engine.
//   seg_runner --dir DIR [--classes 40,88,132] [--schedule 88,88,88,88,88,88,88,40,40,40] [--slot-us 500] [--runs 30]
//              [--lookahead 1] [--fixed N] [--full full_sm132.engine] [--nseg 18]
// DIR holds seg<k>_sm<N>.engine (fp16 I/O) and manifest.json (tensor names/shapes). Classes = green-context partitions split
// from the full device (8-SM granularity). --fixed N runs every segment on class N (segmentation overhead only).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <NvInfer.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s; cuGetErrorString(r, &s); printf("%s: %s\n", #x, s); exit(1); } } while (0)
#define CR(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
struct Logger : nvinfer1::ILogger { void log(Severity s, const char* m) noexcept override { if (s <= Severity::kWARNING) printf("[trt] %s\n", m); } } gLogger;
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::vector<char> slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); return std::vector<char>((std::istreambuf_iterator<char>(f)), {}); }
static size_t numel(const nvinfer1::Dims& d) { size_t n = 1; for (int i = 0; i < d.nbDims; i++) n *= (size_t)d.d[i]; return n; }
struct Cls { int sm; CUgreenCtx g; CUcontext ctx; cudaStream_t s; };
struct Seg { std::vector<nvinfer1::ICudaEngine*> eng; std::vector<nvinfer1::IExecutionContext*> ctx; std::vector<std::string> io; };
int main(int argc, char** argv) {
    std::string dir, full; std::vector<int> classes = {40, 88, 132}; std::vector<int> sched = {88, 88, 88, 88, 88, 88, 88, 40, 40, 40};
    int slot_us = 500, runs = 30, lookahead = 1, fixed = 0, nseg = 18; std::string input;
    auto ints = [](const std::string& s) { std::vector<int> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, ',')) v.push_back(atoi(t.c_str())); return v; };
    for (int i = 1; i < argc; i++) { std::string a = argv[i];
        if (a == "--dir") dir = argv[++i]; else if (a == "--classes") classes = ints(argv[++i]); else if (a == "--schedule") sched = ints(argv[++i]);
        else if (a == "--slot-us") slot_us = atoi(argv[++i]); else if (a == "--runs") runs = atoi(argv[++i]); else if (a == "--lookahead") lookahead = atoi(argv[++i]);
        else if (a == "--fixed") fixed = atoi(argv[++i]); else if (a == "--full") full = argv[++i]; else if (a == "--nseg") nseg = atoi(argv[++i]); else if (a == "--input") input = argv[++i]; }
    if (dir.empty()) { printf("--dir required\n"); return 2; }
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev, 0)); CUcontext prim; CK(cuDevicePrimaryCtxRetain(&prim, dev)); CK(cuCtxSetCurrent(prim));
    CUdevResource all = {}; CK(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM));
    std::map<int, Cls> cls;
    for (int sm : classes) { CUdevResource part = {}, rem = {}; unsigned ng = 1; CK(cuDevSmResourceSplitByCount(&part, &ng, &all, &rem, 0, sm));
        CUdevResourceDesc d; CK(cuDevResourceGenerateDesc(&d, &part, 1)); Cls c; c.sm = sm; CK(cuGreenCtxCreate(&c.g, d, dev, CU_GREEN_CTX_DEFAULT_STREAM)); CK(cuCtxFromGreenCtx(&c.ctx, c.g));
        CK(cuCtxSetCurrent(c.ctx)); CR(cudaStreamCreateWithFlags(&c.s, cudaStreamNonBlocking)); cls[sm] = c; printf("class %d: green context with %u SMs\n", sm, part.sm.smCount); }
    CK(cuCtxSetCurrent(prim));
    nvinfer1::IRuntime* rt = nvinfer1::createInferRuntime(gLogger);
    // engines per (segment, class) + one device buffer per tensor name
    std::vector<Seg> segs(nseg); std::map<std::string, void*> buf; std::map<std::string, size_t> bytes;
    auto load = [&](const std::string& path, int sm, Seg& sg) {
        auto blob = slurp(path); if (blob.empty()) { printf("missing %s\n", path.c_str()); exit(1); }
        CK(cuCtxSetCurrent(cls[sm].ctx)); nvinfer1::ICudaEngine* e = rt->deserializeCudaEngine(blob.data(), blob.size()); if (!e) { printf("deserialize failed %s\n", path.c_str()); exit(1); }
        nvinfer1::IExecutionContext* c = e->createExecutionContext(); if (!c) { printf("context failed %s\n", path.c_str()); exit(1); }
        for (int i = 0; i < e->getNbIOTensors(); i++) { const char* n = e->getIOTensorName(i); nvinfer1::DataType dt = e->getTensorDataType(n); size_t esz = dt == nvinfer1::DataType::kHALF ? 2 : dt == nvinfer1::DataType::kINT64 ? 8 : (dt == nvinfer1::DataType::kBOOL || dt == nvinfer1::DataType::kINT8 || dt == nvinfer1::DataType::kUINT8) ? 1 : 4;
            size_t nb = numel(e->getTensorShape(n)) * esz;
            if (!buf.count(n)) { void* p; CR(cudaMalloc(&p, nb)); CR(cudaMemset(p, 0, nb)); buf[n] = p; bytes[n] = nb; }
            else if (bytes[n] != nb) { printf("tensor %s size mismatch %zu vs %zu\n", n, bytes[n], nb); exit(1); }
            c->setTensorAddress(n, buf[n]); if (std::find(sg.io.begin(), sg.io.end(), n) == sg.io.end()) sg.io.push_back(n); }
        sg.eng.push_back(e); sg.ctx.push_back(c); };
    std::vector<int> cl(classes); for (int k = 0; k < nseg; k++) for (int sm : cl) { char p[512]; snprintf(p, sizeof p, "%s/seg%02d_sm%d.engine", dir.c_str(), k, sm); load(p, sm, segs[k]); }
    auto cidx = [&](int sm) { return (int)(std::find(cl.begin(), cl.end(), sm) - cl.begin()); };
    // deterministic input (fp16 'images')
    { size_t n = bytes["images"] / 2; std::vector<__half> h(n);
      if (!input.empty()) { auto b = slurp(input); if (b.size() != n * 2) { printf("input file size %zu != %zu\n", b.size(), n * 2); return 1; } memcpy(h.data(), b.data(), n * 2); }
      else for (size_t i = 0; i < n; i++) h[i] = __float2half(((i * 7919u) % 256) / 255.0f);
      CR(cudaMemcpy(buf["images"], h.data(), n * 2, cudaMemcpyHostToDevice)); }
    size_t out_n = bytes["output0"] / 2; std::vector<__half> out_seg(out_n), out_full(out_n);
    std::vector<cudaEvent_t> evs(nseg), eve(nseg); for (int k = 0; k < nseg; k++) { CR(cudaEventCreate(&evs[k])); CR(cudaEventCreate(&eve[k])); }
    // ---- segmented runs (just-in-time class choice by the slot clock)
    std::vector<double> lat; std::vector<std::vector<float>> segms(nseg); std::vector<std::map<int, int>> cls_hist(nseg); long switches = 0;
    double t_origin = now_us();
    for (int r = 0; r < runs + 3; r++) {
        CK(cuCtxSetCurrent(prim)); double t0 = now_us(); int prev = -1; std::vector<int> used(nseg);
        for (int k = 0; k < nseg; k++) {
            if (k > lookahead) CR(cudaEventSynchronize(eve[k - 1 - lookahead]));            // keep at most lookahead+1 segments queued
            int pos = (int)(((long long)(now_us() - t_origin) / slot_us) % 10); int sm = fixed ? fixed : sched[pos]; int ci = cidx(sm); used[k] = sm;
            cudaStream_t s = cls[sm].s; if (prev >= 0 && sm != prev) { CR(cudaStreamWaitEvent(s, eve[k - 1], 0)); if (r >= 3) switches++; } prev = sm;
            CR(cudaEventRecord(evs[k], s)); if (!segs[k].ctx[ci]->enqueueV3(s)) { printf("enqueue failed seg %d class %d\n", k, sm); return 1; } CR(cudaEventRecord(eve[k], s));
        }
        CR(cudaEventSynchronize(eve[nseg - 1])); double t1 = now_us();
        if (r >= 3) { lat.push_back(t1 - t0); for (int k = 0; k < nseg; k++) { float ms; CR(cudaEventElapsedTime(&ms, evs[k], eve[k])); segms[k].push_back(ms); cls_hist[k][used[k]]++; } }
    }
    CR(cudaMemcpy(out_seg.data(), buf["output0"], out_n * 2, cudaMemcpyDeviceToHost));
    std::sort(lat.begin(), lat.end());
    printf("segmented (%d segs, lookahead %d, %s): latency p50 %.2f ms p90 %.2f max %.2f | class switches per run %.1f\n", nseg, lookahead, fixed ? "fixed class" : "slot schedule", lat[lat.size() / 2] / 1e3, lat[lat.size() * 9 / 10] / 1e3, lat.back() / 1e3, switches / (double)runs);
    float tot_gpu = 0; for (int k = 0; k < nseg; k++) { std::vector<float> v = segms[k]; std::sort(v.begin(), v.end()); float med = v[v.size() / 2]; tot_gpu += med;
        std::string h; for (auto& kv : cls_hist[k]) h += std::to_string(kv.first) + ":" + std::to_string(kv.second) + " ";
        printf("  seg%02d: gpu p50 %.3f ms max %.3f | classes used %s\n", k, med, v.back(), h.c_str()); }
    printf("  sum of segment medians %.2f ms\n", tot_gpu);
    // ---- reference: unsegmented engine on the largest class (and on every class if --fixed)
    if (!full.empty()) { Seg f; int sm = fixed ? fixed : cl.back(); load(full, sm, f);
        std::vector<double> fl; for (int r = 0; r < runs + 3; r++) { double t0 = now_us(); if (!f.ctx[0]->enqueueV3(cls[sm].s)) return 1; CR(cudaStreamSynchronize(cls[sm].s)); if (r >= 3) fl.push_back(now_us() - t0); }
        std::sort(fl.begin(), fl.end()); CR(cudaMemcpy(out_full.data(), buf["output0"], out_n * 2, cudaMemcpyDeviceToHost));
        auto dets = [&](const std::vector<__half>& o) { std::vector<std::array<float, 6>> v; for (size_t r = 0; r + 6 <= out_n; r += 6) { std::array<float, 6> d; for (int j = 0; j < 6; j++) d[j] = __half2float(o[r + j]); if (d[4] > 0.25f) v.push_back(d); } return v; };
        auto A = dets(out_seg), B = dets(out_full); size_t matched = 0; float worst_iou = 1.f, worst_ds = 0.f;
        for (auto& a : A) { float best = 0, ds = 0; for (auto& b : B) { if ((int)a[5] != (int)b[5]) continue; float ix = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0])), iy = std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
            float inter = ix * iy, uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter; float iou = uni > 0 ? inter / uni : 0; if (iou > best) { best = iou; ds = fabsf(a[4] - b[4]); } }
            if (best > 0.9f) { matched++; worst_iou = std::min(worst_iou, best); worst_ds = std::max(worst_ds, ds); } }
        printf("full engine on class %d: latency p50 %.2f ms p90 %.2f | detections (score>0.25): segmented %zu, full %zu, matched (same class, IoU>0.9) %zu, worst IoU %.3f, worst score diff %.3f\n", sm, fl[fl.size() / 2] / 1e3, fl[fl.size() * 9 / 10] / 1e3, A.size(), B.size(), matched, worst_iou, worst_ds); }
    return 0;
}
