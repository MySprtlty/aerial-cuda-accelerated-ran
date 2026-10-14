/*
 * dapp_yolo: TensorRT runner for a YOLO detection engine (ultralytics v8/11/26
 * export: input [N,3,H,W] float RGB in [0,1]). Two output heads are accepted:
 *   raw      [N,4+nc,A] (or [N,A,4+nc]): xywh boxes in input pixels + per-class
 *            scores for every anchor; the CPU takes the class max, thresholds
 *            and runs a class-aware NMS
 *   end2end  [N,K,6]: the YOLO26 NMS-free head (export nms=False): the K best
 *            boxes per image as x1,y1,x2,y2,score,class; the CPU only thresholds,
 *            maps each slot's boxes through its ROI and merges the duplicates
 *            that different tiles see of the same object.
 *
 * One YoloEngine per CUDA context. TensorRT allocates its device memory in the
 * context that is current when the engine is deserialised, so a scheduler that
 * keeps one SM-capped context per SM class needs one engine per (model, class).
 *
 * Two input paths:
 *   batch 1   the whole image letterboxed on the CPU (stb_image, bilinear),
 *             pinned host buffer, H2D on the engine's stream (the original path)
 *   batch N   TILED, like e2e/infer_server.py: the JPEG is decoded on the NVJPG
 *             hardware engines (e2e/nvjpeg_dec.cu) straight into the engine's
 *             device input - N-1 native crops of the input size on a grid plus
 *             whole-image views in the remaining slots - no H2D copy. The boxes
 *             of every slot are mapped back through the slot's source ROI and
 *             merged by one class-aware NMS (the sliced-inference recipe for 4K).
 *   (--decode gpu puts batch-1 engines on the nvjpeg path too.)
 *
 * Pre/post-processing follow ultralytics' defaults for non-PyTorch backends:
 * letterbox with grey 114 padding, conf 0.25, per-class NMS at IoU 0.7, boxes
 * mapped back to the original image. Timings: h2d / net / d2h from CUDA events
 * on the engine stream, dec from the decoder (parse + decode + letterbox).
 *
 * Header-only apart from the stb_image implementation unit and nvjpeg_dec.cu.
 */
#ifndef DAPP_YOLO_HPP
#define DAPP_YOLO_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "stb_image.h"
#ifdef DAPP_SCHED_NVJPEG
#include "dapp_nvjpeg.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace nv {
namespace dapp {
namespace sched {

struct Detection {
    int   cls = 0;
    float conf = 0.f;
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;   // original image pixels
};

// COCO-80 names, the label set of the stock ultralytics detection models.
inline const char* coco_name(int c)
{
    static const char* const names[80] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"};
    return (c >= 0 && c < 80) ? names[c] : "?";
}

// ---------------------------------------------------------------------------
// Input: what the post-processing needs to map every batch slot's boxes back.
// Each slot was filled from a source ROI (x, y, w, h in image pixels) resized
// to fit the input and centred with grey padding: the whole image for the
// usual letterbox, an S x S crop (scale 1) for a native tile.
// ---------------------------------------------------------------------------
struct YoloInput {
    int   net_w = 0, net_h = 0;   // engine input size
    int   img_w = 0, img_h = 0;   // original image size
    int   batch = 1;
    std::vector<int>   rois;      // 4 ints per slot
    std::vector<float> chw;       // CPU path only: batch * 3 * net_h * net_w, RGB, [0,1]
};

// CPU letterbox of one image into a batch-1 input (stb_image decode, bilinear).
inline bool yolo_preprocess(const std::string& path, int net_w, int net_h, YoloInput& in, std::string& err)
{
    int w = 0, h = 0, c = 0;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!px) {
        const char* why = stbi_failure_reason();
        err = "cannot decode " + path + ": " + (why ? why : "unknown");
        return false;
    }
    in.net_w = net_w; in.net_h = net_h;
    in.img_w = w;     in.img_h = h;
    in.batch = 1;
    in.rois  = {0, 0, w, h};
    const float scale = std::min(static_cast<float>(net_w) / static_cast<float>(w),
                                 static_cast<float>(net_h) / static_cast<float>(h));
    const int new_w = static_cast<int>(std::lround(w * scale));
    const int new_h = static_cast<int>(std::lround(h * scale));
    const int x0 = (net_w - new_w) / 2;   // same placement as the GPU letterbox kernel
    const int y0 = (net_h - new_h) / 2;

    in.chw.assign(static_cast<size_t>(3) * net_w * net_h, 114.f / 255.f);
    const size_t plane = static_cast<size_t>(net_w) * net_h;

    // bilinear resize straight into the letterbox
    for (int y = 0; y < new_h; ++y) {
        float sy = (y + 0.5f) / scale - 0.5f;
        sy = std::min(std::max(sy, 0.f), static_cast<float>(h - 1));
        const int   ya = static_cast<int>(sy);
        const int   yb = std::min(ya + 1, h - 1);
        const float fy = sy - ya;
        for (int x = 0; x < new_w; ++x) {
            float sx = (x + 0.5f) / scale - 0.5f;
            sx = std::min(std::max(sx, 0.f), static_cast<float>(w - 1));
            const int   xa = static_cast<int>(sx);
            const int   xb = std::min(xa + 1, w - 1);
            const float fx = sx - xa;
            const unsigned char* p00 = px + (static_cast<size_t>(ya) * w + xa) * 3;
            const unsigned char* p01 = px + (static_cast<size_t>(ya) * w + xb) * 3;
            const unsigned char* p10 = px + (static_cast<size_t>(yb) * w + xa) * 3;
            const unsigned char* p11 = px + (static_cast<size_t>(yb) * w + xb) * 3;
            const size_t o = static_cast<size_t>(y0 + y) * net_w + static_cast<size_t>(x0 + x);
            for (int ch = 0; ch < 3; ++ch) {
                const float v = (1.f - fy) * ((1.f - fx) * p00[ch] + fx * p01[ch]) +
                                fy         * ((1.f - fx) * p10[ch] + fx * p11[ch]);
                in.chw[static_cast<size_t>(ch) * plane + o] = v / 255.f;
            }
        }
    }
    stbi_image_free(px);
    return true;
}

// ---------------------------------------------------------------------------
// Output: candidates of every slot -> slot ROI -> original image pixels ->
// one class-aware NMS over all slots (merges the duplicates of overlapping
// tiles and the global view). Slots that repeat an earlier ROI are skipped.
// ---------------------------------------------------------------------------
inline std::vector<Detection> yolo_postprocess(const float* out, int nc, int na, bool anchors_first,
                                               const YoloInput& in, float conf_thr, float iou_thr,
                                               int max_det = 300)
{
    struct Cand { float x1, y1, x2, y2, conf; int cls; };
    std::vector<Cand> cand;
    static thread_local std::vector<float> best_buf;   // scratch of the row-wise class max (na floats)
    static thread_local std::vector<int>   cls_buf;
    const int    stride     = 4 + nc;
    const size_t slot_elems = static_cast<size_t>(stride) * static_cast<size_t>(na);
    const float  fw = static_cast<float>(in.img_w), fh = static_cast<float>(in.img_h);

    for (int s = 0; s < in.batch; ++s) {
        const int* roi = &in.rois[static_cast<size_t>(4) * s];
        bool dup = false;
        for (int t = 0; t < s && !dup; ++t) { dup = std::equal(roi, roi + 4, &in.rois[static_cast<size_t>(4) * t]); }
        if (dup) { continue; }
        const int   rx = roi[0], ry = roi[1], rw = roi[2], rh = roi[3];
        const float scale = std::min(static_cast<float>(in.net_w) / static_cast<float>(rw),
                                     static_cast<float>(in.net_h) / static_cast<float>(rh));
        const int   nw = static_cast<int>(std::lround(rw * scale));
        const int   nh = static_cast<int>(std::lround(rh * scale));
        const float ox = static_cast<float>((in.net_w - nw) / 2);
        const float oy = static_cast<float>((in.net_h - nh) / 2);
        const float* o = out + slot_elems * static_cast<size_t>(s);
        auto mx = [&](float v) { return std::min(std::max((v - ox) / scale + rx, 0.f), fw); };
        auto my = [&](float v) { return std::min(std::max((v - oy) / scale + ry, 0.f), fh); };
        if (!anchors_first) {
            // [4+nc][na]: class rows are contiguous, so take the max over classes row by row (vectorisable,
            // sequential memory) instead of walking nc strided values per anchor (~20 ms per 4K tiled request)
            std::vector<float>& best = best_buf;
            std::vector<int>&   bcls = cls_buf;
            best.assign(o + 4 * static_cast<size_t>(na), o + 5 * static_cast<size_t>(na));
            bcls.assign(static_cast<size_t>(na), 0);
            for (int k = 1; k < nc; ++k) {
                const float* row = o + static_cast<size_t>(4 + k) * na;
                float* b = best.data();
                int*   c = bcls.data();
                for (int a = 0; a < na; ++a) {
                    const float v = row[a];
                    if (v > b[a]) { b[a] = v; c[a] = k; }
                }
            }
            const float* cxr = o;
            const float* cyr = o + static_cast<size_t>(na);
            const float* wr  = o + 2 * static_cast<size_t>(na);
            const float* hr  = o + 3 * static_cast<size_t>(na);
            for (int a = 0; a < na; ++a) {
                if (best[a] < conf_thr) { continue; }
                cand.push_back(Cand{mx(cxr[a] - wr[a] / 2), my(cyr[a] - hr[a] / 2), mx(cxr[a] + wr[a] / 2), my(cyr[a] + hr[a] / 2),
                                    best[a], bcls[a]});
            }
        } else {
            // [na][4+nc]: one anchor's values are contiguous
            for (int a = 0; a < na; ++a) {
                const float* p = o + static_cast<size_t>(a) * stride;
                int   best = 0;
                float bs   = p[4];
                for (int k = 1; k < nc; ++k) {
                    if (p[4 + k] > bs) { bs = p[4 + k]; best = k; }
                }
                if (bs < conf_thr) { continue; }
                cand.push_back(Cand{mx(p[0] - p[2] / 2), my(p[1] - p[3] / 2), mx(p[0] + p[2] / 2), my(p[1] + p[3] / 2), bs, best});
            }
        }
    }
    std::sort(cand.begin(), cand.end(), [](const Cand& p, const Cand& q) { return p.conf > q.conf; });
    if (cand.size() > 5000) { cand.resize(5000); }

    auto iou = [](const Cand& p, const Cand& q) {
        const float ix = std::max(0.f, std::min(p.x2, q.x2) - std::max(p.x1, q.x1));
        const float iy = std::max(0.f, std::min(p.y2, q.y2) - std::max(p.y1, q.y1));
        const float inter = ix * iy;
        const float uni = (p.x2 - p.x1) * (p.y2 - p.y1) + (q.x2 - q.x1) * (q.y2 - q.y1) - inter;
        return uni > 0.f ? inter / uni : 0.f;
    };

    std::vector<Detection> res;
    std::vector<char> dead(cand.size(), 0);
    for (size_t i = 0; i < cand.size() && static_cast<int>(res.size()) < max_det; ++i) {
        if (dead[i]) { continue; }
        for (size_t j = i + 1; j < cand.size(); ++j) {
            if (!dead[j] && cand[j].cls == cand[i].cls && iou(cand[i], cand[j]) > iou_thr) { dead[j] = 1; }
        }
        Detection d;
        d.cls  = cand[i].cls;
        d.conf = cand[i].conf;
        d.x1 = cand[i].x1; d.y1 = cand[i].y1;
        d.x2 = cand[i].x2; d.y2 = cand[i].y2;
        res.push_back(d);
    }
    return res;
}

// End-to-end head: per slot the K boxes [x1,y1,x2,y2,score,class] are already one-to-one, so only the
// threshold, the slot ROI mapping and a class-aware NMS BETWEEN slots (tile overlaps and the global views
// see the same object) are left. Boxes of the same slot never suppress each other.
inline std::vector<Detection> yolo_postprocess_e2e(const float* out, int k_per_slot, const YoloInput& in,
                                                   float conf_thr, float iou_thr, int max_det = 300)
{
    struct Cand { float x1, y1, x2, y2, conf; int cls, slot; };
    std::vector<Cand> cand;
    const float fw = static_cast<float>(in.img_w), fh = static_cast<float>(in.img_h);
    for (int s = 0; s < in.batch; ++s) {
        const int* roi = &in.rois[static_cast<size_t>(4) * s];
        bool dup = false;
        for (int t = 0; t < s && !dup; ++t) { dup = std::equal(roi, roi + 4, &in.rois[static_cast<size_t>(4) * t]); }
        if (dup) { continue; }
        const int   rx = roi[0], ry = roi[1], rw = roi[2], rh = roi[3];
        const float scale = std::min(static_cast<float>(in.net_w) / static_cast<float>(rw),
                                     static_cast<float>(in.net_h) / static_cast<float>(rh));
        const int   nw = static_cast<int>(std::lround(rw * scale));
        const int   nh = static_cast<int>(std::lround(rh * scale));
        const float ox = static_cast<float>((in.net_w - nw) / 2);
        const float oy = static_cast<float>((in.net_h - nh) / 2);
        auto mx = [&](float v) { return std::min(std::max((v - ox) / scale + rx, 0.f), fw); };
        auto my = [&](float v) { return std::min(std::max((v - oy) / scale + ry, 0.f), fh); };
        const float* o = out + static_cast<size_t>(s) * k_per_slot * 6;
        for (int k = 0; k < k_per_slot; ++k) {
            const float* p = o + static_cast<size_t>(k) * 6;
            if (p[4] < conf_thr) { continue; }      // the head sorts by score: the rest of this slot is below too
            cand.push_back(Cand{mx(p[0]), my(p[1]), mx(p[2]), my(p[3]), p[4], static_cast<int>(p[5] + 0.5f), s});
        }
    }
    std::sort(cand.begin(), cand.end(), [](const Cand& p, const Cand& q) { return p.conf > q.conf; });
    auto iou = [](const Cand& p, const Cand& q) {
        const float ix = std::max(0.f, std::min(p.x2, q.x2) - std::max(p.x1, q.x1));
        const float iy = std::max(0.f, std::min(p.y2, q.y2) - std::max(p.y1, q.y1));
        const float inter = ix * iy;
        const float uni = (p.x2 - p.x1) * (p.y2 - p.y1) + (q.x2 - q.x1) * (q.y2 - q.y1) - inter;
        return uni > 0.f ? inter / uni : 0.f;
    };
    std::vector<Detection> res;
    std::vector<char> dead(cand.size(), 0);
    for (size_t i = 0; i < cand.size() && static_cast<int>(res.size()) < max_det; ++i) {
        if (dead[i]) { continue; }
        for (size_t j = i + 1; j < cand.size(); ++j) {
            if (!dead[j] && cand[j].slot != cand[i].slot && cand[j].cls == cand[i].cls && iou(cand[i], cand[j]) > iou_thr) { dead[j] = 1; }
        }
        Detection d;
        d.cls  = cand[i].cls;
        d.conf = cand[i].conf;
        d.x1 = cand[i].x1; d.y1 = cand[i].y1;
        d.x2 = cand[i].x2; d.y2 = cand[i].y2;
        res.push_back(d);
    }
    return res;
}

// ---------------------------------------------------------------------------
// Engine bound to the CUDA context that is current at load()
// ---------------------------------------------------------------------------
struct InferTiming {
    float dec_ms = 0.f;                    // nvjpeg path: parse + decode + letterbox/tiles (host-timed, synchronous)
    float h2d_ms = 0.f, net_ms = 0.f, d2h_ms = 0.f;   // CUDA events on the engine stream
    float gpu_ms() const { return h2d_ms + net_ms + d2h_ms; }
};

enum class DecodeMode { Auto = 0, Cpu = 1, Gpu = 2 };   // Auto: CPU for batch 1, nvjpeg for batch N

class YoloEngine {
public:
    YoloEngine() = default;
    YoloEngine(const YoloEngine&) = delete;
    YoloEngine& operator=(const YoloEngine&) = delete;

    // Call with the target CUDA context current. Runs one warm-up inference,
    // which is also where an engine built for another SM count fails.
    bool load(const std::vector<char>& blob, DecodeMode mode, std::string& err)
    {
        runtime_.reset(nvinfer1::createInferRuntime(logger()));
        if (!runtime_) { err = "createInferRuntime failed"; return false; }
        engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
        if (!engine_) { err = "deserializeCudaEngine failed (engine built for another GPU/TensorRT?)"; return false; }
        ctx_.reset(engine_->createExecutionContext());
        if (!ctx_) { err = "createExecutionContext failed"; return false; }

        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            const auto  dims = engine_->getTensorShape(name);
            if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
                err = std::string("tensor ") + name + " is not float32; export the engine with float32 I/O";
                return false;
            }
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                if (dims.nbDims != 4 || dims.d[0] < 1 || dims.d[1] != 3) {
                    err = std::string("unexpected input shape for ") + name + " (want [N,3,H,W])";
                    return false;
                }
                in_name_ = name;
                batch_ = static_cast<int>(dims.d[0]);
                net_h_ = static_cast<int>(dims.d[2]);
                net_w_ = static_cast<int>(dims.d[3]);
            } else {
                if (dims.nbDims != 3 || dims.d[0] < 1) {
                    err = std::string("unexpected output shape for ") + name + " (want [N,4+nc,A], [N,A,4+nc] or [N,K,6])";
                    return false;
                }
                out_name_ = name;
                out_batch_ = static_cast<int>(dims.d[0]);
                if (dims.d[2] == 6) {
                    // YOLO26 end-to-end head: K boxes per image, x1,y1,x2,y2,score,class
                    e2e_ = true;
                    na_  = static_cast<int>(dims.d[1]);
                    nc_  = 80;
                } else if (dims.d[1] < dims.d[2]) { anchors_first_ = false; nc_ = static_cast<int>(dims.d[1]) - 4; na_ = static_cast<int>(dims.d[2]); }   // ultralytics [N, 4+nc, A]
                else                              { anchors_first_ = true;  na_ = static_cast<int>(dims.d[1]);     nc_ = static_cast<int>(dims.d[2]) - 4; }   // transposed
                out_slot_elems_ = static_cast<size_t>(dims.d[1]) * static_cast<size_t>(dims.d[2]);
            }
        }
        if (in_name_.empty() || out_name_.empty()) { err = "engine needs one input and one output tensor"; return false; }
        if (out_batch_ != batch_) { err = "input batch " + std::to_string(batch_) + " != output batch " + std::to_string(out_batch_); return false; }
        in_elems_  = static_cast<size_t>(batch_) * 3 * net_w_ * net_h_;
        out_elems_ = static_cast<size_t>(batch_) * out_slot_elems_;

        const bool want_decoder = (mode == DecodeMode::Gpu) || (mode == DecodeMode::Auto && batch_ > 1);
        if (!want_decoder && batch_ > 1) {
            err = "batch " + std::to_string(batch_) + " engine needs the nvjpeg (tiled) path; --decode cpu only takes batch 1";
            return false;
        }

        if (cudaMalloc(&d_in_, in_elems_ * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_out_, out_elems_ * sizeof(float)) != cudaSuccess ||
            cudaMallocHost(reinterpret_cast<void**>(&h_out_), out_elems_ * sizeof(float)) != cudaSuccess ||
            (!want_decoder && cudaMallocHost(reinterpret_cast<void**>(&h_in_), in_elems_ * sizeof(float)) != cudaSuccess) ||
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreate(&ev0_) != cudaSuccess || cudaEventCreate(&ev1_) != cudaSuccess ||
            cudaEventCreate(&ev2_) != cudaSuccess || cudaEventCreate(&ev3_) != cudaSuccess) {
            err = std::string("CUDA allocation failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (!ctx_->setTensorAddress(in_name_.c_str(), d_in_) || !ctx_->setTensorAddress(out_name_.c_str(), d_out_)) {
            err = "setTensorAddress failed";
            return false;
        }
        if (want_decoder) {
#ifdef DAPP_SCHED_NVJPEG
            if (net_w_ != net_h_) { err = "the nvjpeg path needs a square input (tiles are S x S)"; return false; }
            dec_ = njd_create(0);   // NVJPG hardware engines; lives in the context current now
            if (!dec_) { err = "nvjpeg hardware decoder unavailable in this CUDA context"; return false; }
#else
            err = "dapp_sched was built without nvjpeg: batch N / --decode gpu engines cannot run";
            return false;
#endif
        }
        // warm-up: first launch pays for lazy module loading (and rejects an engine built for another SM count)
        if (cudaMemsetAsync(d_in_, 0, in_elems_ * sizeof(float), stream_) != cudaSuccess) { err = "cudaMemset failed"; return false; }
        if (h_in_) { std::memset(h_in_, 0, in_elems_ * sizeof(float)); }
        InferTiming t;
        if (!infer(nullptr, t)) { err = "warm-up inference failed"; return false; }
        return true;
    }

    bool has_decoder() const { return dec_ != nullptr; }

    // nvjpeg path: JPEG bytes -> the engine's device input (batch 1: letterbox;
    // batch N: N-1 native tiles + global views). Synchronous on the engine stream.
    bool decode(const unsigned char* jpeg, size_t n, YoloInput& in, float& dec_ms, std::string& err)
    {
#ifdef DAPP_SCHED_NVJPEG
        if (!dec_) { err = "engine has no decoder"; return false; }
        float t[3] = {0.f, 0.f, 0.f};
        int   wh[2] = {0, 0};
        in.rois.assign(static_cast<size_t>(4) * batch_, 0);
        const int st = njd_decode_tiles(dec_, jpeg, n, static_cast<float*>(d_in_), net_w_, batch_, batch_ > 1 ? 1 : 0,
                                        stream_, t, wh, in.rois.data());
        if (st == -1)      { err = "JPEG not supported by the hardware decoder (progressive or 4:4:4?)"; return false; }
        else if (st == -2) { err = "nvjpeg: device allocation failed"; return false; }
        else if (st == -3) { err = "nvjpeg: stream synchronisation failed"; return false; }
        else if (st != 0)  { err = "nvjpeg status " + std::to_string(st); return false; }
        in.net_w = net_w_; in.net_h = net_h_;
        in.img_w = wh[0];  in.img_h = wh[1];
        in.batch = batch_;
        in.chw.clear();
        dec_ms = t[0] + t[1] + t[2];
        return true;
#else
        (void)jpeg; (void)n; (void)in; (void)dec_ms;
        err = "built without nvjpeg";
        return false;
#endif
    }

    // One inference. chw = a host CHW buffer to upload (batch-1 CPU path), or
    // null when the input is already in the device buffer (decoder path or a
    // repeat of the previous input). Timings per stage from events.
    bool infer(const float* chw, InferTiming& t)
    {
        if (cudaEventRecord(ev0_, stream_) != cudaSuccess) { return false; }
        if (chw) {
            if (!h_in_) { return false; }
            std::memcpy(h_in_, chw, in_elems_ * sizeof(float));
            if (cudaMemcpyAsync(d_in_, h_in_, in_elems_ * sizeof(float), cudaMemcpyHostToDevice, stream_) != cudaSuccess) { return false; }
        }
        if (cudaEventRecord(ev1_, stream_) != cudaSuccess) { return false; }
        if (!ctx_->enqueueV3(stream_)) { return false; }
        if (cudaEventRecord(ev2_, stream_) != cudaSuccess) { return false; }
        if (cudaMemcpyAsync(h_out_, d_out_, out_elems_ * sizeof(float), cudaMemcpyDeviceToHost, stream_) != cudaSuccess) { return false; }
        if (cudaEventRecord(ev3_, stream_) != cudaSuccess) { return false; }
        if (cudaEventSynchronize(ev3_) != cudaSuccess) { return false; }
        cudaEventElapsedTime(&t.h2d_ms, ev0_, ev1_);
        cudaEventElapsedTime(&t.net_ms, ev1_, ev2_);
        cudaEventElapsedTime(&t.d2h_ms, ev2_, ev3_);
        return cudaGetLastError() == cudaSuccess;
    }

    std::vector<Detection> detections(const YoloInput& in, float conf, float iou, int max_det = 300) const
    {
        if (e2e_) { return yolo_postprocess_e2e(h_out_, na_, in, conf, iou, max_det); }
        return yolo_postprocess(h_out_, nc_, na_, anchors_first_, in, conf, iou, max_det);
    }
    bool end2end() const { return e2e_; }        // [N,K,6] head
    const char* head() const { return e2e_ ? "end2end" : "raw"; }

    // Call with the engine's CUDA context current, before that context dies.
    void release()
    {
#ifdef DAPP_SCHED_NVJPEG
        if (dec_) { njd_destroy(dec_); dec_ = nullptr; }
#endif
        ctx_.reset();
        engine_.reset();
        runtime_.reset();
        if (ev0_)   { cudaEventDestroy(ev0_);   ev0_ = nullptr; }
        if (ev1_)   { cudaEventDestroy(ev1_);   ev1_ = nullptr; }
        if (ev2_)   { cudaEventDestroy(ev2_);   ev2_ = nullptr; }
        if (ev3_)   { cudaEventDestroy(ev3_);   ev3_ = nullptr; }
        if (stream_){ cudaStreamDestroy(stream_); stream_ = nullptr; }
        if (d_in_)  { cudaFree(d_in_);   d_in_ = nullptr; }
        if (d_out_) { cudaFree(d_out_);  d_out_ = nullptr; }
        if (h_in_)  { cudaFreeHost(h_in_);  h_in_ = nullptr; }
        if (h_out_) { cudaFreeHost(h_out_); h_out_ = nullptr; }
    }
    ~YoloEngine() { release(); }

    int net_w() const { return net_w_; }
    int net_h() const { return net_h_; }
    int batch() const { return batch_; }
    int num_classes() const { return nc_; }
    int num_anchors() const { return na_; }

    static bool read_file(const std::string& path, std::vector<char>& blob, std::string& err)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) { err = "cannot open " + path; return false; }
        blob.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        if (blob.empty()) { err = path + " is empty"; return false; }
        return true;
    }

private:
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity s, const char* msg) noexcept override
        {
            if (s <= Severity::kWARNING) { std::fprintf(stderr, "[trt] %s\n", msg); }
        }
    };
    static Logger& logger() { static Logger l; return l; }

    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    std::string  in_name_, out_name_;
    int          net_w_ = 0, net_h_ = 0, nc_ = 0, na_ = 0, batch_ = 1, out_batch_ = 1;
    bool         anchors_first_ = false, e2e_ = false;
    size_t       in_elems_ = 0, out_elems_ = 0, out_slot_elems_ = 0;
    void*        d_in_ = nullptr;
    void*        d_out_ = nullptr;
    float*       h_in_ = nullptr;
    float*       h_out_ = nullptr;
    void*        dec_ = nullptr;          // nvjpeg decoder of this context (nvjpeg path only)
    cudaStream_t stream_ = nullptr;
    cudaEvent_t  ev0_ = nullptr, ev1_ = nullptr, ev2_ = nullptr, ev3_ = nullptr;
};

} // namespace sched
} // namespace dapp
} // namespace nv

#endif // DAPP_YOLO_HPP
