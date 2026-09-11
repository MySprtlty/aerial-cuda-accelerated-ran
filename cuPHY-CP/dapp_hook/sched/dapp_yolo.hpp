/*
 * dapp_yolo: TensorRT runner for a YOLO detection engine (ultralytics v8/11
 * export: input [1,3,H,W] float RGB in [0,1], output [1,4+nc,A] with xywh
 * boxes in input pixels and per-class scores).
 *
 * One YoloEngine per CUDA context. TensorRT allocates its device memory in the
 * context that is current when the engine is deserialised, so a scheduler that
 * keeps one SM-capped context per SM class needs one engine per class. The
 * weights of the n/s variants are a few MB, so the duplication is cheap and
 * buys a zero-cost switch between SM budgets at request time.
 *
 * Pre/post-processing follow ultralytics' defaults for non-PyTorch backends:
 * letterbox to the engine input (grey 114 padding, bilinear), conf 0.25,
 * per-class NMS at IoU 0.7, boxes mapped back to the original image.
 *
 * Header-only apart from the stb_image implementation unit.
 */
#ifndef DAPP_YOLO_HPP
#define DAPP_YOLO_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "stb_image.h"

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
// Input: decoded, letterboxed, CHW float
// ---------------------------------------------------------------------------
struct YoloInput {
    int   net_w = 0, net_h = 0;   // engine input size
    int   img_w = 0, img_h = 0;   // original image size
    float scale = 1.f;            // letterbox scale
    float pad_x = 0.f, pad_y = 0.f;
    std::vector<float> chw;       // 3 * net_h * net_w, RGB, [0,1]
};

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
    in.scale = std::min(static_cast<float>(net_w) / static_cast<float>(w),
                        static_cast<float>(net_h) / static_cast<float>(h));
    const int new_w = static_cast<int>(std::lround(w * in.scale));
    const int new_h = static_cast<int>(std::lround(h * in.scale));
    in.pad_x = (net_w - new_w) / 2.f;
    in.pad_y = (net_h - new_h) / 2.f;
    const int x0 = static_cast<int>(std::lround(in.pad_x - 0.1f));
    const int y0 = static_cast<int>(std::lround(in.pad_y - 0.1f));

    in.chw.assign(static_cast<size_t>(3) * net_w * net_h, 114.f / 255.f);
    const size_t plane = static_cast<size_t>(net_w) * net_h;

    // bilinear resize straight into the letterbox
    for (int y = 0; y < new_h; ++y) {
        float sy = (y + 0.5f) / in.scale - 0.5f;
        sy = std::min(std::max(sy, 0.f), static_cast<float>(h - 1));
        const int   ya = static_cast<int>(sy);
        const int   yb = std::min(ya + 1, h - 1);
        const float fy = sy - ya;
        for (int x = 0; x < new_w; ++x) {
            float sx = (x + 0.5f) / in.scale - 0.5f;
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
// Output: candidates -> per-class NMS -> original image coordinates
// ---------------------------------------------------------------------------
inline std::vector<Detection> yolo_postprocess(const float* out, int nc, int na, bool anchors_first,
                                               const YoloInput& in, float conf_thr, float iou_thr,
                                               int max_det = 300)
{
    struct Cand { float x1, y1, x2, y2, conf; int cls; };
    std::vector<Cand> cand;
    const int stride = 4 + nc;
    auto at = [&](int a, int k) -> float {
        return anchors_first ? out[static_cast<size_t>(a) * stride + k]
                             : out[static_cast<size_t>(k) * na + a];
    };
    for (int a = 0; a < na; ++a) {
        int   best = 0;
        float bs   = at(a, 4);
        for (int k = 1; k < nc; ++k) {
            const float s = at(a, 4 + k);
            if (s > bs) { bs = s; best = k; }
        }
        if (bs < conf_thr) { continue; }
        const float cx = at(a, 0), cy = at(a, 1), w = at(a, 2), h = at(a, 3);
        cand.push_back(Cand{cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, bs, best});
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
            if (!dead[j] && cand[j].cls == cand[i].cls && iou(cand[i], cand[j]) > iou_thr) { dead[j] = 1; }
        }
        Detection d;
        d.cls  = cand[i].cls;
        d.conf = cand[i].conf;
        // letterbox -> original image, clamped
        auto mx = [&](float v) { return std::min(std::max((v - in.pad_x) / in.scale, 0.f), static_cast<float>(in.img_w)); };
        auto my = [&](float v) { return std::min(std::max((v - in.pad_y) / in.scale, 0.f), static_cast<float>(in.img_h)); };
        d.x1 = mx(cand[i].x1); d.y1 = my(cand[i].y1);
        d.x2 = mx(cand[i].x2); d.y2 = my(cand[i].y2);
        res.push_back(d);
    }
    return res;
}

// ---------------------------------------------------------------------------
// Engine bound to the CUDA context that is current at load()
// ---------------------------------------------------------------------------
class YoloEngine {
public:
    YoloEngine() = default;
    YoloEngine(const YoloEngine&) = delete;
    YoloEngine& operator=(const YoloEngine&) = delete;

    // Call with the target CUDA context current.
    bool load(const std::vector<char>& blob, std::string& err)
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
                if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != 3) {
                    err = std::string("unexpected input shape for ") + name + " (want [1,3,H,W])";
                    return false;
                }
                in_name_ = name;
                net_h_ = static_cast<int>(dims.d[2]);
                net_w_ = static_cast<int>(dims.d[3]);
            } else {
                if (dims.nbDims != 3 || dims.d[0] != 1) {
                    err = std::string("unexpected output shape for ") + name + " (want [1,4+nc,A] or [1,A,4+nc])";
                    return false;
                }
                out_name_ = name;
                // ultralytics exports [1, 4+nc, A]; some converters transpose it
                if (dims.d[1] < dims.d[2]) { anchors_first_ = false; nc_ = static_cast<int>(dims.d[1]) - 4; na_ = static_cast<int>(dims.d[2]); }
                else                       { anchors_first_ = true;  na_ = static_cast<int>(dims.d[1]);     nc_ = static_cast<int>(dims.d[2]) - 4; }
                out_elems_ = static_cast<size_t>(dims.d[1]) * static_cast<size_t>(dims.d[2]);
            }
        }
        if (in_name_.empty() || out_name_.empty()) { err = "engine needs one input and one output tensor"; return false; }
        in_elems_ = static_cast<size_t>(3) * net_w_ * net_h_;

        if (cudaMalloc(&d_in_, in_elems_ * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_out_, out_elems_ * sizeof(float)) != cudaSuccess ||
            cudaMallocHost(reinterpret_cast<void**>(&h_in_), in_elems_ * sizeof(float)) != cudaSuccess ||
            cudaMallocHost(reinterpret_cast<void**>(&h_out_), out_elems_ * sizeof(float)) != cudaSuccess ||
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreate(&ev0_) != cudaSuccess || cudaEventCreate(&ev1_) != cudaSuccess) {
            err = std::string("CUDA allocation failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (!ctx_->setTensorAddress(in_name_.c_str(), d_in_) || !ctx_->setTensorAddress(out_name_.c_str(), d_out_)) {
            err = "setTensorAddress failed";
            return false;
        }
        // warm-up: first launch pays for lazy module loading
        std::memset(h_in_, 0, in_elems_ * sizeof(float));
        float ms = 0.f;
        if (!infer(nullptr, ms)) { err = "warm-up inference failed"; return false; }
        return true;
    }

    // One inference. chw may be null to reuse what is already in the pinned
    // input buffer. gpu_ms covers H2D + network + D2H on the engine's stream.
    bool infer(const float* chw, float& gpu_ms)
    {
        if (chw) { std::memcpy(h_in_, chw, in_elems_ * sizeof(float)); }
        if (cudaEventRecord(ev0_, stream_) != cudaSuccess) { return false; }
        if (cudaMemcpyAsync(d_in_, h_in_, in_elems_ * sizeof(float), cudaMemcpyHostToDevice, stream_) != cudaSuccess) { return false; }
        if (!ctx_->enqueueV3(stream_)) { return false; }
        if (cudaMemcpyAsync(h_out_, d_out_, out_elems_ * sizeof(float), cudaMemcpyDeviceToHost, stream_) != cudaSuccess) { return false; }
        if (cudaEventRecord(ev1_, stream_) != cudaSuccess) { return false; }
        if (cudaEventSynchronize(ev1_) != cudaSuccess) { return false; }
        cudaEventElapsedTime(&gpu_ms, ev0_, ev1_);
        return cudaGetLastError() == cudaSuccess;
    }

    std::vector<Detection> detections(const YoloInput& in, float conf, float iou) const
    {
        return yolo_postprocess(h_out_, nc_, na_, anchors_first_, in, conf, iou);
    }

    // Call with the engine's CUDA context current, before that context dies.
    void release()
    {
        ctx_.reset();
        engine_.reset();
        runtime_.reset();
        if (ev0_)   { cudaEventDestroy(ev0_);   ev0_ = nullptr; }
        if (ev1_)   { cudaEventDestroy(ev1_);   ev1_ = nullptr; }
        if (stream_){ cudaStreamDestroy(stream_); stream_ = nullptr; }
        if (d_in_)  { cudaFree(d_in_);   d_in_ = nullptr; }
        if (d_out_) { cudaFree(d_out_);  d_out_ = nullptr; }
        if (h_in_)  { cudaFreeHost(h_in_);  h_in_ = nullptr; }
        if (h_out_) { cudaFreeHost(h_out_); h_out_ = nullptr; }
    }
    ~YoloEngine() { release(); }

    int net_w() const { return net_w_; }
    int net_h() const { return net_h_; }
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
    int          net_w_ = 0, net_h_ = 0, nc_ = 0, na_ = 0;
    bool         anchors_first_ = false;
    size_t       in_elems_ = 0, out_elems_ = 0;
    void*        d_in_ = nullptr;
    void*        d_out_ = nullptr;
    float*       h_in_ = nullptr;
    float*       h_out_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t  ev0_ = nullptr, ev1_ = nullptr;
};

} // namespace sched
} // namespace dapp
} // namespace nv

#endif // DAPP_YOLO_HPP
