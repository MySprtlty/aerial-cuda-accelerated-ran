// nvjpeg_dec: JPEG bytes -> letterboxed float32 CHW tensor on the device, in one call, for the E2E
// inference server (infer_server.py loads this with ctypes). The decode runs on the GH200's NVJPG
// hardware engines (NVJPEG_BACKEND_HARDWARE, no SMs; baseline single-scan JPEG only) by default;
// the letterbox (bilinear resize + /255 + grey-114 padding) is one small SM kernel (~15 us at 1280).
//
// Build (inside the Aerial container):
//   nvcc -O2 -arch=sm_90 -Xcompiler -fPIC -shared -o libnvjpeg_dec.so nvjpeg_dec.cu -lnvjpeg
//
// C API (all synchronous on the given stream; status 0 = ok, nvjpeg status codes otherwise, -1 = unsupported JPEG):
//   void* njd_create(int backend);                 // 0 = hardware, 1 = hybrid (CPU Huffman), 2 = gpu hybrid
//   int   njd_info(void* ctx, unsigned* engines, unsigned* cores_per_engine);
//   int   njd_decode(void* ctx, const unsigned char* jpeg, size_t n, float* d_tensor, int S,
//                    void* cuda_stream, float t_ms[3], int wh[2]);   // t_ms: parse+host, decode (device), letterbox
//   int   njd_decode_tiles(ctx, jpeg, n, d_tensor, S, batch, tiled, cuda_stream, t_ms, wh, int rois_out[4*batch]);
//         // tiled = 1: S x S native crops on a grid (4K at S = 1280 -> 6 tiles) + whole-image views in the remaining slots
//   void  njd_destroy(void* ctx);
#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <chrono>
#include <cstdio>
#include <cstring>

namespace {

struct Ctx {
    nvjpegHandle_t handle = nullptr;
    nvjpegJpegDecoder_t decoder = nullptr;
    nvjpegJpegState_t state = nullptr;
    nvjpegBufferPinned_t pinned = nullptr;
    nvjpegBufferDevice_t devbuf = nullptr;
    nvjpegJpegStream_t jstream = nullptr;
    nvjpegDecodeParams_t params = nullptr;
    unsigned char* d_rgb = nullptr;      // interleaved RGB, W*3 pitch
    size_t d_rgb_bytes = 0;
    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;
};

// One destination tile (S x S, CHW float /255) from a source ROI (rx, ry, rw, rh) of the interleaved RGB image:
// bilinear resize that keeps the aspect ratio, centred, grey-114 padding. ROI = whole image -> the usual letterbox;
// ROI = S x S native crop -> a plain copy (scale 1), which is how the 4K image is cut into tiles.
__global__ void letterbox_kernel(const unsigned char* src, int spitch, int rx, int ry, int rw, int rh, float* dst, int S, float scale, int ox, int oy, int nw, int nh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= S || y >= S) return;
    float r, g, b;
    int ix = x - ox, iy = y - oy;
    if (ix < 0 || iy < 0 || ix >= nw || iy >= nh) {
        r = g = b = 114.f;
    } else {
        float fx = (ix + 0.5f) * scale - 0.5f, fy = (iy + 0.5f) * scale - 0.5f;
        int x0 = min(rw - 1, max(0, (int)floorf(fx))), y0 = min(rh - 1, max(0, (int)floorf(fy)));
        int x1 = min(rw - 1, x0 + 1), y1 = min(rh - 1, y0 + 1);
        float ax = fminf(fmaxf(fx - x0, 0.f), 1.f), ay = fminf(fmaxf(fy - y0, 0.f), 1.f);
        const unsigned char* p00 = src + (ry + y0) * spitch + (rx + x0) * 3;
        const unsigned char* p01 = src + (ry + y0) * spitch + (rx + x1) * 3;
        const unsigned char* p10 = src + (ry + y1) * spitch + (rx + x0) * 3;
        const unsigned char* p11 = src + (ry + y1) * spitch + (rx + x1) * 3;
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

void launch_tile(const unsigned char* d_rgb, int w, int rx, int ry, int rw, int rh, float* dst, int S, cudaStream_t cs) {
    float r = (float)S / (float)(rw > rh ? rw : rh);
    int nw = (int)(rw * r + 0.5f), nh = (int)(rh * r + 0.5f);
    dim3 blk(16, 16), grd((S + 15) / 16, (S + 15) / 16);
    letterbox_kernel<<<grd, blk, 0, cs>>>(d_rgb, w * 3, rx, ry, rw, rh, dst, S, 1.f / r, (S - nw) / 2, (S - nh) / 2, nw, nh);
}

}  // namespace

extern "C" {

void* njd_create(int backend) {
    Ctx* c = new Ctx();
    nvjpegBackend_t be = backend == 1 ? NVJPEG_BACKEND_HYBRID : backend == 2 ? NVJPEG_BACKEND_GPU_HYBRID : NVJPEG_BACKEND_HARDWARE;
    if (nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, nullptr, nullptr, 0, &c->handle) != NVJPEG_STATUS_SUCCESS) { delete c; return nullptr; }
    nvjpegStatus_t st = nvjpegDecoderCreate(c->handle, be, &c->decoder);
    if (st != NVJPEG_STATUS_SUCCESS) { fprintf(stderr, "nvjpeg_dec: nvjpegDecoderCreate(backend %d) status %d\n", backend, (int)st); nvjpegDestroy(c->handle); delete c; return nullptr; }
    if (nvjpegDecoderStateCreate(c->handle, c->decoder, &c->state) != NVJPEG_STATUS_SUCCESS ||
        nvjpegBufferPinnedCreate(c->handle, nullptr, &c->pinned) != NVJPEG_STATUS_SUCCESS ||
        nvjpegBufferDeviceCreate(c->handle, nullptr, &c->devbuf) != NVJPEG_STATUS_SUCCESS ||
        nvjpegStateAttachPinnedBuffer(c->state, c->pinned) != NVJPEG_STATUS_SUCCESS ||
        nvjpegStateAttachDeviceBuffer(c->state, c->devbuf) != NVJPEG_STATUS_SUCCESS ||
        nvjpegJpegStreamCreate(c->handle, &c->jstream) != NVJPEG_STATUS_SUCCESS ||
        nvjpegDecodeParamsCreate(c->handle, &c->params) != NVJPEG_STATUS_SUCCESS ||
        nvjpegDecodeParamsSetOutputFormat(c->params, NVJPEG_OUTPUT_RGBI) != NVJPEG_STATUS_SUCCESS) {
        fprintf(stderr, "nvjpeg_dec: state/buffer creation failed\n");
        delete c;
        return nullptr;
    }
    cudaEventCreate(&c->e0); cudaEventCreate(&c->e1); cudaEventCreate(&c->e2);
    return c;
}

int njd_info(void* vc, unsigned* engines, unsigned* cores) {
    Ctx* c = (Ctx*)vc;
    return (int)nvjpegGetHardwareDecoderInfo(c->handle, engines, cores);
}

// Decode + fill `batch` tiles of the S x S input tensor. tiled = 0: every slot is the whole image letterboxed
// (batch 1 = the plain case). tiled = 1: the image is cut into a grid of S x S native-resolution crops (evenly
// spread, overlapping where the size is not a multiple of S; 3840x2160 at S = 1280 -> 3 x 2 = 6 tiles) and the
// remaining slots carry the whole image letterboxed (global view). rois_out (4 ints per slot: x, y, w, h in
// source pixels) tells the caller how to map each slot's boxes back.
int njd_decode_tiles(void* vc, const unsigned char* jpeg, size_t n, float* d_tensor, int S, int batch, int tiled,
                     void* vstream, float* t_ms, int* wh, int* rois_out) {
    Ctx* c = (Ctx*)vc;
    cudaStream_t cs = (cudaStream_t)vstream;
    auto a0 = std::chrono::steady_clock::now();
    nvjpegStatus_t st = nvjpegJpegStreamParse(c->handle, jpeg, n, 0, 0, c->jstream);
    if (st != NVJPEG_STATUS_SUCCESS) return (int)st;
    unsigned int nc = 0, w = 0, h = 0;
    nvjpegJpegStreamGetComponentsNum(c->jstream, &nc);
    nvjpegJpegStreamGetComponentDimensions(c->jstream, 0, &w, &h);
    int supported = -1;
    st = nvjpegDecoderJpegSupported(c->decoder, c->jstream, c->params, &supported);
    if (st != NVJPEG_STATUS_SUCCESS) return (int)st;
    if (supported != 0) return -1;     // e.g. progressive JPEG on the hardware backend
    size_t need = (size_t)w * 3 * h;
    if (need > c->d_rgb_bytes) {
        if (c->d_rgb) cudaFree(c->d_rgb);
        if (cudaMalloc((void**)&c->d_rgb, need) != cudaSuccess) { c->d_rgb = nullptr; c->d_rgb_bytes = 0; return -2; }
        c->d_rgb_bytes = need;
    }
    st = nvjpegDecodeJpegHost(c->handle, c->decoder, c->state, c->params, c->jstream);
    if (st != NVJPEG_STATUS_SUCCESS) return (int)st;
    auto a1 = std::chrono::steady_clock::now();
    cudaEventRecord(c->e0, cs);
    st = nvjpegDecodeJpegTransferToDevice(c->handle, c->decoder, c->state, c->jstream, cs);
    if (st != NVJPEG_STATUS_SUCCESS) return (int)st;
    nvjpegImage_t out; memset(&out, 0, sizeof(out));
    out.channel[0] = c->d_rgb; out.pitch[0] = w * 3;
    st = nvjpegDecodeJpegDevice(c->handle, c->decoder, c->state, &out, cs);
    if (st != NVJPEG_STATUS_SUCCESS) return (int)st;
    cudaEventRecord(c->e1, cs);
    int W = (int)w, H = (int)h, slot = 0;
    size_t tile_elems = (size_t)3 * S * S;
    if (tiled && W >= S && H >= S) {
        int nx = (W + S - 1) / S, ny = (H + S - 1) / S;
        for (int j = 0; j < ny && slot < batch - 1; j++) {
            for (int i = 0; i < nx && slot < batch - 1; i++) {
                int rx = nx > 1 ? (int)((long)i * (W - S) / (nx - 1)) : 0;
                int ry = ny > 1 ? (int)((long)j * (H - S) / (ny - 1)) : 0;
                launch_tile(c->d_rgb, W, rx, ry, S, S, d_tensor + slot * tile_elems, S, cs);
                if (rois_out) { rois_out[4 * slot] = rx; rois_out[4 * slot + 1] = ry; rois_out[4 * slot + 2] = S; rois_out[4 * slot + 3] = S; }
                slot++;
            }
        }
    }
    for (; slot < batch; slot++) {      // global view(s)
        launch_tile(c->d_rgb, W, 0, 0, W, H, d_tensor + slot * tile_elems, S, cs);
        if (rois_out) { rois_out[4 * slot] = 0; rois_out[4 * slot + 1] = 0; rois_out[4 * slot + 2] = W; rois_out[4 * slot + 3] = H; }
    }
    cudaEventRecord(c->e2, cs);
    if (cudaStreamSynchronize(cs) != cudaSuccess) return -3;
    float d01 = 0, d12 = 0;
    cudaEventElapsedTime(&d01, c->e0, c->e1); cudaEventElapsedTime(&d12, c->e1, c->e2);
    if (t_ms) { t_ms[0] = std::chrono::duration<float, std::milli>(a1 - a0).count(); t_ms[1] = d01; t_ms[2] = d12; }
    if (wh) { wh[0] = W; wh[1] = H; }
    return 0;
}

int njd_decode(void* vc, const unsigned char* jpeg, size_t n, float* d_tensor, int S, void* vstream, float* t_ms, int* wh) {
    return njd_decode_tiles(vc, jpeg, n, d_tensor, S, 1, 0, vstream, t_ms, wh, nullptr);
}

void njd_destroy(void* vc) {
    Ctx* c = (Ctx*)vc;
    if (!c) return;
    if (c->d_rgb) cudaFree(c->d_rgb);
    if (c->params) nvjpegDecodeParamsDestroy(c->params);
    if (c->jstream) nvjpegJpegStreamDestroy(c->jstream);
    if (c->devbuf) nvjpegBufferDeviceDestroy(c->devbuf);
    if (c->pinned) nvjpegBufferPinnedDestroy(c->pinned);
    if (c->state) nvjpegJpegStateDestroy(c->state);
    if (c->decoder) nvjpegDecoderDestroy(c->decoder);
    if (c->handle) nvjpegDestroy(c->handle);
    delete c;
}

}  // extern "C"
