/*
 * C API of e2e/nvjpeg_dec.cu, compiled into dapp_sched when the CUDA toolkit
 * ships nvjpeg (CMake: DAPP_SCHED_NVJPEG). JPEG bytes -> NVJPG hardware
 * decode -> letterbox / native tiles straight into a device tensor, all on the
 * caller's stream and CUDA context. See nvjpeg_dec.cu for the semantics; the
 * numbers in t_ms are parse (host), decode (device) and letterbox (device) ms.
 */
#ifndef DAPP_NVJPEG_H
#define DAPP_NVJPEG_H

#include <cstddef>

extern "C" {
void* njd_create(int backend);   // 0 = hardware (NVJPG engines), 1 = hybrid, 2 = gpu hybrid; null = unavailable
int   njd_info(void* ctx, unsigned* engines, unsigned* cores_per_engine);
int   njd_decode(void* ctx, const unsigned char* jpeg, size_t n, float* d_tensor, int S, void* cuda_stream,
                 float* t_ms, int* wh);
// tiled = 1: (batch - 1) S x S native crops on a grid, whole-image views in the remaining slots; rois_out gets
// 4 ints per slot (x, y, w, h in source pixels). 0 = ok, -1 = JPEG unsupported by this backend, else nvjpeg status.
int   njd_decode_tiles(void* ctx, const unsigned char* jpeg, size_t n, float* d_tensor, int S, int batch, int tiled,
                       void* cuda_stream, float* t_ms, int* wh, int* rois_out);
void  njd_destroy(void* ctx);
}

#endif // DAPP_NVJPEG_H
