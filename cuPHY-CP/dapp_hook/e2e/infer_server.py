#!/usr/bin/env python3
"""
infer_server: the GH200 side of the E2E latency path (UE -> RU -> gNB -> this -> gNB -> RU -> UE).
Receives JPEG images over TCP, decodes them on the NVJPG hardware engines (libnvjpeg_dec.so), letterboxes
into the detector's input tensor on the device, runs a TensorRT YOLO engine, and replies with the
detections plus the server-side timings, so the E2E number can be split into air/transport and GH200
parts. One GPU worker, FIFO; several engines can be loaded (model selection), one is used per request.
A batch-1 engine sees the whole image letterboxed to its input size; a batch-N engine is used TILED: the image is
cut into native-resolution crops of the input size (3840x2160 at 1280 -> 3 x 2 = 6 tiles), the remaining slots carry
the whole image letterboxed, and the boxes of all slots are merged by one NMS (the usual sliced-inference recipe for
small objects in 4K frames).

    python3 infer_server.py --engine 26x1280=/opt/.../yolo26x_1280_b1_fp16_atp100.engine \
        [--engine 26n640=/opt/.../yolo26n_b1_fp16_atp100.engine] [--default 26x1280] \
        [--port 9000] [--backend hw] [--log server.csv] [--conf 0.25] [--iou 0.7]

Run inside the Aerial container as the same user as cuPHY's MPS clients (root): MPS serves one user at a
time, a client of another user waits until the first user's server is gone. Cap/priority come from the
usual MPS environment; the engine must match the cap it was built with.

Protocol (TCP, little endian):
    request  '<4sIII'  b'DAPQ', req_id, engine_sel (0xFFFFFFFF = server policy, else engine index), n_bytes ; JPEG
    reply    '<4sIII'  b'DAPR', req_id, status (0 ok, else decode/engine error), n_json ; JSON
JSON: {req_id, status, engine, n_det, dets: [[x1,y1,x2,y2,conf,cls] ... up to --max-det], img: [w,h],
       t_ms: {net_in, queue, parse, decode, letterbox, infer, post, server}} with
    net_in   = first header byte received -> last payload byte received   (uplink tail seen by the server)
    queue    = payload complete -> GPU worker picked it up
    server   = first header byte received -> reply handed to the socket   (what the UE's RTT contains besides the air)
Log (--log): one csv row per request with the same fields.
"""
import argparse
import ctypes
import json
import os
import queue
import socket
import struct
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sweep"))
from yolo_worker import TrtRunner, chk, cudart  # noqa: E402

HDR = struct.Struct("<4sIII")
MAGIC_Q, MAGIC_R = b"DAPQ", b"DAPR"
SEL_POLICY = 0xFFFFFFFF


class Decoder:
    def __init__(self, backend):
        self.lib = ctypes.CDLL(os.path.join(HERE, "libnvjpeg_dec.so"))
        self.lib.njd_create.restype = ctypes.c_void_p
        self.lib.njd_create.argtypes = [ctypes.c_int]
        self.lib.njd_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
        self.lib.njd_decode.restype = ctypes.c_int
        self.lib.njd_decode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)]
        self.lib.njd_decode_tiles.restype = ctypes.c_int
        self.lib.njd_decode_tiles.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                              ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.lib.njd_destroy.argtypes = [ctypes.c_void_p]
        self.ctx = self.lib.njd_create({"hw": 0, "hybrid": 1, "gpu": 2}[backend])
        if not self.ctx:
            raise RuntimeError("nvjpeg decoder (%s backend) unavailable here" % backend)
        e, c = ctypes.c_uint(0), ctypes.c_uint(0)
        self.lib.njd_info(self.ctx, ctypes.byref(e), ctypes.byref(c))
        self.engines, self.cores = e.value, c.value
        self.t = (ctypes.c_float * 3)()
        self.wh = (ctypes.c_int * 2)()

    def decode(self, data, d_tensor, size, stream, batch=1, tiled=False):
        """-> status, (parse, decode, letterbox) ms, (w, h), rois [(x, y, w, h) per batch slot, source pixels]"""
        rois = (ctypes.c_int * (4 * batch))()
        st = self.lib.njd_decode_tiles(self.ctx, data, len(data), d_tensor, size, batch, 1 if tiled else 0, stream, self.t, self.wh, rois)
        return st, (self.t[0], self.t[1], self.t[2]), (self.wh[0], self.wh[1]), [tuple(rois[4 * i:4 * i + 4]) for i in range(batch)]


class Engine(TrtRunner):
    """TrtRunner whose input tensor is written on the device by the decoder: run = network + D2H only."""
    def __init__(self, path, batch=1):
        super().__init__(path, batch, None)
        self.size = int(self.in_shape[-1])
        self.batch = int(self.in_shape[0])       # > 1: one request fills the batch with tiles of the image (4K -> 6 x 1280 crops + global views)
        self.d_in = int(self.bufs[self.in_name][0])

    def run(self):
        d_out, h_out, n_out, _, _ = self.bufs[self.out_name]
        chk(cudart.cudaEventRecord(self.ev0, self.stream))
        if not self.ctx.execute_async_v3(int(self.stream)):
            raise RuntimeError("execute_async_v3 failed")
        chk(cudart.cudaMemcpyAsync(h_out.ctypes.data, int(d_out), n_out, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        chk(cudart.cudaEventRecord(self.ev1, self.stream))
        chk(cudart.cudaEventSynchronize(self.ev1))
        return chk(cudart.cudaEventElapsedTime(self.ev0, self.ev1)), h_out


def postprocess(out, size, rois, conf_th, iou_th, max_det):
    """ultralytics detect head (batch, 4+nc, na) -> boxes of every batch slot mapped back through its ROI to original
    image pixels -> one per-class NMS over all slots (merges the duplicates of overlapping tiles and the global view)"""
    all_xyxy, all_conf, all_cls = [], [], []
    for bi, (rx, ry, rw, rh) in enumerate(rois):
        o = out[bi]
        if o.shape[0] > o.shape[1]:      # (na, 4+nc) -> rows = channels
            o = o.T
        conf_all = o[4:].max(axis=0)     # (4+nc, na) is C-contiguous per channel: one pass over nc x na, then only the survivors are touched
        idx = np.flatnonzero(conf_all >= conf_th)
        if len(idx) == 0:
            continue
        boxes = o[:4, idx].T
        conf = conf_all[idx]
        cls = o[4:, idx].argmax(axis=0)
        r = size / max(rw, rh)
        nw, nh = int(rw * r + 0.5), int(rh * r + 0.5)
        ox, oy = (size - nw) // 2, (size - nh) // 2
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - ox) / r + rx
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - oy) / r + ry
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2 - ox) / r + rx
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2 - oy) / r + ry
        all_xyxy.append(xyxy); all_conf.append(conf); all_cls.append(cls)
        if len(set(rois)) == 1:          # every slot is the same view (plain batch-1 case or padding): one is enough
            break
    if not all_conf:
        return []
    xyxy, conf, cls = np.concatenate(all_xyxy), np.concatenate(all_conf), np.concatenate(all_cls)
    order = conf.argsort()[::-1][:5000]
    xyxy, conf, cls = xyxy[order], conf[order], cls[order]
    span = float(max(xyxy[:, 2].max(), xyxy[:, 3].max(), 1.0)) * 2
    b = xyxy + cls.astype(np.float32)[:, None] * span        # class offset trick: one NMS over all classes
    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    picked = []
    idx = np.arange(len(b))
    while len(idx) and len(picked) < max_det:
        i = idx[0]
        picked.append(i)
        if len(idx) == 1:
            break
        rest = idx[1:]
        ix1 = np.maximum(b[i, 0], b[rest, 0]); iy1 = np.maximum(b[i, 1], b[rest, 1])
        ix2 = np.minimum(b[i, 2], b[rest, 2]); iy2 = np.minimum(b[i, 3], b[rest, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        idx = rest[iou <= iou_th]
    return [[round(float(xyxy[i, 0]), 1), round(float(xyxy[i, 1]), 1), round(float(xyxy[i, 2]), 1), round(float(xyxy[i, 3]), 1),
             round(float(conf[i]), 3), int(cls[i])] for i in picked]


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed")
        got += k
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", action="append", required=True, help="name=path (repeatable); the first is the default unless --default")
    ap.add_argument("--default", default="")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--backend", choices=["hw", "hybrid", "gpu"], default="hw")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--log", default="")
    ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()

    engines, names = [], []
    for e in a.engine:
        name, path = e.split("=", 1)
        engines.append(Engine(path))
        names.append(name)
        print("engine %s: %s input %dx%dx%d%s" % (name, os.path.basename(path), engines[-1].batch, engines[-1].size, engines[-1].size,
                                                    " (tiled: grid of native crops + global view)" if engines[-1].batch > 1 else ""), flush=True)
    default = names.index(a.default) if a.default else 0
    dec = Decoder(a.backend)
    print("nvjpeg %s backend: %u hardware engines x %u cores visible" % (a.backend, dec.engines, dec.cores), flush=True)
    # warm-up with a bundled 4K image if present
    sample = os.path.join(HERE, "..", "models", "img4k", "uhd_0_q85.jpg")
    if os.path.exists(sample):
        data = open(sample, "rb").read()
        for eng in engines:
            for _ in range(a.warmup):
                st, _, wh, _ = dec.decode(data, eng.d_in, eng.size, int(eng.stream), eng.batch, eng.batch > 1)
                eng.run()
        print("warm-up done (%s, %dx%d)" % (os.path.basename(sample), *wh), flush=True)

    log = open(a.log, "w") if a.log else None
    if log:
        log.write("t_done_ns,peer,req_id,engine,status,n_det,img_w,img_h,n_bytes,net_in_ms,queue_ms,parse_ms,decode_ms,letterbox_ms,infer_ms,post_ms,server_ms,depth\n")
    q = queue.Queue()
    stop = threading.Event()

    def policy(depth):
        """engine for the next request when the client leaves the choice to the server (hook for the dApp scheduler)"""
        return default

    def worker():
        while not stop.is_set():
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            conn, peer, req_id, sel, data, t_first, t_full = item
            t_pick = time.perf_counter()
            idx = policy(q.qsize()) if sel == SEL_POLICY else min(int(sel), len(engines) - 1)
            eng = engines[idx]
            st, (t_parse, t_dec, t_lb), wh, rois = dec.decode(data, eng.d_in, eng.size, int(eng.stream), eng.batch, eng.batch > 1)
            t_inf = t_post = 0.0
            dets = []
            if st == 0:
                t_inf, out = eng.run()
                t1 = time.perf_counter()
                dets = postprocess(out, eng.size, rois, a.conf, a.iou, a.max_det)
                t_post = (time.perf_counter() - t1) * 1e3
            t_send = time.perf_counter()
            tm = {"net_in": round((t_full - t_first) * 1e3, 3), "queue": round((t_pick - t_full) * 1e3, 3), "parse": round(t_parse, 3), "decode": round(t_dec, 3),
                  "letterbox": round(t_lb, 3), "infer": round(t_inf, 3), "post": round(t_post, 3), "server": round((t_send - t_first) * 1e3, 3)}
            body = json.dumps({"req_id": req_id, "status": st, "engine": names[idx], "n_det": len(dets), "dets": dets, "img": list(wh), "t_ms": tm}).encode()
            try:
                conn.sendall(HDR.pack(MAGIC_R, req_id, st & 0xFFFFFFFF, len(body)) + body)
            except OSError:
                pass
            if log:
                log.write("%d,%s,%d,%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n" % (
                    time.time_ns(), peer, req_id, names[idx], st, len(dets), wh[0], wh[1], len(data), tm["net_in"], tm["queue"], tm["parse"], tm["decode"],
                    tm["letterbox"], tm["infer"], tm["post"], tm["server"], q.qsize()))
                log.flush()

    def serve(conn, peer):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while not stop.is_set():
                first = conn.recv(HDR.size)
                if not first:
                    break
                t_first = time.perf_counter()
                hdr = first + (recv_exact(conn, HDR.size - len(first)) if len(first) < HDR.size else b"")
                magic, req_id, sel, n = HDR.unpack(hdr)
                if magic != MAGIC_Q:
                    print("bad magic from %s, closing" % peer, flush=True)
                    break
                data = recv_exact(conn, n)
                t_full = time.perf_counter()
                q.put((conn, peer, req_id, sel, data, t_first, t_full))
        except (ConnectionError, OSError) as e:
            print("connection %s closed: %s" % (peer, e), flush=True)
        finally:
            conn.close()

    threading.Thread(target=worker, daemon=True).start()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.bind, a.port))
    srv.listen(8)
    print("listening on %s:%d, engines %s (default %s), log %s" % (a.bind, a.port, names, names[default], a.log or "-"), flush=True)
    try:
        while True:
            conn, addr = srv.accept()
            peer = "%s:%d" % addr
            print("client %s" % peer, flush=True)
            threading.Thread(target=serve, args=(conn, peer), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if log:
            log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
