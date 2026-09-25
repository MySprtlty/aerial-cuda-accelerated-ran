#!/usr/bin/env python3
"""
yolo_worker: the "tenant". A saturating TensorRT inference loop that runs as
one MPS client next to cuPHY. The launcher sets the MPS environment
(CUDA_MPS_PIPE_DIRECTORY, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = cap,
CUDA_MPS_CLIENT_PRIORITY) and pins the process with taskset.

    python3 yolo_worker.py --engine yolo26n_b1_fp16_atp10.engine --input bus_640.npy \
        --batch 1 --log worker.csv --seconds 200 --ready /path/ready

The input image is preprocessed once (letterboxed CHW float32 .npy) and copied
H2D for every inference, then the network runs and the output is copied D2H,
so one "inference" costs what dapp_sched measures. Every second the worker
appends: ts_ns, n, fps, lat_ms_mean, lat_ms_p50, lat_ms_p99, lat_ms_max.
Runs inside the Aerial container (tensorrt 10.3, cuda-python 12.9, numpy).
"""
import argparse
import os
import signal
import sys
import time

import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:  # older cuda-python
    from cuda import cudart


def chk(res):
    err = res[0]
    if int(err) != 0:
        raise RuntimeError("CUDA error %s" % err)
    return res[1] if len(res) > 1 else None


class TrtRunner:
    def __init__(self, engine_path, batch, input_npy):
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        blob = open(engine_path, "rb").read()
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError("deserialize failed: " + engine_path)
        self.ctx = self.engine.create_execution_context()
        self.stream = chk(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))
        self.bufs = {}
        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)
            if -1 in shape:  # dynamic: fix batch
                shape = tuple(batch if d == -1 else d for d in shape)
                if mode == trt.TensorIOMode.INPUT:
                    self.ctx.set_input_shape(name, shape)
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            dptr = chk(cudart.cudaMalloc(nbytes))
            host = np.zeros(shape, dtype=dtype)
            chk(cudart.cudaHostRegister(host.ctypes.data, nbytes, 0))
            self.bufs[name] = (dptr, host, nbytes, shape, mode)
            self.ctx.set_tensor_address(name, int(dptr))
            if mode == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_name = name
        d_in, h_in, n_in, shp, _ = self.bufs[self.in_name]
        img = np.load(input_npy).astype(h_in.dtype)
        if img.ndim == 3:
            img = img[None]
        reps = shp[0] // img.shape[0]
        h_in[...] = np.tile(img, (reps, 1, 1, 1))[: shp[0]]
        self.ev0 = chk(cudart.cudaEventCreateWithFlags(0))
        self.ev1 = chk(cudart.cudaEventCreateWithFlags(0))
        self.in_shape = shp
        self.out_shape = self.bufs[self.out_name][3]

    def infer(self):
        d_in, h_in, n_in, _, _ = self.bufs[self.in_name]
        d_out, h_out, n_out, _, _ = self.bufs[self.out_name]
        chk(cudart.cudaEventRecord(self.ev0, self.stream))
        chk(cudart.cudaMemcpyAsync(int(d_in), h_in.ctypes.data, n_in, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        if not self.ctx.execute_async_v3(int(self.stream)):
            raise RuntimeError("execute_async_v3 failed")
        chk(cudart.cudaMemcpyAsync(h_out.ctypes.data, int(d_out), n_out, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        chk(cudart.cudaEventRecord(self.ev1, self.stream))
        chk(cudart.cudaEventSynchronize(self.ev1))
        return chk(cudart.cudaEventElapsedTime(self.ev0, self.ev1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--log", required=True)
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds of measurement (0 = until SIGTERM)")
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--ready", default="", help="file to create when warmed up")
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()

    cap = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100")
    prio = os.environ.get("CUDA_MPS_CLIENT_PRIORITY", "-")
    t_load = time.time()
    r = TrtRunner(a.engine, a.batch, a.input)
    load_s = time.time() - t_load
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))

    # warm-up
    t0 = time.time()
    n_w = 0
    while time.time() - t0 < a.warmup and not stop:
        r.infer()
        n_w += 1
    if a.ready:
        with open(a.ready, "w") as f:
            f.write("%d\n" % os.getpid())
    log = open(a.log, "w")
    log.write("ts_ns,n,fps,lat_ms_mean,lat_ms_p50,lat_ms_p99,lat_ms_max,cap_pct,prio,batch\n")
    log.flush()
    print("worker: engine %s in %dx%dx%dx%d -> %s, load %.1f s, warm-up %d inferences, cap=%s prio=%s batch=%d" % (
        os.path.basename(a.engine), *r.in_shape, "x".join(map(str, r.out_shape)), load_s, n_w, cap, prio, a.batch), flush=True)

    t_start = time.time()
    t_win = t_start
    lats = []
    total = 0
    while not stop:
        ms = r.infer()
        lats.append(ms)
        total += 1
        now = time.time()
        if now - t_win >= a.interval:
            arr = np.array(lats)
            log.write("%d,%d,%.1f,%.3f,%.3f,%.3f,%.3f,%s,%s,%d\n" % (
                time.time_ns(), len(arr), len(arr) * a.batch / (now - t_win), arr.mean(), np.percentile(arr, 50),
                np.percentile(arr, 99), arr.max(), cap, prio, a.batch))
            log.flush()
            lats = []
            t_win = now
        if a.seconds and now - t_start >= a.seconds:
            break
    log.close()
    print("worker: done, %d inferences in %.1f s" % (total, time.time() - t_start), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
