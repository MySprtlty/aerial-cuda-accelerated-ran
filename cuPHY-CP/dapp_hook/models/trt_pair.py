# trt_pair.py [--per-runtime] engine1 engine2 ... (2026-11-10) - TensorRT engine coexistence test (python3 of the Aerial container): deserialize all engines (one IRuntime, or one per engine), then execute each once; prints per-engine ok/err
import ctypes, sys, numpy as np, tensorrt as trt
per = "--per-runtime" in sys.argv; files = [a for a in sys.argv[1:] if not a.startswith("--")]
cu = ctypes.CDLL("libcudart.so.12"); cu.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cu.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]; cu.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]; cu.cudaGetErrorString.restype = ctypes.c_char_p
logger = trt.Logger(trt.Logger.WARNING); rt0 = trt.Runtime(logger); ctxs = []
for f in files:
    rt = trt.Runtime(logger) if per else rt0
    eng = rt.deserialize_cuda_engine(open(f, "rb").read()); x = eng.create_execution_context(); bufs = []
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i); nb = int(np.prod(eng.get_tensor_shape(n))) * trt.nptype(eng.get_tensor_dtype(n))().itemsize
        p = ctypes.c_void_p(); cu.cudaMalloc(ctypes.byref(p), nb); x.set_tensor_address(n, p.value); bufs.append(p)
    ctxs.append((f, rt, eng, x, bufs)); print("loaded", f, "batch", eng.get_tensor_shape("images")[0], flush=True)
s = ctypes.c_void_p(); cu.cudaStreamCreate(ctypes.byref(s))
for f, rt, eng, x, bufs in ctxs:
    ok = x.execute_async_v3(s.value); e = cu.cudaStreamSynchronize(s.value); le = cu.cudaGetLastError()
    print("exec", f, "ok" if ok else "FAILED", "| sync", cu.cudaGetErrorString(e).decode(), "| last", cu.cudaGetErrorString(le).decode(), flush=True)
