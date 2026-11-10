# trt_seq.py <mode> (2026-11-10): the tenant's start-up order in Python. Green contexts for 80 and 64 SMs are created first (each made current for a
# cudart stream creation, then the primary restored), THEN the TensorRT runtime is created and fhd26n/seg00_sm80.engine is deserialized in the primary
# context and executed on the 80-SM stream. mode=tenant: as described; mode=trtfirst: the runtime is created and one engine deserialized+deleted
# BEFORE the green contexts exist (the proposed fix); mode=touch: like tenant, but the primary context is "touched" by a cudart call after the green
# contexts (control). Prints the segment-K GEMM grid via nsys when run under it; here just ok/FAILED.
import ctypes, sys, numpy as np, tensorrt as trt
mode = sys.argv[1]; f80 = "fhd26n/seg00_sm80.engine"
cu = ctypes.CDLL("libcudart.so.12"); cu.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]; cu.cudaFree.argtypes = [ctypes.c_void_p]
cu.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]; cu.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]; cu.cudaGetErrorString.restype = ctypes.c_char_p
dr = ctypes.CDLL("libcuda.so.1"); V = ctypes.c_void_p
def ck(r, w):
    if r != 0: s = ctypes.c_char_p(); dr.cuGetErrorString(r, ctypes.byref(s)); raise SystemExit("%s -> %s" % (w, s.value))
ck(dr.cuInit(0), "cuInit"); dev = ctypes.c_int(); ck(dr.cuDeviceGet(ctypes.byref(dev), 0), "dev"); prim = V(); ck(dr.cuDevicePrimaryCtxRetain(ctypes.byref(prim), dev), "retain"); ck(dr.cuCtxSetCurrent(prim), "cur")
class Res(ctypes.Structure): _fields_ = [("type", ctypes.c_uint), ("pad", ctypes.c_ubyte * 92), ("smCount", ctypes.c_uint), ("rest", ctypes.c_ubyte * 44)]
allr = Res(); ck(dr.cuDeviceGetDevResource(dev, ctypes.byref(allr), 1), "res")
logger = trt.Logger(trt.Logger.WARNING); rt = None
if mode == "trtfirst":
    rt = trt.Runtime(logger); e = rt.deserialize_cuda_engine(open(f80, "rb").read()); del e; print("TRT runtime created and one engine deserialized+deleted before any green context")
def green(c):
    part = Res(); rem = Res(); n = ctypes.c_uint(1); ck(dr.cuDevSmResourceSplitByCount(ctypes.byref(part), ctypes.byref(n), ctypes.byref(allr), ctypes.byref(rem), 0, c), "split")
    desc = V(); ck(dr.cuDevResourceGenerateDesc(ctypes.byref(desc), ctypes.byref(part), 1), "desc"); g = V(); ck(dr.cuGreenCtxCreate(ctypes.byref(g), desc, dev, 1), "gctx")
    ctx = V(); ck(dr.cuCtxFromGreenCtx(ctypes.byref(ctx), g), "from"); ck(dr.cuCtxSetCurrent(ctx), "set"); s = V(); cu.cudaStreamCreateWithFlags(ctypes.byref(s), 1); ck(dr.cuCtxSetCurrent(prim), "prim")
    print("green ctx %d SMs created (cudart stream made inside it, primary restored)" % part.smCount); return s
s80 = green(80); s64 = green(64)          # the tenant: partitions in gap order, the last one is a 64-SM (or 72-SM) partition
if mode == "touch": d = V(); cu.cudaMalloc(ctypes.byref(d), 16); print("cudaMalloc on the primary after the green contexts")
if rt is None: rt = trt.Runtime(logger)
eng = rt.deserialize_cuda_engine(open(f80, "rb").read()); x = eng.create_execution_context()
for i in range(eng.num_io_tensors):
    n = eng.get_tensor_name(i); nb = int(np.prod(eng.get_tensor_shape(n))) * trt.nptype(eng.get_tensor_dtype(n))().itemsize; p = V(); cu.cudaMalloc(ctypes.byref(p), nb); x.set_tensor_address(n, p.value)
ok = x.execute_async_v3(s80.value); e = cu.cudaStreamSynchronize(s80.value)
print("mode %s: exec n80 on the 80-SM stream (deserialized in the primary):" % mode, "ok" if ok else "FAILED", "| sync", cu.cudaGetErrorString(e).decode())
