# trt_perf.py <engine> <sm> [--deser-green] (2026-11-10): time one engine on a green context of <sm> SMs, engine deserialized in the primary (default) or in that green context
import ctypes, sys, time, numpy as np, tensorrt as trt
f = sys.argv[1]; c = int(sys.argv[2]); dg = "--deser-green" in sys.argv
cu = ctypes.CDLL("libcudart.so.12"); cu.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]; cu.cudaFree.argtypes = [ctypes.c_void_p]; cu.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
dr = ctypes.CDLL("libcuda.so.1"); V = ctypes.c_void_p
def ck(r, w):
    if r != 0: raise SystemExit("%s -> %d" % (w, r))
cu.cudaFree(None); dev = ctypes.c_int(); ck(dr.cuDeviceGet(ctypes.byref(dev), 0), "dev"); prim = V(); ck(dr.cuCtxGetCurrent(ctypes.byref(prim)), "cur")
class Res(ctypes.Structure): _fields_ = [("type", ctypes.c_uint), ("pad", ctypes.c_ubyte * 92), ("smCount", ctypes.c_uint), ("rest", ctypes.c_ubyte * 44)]
allr = Res(); ck(dr.cuDeviceGetDevResource(dev, ctypes.byref(allr), 1), "res"); part = Res(); rem = Res(); n = ctypes.c_uint(1)
ck(dr.cuDevSmResourceSplitByCount(ctypes.byref(part), ctypes.byref(n), ctypes.byref(allr), ctypes.byref(rem), 0, c), "split"); desc = V(); ck(dr.cuDevResourceGenerateDesc(ctypes.byref(desc), ctypes.byref(part), 1), "desc")
g = V(); ck(dr.cuGreenCtxCreate(ctypes.byref(g), desc, dev, 1), "gctx"); s = V(); ck(dr.cuGreenCtxStreamCreate(ctypes.byref(s), g, 1, 0), "stream"); gctx = V(); ck(dr.cuCtxFromGreenCtx(ctypes.byref(gctx), g), "from")
if dg: ck(dr.cuCtxSetCurrent(gctx), "set green")
logger = trt.Logger(trt.Logger.ERROR); rt = trt.Runtime(logger); eng = rt.deserialize_cuda_engine(open(f, "rb").read()); x = eng.create_execution_context()
for i in range(eng.num_io_tensors):
    nm = eng.get_tensor_name(i); nb = int(np.prod(eng.get_tensor_shape(nm))) * trt.nptype(eng.get_tensor_dtype(nm))().itemsize; p = V(); cu.cudaMalloc(ctypes.byref(p), nb); x.set_tensor_address(nm, p.value)
ck(dr.cuCtxSetCurrent(prim), "prim")
for i in range(10): x.execute_async_v3(s.value)
cu.cudaStreamSynchronize(s.value); t = []
for i in range(50):
    t0 = time.perf_counter(); ok = x.execute_async_v3(s.value); cu.cudaStreamSynchronize(s.value); t.append((time.perf_counter() - t0) * 1e3)
t.sort(); print("%s on %d SMs, deserialized in %s: p50 %.2f ms  min %.2f  p90 %.2f  (%s)" % (f.split("/")[-2] + "/" + f.split("/")[-1], part.smCount, "green" if dg else "primary", t[25], t[0], t[45], "ok" if ok else "FAILED"))
