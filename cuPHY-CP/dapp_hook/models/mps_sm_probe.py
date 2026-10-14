#!/usr/bin/env python3
"""Print the SM count the driver grants to this process (cuCtxGetExecAffinity on the primary context).
Under an MPS percentage cap (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) the GH200 grants floor(pct/100 * 66 TPC) * 2 SMs:
12 % -> 14, 30 % -> 38, 37 % -> 48, 43 % -> 56, 49 % -> 64, 50 % -> 66, 61 % -> 80, 100 % -> 132. The engine
builder (trtexec) sees the same count, which is what build_engines.sh puts into the _sm<N> engine name."""
import ctypes, os, sys
cu = ctypes.CDLL("libcuda.so.1")
if cu.cuInit(0) != 0:
    sys.exit("cuInit failed")
dev = ctypes.c_int(); cu.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p(); cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev); cu.cuCtxSetCurrent(ctx)
class Aff(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("val", ctypes.c_uint), ("pad", ctypes.c_uint * 7)]
a = Aff()
rc = cu.cuCtxGetExecAffinity(ctypes.byref(a), 0)      # 0 = CU_EXEC_AFFINITY_TYPE_SM_COUNT
if rc != 0:
    sys.exit("cuCtxGetExecAffinity rc %d" % rc)
print(a.val if len(sys.argv) < 2 else "pct %s -> %u SM" % (os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "unset"), a.val))
