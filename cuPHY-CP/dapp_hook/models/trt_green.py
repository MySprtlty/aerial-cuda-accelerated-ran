# trt_green.py [options] engine... (2026-11-10) - TensorRT engine coexistence test with CUDA green contexts (python3 of the Aerial container).
#   Each seg00_sm<c>.engine gets a green context of c SMs and is executed on that context's stream; the engines are deserialized in the primary
#   context unless --deser-green (the tenant's behaviour before 2026-11-10: the class's own green context -> the first class loaded fixes the
#   Myelin segment-K GEMM grids of every later class, see tenant/dapp_tenant.cu load_set). Options: --deser-green | --prim-exec (control: run on a
#   primary-context stream) | --union [--skip] [--groups=i,j,...] (green context = union of 8-SM groups like the tenant's union layout) | --sig
#   (deserialize+delete pass first) | --aux (bind tenant-style aux streams) | --rtstream (cudart stream created inside the green context) |
#   --sharebuf (one buffer per tensor name per model dir) | --pre (all green contexts before any TensorRT call) | --exec-order=i,j,... Prints ok/FAILED.
# (the tenant's default), otherwise in the primary context; --prim-exec executes on a primary-context stream instead (control).
import ctypes, re, sys, numpy as np, tensorrt as trt
dg = "--deser-green" in sys.argv; pe = "--prim-exec" in sys.argv; un = "--union" in sys.argv; sg = "--sig" in sys.argv; sk = "--skip" in sys.argv; ax = "--aux" in sys.argv; rs = "--rtstream" in sys.argv; sb = "--sharebuf" in sys.argv; gsel = [int(v) for a in sys.argv if a.startswith("--groups=") for v in a.split("=")[1].split(",")]; pre = "--pre" in sys.argv; xo = [int(v) for a in sys.argv if a.startswith("--exec-order=") for v in a.split("=")[1].split(",")]; shared = {}
files = [a for a in sys.argv[1:] if not a.startswith("--")]
cu = ctypes.CDLL("libcudart.so.12"); cu.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]; cu.cudaFree.argtypes = [ctypes.c_void_p]
cu.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]; cu.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]; cu.cudaGetErrorString.restype = ctypes.c_char_p
dr = ctypes.CDLL("libcuda.so.1"); V = ctypes.c_void_p
def ck(r, what):
    if r != 0: s = ctypes.c_char_p(); dr.cuGetErrorString(r, ctypes.byref(s)); raise SystemExit("%s -> %s" % (what, s.value))
cu.cudaFree(None)                                                          # primary context
dev = ctypes.c_int(); ck(dr.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet"); prim = V(); ck(dr.cuCtxGetCurrent(ctypes.byref(prim)), "cuCtxGetCurrent")
class Res(ctypes.Structure): _fields_ = [("type", ctypes.c_uint), ("pad", ctypes.c_ubyte * 92), ("smCount", ctypes.c_uint), ("rest", ctypes.c_ubyte * 44)]
allr = Res(); ck(dr.cuDeviceGetDevResource(dev, ctypes.byref(allr), 1), "cuDeviceGetDevResource"); print("device SMs", allr.smCount)
greens = {}; greens_g = {}
def green(c):
    if c in greens: return greens[c]
    if un:   # the tenant's way: 8-SM groups, a partition = union of c/8 groups (scattered like the union layout when --skip: every other group)
        n = ctypes.c_uint(0); rem = Res(); ck(dr.cuDevSmResourceSplitByCount(None, ctypes.byref(n), ctypes.byref(allr), ctypes.byref(rem), 0, 8), "count groups")
        groups = (Res * n.value)(); ck(dr.cuDevSmResourceSplitByCount(groups, ctypes.byref(n), ctypes.byref(allr), ctypes.byref(rem), 0, 8), "split groups")
        idx = list(range(n.value)); idx = (idx[1::2] + idx[0::2]) if sk else idx; idx = gsel if gsel else idx; sel = (Res * (c // 8))(*[groups[i] for i in idx[: c // 8]])
        desc = V(); ck(dr.cuDevResourceGenerateDesc(ctypes.byref(desc), sel, c // 8), "desc"); part = Res(); part.smCount = sum(groups[i].smCount for i in idx[: c // 8])
        print("groups: %d of %d SMs (remaining %d), partition %d uses groups %s" % (n.value, groups[0].smCount, rem.smCount, c, idx[: c // 8]))
    else:
        part = Res(); rem = Res(); n = ctypes.c_uint(1)
        ck(dr.cuDevSmResourceSplitByCount(ctypes.byref(part), ctypes.byref(n), ctypes.byref(allr), ctypes.byref(rem), 0, c), "split %d" % c)
        desc = V(); ck(dr.cuDevResourceGenerateDesc(ctypes.byref(desc), ctypes.byref(part), 1), "desc")
    g = V(); ck(dr.cuGreenCtxCreate(ctypes.byref(g), desc, dev, 1), "cuGreenCtxCreate"); s = V(); ctx = V(); ck(dr.cuCtxFromGreenCtx(ctypes.byref(ctx), g), "cuCtxFromGreenCtx")
    if rs:   # the tenant's way: a cudart stream created while the green context is current
        ck(dr.cuCtxSetCurrent(ctx), "set green"); cu.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]; cu.cudaStreamCreateWithFlags(ctypes.byref(s), 1); ck(dr.cuCtxSetCurrent(prim), "set prim")
    else: ck(dr.cuGreenCtxStreamCreate(ctypes.byref(s), g, 1, 0), "stream")
    greens[c] = (ctx, s, part.smCount); greens_g[c] = g; print("green ctx for class %d: %d SMs, ctx %s (primary %s)" % (c, part.smCount, hex(ctx.value), hex(prim.value))); return greens[c]
if pre:
    for f in files: green(int(re.search(r"_sm(\d+)", f).group(1)))
    print("all green contexts created up front")
logger = trt.Logger(trt.Logger.WARNING); rt = trt.Runtime(logger); ctxs = []
if sg:
    for f in files: e = rt.deserialize_cuda_engine(open(f, "rb").read()); n = e.num_io_tensors; del e
    print("signature pass done (deserialize + delete, primary context)")
for f in files:
    c = int(re.search(r"_sm(\d+)", f).group(1)); gctx, gs, nsm = green(c)
    if dg: ck(dr.cuCtxSetCurrent(gctx), "set green")
    eng = rt.deserialize_cuda_engine(open(f, "rb").read()); x = eng.create_execution_context(); bufs = []
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i); nb = int(np.prod(eng.get_tensor_shape(n))) * trt.nptype(eng.get_tensor_dtype(n))().itemsize
        key = (f.split("/")[-2], n)
        if sb and key in shared: p = shared[key]
        else:
            p = ctypes.c_void_p(); cu.cudaMalloc(ctypes.byref(p), nb); cu.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]; cu.cudaMemset(p, 0, nb); shared[key] = p
        x.set_tensor_address(n, p.value); bufs.append(p)
    if ax:   # the tenant's bind_aux: 8 streams created in the partition's green context, bound as TensorRT aux streams
        auxs = []
        for i in range(8): a = V(); ck(dr.cuGreenCtxStreamCreate(ctypes.byref(a), greens_g[c], 1, 0), "aux stream"); auxs.append(a.value)
        x.set_aux_streams(auxs[: max(eng.num_aux_streams, 0)] if eng.num_aux_streams > 0 else []); print("aux streams for", f.split("/")[-1], eng.num_aux_streams)
    ck(dr.cuCtxSetCurrent(prim), "set prim"); ctxs.append((f, eng, x, bufs, gctx, gs))
ps = ctypes.c_void_p(); cu.cudaStreamCreate(ctypes.byref(ps))
if xo: ctxs = [ctxs[i] for i in xo]
for f, eng, x, bufs, gctx, gs in ctxs:
    s = ps.value if pe else gs.value
    ok = x.execute_async_v3(s); e = cu.cudaStreamSynchronize(s); le = cu.cudaGetLastError()
    print("exec", f.split("/")[-2], f.split("/")[-1], "on", "primary stream" if pe else "green stream", "ok" if ok else "FAILED", "| sync", cu.cudaGetErrorString(e).decode(), "| last", cu.cudaGetErrorString(le).decode(), flush=True)
