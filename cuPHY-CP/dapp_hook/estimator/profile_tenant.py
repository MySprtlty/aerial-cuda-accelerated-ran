#!/usr/bin/env python3
"""
profile_tenant: measure a tenant profile (profiles/<name>.yaml) for the estimator.

  dur_ms[batch][cap]  one inference in isolation under an SM cap, for cap in --caps
  sm_fill[batch][cap] fraction of the cap the tenant really keeps busy   (nsys, else "default")
  kernel_max_us       longest single kernel                               (nsys, else "default")

TensorRT (preferred): runs dapp_sched --sweep with one SM class per cap
    python3 profile_tenant.py --name yolov8n_b1 --batch 1 --engine models/yolov8n_fp16_sm16.engine \
        --image tools/bus.jpg --sched ../../../build.aarch64/cuPHY-CP/dapp_hook/dapp_sched --out profiles/yolov8n_b1.yaml
PyTorch fallback: runs ultralytics under CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=<cap> per cap
    python3 profile_tenant.py --name yolov8m_b8 --batch 8 --pytorch --weights yolov8m.pt \
        --python ~/.venvs/dapp_yolo/bin/python --out profiles/yolov8m_b8.yaml
--nsys wraps each cap run in nsys (CUDA trace of the tenant only; never the L1) and fills
sm_fill (time-fill proxy: kernel busy time / inference wall time) and kernel_max_us.
--from-sweep parses an existing dapp_sched sweep log instead of running anything.
Both need MPS to be up (CUDA_MPS_PIPE_DIRECTORY, e.g. /var).
"""
import argparse
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

DEFAULT_CAPS = (20, 40, 60, 80, 100)
DEFAULT_SM_TOTAL = 132
DEFAULT_FILL = {1: 0.5, 8: 1.0}
DEFAULT_KERNEL_MAX_US = 100.0
_SWEEP = re.compile(r"\[sweep\]\s+(?:SM=\s*(\d+)|(uncapped))\s+median=([0-9.]+) ms\s+min=([0-9.]+) ms\s+max=([0-9.]+) ms")

PYTORCH_SNIPPET = r"""
import sys, time, statistics, torch
from ultralytics import YOLO
w, res, batch, runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
m = YOLO(w)
x = torch.zeros(batch, 3, res, res, device="cuda")
for _ in range(5): m.model(x) if hasattr(m, "model") else m(x)
torch.cuda.synchronize()
ms = []
for _ in range(runs):
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.no_grad(): m.model(x)
    torch.cuda.synchronize(); ms.append((time.perf_counter() - t) * 1e3)
print("[pytorch] median=%.3f ms min=%.3f ms max=%.3f ms" % (statistics.median(ms), min(ms), max(ms)))
"""


def sm_for_cap(cap, sm_total):
    return max(1, int(round(cap / 100.0 * sm_total)))


def parse_sweep(text, sm_total):
    """-> {cap_pct: median_ms} from dapp_sched sweep lines."""
    out = {}
    for m in _SWEEP.finditer(text):
        med = float(m.group(3))
        if m.group(2):
            out[100] = med
        else:
            out[int(round(100.0 * int(m.group(1)) / sm_total))] = med
    return out


def run(cmd, env=None, cwd=None):
    p = subprocess.run(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError("command failed (%d): %s\n%s" % (p.returncode, " ".join(cmd), p.stdout[-2000:]))
    return p.stdout


def nsys_kernel_stats(sqlite_path):
    """(kernel_max_us, time_fill) from CUPTI_ACTIVITY_KIND_KERNEL of an nsys sqlite export."""
    db = sqlite3.connect(sqlite_path)
    rows = db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    db.close()
    if not rows:
        return None, None
    kmax = max(e - s for s, e in rows) / 1000.0
    busy = 0
    cur_s, cur_e = rows[0]
    for s, e in rows[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
    busy += cur_e - cur_s
    span = rows[-1][1] - rows[0][0]
    return kmax, (busy / span if span > 0 else None)


def with_nsys(cmd, workdir, tag):
    base = os.path.join(workdir, tag)
    out = run(["nsys", "profile", "-t", "cuda", "--cuda-graph-trace=node", "-o", base, "--force-overwrite", "true"] + cmd)
    run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", base + ".sqlite", base + ".nsys-rep"])
    kmax, fill = nsys_kernel_stats(base + ".sqlite")
    return out, kmax, fill


def measure_tensorrt(a, caps):
    env = dict(os.environ)
    dur = {}
    kmax_all = []
    fill = {}
    if a.nsys:
        work = tempfile.mkdtemp(prefix="dapp_prof_")
        for cap in caps:
            cmd = [a.sched, "--engine", a.engine, "--image", a.image, "--classes", str(sm_for_cap(cap, a.sm_total)),
                   "--sweep", str(a.runs)]
            out, kmax, tf = with_nsys(cmd, work, "cap%d" % cap)
            got = parse_sweep(out, a.sm_total)
            if got:
                dur[cap] = list(got.values())[0]
            if kmax:
                kmax_all.append(kmax)
            if tf is not None:
                fill[cap] = round(tf, 3)
        shutil.rmtree(work, ignore_errors=True)
    else:
        classes = ",".join(str(sm_for_cap(c, a.sm_total)) for c in caps)
        out = run([a.sched, "--engine", a.engine, "--image", a.image, "--classes", classes, "--sweep", str(a.runs)], env=env)
        got = parse_sweep(out, a.sm_total)
        for cap in caps:
            near = min(got, key=lambda k: abs(k - cap)) if got else None
            if near is not None and abs(near - cap) <= 2:
                dur[cap] = got[near]
    return dur, (max(kmax_all) if kmax_all else None), fill


def measure_pytorch(a, caps):
    dur = {}
    kmax_all = []
    fill = {}
    work = tempfile.mkdtemp(prefix="dapp_prof_")
    snippet = os.path.join(work, "bench.py")
    with open(snippet, "w") as f:
        f.write(PYTORCH_SNIPPET)
    for cap in caps:
        env = dict(os.environ)
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(cap)
        cmd = [a.python, snippet, a.weights, str(a.res), str(a.batch), str(a.runs)]
        if a.nsys:
            out, kmax, tf = with_nsys(cmd, work, "cap%d" % cap)
            if kmax:
                kmax_all.append(kmax)
            if tf is not None:
                fill[cap] = round(tf, 3)
        else:
            out = run(cmd, env=env)
        m = re.search(r"\[pytorch\] median=([0-9.]+) ms", out)
        if m:
            dur[cap] = float(m.group(1))
    shutil.rmtree(work, ignore_errors=True)
    return dur, (max(kmax_all) if kmax_all else None), fill


def write_profile(path, a, caps, dur, kmax, fill, source):
    b = a.batch
    dflt = DEFAULT_FILL.get(b, 1.0 if b >= 8 else 0.5)
    fill_status = "nsys_time_fill" if fill else "default"
    kmax_status = "nsys" if kmax else "default"
    lines = ["# Tenant profile written by profile_tenant.py on %s" % time.strftime("%Y-%m-%d %H:%M"),
             "# source: %s" % source,
             "name: %s" % a.name,
             "framework: %s" % ("pytorch" if a.pytorch else "tensorrt"),
             "input_res: %d" % a.res,
             "batches: [%d]" % b,
             "# dur_ms[batch][cap]: median of %d runs per cap. status: measured" % a.runs,
             "dur_ms:",
             "  %d: {%s}" % (b, ", ".join("%d: %.3f" % (c, dur[c]) for c in caps if c in dur)),
             "# sm_fill[batch][cap]: status: %s" % fill_status,
             "sm_fill:",
             "  %d: {%s}" % (b, ", ".join("%d: %.3f" % (c, fill.get(c, dflt)) for c in caps)),
             "kernel_max_us: %.1f        # status: %s" % (kmax if kmax else DEFAULT_KERNEL_MAX_US, kmax_status),
             "status: {dur_ms: measured, sm_fill: %s, kernel_max_us: %s}" % (fill_status, kmax_status), ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--caps", default=",".join(str(c) for c in DEFAULT_CAPS))
    ap.add_argument("--sm-total", type=int, default=DEFAULT_SM_TOTAL)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--sched", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "build.aarch64", "cuPHY-CP", "dapp_hook", "dapp_sched"))
    ap.add_argument("--pytorch", action="store_true")
    ap.add_argument("--weights", default="")
    ap.add_argument("--python", default=os.path.expanduser("~/.venvs/dapp_yolo/bin/python"))
    ap.add_argument("--nsys", action="store_true")
    ap.add_argument("--from-sweep", default="", help="parse this dapp_sched sweep log instead of running")
    a = ap.parse_args()
    caps = [int(c) for c in a.caps.split(",")]
    if a.nsys and not shutil.which("nsys"):
        print("nsys not found: sm_fill/kernel_max_us stay default")
        a.nsys = False
    if a.from_sweep:
        dur = parse_sweep(open(a.from_sweep).read(), a.sm_total)
        dur = {c: dur[min(dur, key=lambda k: abs(k - c))] for c in caps if dur}
        kmax, fill, source = None, {}, "sweep log %s" % a.from_sweep
    elif a.pytorch:
        if not a.weights:
            ap.error("--pytorch needs --weights")
        dur, kmax, fill = measure_pytorch(a, caps)
        source = "pytorch %s batch %d" % (a.weights, a.batch)
    else:
        if not (a.engine and a.image):
            ap.error("TensorRT mode needs --engine and --image (or --pytorch / --from-sweep)")
        dur, kmax, fill = measure_tensorrt(a, caps)
        source = "dapp_sched --sweep %s" % a.engine
    missing = [c for c in caps if c not in dur]
    if missing:
        print("WARNING: no measurement for caps %s" % missing)
    write_profile(a.out, a, caps, dur, kmax, fill, source)
    print("wrote %s: dur_ms=%s kernel_max_us=%s sm_fill=%s" % (a.out, dur, kmax or "default", fill or "default"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
