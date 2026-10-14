#!/usr/bin/env python3
"""
profile_tenant: measure a tenant profile (profiles/<name>.yaml) for the estimator.

  dur_ms[batch][cap]  one inference in isolation under an SM cap, for cap in --caps   (dapp_sched --sweep, GPU time
                      h2d + network + d2h of the engine stream; the nvjpeg decode of the tiled path is reported
                      separately as dec_ms, it runs on the NVJPG engines, not on SMs)
  sm_fill[batch][cap] fraction of the cap the tenant really keeps busy   (nsys time-fill proxy, else "default")
  kernel_max_us       longest single kernel                               (nsys, else "default")

TensorRT (preferred): runs dapp_sched --sweep, one SM class per cap. A TensorRT engine only runs near the SM
count it was built for (README "Engines and SM classes"), so give the class of every cap and its engine:
    python3 profile_tenant.py --name yolo26x_b8 --batch 8 --res 1280 --ctrl-name 26x \\
        --caps 43,61,100 --classes 56,80,132 \\
        --engine-sm "56=../models/yolo26x_1280_b8_fp16_sm56.engine,80=...sm80.engine,132=...sm132.engine" \\
        --image ../models/img4k/uhd_0_q85.jpg --nsys --out profiles/yolo26x_b8.yaml
    (--engine FILE instead of --engine-sm loads that one file in every class; without --classes the class of a
     cap is round(cap/100 * sm_total), which only works when an engine exists for that count)
PyTorch fallback: runs ultralytics under CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=<cap> per cap
    python3 profile_tenant.py --name yolov8m_b8 --batch 8 --pytorch --weights yolov8m.pt \\
        --python ~/.venvs/dapp_yolo/bin/python --out profiles/yolov8m_b8.yaml
--nsys wraps one extra sweep per cap in nsys (CUDA trace of the tenant only; never the L1) and fills
sm_fill (time-fill proxy: kernel busy time / span of the traced kernels) and kernel_max_us; dur_ms comes
from the plain sweep. --from-sweep parses an existing dapp_sched sweep log instead of running anything.
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

DEFAULT_CAPS = (43, 61, 100)
DEFAULT_SM_TOTAL = 132
DEFAULT_FILL = {1: 0.5, 8: 1.0}
DEFAULT_KERNEL_MAX_US = 100.0
_SWEEP = re.compile(r"\[sweep\]\s+(?:SM=\s*(\d+)|(uncapped))\s+median=([0-9.]+) ms\s+min=([0-9.]+) ms\s+max=([0-9.]+) ms"
                    r"(?:.*?\bdec=([0-9.]+))?")

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


def parse_sweep(text, sm_to_cap):
    """-> ({cap_pct: median_ms}, {cap_pct: dec_ms}) from dapp_sched sweep lines. sm_to_cap maps the SM class of a
    line to the cap it stands for (an uncapped line is cap 100)."""
    dur, dec = {}, {}
    for m in _SWEEP.finditer(text):
        cap = 100 if m.group(2) else sm_to_cap.get(int(m.group(1)))
        if cap is None:
            continue
        dur[cap] = float(m.group(3))
        if m.group(6):
            dec[cap] = float(m.group(6))
    return dur, dec


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


def engine_args(a, sms):
    """dapp_sched --engine NAME:SM=file arguments: one file for every class (--engine) or per-class files (--engine-sm)."""
    if a.engine_sm:
        return sum((["--engine", "prof:%s" % kv.strip()] for kv in a.engine_sm.split(",") if kv.strip()), [])
    return sum((["--engine", "prof:%d=%s" % (sm, a.engine)] for sm in sms), [])


def sched_cmd(a, sms):
    return [a.sched] + engine_args(a, sms) + ["--image", a.image, "--classes", ",".join(map(str, sms)),
                                             "--sweep", str(a.runs), "--sweep-skip", "2"]


def measure_tensorrt(a, caps, classes):
    env = dict(os.environ)
    sm_to_cap = dict(zip(classes, caps))
    # dur_ms: one plain sweep over every class (no profiler in the way)
    out = run(sched_cmd(a, classes), env=env)
    dur, dec = parse_sweep(out, sm_to_cap)
    kmax_all = []
    fill = {}
    if a.nsys:
        work = tempfile.mkdtemp(prefix="dapp_prof_")
        for cap, sm in zip(caps, classes):
            _, kmax, tf = with_nsys(sched_cmd(a, [sm]), work, "cap%d" % cap)
            if kmax:
                kmax_all.append(kmax)
            if tf is not None:
                fill[cap] = round(tf, 3)
        shutil.rmtree(work, ignore_errors=True)
    return dur, dec, (max(kmax_all) if kmax_all else None), fill


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
    return dur, {}, (max(kmax_all) if kmax_all else None), fill


def write_profile(path, a, caps, dur, dec, kmax, fill, source):
    b = a.batch
    dflt = DEFAULT_FILL.get(b, 1.0 if b >= 8 else 0.5)
    fill_status = "nsys_time_fill" if fill else "default"
    kmax_status = "nsys" if kmax else "default"
    lines = ["# Tenant profile written by profile_tenant.py on %s" % time.strftime("%Y-%m-%d %H:%M"),
             "# source: %s" % source,
             "name: %s" % a.name,
             "ctrl_name: %s" % (a.ctrl_name or a.name[:8]),
             "framework: %s" % ("pytorch" if a.pytorch else "tensorrt"),
             "input_res: %d" % a.res,
             "batches: [%d]" % b,
             "# dur_ms[batch][cap]: median GPU time (h2d + network + d2h) of %d runs per cap, first 2 excluded. status: measured" % a.runs,
             "dur_ms:",
             "  %d: {%s}" % (b, ", ".join("%d: %.3f" % (c, dur[c]) for c in caps if c in dur))]
    if dec:
        lines += ["# dec_ms[batch][cap]: nvjpeg decode + tiles of the tiled path (NVJPG engines, cap independent), not in dur_ms",
                  "dec_ms:",
                  "  %d: {%s}" % (b, ", ".join("%d: %.3f" % (c, dec[c]) for c in caps if c in dec))]
    lines += ["# sm_fill[batch][cap]: status: %s" % fill_status,
              "sm_fill:",
              "  %d: {%s}" % (b, ", ".join("%d: %.3f" % (c, fill.get(c, dflt)) for c in caps)),
              "kernel_max_us: %.1f        # status: %s" % (kmax if kmax else DEFAULT_KERNEL_MAX_US, kmax_status),
              "status: {dur_ms: measured, sm_fill: %s, kernel_max_us: %s}" % (fill_status, kmax_status), ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ctrl-name", default="", help="name in the control block / RUN model=NAME (max 8 chars; default: name)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--caps", default=",".join(str(c) for c in DEFAULT_CAPS))
    ap.add_argument("--classes", default="", help="SM class per cap, same order as --caps (default round(cap/100*sm_total))")
    ap.add_argument("--sm-total", type=int, default=DEFAULT_SM_TOTAL)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="", help="one engine file, loaded in every class")
    ap.add_argument("--engine-sm", default="", help="per-class engines 'SM=file,SM=file,...' (instead of --engine)")
    ap.add_argument("--image", default="")
    ap.add_argument("--sched", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "build.aarch64", "cuPHY-CP", "dapp_hook", "dapp_sched"))
    ap.add_argument("--pytorch", action="store_true")
    ap.add_argument("--weights", default="")
    ap.add_argument("--python", default=os.path.expanduser("~/.venvs/dapp_yolo/bin/python"))
    ap.add_argument("--nsys", action="store_true")
    ap.add_argument("--from-sweep", default="", help="parse this dapp_sched sweep log instead of running")
    a = ap.parse_args()
    caps = [int(c) for c in a.caps.split(",")]
    classes = [int(c) for c in a.classes.split(",")] if a.classes else [sm_for_cap(c, a.sm_total) for c in caps]
    if len(classes) != len(caps):
        ap.error("--classes must list one SM class per cap")
    if a.nsys and not shutil.which("nsys"):
        print("nsys not found: sm_fill/kernel_max_us stay default")
        a.nsys = False
    dec = {}
    if a.from_sweep:
        dur, dec = parse_sweep(open(a.from_sweep).read(), dict(zip(classes, caps)))
        kmax, fill, source = None, {}, "sweep log %s" % a.from_sweep
    elif a.pytorch:
        if not a.weights:
            ap.error("--pytorch needs --weights")
        dur, dec, kmax, fill = measure_pytorch(a, caps)
        source = "pytorch %s batch %d" % (a.weights, a.batch)
    else:
        if not ((a.engine or a.engine_sm) and a.image):
            ap.error("TensorRT mode needs --engine (or --engine-sm) and --image (or --pytorch / --from-sweep)")
        dur, dec, kmax, fill = measure_tensorrt(a, caps, classes)
        source = "dapp_sched --sweep %s, caps %s = SM classes %s, image %s" % (
            a.engine_sm or a.engine, caps, classes, os.path.basename(a.image))
    missing = [c for c in caps if c not in dur]
    if missing:
        print("WARNING: no measurement for caps %s" % missing)
    write_profile(a.out, a, caps, dur, dec, kmax, fill, source)
    print("wrote %s: dur_ms=%s dec_ms=%s kernel_max_us=%s sm_fill=%s" % (a.out, dur, dec or "-", kmax or "default", fill or "default"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
