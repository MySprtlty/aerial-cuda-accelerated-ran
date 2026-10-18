#!/usr/bin/env python3
"""jpeg_cpu_bench.py --images DIR [--glob 'uhd_*_q60.jpg'] [--threads 1,4,8] [--rounds 10] [--lib cv2|pil]
CPU JPEG decode of the demo 4K q60 frames (libjpeg-turbo through OpenCV, or Pillow): per-image decode ms p50 / p95 per thread
count (threads decode different images concurrently; a thread pool of N workers). Pin with taskset outside the L1/RU/testMAC cores,
e.g.  taskset -c 64-71 python3 jpeg_cpu_bench.py --images cuPHY-CP/dapp_hook/models/img4k"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def pct(v, p):
    s = sorted(v); return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--images", required=True); ap.add_argument("--glob", default="uhd_*_q60.jpg")
    ap.add_argument("--threads", default="1,4,8"); ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--lib", default="cv2")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.images, a.glob)))
    blobs = [open(f, "rb").read() for f in files]
    if not blobs:
        print("no images"); return 1
    if a.lib == "cv2":
        import cv2
        cv2.setNumThreads(1)
        def dec(b):
            t = time.perf_counter(); img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR); return (time.perf_counter() - t) * 1e3, img.shape
        info = "cv2 %s, libjpeg-turbo %s" % (cv2.__version__, "yes" if "libjpeg-turbo" in cv2.getBuildInformation() else "?")
    else:
        from PIL import Image
        import io
        def dec(b):
            t = time.perf_counter(); img = Image.open(io.BytesIO(b)); img.load(); return (time.perf_counter() - t) * 1e3, img.size
        import PIL; info = "Pillow %s" % PIL.__version__
    print("%d images, %.2f MB mean, %s, affinity %s" % (len(blobs), sum(map(len, blobs)) / len(blobs) / 1e6, info, sorted(os.sched_getaffinity(0))))
    print("| threads | decodes | per-image ms p50 / p95 / max | aggregate images/s |"); print("|---|---|---|---|")
    for n in [int(x) for x in a.threads.split(",")]:
        work = [blobs[i % len(blobs)] for i in range(a.rounds * len(blobs) * max(1, n))]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            res = list(ex.map(dec, work))
        el = time.perf_counter() - t0
        ms = [r[0] for r in res]
        print("| %d | %d | %.1f / %.1f / %.1f | %.1f |" % (n, len(ms), pct(ms, 50), pct(ms, 95), max(ms), len(ms) / el))
    return 0


if __name__ == "__main__":
    sys.exit(main())
