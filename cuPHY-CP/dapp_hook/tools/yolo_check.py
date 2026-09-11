#!/usr/bin/env python3
"""
Compare the detections dapp_sched prints (its "--sweep" or "--verbose" output)
with the ultralytics reference written by models/export_yolo.py.

    dapp_sched --engine yolov8n_fp16.engine --image bus.jpg --sweep 5 \
        | python3 yolo_check.py models/reference_detections.json bus.jpg

A detection matches when the class is the same, the box IoU is >= 0.7 and the
confidence differs by < 0.1. FP16 and a slightly different resize make small
differences unavoidable; a wrong class or a missing object is what this is
meant to catch.
"""
import json
import re
import sys


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ref = json.load(open(sys.argv[1]))[sys.argv[2]]
    pat = re.compile(r"^\s+(\S.*?)\s+(\d\.\d\d)\s+\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]\s*$")
    got = []
    for line in sys.stdin:
        m = pat.match(line)
        if m:
            got.append({"name": m.group(1), "conf": float(m.group(2)),
                        "xyxy": [float(m.group(i)) for i in range(3, 7)]})
        if got and not m and line.startswith("[sweep] SM="):
            break  # only the first block of detections is printed
    if not got:
        print("no detections found in input")
        return 1

    unmatched = list(ref)
    bad = 0
    for g in got:
        best, best_iou = None, 0.0
        for r in unmatched:
            if r["name"] != g["name"]:
                continue
            v = iou(g["xyxy"], r["xyxy"])
            if v > best_iou:
                best, best_iou = r, v
        ok = best is not None and best_iou >= 0.7 and abs(best["conf"] - g["conf"]) < 0.1
        print("%-5s %-12s conf %.2f vs %s  iou %.2f" % (
            "ok" if ok else "DIFF", g["name"], g["conf"],
            ("%.2f" % best["conf"]) if best else "-", best_iou))
        if ok:
            unmatched.remove(best)
        else:
            bad += 1
    for r in unmatched:
        print("MISS  %-12s conf %.2f (reference only)" % (r["name"], r["conf"]))
    bad += len(unmatched)
    print("YOLO CHECK %s: %d detections, %d reference, %d differences" % (
        "PASS" if bad == 0 else "FAIL", len(got), len(ref), bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
