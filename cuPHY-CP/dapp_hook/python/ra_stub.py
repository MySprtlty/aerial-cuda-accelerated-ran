#!/usr/bin/env python3
"""ra_stub.py (2026-11-02): stand-in for the Resource Allocator xApp (docs/XAPP_INTERFACE.md §2) and the load generator for
KPI runs. Subscribes to the per-slot estimates, picks a model for every request by a policy, sends the 4K JPEGs at a
given rate, logs every result to csv and prints the KPI summary (mean E2E / server latency per model, request-weighted
COCO mAP against the 26x baseline).

  ra_stub.py --images DIR --rate 4 --duration 60 --policy deadline --deadline-ms 40 --csv out.csv
  policies: fixed:26x | deadline (largest model whose est_ms <= deadline, else the smallest) | cycle (26n..26x round robin)
            | map (largest model whose est_ms <= deadline while the running request-weighted mAP stays >= --min-map)
"""
import argparse
import csv
import glob
import os
import random
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xapp_client import DappAgent  # noqa: E402

MODELS = ["26n", "26s", "26m", "26l", "26x"]
MAP = {"26n": 40.9, "26s": 48.6, "26m": 53.1, "26l": 55.0, "26x": 57.5}   # COCO mAP50-95 of the YOLO26 family (xjudge.py)

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=9810)
ap.add_argument("--images", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "img4k"))
ap.add_argument("--rate", type=float, default=4.0, help="requests per second (Poisson arrivals)")
ap.add_argument("--duration", type=float, default=60.0); ap.add_argument("--policy", default="deadline")
ap.add_argument("--deadline-ms", type=float, default=40.0, help="server-side budget used by the deadline/map policies")
ap.add_argument("--min-map", type=float, default=51.75, help="map policy: request-weighted mAP floor (0.9 x 26x)")
ap.add_argument("--csv"); ap.add_argument("--want-dets", action="store_true"); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--max-inflight", type=int, default=4)
a = ap.parse_args()

images = [p for p in sorted(glob.glob(os.path.join(a.images, "*.jpg"))) if "prog" not in os.path.basename(p)]   # NVJPG hardware: baseline JPEG only
if not images:
    sys.exit("no baseline .jpg images in " + a.images)
data = [open(p, "rb").read() for p in images]
random.seed(a.seed)

c = DappAgent(a.host, a.port, timeout=10.0)
est = {"msg": None}
def on_msg(m):
    if m.get("type") == "estimate":
        est["msg"] = m
print("welcome:", c.hello("ra", "ra_stub", ["estimate"], every=1, on_message=on_msg))

rows = []; lock = threading.Lock(); inflight = threading.Semaphore(a.max_inflight)
map_sum = [0.0, 0]   # running request-weighted mAP for the map policy
cycle = [0]

def choose():
    e = est["msg"]
    ests = (e or {}).get("est_ms", {}) if e else {}
    if a.policy.startswith("fixed:"):
        return a.policy.split(":", 1)[1], ests
    if a.policy == "cycle":
        m = MODELS[cycle[0] % len(MODELS)]; cycle[0] += 1; return m, ests
    if a.policy in ("deadline", "map"):
        cand = [m for m in reversed(MODELS) if ests.get(m, 1e9) <= a.deadline_ms]
        pick = cand[0] if cand else "26n"
        if a.policy == "map" and map_sum[1] > 0:
            # keep the running mean mAP above the floor: when the pick would drop below it, step up until it does not
            for m in MODELS[MODELS.index(pick):]:
                if (map_sum[0] + MAP[m]) / (map_sum[1] + 1) >= a.min_map:
                    pick = m; break
        return pick, ests
    return "26x", ests

def one(i, img_idx, t_arr):
    model, ests = choose()
    est_ms = (ests or {}).get(model)
    with inflight:
        t_send = time.time()
        r = c.run(data[img_idx], model=model, qos={"deadline_ms": a.deadline_ms}, want_dets=a.want_dets)
    with lock:
        map_sum[0] += MAP.get(model, 0.0); map_sum[1] += 1
        rows.append({"i": i, "t_arrival": t_arr, "t_send": t_send, "image": os.path.basename(images[img_idx]), "model": model, "est_ms": est_ms,
                     "status": r.get("status"), "reason": r.get("reason", ""), "sm": r.get("sm"), "sm_mean": r.get("sm_mean"), "wait_ms": r.get("wait_ms"),
                     "decode_ms": r.get("decode_ms"), "infer_ms": r.get("infer_ms"), "server_ms": r.get("latency_ms"), "e2e_ms": r.get("e2e_ms"),
                     "queue_wait_ms": (t_send - t_arr) * 1000.0, "detections": r.get("detections")})

threads = []; t0 = time.time(); i = 0
while time.time() - t0 < a.duration:
    gap = random.expovariate(a.rate) if a.rate > 0 else 1.0
    time.sleep(gap)
    th = threading.Thread(target=one, args=(i, random.randrange(len(data)), time.time())); th.start(); threads.append(th); i += 1
for th in threads:
    th.join()
c.close()

if a.csv:
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["i"]); w.writeheader(); w.writerows(rows)
ok = [r for r in rows if r["status"] == "success"]
print("requests %d, success %d, fail %d (%s)" % (len(rows), len(ok), len(rows) - len(ok), {r["reason"] for r in rows if r["status"] != "success"}))
if ok:
    e2e = [r["e2e_ms"] for r in ok]; srv = [r["server_ms"] for r in ok if r["server_ms"] is not None]
    print("E2E (client->result) mean %.1f p50 %.1f p90 %.1f max %.1f ms | server mean %.1f p50 %.1f | client queue wait mean %.1f ms" % (
        statistics.mean(e2e), statistics.median(e2e), sorted(e2e)[int(0.9 * (len(e2e) - 1))], max(e2e), statistics.mean(srv), statistics.median(srv), statistics.mean(r["queue_wait_ms"] for r in ok)))
    wmap = sum(MAP[r["model"]] for r in ok) / len(ok)
    print("models:", {m: sum(1 for r in ok if r["model"] == m) for m in MODELS}, "request-weighted mAP %.2f (26x 57.5, -%.1f %%)" % (wmap, 100 * (1 - wmap / 57.5)))
    for m in MODELS:
        rs = [r for r in ok if r["model"] == m]
        if rs:
            print("  %s: n %d server mean %.1f ms (decode %.1f infer %.1f wait %.1f) sm mean %.0f est mean %s" % (m, len(rs), statistics.mean(r["server_ms"] for r in rs), statistics.mean(r["decode_ms"] or 0 for r in rs),
                  statistics.mean(r["infer_ms"] or 0 for r in rs), statistics.mean(r["wait_ms"] or 0 for r in rs), statistics.mean(r["sm_mean"] or 0 for r in rs),
                  ("%.1f" % statistics.mean(r["est_ms"] for r in rs if r["est_ms"] is not None)) if any(r["est_ms"] is not None for r in rs) else "-"))
