#!/usr/bin/env python3
"""
openloop_worker: the tenant as an OPEN-LOOP request stream (fixed offered rate, FIFO, one request at a
time), for the request-rate -> latency curve of one engine on a given SM budget. Unlike yolo_worker.py
(closed loop, saturating), the arrival times are fixed in advance, so a slow server builds a queue
instead of generating fewer requests; the same schedule can therefore be offered to the static-MIG
baseline and to the shared GPU and the two compared fairly.

    python3 openloop_worker.py --engine yolo26x_1280_b1_fp16_atp100.engine --input bus_1280.npy \
        --rates 5,10,20,30,40,60,80 --seconds-per-rate 50 --log ol.csv --summary ol_summary.csv --ready /path/ready

Per request (ol.csv): rate, i, t_arrival_ns, queue_ms (arrival -> start), service_ms (H2D + network + D2H,
CUDA events), total_ms (arrival -> done, wall clock). Per rate (summary csv): offered, served, achieved rate,
backlog left when the stage ended, latency mean/p50/p99/max, queue mean/p99, service p50/p99, busy fraction.
Arrivals: --arrivals fixed (default, 1/rate apart) or poisson (exponential gaps, --seed). Service is batch 1
(one request = one image); dynamic batching is the dispatcher's job and is not modelled here.
Same launcher contract as yolo_worker.py (--engine/--input/--batch/--log/--ready/--seconds), so run_one.py
can start it in place of the closed-loop worker (tenant.worker / tenant.args in matrix.yaml).
"""
import argparse
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_worker import TrtRunner  # noqa: E402


def wait_until(t_target):
    """sleep, then spin, until time.perf_counter() >= t_target"""
    while True:
        d = t_target - time.perf_counter()
        if d <= 0:
            return
        if d > 0.0015:
            time.sleep(d - 0.001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--rates", default="5,10,20,30,40,60,80", help="offered request rates (req/s), run in this order")
    ap.add_argument("--seconds-per-rate", type=float, default=50.0)
    ap.add_argument("--arrivals", choices=["fixed", "poisson"], default="fixed")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log", required=True, help="per-request csv")
    ap.add_argument("--summary", default="", help="per-rate csv (default: <log>_summary.csv)")
    ap.add_argument("--seconds", type=float, default=0, help="overall budget from the launcher; stages beyond it are skipped")
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--ready", default="")
    ap.add_argument("--drain-s", type=float, default=2.0, help="idle gap between rates so the queue empties")
    a = ap.parse_args()

    cap = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100")
    rates = [float(x) for x in a.rates.split(",") if x]
    summary_path = a.summary or (os.path.splitext(a.log)[0] + "_summary.csv")
    r = TrtRunner(a.engine, a.batch, a.input)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))

    t0 = time.perf_counter()
    n_w = 0
    while time.perf_counter() - t0 < a.warmup and not stop:
        r.infer()
        n_w += 1
    if a.ready:
        with open(a.ready, "w") as f:
            f.write("%d\n" % os.getpid())
    print("openloop: engine %s in %s, cap=%s batch=%d, warm-up %d, rates %s x %.0f s, arrivals %s" % (
        os.path.basename(a.engine), "x".join(map(str, r.in_shape)), cap, a.batch, n_w, rates, a.seconds_per_rate, a.arrivals), flush=True)

    log = open(a.log, "w")
    log.write("rate,i,t_arrival_ns,queue_ms,service_ms,total_ms\n")
    summ = open(summary_path, "w")
    summ.write("rate,arrivals,served,achieved_rate,backlog_end,lat_mean_ms,lat_p50_ms,lat_p99_ms,lat_max_ms,queue_mean_ms,queue_p99_ms,service_p50_ms,service_p99_ms,busy_frac,cap_pct,batch,engine\n")
    rng = np.random.default_rng(a.seed)
    t_begin = time.perf_counter()
    for rate in rates:
        if stop or (a.seconds and time.perf_counter() - t_begin > a.seconds - a.seconds_per_rate):
            print("openloop: budget exhausted before rate %g" % rate, flush=True)
            break
        n = int(rate * a.seconds_per_rate)
        if a.arrivals == "fixed":
            gaps = np.full(n, 1.0 / rate)
        else:
            gaps = rng.exponential(1.0 / rate, n)
        arrivals = np.cumsum(gaps) - gaps[0]
        base = time.perf_counter() + 0.05
        t_end = base + a.seconds_per_rate
        q, s, tot = [], [], []
        busy = 0.0
        served = 0
        i = 0
        while i < n and not stop:
            ta = base + arrivals[i]
            if ta > t_end:          # arrivals after the stage window are not offered
                break
            wait_until(ta)
            now = time.perf_counter()
            if now > t_end:         # stage time is over: whatever is still queued is backlog
                break
            ms = r.infer()
            done = time.perf_counter()
            q.append((now - ta) * 1e3)
            s.append(ms)
            tot.append((done - ta) * 1e3)
            busy += done - now
            log.write("%g,%d,%d,%.3f,%.3f,%.3f\n" % (rate, i, int((ta - t_begin) * 1e9), q[-1], s[-1], tot[-1]))
            served += 1
            i += 1
        offered = int(np.searchsorted(arrivals, min(t_end, time.perf_counter()) - base, side="right"))
        backlog = max(0, offered - served)
        wall = max(1e-9, min(time.perf_counter(), t_end) - base)
        qa, sa, ta_ = np.array(q), np.array(s), np.array(tot)
        row = (rate, offered, served, served / wall, backlog,
               ta_.mean() if served else 0, np.percentile(ta_, 50) if served else 0, np.percentile(ta_, 99) if served else 0, ta_.max() if served else 0,
               qa.mean() if served else 0, np.percentile(qa, 99) if served else 0,
               np.percentile(sa, 50) if served else 0, np.percentile(sa, 99) if served else 0, busy / wall, cap, a.batch, os.path.basename(a.engine))
        summ.write("%g,%d,%d,%.2f,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%d,%s\n" % row)
        summ.flush()
        log.flush()
        print("openloop: rate %5.1f  offered %5d served %5d (%.1f/s) backlog %4d  lat mean %.2f p50 %.2f p99 %.2f max %.2f ms  queue mean %.2f  service p50 %.2f  busy %.2f" % (
            rate, offered, served, served / wall, backlog, row[5], row[6], row[7], row[8], row[9], row[11], row[13]), flush=True)
        # drain: let the queue empty before the next rate
        t_d = time.perf_counter()
        while time.perf_counter() - t_d < a.drain_s and not stop:
            time.sleep(0.01)
    log.close()
    summ.close()
    print("openloop: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
