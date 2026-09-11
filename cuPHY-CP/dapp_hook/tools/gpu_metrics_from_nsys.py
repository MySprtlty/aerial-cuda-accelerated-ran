#!/usr/bin/env python3
"""
Summarise the GPU metrics sampled by Nsight Systems (--gpu-metrics-devices)
from the sqlite export of a report:

    nsys export --type sqlite -o l1.sqlite l1.nsys-rep
    python3 gpu_metrics_from_nsys.py l1.sqlite [--slot-us 500]

Prints, for every sampled metric (SMs Active, SM Warp Occupancy, Tensor
Active, DRAM bandwidth, ...), the mean / median / p95 / max over the capture
and the distribution per slot-sized bin. "SMs Active" is the share of SMs
with at least one resident warp, i.e. how much of the GPU the RAN actually
occupies; it is not affected by GPU-side busy-waiting kernels the way
nvidia-smi's utilization.gpu is.
"""
import argparse
import sqlite3
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--slot-us", type=float, default=500.0)
    ap.add_argument("--from", dest="t_from", default="", help="only samples after this UTC time, HH:MM:SS[.f] of the capture day (e.g. testMAC's first stats line)")
    ap.add_argument("--to", dest="t_to", default="", help="only samples before this UTC time (e.g. testMAC's 'Finished running' line)")
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    if "GPU_METRICS" not in tables or "TARGET_INFO_GPU_METRICS" not in tables:
        print("no GPU_METRICS tables in this export; profile with --gpu-metrics-devices=0")
        print("tables:", ", ".join(sorted(tables)))
        return 1
    names = {}
    for typeId, metricId, name in db.execute("select typeId, metricId, metricName from TARGET_INFO_GPU_METRICS"):
        names[(typeId, metricId)] = name
    series = defaultdict(list)
    for typeId, metricId, ts, val in db.execute("select typeId, metricId, timestamp, value from GPU_METRICS order by timestamp"):
        series[names.get((typeId, metricId), "%s/%s" % (typeId, metricId))].append((ts, val))
    if not series:
        print("GPU_METRICS is empty")
        return 1
    if a.t_from or a.t_to:
        import calendar
        epoch = db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
        day = db.execute("select utcTime from TARGET_INFO_SESSION_START_TIME").fetchone()[0][:10]
        y, mo, d = (int(x) for x in day.split("-"))
        def abs_ns(hms):
            h, m, sec = hms.split(":")
            return int((calendar.timegm((y, mo, d, int(h), int(m), 0, 0, 0, 0)) + float(sec)) * 1e9) - epoch
        lo = abs_ns(a.t_from) if a.t_from else -1 << 62
        hi = abs_ns(a.t_to) if a.t_to else 1 << 62
        series = {k: [(t, v) for t, v in s if lo <= t <= hi] for k, s in series.items()}
        series = {k: s for k, s in series.items() if s}
        if not series:
            print("no samples inside --from/--to")
            return 1
    ts_all = [t for s in series.values() for t, _ in s]
    t0, t1 = min(ts_all), max(ts_all)
    print("capture window %.3f s, %d metrics, %d samples each" % ((t1 - t0) / 1e9, len(series), len(next(iter(series.values())))))
    slot_ns = int(a.slot_us * 1000)
    for name in sorted(series):
        vals = [v for _, v in series[name]]
        sv = sorted(vals)
        n = len(sv)
        q = lambda p: sv[min(n - 1, int(p * n))]
        # per-slot mean of the samples in each slot bin
        bins = defaultdict(list)
        for t, v in series[name]:
            bins[(t - t0) // slot_ns].append(v)
        slot_means = sorted(sum(b) / len(b) for b in bins.values())
        m = len(slot_means)
        qs = lambda p: slot_means[min(m - 1, int(p * m))]
        print("%-34s mean %6.2f  med %6.2f  p95 %6.2f  max %6.2f | per-slot mean: med %6.2f p90 %6.2f max %6.2f" % (
            name[:34], sum(vals) / n, q(0.5), q(0.95), sv[-1], qs(0.5), qs(0.9), slot_means[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
