#!/usr/bin/env python3
"""
GPU busy time of an L1 run from an Nsight Systems trace.

    nsys stats --report cuda_gpu_trace --format csv -o trace  l1.nsys-rep
    python3 gpu_busy_from_nsys.py trace_cuda_gpu_trace.csv [--slot-us 500]

Reports
  * the fraction of the captured window during which at least one kernel or
    copy was running on the GPU (union of all intervals across streams), which
    is what "GPU utilization" means once concurrent streams are folded;
  * the same per slot (500 us bins at mu=1) as a distribution, which is the
    per-slot GPU time a load predictor would be trained on;
  * time per cuPHY channel, grouped by kernel name (sum of durations, so
    concurrent kernels count twice; the union column does not).
"""
import argparse
import csv
import re
import sys
from collections import defaultdict

GROUPS = [
    ("PUSCH",  re.compile(r"pusch|ldpc.*dec|chest|ch_est|equal|noise|derateM|uci_on|softDemap|cfoTa|rsrp|srsChest", re.I)),
    ("PDSCH",  re.compile(r"pdsch|ldpc.*enc|modulation|dmrs.*tx|rate_?match|crc.*enc|prepareCrc|scramble|precod", re.I)),
    ("PDCCH",  re.compile(r"pdcch|polar.*enc|dci", re.I)),
    ("PUCCH",  re.compile(r"pucch", re.I)),
    ("PRACH",  re.compile(r"prach", re.I)),
    ("SRS",    re.compile(r"srs", re.I)),
    ("CSI-RS", re.compile(r"csirs|csi_rs", re.I)),
    ("SSB",    re.compile(r"ssb|pbch", re.I)),
    ("BFW",    re.compile(r"bfw|beamform", re.I)),
    ("copy",   re.compile(r"memcpy|memset|\[CUDA ", re.I)),
]


def group_of(name):
    for g, rx in GROUPS:
        if rx.search(name):
            return g
    return "other"


def union_busy(intervals):
    """Total length of the union of [start, end) intervals (sorted by start)."""
    total = 0
    cur_s, cur_e = None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--slot-us", type=float, default=500.0)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    rows = []
    with open(a.csv, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                s = int(r["Start (ns)"])
                d = int(r["Duration (ns)"])
            except (KeyError, ValueError):
                continue
            rows.append((s, s + d, r.get("Name", "?")))
    if not rows:
        print("no GPU rows found (expected columns 'Start (ns)', 'Duration (ns)', 'Name')")
        return 1
    rows.sort()
    t0 = rows[0][0]
    t1 = max(e for _, e, _ in rows)
    window = t1 - t0

    busy = union_busy([(s, e) for s, e, _ in rows])
    print("capture window %.3f s, %d GPU operations" % (window / 1e9, len(rows)))
    print("GPU busy (union over streams): %.3f s = %.1f%% of the window" % (busy / 1e9, 100.0 * busy / window))

    # per slot bins
    slot_ns = int(a.slot_us * 1000)
    nbins = window // slot_ns + 1
    per_bin = defaultdict(list)
    for s, e, _ in rows:
        b = (s - t0) // slot_ns
        while b < nbins and (t0 + b * slot_ns) < e:
            bs = t0 + b * slot_ns
            be = bs + slot_ns
            per_bin[b].append((max(s, bs), min(e, be)))
            b += 1
    busy_us = []
    for b in range(nbins):
        iv = sorted(per_bin.get(b, []))
        busy_us.append(union_busy(iv) / 1000.0)
    busy_us_sorted = sorted(busy_us)
    n = len(busy_us_sorted)
    pct = lambda p: busy_us_sorted[min(n - 1, int(p * n))]
    print("per-%dus slot GPU busy: mean %.0f us  median %.0f  p90 %.0f  p99 %.0f  max %.0f  (%d slots, %d idle)" % (
        a.slot_us, sum(busy_us) / n, pct(0.5), pct(0.9), pct(0.99), busy_us_sorted[-1], n,
        sum(1 for v in busy_us if v == 0)))
    hist = defaultdict(int)
    for v in busy_us:
        hist[int(min(100, 100.0 * v / a.slot_us) // 10) * 10] += 1
    print("slot busy histogram (%% of slot -> slots): " + ", ".join(
        "%d-%d%%: %d" % (k, k + 10, hist[k]) for k in sorted(hist)))

    # per channel group
    by_group = defaultdict(lambda: [0, 0])
    by_name = defaultdict(lambda: [0, 0])
    for s, e, name in rows:
        g = group_of(name)
        by_group[g][0] += e - s
        by_group[g][1] += 1
        by_name[name][0] += e - s
        by_name[name][1] += 1
    print("\ntime per channel group (sum of durations; concurrent work counts more than once):")
    for g, (t, c) in sorted(by_group.items(), key=lambda kv: -kv[1][0]):
        print("  %-7s %8.3f s  %6.1f%% of window  %7d ops  avg %6.1f us" % (g, t / 1e9, 100.0 * t / window, c, t / c / 1000.0))
    print("\ntop kernels:")
    for name, (t, c) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print("  %6.1f%%  %7d x %7.1f us  %s" % (100.0 * t / window, c, t / c / 1000.0, name[:90]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
