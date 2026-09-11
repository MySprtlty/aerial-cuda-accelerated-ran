#!/usr/bin/env python3
# Align nsys GPU-metrics samples (--gpu-metrics-devices, sqlite export) with the dApp ring's SLOT_END records
# and report SM activity per slot type relative to the slot boundary T0.
import sys, sqlite3, statistics as st
from collections import defaultdict
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import dapp_ring as dr

sq = sys.argv[1]
db = sqlite3.connect(sq)
epoch = db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
names = {m: n for _, m, n in db.execute("select typeId, metricId, metricName from TARGET_INFO_GPU_METRICS")}
want = {"SMs Active [Throughput %]": "sm_active", "GR Active [Throughput %]": "gr_active",
        "SM Issue [Throughput %]": "sm_issue", "Tensor Active [Throughput %]": "tensor",
        "Compute Warps in Flight [Avg]": "warps", "DRAM Read Bandwidth [Throughput %]": "dram_r",
        "DRAM Write Bandwidth [Throughput %]": "dram_w"}
ids = {m: want[n] for m, n in names.items() if n in want}
series = defaultdict(list)
for m, ts, v in db.execute("select metricId, timestamp, value from GPU_METRICS order by timestamp"):
    if m in ids:
        series[ids[m]].append((epoch + ts, v))
t = np.array([x[0] for x in series["sm_active"]], dtype=np.int64)
sm = np.array([x[1] for x in series["sm_active"]], dtype=np.float64)
gr = np.array([x[1] for x in series["gr_active"]], dtype=np.float64)
issue = np.array([x[1] for x in series["sm_issue"]], dtype=np.float64)
warps = np.array([x[1] for x in series["warps"]], dtype=np.float64)
dr_r = np.array([x[1] for x in series["dram_r"]], dtype=np.float64)
dr_w = np.array([x[1] for x in series["dram_w"]], dtype=np.float64)
import datetime
def hms(ns): return datetime.datetime.fromtimestamp(ns / 1e9).strftime("%H:%M:%S.%f")[:-3]
print("capture %s .. %s (%.1f s, %d samples)" % (hms(t[0]), hms(t[-1]), (t[-1] - t[0]) / 1e9, len(t)))

# ---- ring: slots in the window ----
ring = dr.DappRing("/aerial_dapp_ring")
recs = np.array(ring._recs); recs = recs[recs["seq"] > 0]; recs = recs[np.argsort(recs["seq"])]
slots = []   # (t0_ns, kind, prb_layers, tb)
ul_by_key = {}
buf = ring.buf
SE = dr.TYPE_DTYPES[dr.REC_SLOT_END]; UT = dr.TYPE_DTYPES[dr.REC_UL_TTI]
raw = np.frombuffer(buf, dtype=np.uint8, count=ring.ring_len * dr.REC_SIZE, offset=dr.HDR_SIZE)
se = np.frombuffer(raw.tobytes(), dtype=SE); ut = np.frombuffer(raw.tobytes(), dtype=UT)
for r in ut:
    if int(r["seq"]) > 0 and int(r["type"]) == dr.REC_UL_TTI:
        ul_by_key[(int(r["sfn"]), int(r["slot"]))] = (int(r["tot_pusch_prb_layers"]), int(r["tot_pusch_tb_bytes"]))
for r in se:
    if int(r["seq"]) > 0 and int(r["type"]) == dr.REC_SLOT_END:
        kind = ("UL" if int(r["is_ul"]) else "") + ("+" if int(r["is_ul"]) and int(r["is_dl"]) else "") + ("DL" if int(r["is_dl"]) else "")
        slots.append((int(r["t0_ns"]), kind, (int(r["sfn"]), int(r["slot"]))))
slots.sort()
in_win = [s for s in slots if t[0] + 3_000_000 < s[0] < t[-1] - 4_000_000]
print("ring slots inside capture: %d (%s .. %s)" % (len(in_win), hms(in_win[0][0]) if in_win else "-", hms(in_win[-1][0]) if in_win else "-"))

# ---- 1 s timeline ----
print("\n1 s timeline: mean SMs Active %, GR Active %, SM issue %, warps in flight, DRAM r/w %")
t_rel = (t - t[0]) / 1e9
for b in range(int(t_rel[-1]) + 1):
    m = (t_rel >= b) & (t_rel < b + 1)
    if m.sum() == 0: continue
    ns = [s for s in slots if t[0] + b * 10**9 <= s[0] < t[0] + (b + 1) * 10**9]
    print("  +%2ds %s  sm=%5.2f  gr=%5.2f  issue=%5.2f  warps=%6.2f  dram=%4.1f/%4.1f  slots_in_ring=%d" % (
        b, hms(t[0] + b * 10**9), sm[m].mean(), gr[m].mean(), issue[m].mean(), warps[m].mean(), dr_r[m].mean(), dr_w[m].mean(), len(ns)))

# ---- traffic window stats: samples that fall within a ring slot's [T0-1ms, T0+3ms] ----
if in_win:
    lo = in_win[0][0] - 1_000_000; hi = in_win[-1][0] + 3_000_000
    m = (t >= lo) & (t <= hi)
    v = sm[m]
    print("\nTraffic window %s .. %s: SMs Active mean %.2f%% (= %.1f of 132 SMs), median %.2f, p95 %.2f, p99 %.2f, max %.2f; samples>0: %.1f%%" % (
        hms(lo), hms(hi), v.mean(), v.mean() * 1.32, np.median(v), np.percentile(v, 95), np.percentile(v, 99), v.max(), 100.0 * (v > 0).mean()))
    print("   GR Active mean %.2f%%, SM issue mean %.2f%%, warps in flight mean %.2f, DRAM r/w mean %.2f/%.2f%%" % (
        gr[m].mean(), issue[m].mean(), warps[m].mean(), dr_r[m].mean(), dr_w[m].mean()))

    # ---- profile relative to T0 per slot kind ----
    bins = np.arange(-1500, 3501, 250)  # us
    prof = defaultdict(lambda: np.zeros(len(bins) - 1)); cnt = defaultdict(int)
    per_slot_mean = defaultdict(list)
    for t0, kind, key in in_win:
        idx0 = np.searchsorted(t, t0 - 1_500_000); idx1 = np.searchsorted(t, t0 + 3_500_000)
        tt = (t[idx0:idx1] - t0) / 1000.0; vv = sm[idx0:idx1]
        h, _ = np.histogram(tt, bins=bins, weights=vv); n, _ = np.histogram(tt, bins=bins)
        prof[kind] += np.where(n > 0, h / np.maximum(n, 1), 0); cnt[kind] += 1
        w = (tt >= -500) & (tt < 3000)
        per_slot_mean[kind].append(vv[w].mean() if w.any() else 0.0)
    print("\nSMs Active %% vs time relative to slot boundary T0 (250 us bins), averaged per slot kind:")
    print("   kind    n    " + " ".join("%5d" % b for b in bins[:-1]))
    for kind in sorted(prof):
        print("   %-6s %4d  " % (kind, cnt[kind]) + " ".join("%5.1f" % v for v in prof[kind] / max(cnt[kind], 1)))
    print("\nper-slot mean SMs Active over [T0-0.5ms, T0+3ms):")
    for kind in sorted(per_slot_mean):
        a = np.array(per_slot_mean[kind])
        print("   %-6s n=%4d  mean %.2f%%  median %.2f  p95 %.2f  max %.2f" % (kind, len(a), a.mean(), np.median(a), np.percentile(a, 95), a.max()))
