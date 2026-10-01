#!/usr/bin/env python3
"""e_analysis.py: hook cost on the L2-adapter path. For each run name given, parse the L2A.PROCESSING_TIMES lines of
prof/run61/phy_<name>.log (one per slot) and the console log for drops. First second excluded."""
import sys, os, re, collections
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
R = os.path.join(REPO, "prof", "run61")
S = os.environ.get("FS_OUT", R)
def num(s, key, end=" "):
    i = s.find(key)
    if i < 0: return None
    i += len(key); j = s.find(end, i)
    return int(s[i:j] if j > 0 else s[i:].strip())
def q(v, p): return np.percentile(v, p)
res = {}
for name in sys.argv[1:]:
    ticks, enq, fapi, l2est, pos, l2a_start, l2a_end = [], [], [], [], [], [], []
    with open(os.path.join(R, "phy_%s.log" % name), errors="replace") as f:
        for ln in f:
            if "[L2A.PROCESSING_TIMES]" not in ln: continue
            t = num(ln, "l1_slot_ind_tick="); e = num(ln, "l1_enqueue_complete_time="); s0 = num(ln, "l2a_start_time="); s1 = num(ln, "l2a_end_time=")
            if t is None or e is None: continue
            i = ln.find("SFN "); j = ln.find(" ", i + 4); slot = int(ln[i + 4:j].split(".")[1])
            ticks.append(t); enq.append(e - t); fapi.append(s1 - s0); l2est.append(num(ln, "L2 Estimate=", "ns") or 0); pos.append(slot % 10); l2a_start.append(s0 - t); l2a_end.append(s1 - t)
    ticks = np.array(ticks); enq = np.array(enq) / 1e3; fapi = np.array(fapi) / 1e3; l2est = np.array(l2est) / 1e3; pos = np.array(pos); l2a_start = np.array(l2a_start) / 1e3; l2a_end = np.array(l2a_end) / 1e3
    keep = ticks - ticks.min() > 1_000_000_000
    enq, fapi, l2est, pos, l2a_start, l2a_end = enq[keep], fapi[keep], l2est[keep], pos[keep], l2a_start[keep], l2a_end[keep]
    con = open(os.path.join(S, "run2_%s.log" % name), errors="replace").read()
    drops = re.findall(r"(\d\d:\d\d:\d\d\.\d+) WRN .*Dropping the slot command for sfn (\d+) slot (\d+)", con)
    first_tick = re.search(r"(\d\d:\d\d:\d\d\.\d+) CON timer_thread .*tick_received", con)
    def ts(s): h, m, r = s.split(":"); return int(h) * 3600 + int(m) * 60 + float(r)
    t0 = ts(first_tick.group(1)) if first_tick else 0
    late_drops = [(d[1], d[2], round(ts(d[0]) - t0, 3)) for d in drops if ts(d[0]) - t0 > 1.0]
    res[name] = dict(n=len(enq), enq=enq, fapi=fapi, l2est=l2est, pos=pos, l2a_start=l2a_start, l2a_end=l2a_end, drops_all=len(drops), drops_late=late_drops)
    print("=== %s: %d slots after the first second; drops total %d, after the first second %d %s" % (name, len(enq), len(drops), len(late_drops), late_drops))
    print("    enqueue - tick (us):      p50 %6.1f  p99 %6.1f  p99.9 %6.1f  max %6.1f   share > 600 us: %.4f %%" % (q(enq, 50), q(enq, 99), q(enq, 99.9), enq.max(), 100 * (enq > 600).mean()))
    print("    L2A end - tick (us):      p50 %6.1f  p99 %6.1f  p99.9 %6.1f  max %6.1f   share > 600 us: %.4f %%  (drop criterion, check_time_threshold)" % (q(l2a_end, 50), q(l2a_end, 99), q(l2a_end, 99.9), l2a_end.max(), 100 * (l2a_end > 600).mean()))
    print("    L2A start - tick (us):    p50 %6.1f  p99 %6.1f  p99.9 %6.1f  max %6.1f   (first FAPI message seen by the adapter)" % (q(l2a_start, 50), q(l2a_start, 99), q(l2a_start, 99.9), l2a_start.max()))
    print("    last FAPI msg - tick (us): p50 %6.1f  p99 %6.1f  p99.9 %6.1f  max %6.1f   (L2A receive time of the slot's last message)" % (q(l2est, 50), q(l2est, 99), q(l2est, 99.9), l2est.max()))
    print("    FAPI proc dur (us):       p50 %6.1f  p99 %6.1f  p99.9 %6.1f  max %6.1f   (L2A parse + hook, overlaps arrivals)" % (q(fapi, 50), q(fapi, 99), q(fapi, 99.9), fapi.max()))
    print("    enqueue - tick by position p50/p99: %s" % {p: "%.0f/%.0f" % (q(enq[pos == p], 50), q(enq[pos == p], 99)) for p in range(10) if (pos == p).any()})
    print("    L2A end - tick by position p50/p99/p99.9: %s" % {p: "%.0f/%.0f/%.0f" % (q(l2a_end[pos == p], 50), q(l2a_end[pos == p], 99), q(l2a_end[pos == p], 99.9)) for p in range(10) if (pos == p).any()})
if len(res) >= 2:
    names = list(res)
    print("\n=== differences vs '%s' (us): enqueue-tick p50 / p99 / p99.9 / max ; FAPI proc dur p50 / p99 ; L2A end-tick p50 / p99 / p99.9 / max" % names[-1])
    ref = res[names[-1]]
    for n in names[:-1]:
        r = res[n]
        print("  %-6s %+6.1f / %+6.1f / %+6.1f / %+6.1f ; %+6.1f / %+6.1f ; %+6.1f / %+6.1f / %+6.1f / %+6.1f" % (n, q(r["enq"], 50) - q(ref["enq"], 50), q(r["enq"], 99) - q(ref["enq"], 99), q(r["enq"], 99.9) - q(ref["enq"], 99.9), r["enq"].max() - ref["enq"].max(), q(r["fapi"], 50) - q(ref["fapi"], 50), q(r["fapi"], 99) - q(ref["fapi"], 99), q(r["l2a_end"], 50) - q(ref["l2a_end"], 50), q(r["l2a_end"], 99) - q(ref["l2a_end"], 99), q(r["l2a_end"], 99.9) - q(ref["l2a_end"], 99.9), r["l2a_end"].max() - ref["l2a_end"].max()))
        for p in (3, 4, 5, 6):
            a, b = r["l2a_end"][r["pos"] == p], ref["l2a_end"][ref["pos"] == p]
            print("         pos %d: L2A end-tick p50 %+6.1f p99 %+6.1f p99.9 %+6.1f" % (p, q(a, 50) - q(b, 50), q(a, 99) - q(b, 99), q(a, 99.9) - q(b, 99.9)))
