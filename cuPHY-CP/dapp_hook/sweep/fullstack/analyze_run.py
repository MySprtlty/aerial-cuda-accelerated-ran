#!/usr/bin/env python3
"""analyze_run.py <name>: summarize one full-stack run in run61/ (L1 log events, testMAC errors, RU late table,
hook ring per-slot timing, DCGM). Read-only."""
import sys, os, re, collections, datetime
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
S = os.environ.get("FS_OUT", os.path.join(R, "prof", "run61"))
name = sys.argv[1]
def ts(s):
    h, m, r = s.split(":"); return int(h) * 3600 + int(m) * 60 + float(r)

# ---- L1 log ----
lines = open(os.path.join(S, "run2_%s.log" % name), errors="replace").read().splitlines()
t0 = None; ev = []
for ln in lines:
    m = re.match(r"(\d\d:\d\d:\d\d\.\d+) (\w+) (\S+) \d+ (.*)", ln)
    if not m: continue
    t, lvl, msg = ts(m.group(1)), m.group(2), m.group(4)
    if t0 is None and "tick_received" in msg: t0 = t; tick_str = m.group(1)
    if t0 is not None and lvl in ("WRN", "ERR"): ev.append((t - t0, lvl, msg))
print("=== L1 (%s): first tick %s, WRN/ERR after it: %d" % (name, tick_str if t0 else "?", len(ev)))
cl = []
for e in ev:
    if cl and e[0] - cl[-1][-1][0] < 0.05: cl[-1].append(e)
    else: cl.append([e])
for c in cl:
    kinds = collections.Counter(re.sub(r"0x[0-9a-fA-F]+|\d+", "#", x[2])[:70] for x in c)
    print("  cluster +%9.3f .. +%9.3f s  n=%d" % (c[0][0], c[-1][0], len(c)))
    for k, v in kinds.most_common(5): print("      x%-4d %s" % (v, k))
drops = [x for x in ev if "Dropping the slot" in x[2]]
print("  drops: %d  %s" % (len(drops), [(round(x[0], 3), re.search(r"sfn (\d+) slot (\d+)", x[2]).groups()) for x in drops]))
for key in ("PrePrepare", "waiting for order kernel", "sendCPlane Timing error", "L2+L2A processing taking"):
    v = [x for x in ev if key in x[2]]
    print("  %-28s %d  (after first second: %d)" % (key, len(v), sum(1 for x in v if x[0] > 1.0)))

# ---- testMAC ----
sec = collections.OrderedDict()
for ln in open(os.path.join(S, "run3_%s.log" % name), errors="replace"):
    m = re.match(r"(\d\d:\d\d:\d\d)\.\d+ CON \d+ \d+ \[MAC.FAPI\] Cell\s+(\d+) \| DL\s+([\d.]+) Mbps\s+(\d+) Slots \| UL\s+([\d.]+) Mbps\s+(\d+) Slots .*\| ERR\s+(\d+) ", ln)
    if m: sec.setdefault(m.group(1), collections.Counter())[int(m.group(2))] += int(m.group(7))
tot = sum(sum(c.values()) for c in sec.values()); first = sum(list(sec.values())[0].values()) if sec else 0
print("=== testMAC: %d stat seconds, Err.ind total %d, in the first second %d, later %d" % (len(sec), tot, first, tot - first))

# ---- RU emulator ----
txt = open(os.path.join(S, "run1_%s.log" % name), errors="replace").read().splitlines()
tab = {}
for ln in txt:
    m = re.search(r"\| AVG ONTIME U Slot\s+(\d+) %\s+\|\s+([\d.]+)%", ln)
    if m: tab[int(m.group(1))] = float(m.group(2))
def per(key):
    for ln in txt:
        if "| %s " % key in ln:
            return [int(x) for x in re.findall(r"\|\s+(\d+) ", ln.split(key)[1])]
    return []
u = per("DL U LATE SLOTS"); c = per("DL C LATE SLOTS"); on = per("DL U ON TIME SLOTS")
bad = {i: v for i, v in tab.items() if 0 < v < 100}
pos = collections.defaultdict(list)
for i, v in bad.items(): pos[i % 10].append(v)
print("=== RU: DL U late slots/cell %s (of %s on-time+late per cell -> %.3f %%), DL C late/cell %s" % (
    (min(u), max(u)) if u else "?", (on[0] + u[0]) if on and u else "?", 100.0 * np.mean(u) / (on[0] + u[0]) if on and u else -1, (min(c), max(c)) if c else "?"))
print("    below-100%% slot indices by TDD position: %s" % {p: "%d idx, worst %.2f%%" % (len(v), min(v)) for p, v in sorted(pos.items())})

# ---- ring ----
sys.path.insert(0, os.path.join(R, "cuPHY-CP/dapp_hook/estimator"))
from dapp_est import ringio as RI
path = os.path.join(R, "prof", "ring_16C_%s.bin" % name)
if os.path.exists(path):
    first_msg = {}; ends = []
    n = 0
    for rec in RI.RecordFile(path):
        n += 1
        k = (rec.sfn, rec.slot)
        if rec.type in (RI.REC_DL_TTI, RI.REC_UL_TTI):
            if k not in first_msg or rec.ts_ns < first_msg[k]: first_msg[k] = rec.ts_ns
        elif rec.type == RI.REC_SLOT_END:
            ends.append((k, rec.t0_ns, rec.tick_original_ns, rec.ts_ns, int(rec.enqueued), int(rec.is_ul), int(rec.is_dl)))
    ends.sort(key=lambda e: e[1])
    t_start = ends[0][1]
    body = [e for e in ends if e[1] - t_start > 1_000_000_000]      # statistics exclude the first second
    ndrop_all = sum(1 for e in ends if e[4] == 0); ndrop = sum(1 for e in body if e[4] == 0)
    print("=== ring: %d records, %d SLOT_END (%.1f s); not enqueued: %d total, %d after the first second" % (n, len(ends), (ends[-1][1] - t_start) / 1e9, ndrop_all, ndrop))
    if ndrop_all:
        print("    dropped slots: %s" % [(e[0], round((e[1] - t_start) / 1e9, 3)) for e in ends if e[4] == 0][:10])
    l2 = np.array([(first_msg[e[0]] - e[2]) / 1e3 for e in body if e[0] in first_msg])
    enq = np.array([(e[3] - e[2]) / 1e3 for e in body])
    print("    L2 latency (first request - tick), us: p50 %.0f p99 %.0f p99.9 %.0f max %.0f  | enqueue - tick: p50 %.0f p99 %.0f p99.9 %.0f max %.0f (budget 600)" % (
        np.percentile(l2, 50), np.percentile(l2, 99), np.percentile(l2, 99.9), l2.max(), np.percentile(enq, 50), np.percentile(enq, 99), np.percentile(enq, 99.9), enq.max()))
    bypos = collections.defaultdict(list)
    for e in body: bypos[e[0][1] % 10].append((e[3] - e[2]) / 1e3)
    print("    enqueue - tick by TDD position (p50/p99 us): %s" % {p: "%.0f/%.0f" % (np.percentile(v, 50), np.percentile(v, 99)) for p, v in sorted(bypos.items())})
else:
    print("=== ring: no dump at", path)

# ---- DCGM ----
f = os.path.join(S, "dcgm_%s.csv" % name)
if os.path.exists(f):
    hdr = None; rows = []
    for ln in open(f, errors="replace"):
        if ln.startswith("#Entity"): hdr = ln.replace("#", "").split()[1:]
        elif ln.startswith("GPU") and hdr:
            p = ln.split()[2:]
            if len(p) >= len(hdr):
                try: rows.append({hdr[i]: float(p[i]) for i in range(len(hdr))})
                except ValueError: pass
    if rows:
        sm = np.array([r.get("SMACT", 0) for r in rows]); print("=== DCGM: %d samples, SMACT mean %.2f p95 %.2f max %.2f, POWER mean %.0f W" % (len(rows), sm.mean(), np.percentile(sm, 95), sm.max(), np.mean([r.get("POWER", 0) for r in rows])))
