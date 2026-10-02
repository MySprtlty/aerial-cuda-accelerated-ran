#!/usr/bin/env python3
"""cpu_report.py <run names...>: E margin, D late share, per-core load and thread placement for the CPU-cause runs
(cpu_runs.sh). Reads prof/run61/{l2a_<name>.log, run1_<name>.log, run2_<name>.log, mpstat_<name>.txt,
threads_affinity_<name>.txt}. First second (E) / first 60 s of mpstat (testMAC start-up) excluded. Numpy only."""
import sys, os, re, collections
import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
S = os.environ.get("FS_OUT", os.path.join(R, "prof", "run61"))
OWNER = {}
for c in (4,): OWNER[c] = "L1 low/dpdk"
OWNER[5] = "L1 h2d"; OWNER[6] = OWNER[7] = "L1 UL worker"; OWNER[8] = OWNER[9] = OWNER[10] = "L1 DL worker"
OWNER[11] = "L2A timer"; OWNER[12] = "L2A msg (hook)"; OWNER[13] = "L1 debug"; OWNER[14] = "L2A pcap"; OWNER[19] = "L1 ul pcap"
OWNER[15] = "testMAC low"; OWNER[16] = "testMAC sched"; OWNER[17] = "testMAC recv"; OWNER[18] = "testMAC builder"
for c in range(20, 31): OWNER[c] = "RU UL"
for c in range(31, 41): OWNER[c] = "RU DL"
OWNER[41] = "RU low"; OWNER[42] = "RU dpdk"
OWNER[43] = "dApp export (async)"
def owner(c, ru2=False):
    if ru2:
        if 20 <= c <= 41: return "RU UL (x2)"
        if 42 <= c <= 61: return "RU DL (x2)"
        if c == 62: return "RU low (x2)"
        if c == 63: return "RU dpdk (x2)"
    return OWNER.get(c, "housekeeping" if c < 4 or c > 64 else "free isolated")

def num(s, key, end=" "):
    i = s.find(key)
    if i < 0: return None
    i += len(key); j = s.find(end, i)
    return int(s[i:j] if j > 0 else s[i:].strip())
def q(v, p): return np.percentile(v, p)

def e_stats(name):
    src = os.path.join(S, "phy_%s.log" % name)
    if not os.path.exists(src): src = os.path.join(S, "l2a_%s.log" % name)
    ticks, l2a_end, enq, last, pos = [], [], [], [], []
    with open(src, errors="replace") as f:
        for ln in f:
            if "[L2A.PROCESSING_TIMES]" not in ln: continue
            t = num(ln, "l1_slot_ind_tick="); e = num(ln, "l1_enqueue_complete_time="); s1 = num(ln, "l2a_end_time="); lf = num(ln, "last_fapi_msg_tick=")
            if t is None or e is None: continue
            i = ln.find("SFN "); j = ln.find(" ", i + 4); slot = int(ln[i + 4:j].split(".")[1])
            ticks.append(t); l2a_end.append(s1 - t); enq.append(e - t); last.append(lf - t); pos.append(slot % 10)
    ticks = np.array(ticks); keep = ticks - ticks.min() > 1_000_000_000
    l2a_end = np.array(l2a_end)[keep] / 1e3; enq = np.array(enq)[keep] / 1e3; last = np.array(last)[keep] / 1e3; pos = np.array(pos)[keep]
    con = open(os.path.join(S, "run2_%s.log" % name), errors="replace").read()
    drops = re.findall(r"(\d\d:\d\d:\d\d\.\d+) WRN .*Dropping the slot command for sfn (\d+) slot (\d+)", con)
    first_tick = re.search(r"(\d\d:\d\d:\d\d\.\d+) CON timer_thread .*tick_received", con)
    def ts(s): h, m, r = s.split(":"); return int(h) * 3600 + int(m) * 60 + float(r)
    t0 = ts(first_tick.group(1)) if first_tick else 0
    late_drops = [(d[1], d[2], round(ts(d[0]) - t0, 1)) for d in drops if ts(d[0]) - t0 > 1.0]
    return dict(n=len(l2a_end), l2a_end=l2a_end, enq=enq, last=last, pos=pos, drops=late_drops, drops_all=len(drops))

def d_stats(name):
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
    u = per("DL U LATE SLOTS"); on = per("DL U ON TIME SLOTS"); c = per("DL C LATE SLOTS")
    bypos = collections.defaultdict(list)
    for i, v in tab.items():
        if v > 0: bypos[i % 10].append(100.0 - v)          # UL positions report 0 % on-time (no DL U-plane) -> skipped
    return dict(late=u, ontime=on, clate=c, bypos={p: float(np.mean(v)) for p, v in bypos.items()}, nslot_idx=len(tab))

def mpstat_stats(name, skip_s=60, tail_s=10):
    p = os.path.join(S, "mpstat_%s.txt" % name)
    if not os.path.exists(p): return None
    per = collections.defaultdict(list); ci = None
    for ln in open(p, errors="replace"):
        f = ln.split()
        if "CPU" in f: ci = f.index("CPU"); continue          # header: "HH:MM:SS [AM|PM] CPU %usr ... %idle"
        if ci is None or len(f) <= ci + 1 or not f[ci].isdigit(): continue
        try: idle = float(f[-1])
        except ValueError: continue
        c = int(f[ci]); per[c].append(100.0 - idle)
    if not per: return None
    n = min(len(v) for v in per.values())
    res = {}
    for c, v in per.items():
        v = np.array(v[skip_s:max(skip_s + 1, n - tail_s)])
        res[c] = (v.mean(), q(v, 90), v.max(), len(v))
    return res

def threads(name):
    p = os.path.join(S, "threads_affinity_%s.txt" % name)
    if not os.path.exists(p): return None
    out = collections.defaultdict(list)
    for ln in open(p, errors="replace"):
        f = ln.split()
        if len(f) < 5: continue
        proc, tid, comm = f[0], f[1], f[2]
        allowed = f[3].split("=")[1]; last = f[4].split("=")[1]
        out[proc].append((comm, allowed, int(last)))
    return out

names = sys.argv[1:]
print("== E (L2A path; us after the slot-indication tick; drop line = 600)")
print("  %-9s %7s  %-34s %-34s %-24s %s" % ("run", "slots", "L2A end-tick p50/p99/p99.9/max", "enqueue-tick p50/p99/p99.9/max", "margin 600-p99.9 / -max", "drops>1s ; last FAPI msg-tick p50/p99.9"))
E = {}
for n in names:
    e = e_stats(n); E[n] = e
    print("  %-9s %7d  %6.1f/%6.1f/%6.1f/%6.1f           %6.1f/%6.1f/%6.1f/%6.1f           %6.1f / %6.1f           %d %s ; %5.1f/%5.1f" % (
        n, e["n"], q(e["l2a_end"], 50), q(e["l2a_end"], 99), q(e["l2a_end"], 99.9), e["l2a_end"].max(),
        q(e["enq"], 50), q(e["enq"], 99), q(e["enq"], 99.9), e["enq"].max(), 600 - q(e["l2a_end"], 99.9), 600 - e["l2a_end"].max(),
        len(e["drops"]), e["drops"][:3], q(e["last"], 50), q(e["last"], 99.9)))
print("  L2A end-tick by TDD position (p50/p99.9):")
for n in names:
    e = E[n]
    print("    %-9s %s" % (n, "  ".join("%d:%.0f/%.0f" % (p, q(e["l2a_end"][e["pos"] == p], 50), q(e["l2a_end"][e["pos"] == p], 99.9)) for p in range(10) if (e["pos"] == p).any())))
print("\n== D (RU emulator DL U-plane late slots, from the RU PACKET_SUMMARY; per cell over the whole run)")
print("  %-9s %-28s %-10s %s" % ("run", "late slots/cell min..max", "late %", "late % by TDD position (100 - avg on-time)"))
for n in names:
    d = d_stats(n)
    tot = (d["ontime"][0] + d["late"][0]) if d["ontime"] and d["late"] else 0
    print("  %-9s %-28s %-10s %s" % (n, "%d..%d" % (min(d["late"]), max(d["late"])) if d["late"] else "?", "%.3f" % (100.0 * np.mean(d["late"]) / tot) if tot else "?",
          "  ".join("%d:%.3f" % (p, v) for p, v in sorted(d["bypos"].items()))))
print("\n== per-core load (mpstat, % busy = 100 - idle, traffic window; cores >= 50 % mean listed, >= 90 % marked *)")
for n in names:
    m = mpstat_stats(n)
    if m is None: print("  %-9s no mpstat" % n); continue
    ru2 = n.endswith("ru2")
    hot = sorted((c for c in m if m[c][0] >= 50), key=lambda c: -m[c][0])
    print("  %-9s samples/core %d; cores >= 90 %% mean: %s" % (n, m[hot[0]][3] if hot else 0, ", ".join("%d(%s)" % (c, owner(c, ru2)) for c in hot if m[c][0] >= 90) or "none"))
    for c in hot:
        print("     core %2d %-16s mean %5.1f  p90 %5.1f  max %5.1f %s" % (c, owner(c, ru2), m[c][0], m[c][1], m[c][2], "*" if m[c][0] >= 90 else ""))
    cold = sorted(c for c in m if m[c][0] < 50 and 4 <= c <= 64 and owner(c, ru2) not in ("free isolated",))
    print("     isolated cores assigned but < 50 %%: %s" % ", ".join("%d(%s %.0f%%)" % (c, owner(c, ru2), m[c][0]) for c in cold))
print("\n== thread placement snapshot (allowed cores of every thread; '!' = not pinned to one core, 'H' = housekeeping core)")
for n in names:
    t = threads(n)
    if t is None: print("  %-9s no snapshot" % n); continue
    for proc, lst in t.items():
        single = collections.Counter(a for _, a, _ in lst if "-" not in a and "," not in a)
        multi = [(c, a, l) for c, a, l in lst if "-" in a or "," in a]
        print("  %-9s %-12s %3d threads; pinned cores: %s" % (n, proc, len(lst), " ".join("%s x%d" % (a, k) for a, k in sorted(single.items(), key=lambda x: int(x[0])))))
        if multi:
            print("     ! unpinned (%d): %s" % (len(multi), ", ".join("%s[%s last %d%s]" % (c, a, l, " H" if (l < 4 or l > 64) else "") for c, a, l in multi[:14])))
