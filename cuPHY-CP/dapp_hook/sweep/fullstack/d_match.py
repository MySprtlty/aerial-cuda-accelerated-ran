#!/usr/bin/env python3
"""d_match.py <phy.csv> <ru_slots.csv> <ru_late.csv> [out.csv]: pair every RU-emulator DL slot (ru_late.py output)
with the L1 per-slot row (parse_phylog.py output) by slot start time (same host clock; keys rounded to the
500 us slot grid, SFN mod 256 / slot checked: the RU frame id is 8 bits), then compare the L1 stage times of the slots whose DL U-plane arrived
late at the RU with the normal slots at the same TDD position. First second excluded. Numpy only."""
import sys, csv, collections
import numpy as np

phy, slots, late = sys.argv[1], sys.argv[2], sys.argv[3]
out = sys.argv[4] if len(sys.argv) > 4 else None

def read_csv(p):
    with open(p) as f:
        r = csv.reader(f); hdr = next(r); rows = list(r)
    return hdr, rows

def col(hdr, rows, name, dtype=float):
    i = hdr.index(name)
    return np.array([(dtype(x) if x != "" else np.nan) if dtype is float else dtype(x) for x in (row[i] for row in rows)])

# ---- RU: per-slot aggregate over cells ----
h, rows = read_csv(slots)
ru_t0 = col(h, rows, "t0_ns", int); ru_cell_max = col(h, rows, "max_ns", int); ru_late = col(h, rows, "late", int)
ru_frame = col(h, rows, "frame", int); ru_slot = col(h, rows, "slot", int)
tmin = ru_t0.min()
keep = ru_t0 > tmin + 1_000_000_000
ru_t0, ru_cell_max, ru_late, ru_frame, ru_slot = ru_t0[keep], ru_cell_max[keep], ru_late[keep], ru_frame[keep], ru_slot[keep]
key = np.rint(ru_t0 / 500_000).astype(np.int64)
agg = {}
for k, mx, lt, fr, sl in zip(key, ru_cell_max, ru_late, ru_frame, ru_slot):
    a = agg.get(k)
    if a is None:
        agg[k] = [mx, int(lt > 0), int(lt), 1, fr, sl]
    else:
        a[0] = max(a[0], mx); a[1] += int(lt > 0); a[2] += int(lt); a[3] += 1

# late symbol pattern per slot (from the _late table)
h2, lrows = read_csv(late)
pat = collections.defaultdict(lambda: np.zeros(14, dtype=np.int64))
for row in lrows:
    k = int(round(int(row[h2.index("t0_ns")]) / 500_000))
    pat[k] += np.array([int(row[h2.index("sym%d" % i)]) for i in range(14)])
def kind(k):
    s = pat.get(k)
    if s is None or s.sum() == 0: return ""
    nz = np.nonzero(s)[0]
    if nz[0] == 0 and nz[-1] == 13: return "all"
    if nz[0] == 0: return "head0-%d" % nz[-1]
    if nz[-1] == 13: return "tail%d-13" % nz[0]
    return "mid%d-%d" % (nz[0], nz[-1])

# ---- L1 rows ----
h, rows = read_csv(phy)
l_t0 = col(h, rows, "t0_ns", int); l_sfn = col(h, rows, "sfn", int); l_slot = col(h, rows, "slot", int); l_pos = col(h, rows, "pos", int)
l_dl = col(h, rows, "dl", int)
stages = ["enq_us", "pdsch_run_end", "pdsch_compl_end", "comp_wait_end", "cplane_end_max", "uprep_end_max", "utx_end_max"]
L = {c: col(h, rows, c) for c in stages}
L["utx_start_min"] = col(h, rows, "utx_start_min"); L["utx_start_min"][L["utx_start_min"] < -100000] = np.nan   # START: 0 for a cell
lkey = np.rint(l_t0 / 500_000).astype(np.int64)
sel = (l_t0 > tmin + 1_000_000_000) & (l_dl == 1)
idx = {k: i for i, k in enumerate(lkey) if sel[i]}
matched = [(k, idx[k]) for k in agg if k in idx]
ok = sum(1 for k, i in matched if l_sfn[i] % 256 == agg[k][4] and l_slot[i] == agg[k][5])   # eCPRI frameId is 8 bits
print("RU DL slots %d, L1 DL rows %d, paired %d, SFN/slot agree %d" % (len(agg), sel.sum(), len(matched), ok))

mk = np.array([k for k, i in matched]); mi = np.array([i for k, i in matched])
late_cells = np.array([agg[k][1] for k in mk]); late_pkts = np.array([agg[k][2] for k in mk]); ru_max = np.array([agg[k][0] for k in mk]) / 1e3
pos = l_pos[mi]
def q(v, p): v = v[~np.isnan(v)]; return np.nanpercentile(v, p) if len(v) else float("nan")
def mx(v): v = v[~np.isnan(v)]; return v.max() if len(v) else float("nan")
for p in (6, 7, 8, 9, 0, 1):
    s = pos == p; A = s & (late_cells == 0); B = s & (late_cells > 0)
    if B.sum() == 0: continue
    print("\n== pos %d: normal %d, late %d slots.  RU latest-packet offset (us): normal p50 %.0f p99.9 %.0f | late p50 %.0f p90 %.0f max %.0f" % (
        p, A.sum(), B.sum(), q(ru_max[A], 50), q(ru_max[A], 99.9), q(ru_max[B], 50), q(ru_max[B], 90), mx(ru_max[B])))
    print("   %-16s %8s %8s %8s | %8s %8s %8s   (us vs the slot's t0)" % ("stage", "n p50", "n p99.9", "n max", "late p50", "late p90", "late max"))
    for c in stages + ["utx_start_min"]:
        a = L[c][mi[A]]; b = L[c][mi[B]]
        print("   %-16s %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f" % (c, q(a, 50), q(a, 99.9), mx(a), q(b, 50), q(b, 90), mx(b)))
    kinds = collections.Counter(kind(k) for k in mk[B])
    print("   late-symbol pattern: " + ", ".join("%s:%d" % kv for kv in kinds.most_common(8)))
    # is the L1 U-plane tx end later than normal in the late slots?  (per-slot excess vs the position's p50)
    exc = L["utx_end_max"][mi[B]] - q(L["utx_end_max"][mi[A]], 50)
    print("   utx_end excess over normal p50 in late slots (us): p50 %.1f p90 %.1f max %.1f; share > +30 us: %.0f %%" % (q(exc, 50), q(exc, 90), mx(exc), 100 * np.nanmean(exc > 30)))

if out:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t0_ns", "sfn", "slot", "pos", "late_cells", "late_pkts", "ru_max_us", "pattern"] + stages + ["utx_start_min"])
        for k, i in matched:
            w.writerow([l_t0[i], l_sfn[i], l_slot[i], l_pos[i], agg[k][1], agg[k][2], agg[k][0] / 1e3, kind(k)] + [L[c][i] for c in stages] + [L["utx_start_min"][i]])
