#!/usr/bin/env python3
"""ru_late.py <ru.log> <out_prefix>: RU emulator RU.LATE_PACKETS (tag 503, INFO) -> per-slot/cell table of the DL
U-plane arrival window and the late-packet symbol histogram.

Input lines (ru_emulator packet_timings.cpp), two per cell and slot:
  ... [RU.LATE_PACKETS] F36S6S1 Cell 15 DL U Plane T0 <ns> num pkts 148 max -323672 min -343618 early 0, ontime 148,
      late 0, earliest packet <ns> sym 0, latest packet <ns> sym 5, duration 195324
  ... [RU.LATE_PACKETS] F36S6S1 Cell 15 DL U Plane T0 <ns> late packets Sym 0 0 Sym 1 0 ... Sym 13 0
F = frame (SFN), S = subframe, S = slot in subframe (mu 1: slot = subframe*2 + s). T0 = slot start, CLOCK_REALTIME ns.
max/min = packet arrival time minus the expected start of its symbol (ns); late = arrival after
-(dl_u_plane_timing_delay) + window.

Outputs
  <out_prefix>_slots.csv   one row per (frame, slot, cell): t0_ns, n_pkts, max_ns, min_ns, late, earliest/latest sym
  <out_prefix>_late.csv    rows with late > 0 plus the per-symbol late counts sym0..sym13
  stdout                   per-TDD-position summary and the symbol distribution of the late packets
"""
import sys, re, csv, collections
import numpy as np

src, out = sys.argv[1], sys.argv[2]
re_sum = re.compile(r"F(\d+)S(\d+)S(\d+) Cell (\d+) DL U Plane T0 (\d+) num pkts (\d+) max (-?\d+) min (-?\d+) early (\d+), ontime (\d+), late (\d+), earliest packet (\d+) sym (\d+), latest packet (\d+) sym (\d+), duration (\d+)")
re_sym = re.compile(r"F(\d+)S(\d+)S(\d+) Cell (\d+) DL U Plane T0 (\d+) late packets ((?:Sym \d+ \d+ ?)+)")

rows = []            # (frame, slot, cell, t0, n, max, min, late, esym, lsym)
late_rows = []       # (frame, slot, cell, t0, late, [sym counts])
pending = {}         # (frame, slot, cell) -> late count, for the symbol line that follows
n_lines = 0
with open(src, errors="replace") as f:
    for ln in f:
        if "DL U Plane T0" not in ln:
            continue
        n_lines += 1
        m = re_sum.search(ln)
        if m:
            fr, sf, sl, cell, t0, n, mx, mn, early, ontime, late, ep, es, lp, ls, dur = map(int, m.groups())
            slot = sf * 2 + sl
            rows.append((fr, slot, cell, t0, n, mx, mn, late, es, ls))
            if late > 0:
                pending[(fr, slot, cell)] = late
            continue
        if pending:
            m = re_sym.search(ln)
            if m:
                fr, sf, sl, cell = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                key = (fr, sf * 2 + sl, cell)
                if key in pending:
                    syms = [int(x) for x in re.findall(r"Sym \d+ (\d+)", m.group(6))]
                    late_rows.append((fr, key[1], cell, int(m.group(5)), pending.pop(key), syms))

with open(out + "_slots.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "slot", "cell", "t0_ns", "n_pkts", "max_ns", "min_ns", "late", "earliest_sym", "latest_sym"])
    w.writerows(rows)
with open(out + "_late.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "slot", "cell", "t0_ns", "late"] + ["sym%d" % i for i in range(14)])
    for fr, slot, cell, t0, late, syms in late_rows:
        w.writerow([fr, slot, cell, t0, late] + syms)

a = np.array([(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows], dtype=np.int64)
t0 = a[:, 3]
keep = t0 > t0.min() + 1_000_000_000          # first second excluded
a = a[keep]
pos = a[:, 1] % 10
print("lines %d, (slot, cell) rows %d after the first second, span %.1f s, cells %d" % (n_lines, len(a), (a[:, 3].max() - a[:, 3].min()) / 1e9, len(set(a[:, 2]))))
print("DL U-plane arrival vs symbol start (us), per TDD position: max = latest packet of the slot")
print("  pos  rows    late_rows  late_rows%%  late_pkts  max p50   max p99   max p99.9  max max    min p50")
for p in range(10):
    s = a[pos == p]
    if len(s) == 0:
        continue
    mx = s[:, 5] / 1e3
    print("  %d  %8d  %8d  %8.3f  %8d  %8.1f  %8.1f  %8.1f  %8.1f  %8.1f" % (p, len(s), (s[:, 7] > 0).sum(), 100 * (s[:, 7] > 0).mean(), s[:, 7].sum(),
          np.percentile(mx, 50), np.percentile(mx, 99), np.percentile(mx, 99.9), mx.max(), np.percentile(s[:, 6] / 1e3, 50)))
if late_rows:
    L = np.array([r[5] for r in late_rows if r[3] > t0.min() + 1_000_000_000])
    tot = L.sum(axis=0)
    print("late packets by symbol (all late rows after the first second): total %d" % tot.sum())
    print("  " + " ".join("s%d:%d" % (i, tot[i]) for i in range(14)))
    print("  share sym 0-1: %.1f %%, sym 2-13: %.1f %%" % (100 * tot[:2].sum() / max(tot.sum(), 1), 100 * tot[2:].sum() / max(tot.sum(), 1)))
    first_sym = np.array([next(i for i, v in enumerate(s) if v > 0) for s in L])
    print("  first late symbol of each late (slot, cell): " + " ".join("s%d:%d" % (i, (first_sym == i).sum()) for i in range(14) if (first_sym == i).sum()))
    # cells per late slot: a send-start delay hits all cells of the slot, a per-cell wire/pacing effect hits few
    by_slot = collections.Counter((r[0], r[1]) for r in late_rows if r[3] > t0.min() + 1_000_000_000)
    c = collections.Counter(by_slot.values())
    print("  cells late per late slot: " + " ".join("%d cells:%d slots" % (k, c[k]) for k in sorted(c)))
