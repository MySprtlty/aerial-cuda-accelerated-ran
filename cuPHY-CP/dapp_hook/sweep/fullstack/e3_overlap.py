#!/usr/bin/env python3
"""e3_overlap.py --dir D --name NAME [--first-s 10] [--pre-us 470] [--post-us 500] [--window-us 294]
Hypothesis check (E3): do the RU emulator's late DL U-plane slots sit at TDD positions 6/7 (the DL slots right after the
two UL slots), and is their lateness tied to the PUSCH GPU work of the preceding UL slots (positions 4/5)?

Inputs (r0_ladder.sh outputs): ru_<name>_late.csv (ru_late.py: frame, slot, cell, t0_ns = slot start CLOCK_REALTIME,
late, max_ns, sym0..sym13) and phy_<name>.csv (parse_phylog.py: sfn, slot, t0_ns = slot air time, pusch_gpu_end /
pusch_gpu_run_us in us relative to t0).
For every DL slot the feature "PUSCH overrun" = latest PUSCH GPU end of the same frame's UL slots minus the DL slot's
transmit-window start (t0 - pre_us), in us; > 0 means PUSCH work was still running when the L1 started sending this slot.
Note: with the 59c timing the PUSCH windows of positions 4/5 span ~2.7 ms, so positions 3 and 6-9 always overlap; the
binary overlap is not informative, the overrun size and its difference between late and on-time slots is.
Output per TDD position: DL slots, late (slot, cell) rows, overrun p50 for on-time vs late slots, rank correlation between
overrun and lateness size (max_ns / 1e3 + window_us), lateness p50 / p99 / max, first late symbol histogram. numpy only."""
import argparse
import collections
import csv
import os

import numpy as np


def rank_corr(x, y):
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--first-s", type=float, default=10.0)
    ap.add_argument("--pre-us", type=float, default=470.0); ap.add_argument("--post-us", type=float, default=500.0)
    ap.add_argument("--window-us", type=float, default=294.0, help="dl_u_plane_timing_delay - dl_u_plane_window_size (us)")
    a = ap.parse_args()
    phy = list(csv.DictReader(open(os.path.join(a.dir, "phy_%s.csv" % a.name))))
    late_p = os.path.join(a.dir, "ru_%s_late.csv" % a.name)
    late = list(csv.DictReader(open(late_p))) if os.path.exists(late_p) else []
    if not phy:
        print("no phy rows"); return 1
    t0min = min(int(r["t0_ns"]) for r in phy)
    phy = [r for r in phy if int(r["t0_ns"]) - t0min >= a.first_s * 1e9]
    # PUSCH GPU end (absolute ns) of every UL slot, sorted by the UL slot's t0; for a DL slot the relevant PUSCH work is
    # that of the UL slots that started within the 3 ms before it (positions 4/5 of the same half frame)
    import bisect
    ul = sorted((int(r["t0_ns"]), int(r["t0_ns"]) + int(float(r["pusch_gpu_end"]) * 1e3)) for r in phy if r.get("pusch_gpu_end"))
    ul_t0 = [u[0] for u in ul]
    def overrun_us(t0):
        i = bisect.bisect_left(ul_t0, t0); j = bisect.bisect_left(ul_t0, t0 - 3_000_000)
        if i == j:
            return float("nan")
        return (max(u[1] for u in ul[j:i]) - (t0 - int(a.pre_us * 1e3))) / 1e3
    dl = {}      # (t0 rounded to 100 us, slot) -> (t0, pos, overrun_us)
    for r in phy:
        if not r.get("pdsch_gpu_run_us"):
            continue
        t0 = int(r["t0_ns"])
        dl[(t0 // 100_000, int(r["slot"]))] = (t0, int(r["pos"]), overrun_us(t0))
    pusch_end = ul
    late_rows = []
    for r in late:
        if "max_ns" not in r or int(r["t0_ns"]) - t0min < a.first_s * 1e9:
            continue
        t0 = int(r["t0_ns"]); slot = int(r["slot"]); k = t0 // 100_000
        d = dl.get((k, slot)) or dl.get((k - 1, slot)) or dl.get((k + 1, slot))
        syms = [int(r.get("sym%d" % i, 0) or 0) for i in range(14)]
        first = next((i for i, v in enumerate(syms) if v > 0), -1)
        late_rows.append((k, slot, int(r["cell"]), slot % 10, int(r["max_ns"]) / 1e3 + a.window_us, d[2] if d else float("nan"), first, int(r["late"])))
    late_keys = {(k, s) for k, s, *_ in late_rows}
    print("## E3 %s: DL slots %d, UL slots with PUSCH %d, late (slot, cell) rows %d (after %.0f s)" % (a.name, len(dl), len(pusch_end), len(late_rows), a.first_s))
    print("| pos | DL slots | late rows (slots) | PUSCH overrun into tx window, us: on-time p50 / late p50 | rank corr(overrun, lateness) | lateness us p50 / p99 / max | late pkts | first late symbol |")
    print("|---|---|---|---|---|---|---|---|")
    for p in range(10):
        slots = [(key, v) for key, v in dl.items() if v[1] == p]
        if not slots:
            continue
        ok = [v[2] for key, v in slots if key not in late_keys and v[2] == v[2]]
        lr = [x for x in late_rows if x[3] == p]
        if lr:
            lat = np.array([x[4] for x in lr]); ov = np.array([x[5] for x in lr]); m = ~np.isnan(ov)
            cnt = collections.Counter(x[6] for x in lr)
            print("| %d | %d | %d (%d) | %.0f / %.0f | %s | %.0f / %.0f / %.0f | %d | %s |" % (
                p, len(slots), len(lr), len({(x[0], x[1]) for x in lr}), np.median(ok) if ok else float("nan"), np.median(ov[m]) if m.any() else float("nan"),
                ("%.2f" % rank_corr(ov[m], lat[m])) if m.sum() >= 3 else "-", np.percentile(lat, 50), np.percentile(lat, 99), lat.max(), sum(x[7] for x in lr),
                " ".join("s%d:%d" % (s, n) for s, n in sorted(cnt.items()))))
        else:
            print("| %d | %d | 0 | %.0f / - | - | - | 0 | - |" % (p, len(slots), np.median(ok) if ok else float("nan")))
    if not late_rows:
        print("(no per-slot late rows: run without tag 503 INFO, or ru_late.py found none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
