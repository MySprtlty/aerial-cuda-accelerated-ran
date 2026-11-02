#!/usr/bin/env python3
"""tn_phases.py <name> [<name> ...] [--dir prof/tenant]: per-phase view of a slow-alternation run (tn_alt.sh): the Peak and
Average phases are told apart by the PUSCH GPU run duration of the U slots in phy_<name>.csv (bimodal: Peak vectors run
~1.5x longer), smoothed over 40-slot windows; the tenant csv (dapp_tenant_<name>.csv) is aligned by its plan_ul_sm column
(44 = Average, 76/100 = Peak) where a tenant ran. Reports, per phase: cuPHY PUSCH completion p99 per position, DL late share
at positions 0/1/2, PUSCH profiles chosen, and the tenant's tiles per 5 ms TDD cycle (= 4K requests per second x 8 / 200)."""
import argparse
import csv
import os

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def q(a, p):
    return np.percentile(a, p) if len(a) else float("nan")


def phases_from_phy(rows):
    """phase per row: 'peak' / 'avg' from the PUSCH GPU run of U slots (threshold = midpoint of the two modes)."""
    runs = np.array([float(r["pusch_gpu_run_us"]) if r.get("pusch_gpu_run_us") not in (None, "", "None", "nan") and int(r["pos"]) in (7, 8, 9) else np.nan for r in rows])
    v = runs[~np.isnan(runs)]
    if len(v) < 100:
        return ["?"] * len(rows), float("nan")
    lo, hi = np.percentile(v, 10), np.percentile(v, 90)
    thr = (lo + hi) / 2.0
    # carry the last U-slot classification forward, smoothed over a 40-slot window (two frames)
    cls = np.full(len(rows), np.nan)
    for i, x in enumerate(runs):
        if not np.isnan(x):
            cls[i] = 1.0 if x > thr else 0.0
    out = []
    last = 0.0
    win = []
    for i in range(len(rows)):
        if not np.isnan(cls[i]):
            win.append(cls[i])
            win = win[-12:]
            last = 1.0 if np.mean(win) > 0.5 else 0.0
        out.append("peak" if last else "avg")
    return out, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--dir", default=os.path.join(R, "prof", "tenant"))
    a = ap.parse_args()
    for n in a.names:
        print("== %s" % n)
        phy = os.path.join(a.dir, "phy_%s.csv" % n)
        if os.path.exists(phy):
            rows = list(csv.DictReader(open(phy)))
            ph, thr = phases_from_phy(rows)
            print("  phases from PUSCH GPU run (threshold %.0f us): %d peak rows, %d avg rows" % (thr, ph.count("peak"), ph.count("avg")))
            for name in ("peak", "avg"):
                sel = [r for r, p in zip(rows, ph) if p == name]
                if not sel:
                    continue
                line = "  %-4s cuPHY:" % name
                for pos in (7, 8, 9):
                    v = np.array([float(r["pusch_gpu_end"]) for r in sel if int(r["pos"]) == pos and r["pusch_gpu_end"] not in ("", "None", "nan")])
                    line += " PUSCH%d p99 %5.0f (>2100: %d)" % (pos, q(v, 99), int((v > 2100).sum()) if len(v) else 0)
                for pos in (0, 1, 2):
                    v = np.array([float(r["comp_wait_end"]) for r in sel if int(r["pos"]) == pos and r["comp_wait_end"] not in ("", "None", "nan")])
                    line += " | DL%d late %.2f%%" % (pos, 100.0 * (v > -470).mean() if len(v) else float("nan"))
                print(line)
        tc = os.path.join(a.dir, "dapp_tenant_%s.csv" % n)
        if os.path.exists(tc):
            rows = [r for r in csv.DictReader(open(tc)) if r.get("class") not in (None, "") and r.get("plan_ul_sm") not in (None, "")]
            if rows and "plan_ul_sm" in rows[0]:
                # phase of each wall slot: the last known U-slot plan (44 -> avg, >= 60 -> peak); -1 = unknown
                phase, last = [], "?"
                for r in rows:
                    u = int(r["plan_ul_sm"])
                    if u >= 76:
                        last = "peak"
                    elif u in (44, 60):
                        last = "avg"
                    phase.append(last)
                for name in ("peak", "avg"):
                    sel = [r for r, p in zip(rows, phase) if p == name]
                    if not sel:
                        continue
                    segs = np.array([float(r["segments"]) for r in sel]); cls = np.array([float(r["class"]) for r in sel])
                    sets = {}
                    for r in sel:
                        sets[r["set"]] = sets.get(r["set"], 0) + int(r["segments"])
                    # tiles per 5 ms cycle: quiet b8 segments / 30 segments x 8 tiles, busy b1 segments / 7 x 1 tile (segment counts per set)
                    b8 = sets.get("8", 0); b1 = sets.get("1", 0)
                    tiles = b8 / 30.0 * 8 + b1 / 7.0
                    cycles = len(sel) / 10.0
                    print("  %-4s tenant: %d slots, mean class %.1f SMs, segments/slot %.2f, b8 segs %d, b1 segs %d -> %.2f tiles per 5 ms cycle = %.1f 4K/s" % (name, len(sel), cls.mean(), segs.mean(), b8, b1, tiles / cycles if cycles else 0, (tiles / cycles * 200 / 8) if cycles else 0))
            else:
                print("  tenant csv without plan columns (old runner)")
        tl = os.path.join(a.dir, "dapp_tenant_%s.log" % n)
        if os.path.exists(tl):
            lines = [ln for ln in open(tl, errors="replace") if ln.startswith("[")]
            if lines:
                print("  tenant last window: " + lines[-1][:400].rstrip())


if __name__ == "__main__":
    main()
