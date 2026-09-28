#!/usr/bin/env python3
"""
pick_minmax: from a no-tenant scan (profile c8_scan) pick, per channel, the TV with the
lowest and the highest measured pipeline latency, and write the two combined points

    c8_min = every channel at its lightest TV      c8_max = every channel at its heaviest TV

    python3 pick_minmax.py --scan runs/c8_scan [--stat mean|p99|p99_99] [--cells 8] [--out points/c8_minmax_points.yaml]

A run is a candidate for channel CH when its point changes CH's TV (or is the default
point). The latency is summary.json channels[CH].nsys[stat] of that run, i.e. the GPU time
of one slot execution of CH's pipeline (all cells grouped) with the other channels at
their defaults. The output is fed to `sweep.py --profile c8_minmax --points-file <out>`.
"""
import argparse
import glob
import json
import os
import sys

import yaml

CHANNELS = ["PUSCH", "PUCCH", "PRACH", "SRS", "PDSCH", "PDCCH"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--stat", default="mean", choices=["mean", "p50", "p99", "p99_99", "max"])
    ap.add_argument("--cells", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cand = {ch: [] for ch in CHANNELS}
    default_ids = {}
    for sp in sorted(glob.glob(os.path.join(a.scan, "*", "spec.yaml"))):
        rd = os.path.dirname(sp)
        st = os.path.join(rd, "status.yaml")
        sj = os.path.join(rd, "summary.json")
        if not (os.path.exists(st) and os.path.exists(sj)) or yaml.safe_load(open(st)).get("state") != "ok":
            continue
        spec = yaml.safe_load(open(sp))
        summ = json.load(open(sj))
        chans = spec["point"]["channels"]
        if spec.get("point_name") == "default":
            default_ids = dict(chans)
        for ch in CHANNELS:
            v = ((summ.get("channels") or {}).get(ch) or {}).get("nsys") or {}
            if chans.get(ch) and v.get(a.stat) is not None:
                cand[ch].append({"point": spec.get("point_name"), "tv": int(chans[ch]), "lat_us": float(v[a.stat]), "n": v.get("n")})
    lo, hi, table = {}, {}, {}
    for ch in CHANNELS:
        # only runs where CH differs from the default (plus the default itself) say something about CH
        rows = [c for c in cand[ch] if c["point"] == "default" or c["tv"] != default_ids.get(ch)]
        best = {}
        for c in rows:                       # one entry per TV (keep the first)
            best.setdefault(c["tv"], c)
        rows = sorted(best.values(), key=lambda c: c["lat_us"])
        if not rows:
            print("no candidate for %s" % ch)
            continue
        lo[ch], hi[ch] = rows[0]["tv"], rows[-1]["tv"]
        table[ch] = rows
        print("%-6s min %s (%s, %.1f us)   max %s (%s, %.1f us)   [%d candidates, stat=%s]" % (
            ch, rows[0]["tv"], rows[0]["point"], rows[0]["lat_us"], rows[-1]["tv"], rows[-1]["point"], rows[-1]["lat_us"], len(rows), a.stat))
    out = a.out or os.path.join(a.scan, "minmax_points.yaml")
    doc = {"generated_from": a.scan, "stat": a.stat,
           "points": {"c8_min": dict({"_cells": a.cells}, **lo), "c8_max": dict({"_cells": a.cells}, **hi)},
           "candidates": table}
    yaml.safe_dump(doc, open(out, "w"), sort_keys=False)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
