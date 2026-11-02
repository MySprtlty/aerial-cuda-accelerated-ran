#!/usr/bin/env python3
"""tn_summary.py <name> [--dir prof/tenant]: summary of one one-process tenant run (fullstack_run.sh ... TENANT=1):
  L1 health (Err.ind / drops / profile fallbacks from the kept log excerpts), cuPHY timing per TDD position from
  phy_<name>.csv (parse_phylog.py: PUSCH completion after t0, PDSCH compression done, PDCCH end), RU DL late packets, and the
  tenant's own statistics (dapp_tenant_<name>.log last line, per-slot csv: class per TDD position, pauses)."""
import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
F = os.path.join(R, "cuPHY-CP", "dapp_hook", "sweep", "fullstack")


def q(a, p):
    return np.percentile(a, p) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--dir", default=os.path.join(R, "prof", "tenant"))
    a = ap.parse_args()
    O = a.dir
    n = a.name
    print("== %s (%s)" % (n, O))
    con = os.path.join(O, "console_%s.log" % n)
    if os.path.exists(con):
        for ln in open(con, errors="replace"):
            if re.search(r"^== |L1 env|L1 did not|Dropping|Finished running|profile fallbacks|dropped", ln):
                print("  " + ln.rstrip()[:200])
    dl = os.path.join(O, "dapp_%s.log" % n)
    if os.path.exists(dl):
        lines = open(dl, errors="replace").read().splitlines()
        for ln in lines:
            if re.search(r"green context with|profile .* pipeline objects|DL context|sizer|profiles", ln) and "requested" not in ln:
                print("  L1: " + re.sub(r"^\S+ \S+ \S+ ", "", ln)[:200])
        fb = [ln for ln in lines if "requested" in ln and "used" in ln]
        print("  L1 profile fallbacks (requested != used): %d" % len(fb))
        for ln in fb[:3]:
            print("     " + ln[:160])
    # L1 errors kept by KEEP_RAW=2 (phy_<name>.log excerpt): Err.ind etc.
    phy = os.path.join(O, "phy_%s.log" % n)
    if os.path.exists(phy):
        errs = {}
        for ln in open(phy, errors="replace"):
            if re.search(r"Send Err.ind|Order kernel timeout|not responsive|No available|Successive|timeout waiting|Dropping the slot", ln):
                k = re.sub(r"^[0-9:.]+ ", "", ln.strip())
                k = re.sub(r"SFN [0-9.]+", "SFN x", k)
                k = re.sub(r"[0-9]+", "N", k)[:120]
                errs[k] = errs.get(k, 0) + 1
        print("  L1 error lines: %s" % (sorted(errs.items(), key=lambda kv: -kv[1])[:5] if errs else "none"))
        csvp = os.path.join(O, "phy_%s.csv" % n)
        if not os.path.exists(csvp):
            subprocess.run([sys.executable, os.path.join(F, "parse_phylog.py"), phy, csvp], check=False)
        if os.path.exists(csvp):
            rows = list(csv.DictReader(open(csvp)))
            print("  cuPHY timing (%d slot rows, us after t0 of the slot):" % len(rows))

            def col(c, pos):
                return np.array([float(r[c]) for r in rows if int(r["pos"]) == pos and r.get(c) not in (None, "", "None", "nan")])
            for pos in (7, 8, 9):
                v = col("pusch_gpu_end", pos)
                print("    PUSCH completion pos %d: n %5d p50 %6.0f p99 %6.0f max %6.0f  > 2100 us: %d" % (pos, len(v), q(v, 50), q(v, 99), v.max() if len(v) else float("nan"), int((v > 2100).sum()) if len(v) else 0))
            for pos in (0, 2, 6):
                v = col("comp_wait_end", pos)
                print("    PDSCH compression done pos %d: n %5d p50 %6.0f p99 %6.0f max %6.0f" % (pos, len(v), q(v, 50), q(v, 99), v.max() if len(v) else float("nan")))
            v = np.array([float(r["pdcch_setup_start"]) + float(r["pdcch_gpu_run_us"]) for r in rows if r.get("pdcch_setup_start") not in (None, "", "None", "nan") and r.get("pdcch_gpu_run_us") not in (None, "", "None", "nan")])
            print("    PDCCH end: n %d p99 %6.0f max %6.0f" % (len(v), q(v, 99), v.max() if len(v) else float("nan")))
            v = col("order_run_us", 8)
            print("    order run pos 8: p99 %6.0f" % q(v, 99))
    ru = os.path.join(O, "ru_%s.log" % n)
    if os.path.exists(ru):
        out = subprocess.run(["grep", "-aE", "DL U ON TIME PACKETS|DL U LATE PACKETS|DL U LATE SLOTS|DL C LATE SLOTS", ru], capture_output=True, text=True).stdout.splitlines()
        for ln in out[-4:]:
            print("  RU: " + ln.strip()[:200])
    tl = os.path.join(O, "dapp_tenant_%s.log" % n)
    if os.path.exists(tl):
        lines = open(tl, errors="replace").read().splitlines()
        for ln in lines:
            if ln.startswith("====") or ln.startswith("  (P") or ln.startswith("  part"):
                print("  tenant: " + ln[:220])
        stats = [ln for ln in lines if ln.startswith("[")]
        if stats:
            print("  tenant last stats: " + stats[-1][:900])
    tc = os.path.join(O, "dapp_tenant_%s.csv" % n)
    if os.path.exists(tc):
        rows = list(csv.DictReader(open(tc)))
        if rows:
            print("  tenant per TDD position (wall-clock slot = position of its own slot number): class histogram, mean segments, paused us")
            for pos in range(10):
                rr = [r for r in rows if int(r["pos"]) == pos]
                if not rr:
                    continue
                h = {}
                for r in rr:
                    h[r["class"]] = h.get(r["class"], 0) + 1
                segs = np.array([float(r["segments"]) for r in rr])
                paused = np.array([float(r["paused_us"]) for r in rr])
                unk = sum(1 for r in rr if r["known"] == "0")
                print("    pos %d: n %5d classes %s segments mean %.1f paused mean %4.0f us  unknown-plan slots %d" % (pos, len(rr), " ".join("%s:%d" % kv for kv in sorted(h.items(), key=lambda kv: -kv[1])), segs.mean(), paused.mean(), unk))


if __name__ == "__main__":
    main()
