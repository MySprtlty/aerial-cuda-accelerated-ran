#!/usr/bin/env python3
"""compare_ab.py a.csv b.csv: per TDD position, where the DL chain spends its time (us relative to the air time t0),
for the 3-core (a) and 6-core (b) DL worker runs. First second of each run excluded."""
import sys, csv, collections
import numpy as np
START_TX = -470.0     # cuphydriver: t0 - (Tcp_adv_dl 125 + T1a_max_up 345)
def load(p):
    rows = list(csv.DictReader(open(p)))
    t0s = [int(r["t0_ns"]) for r in rows]; tmin = min(t0s)
    rows = [r for r in rows if int(r["t0_ns"]) - tmin > 1_000_000_000]
    return rows
def col(rs, k):
    return np.array([float(r[k]) for r in rs if r.get(k) not in ("", None)])
def q(v, p):
    return np.percentile(v, p) if len(v) else float("nan")
runs = {lab: load(p) for lab, p in zip(("a", "b"), sys.argv[1:3])}
print("slots after the first second: %s" % {k: len(v) for k, v in runs.items()})
stages = [("enq_us", "L2A enqueue"), ("pdsch_setup_start", "PDSCH CPU setup start"), ("pdsch_run_end", "PDSCH launched (CPU run end)"),
          ("comp_cpu_start", "compression CPU start"), ("comp_wait_end", "compression done (GPU)"), ("utx_end_max", "U-plane tx handoff done (max cell)"),
          ("cplane_end_max", "C-plane done (max cell)")]
for pos in (2, 6, 7, 8, 0):
    print("\n=== TDD position %d: p50 / p99 / max (us from t0; FH TX start deadline %.0f) ===" % (pos, START_TX))
    print("%-38s %26s   %26s" % ("stage", "(a) 3 DL cores", "(b) 6 DL cores"))
    for k, name in stages:
        vals = []
        for lab in ("a", "b"):
            rs = [r for r in runs[lab] if int(r["pos"]) == pos]
            v = col(rs, k)
            vals.append("%6.0f /%6.0f /%6.0f" % (q(v, 50), q(v, 99), q(v, 100)) if len(v) else "        -")
        print("%-38s %26s   %26s" % (name, vals[0], vals[1]))
    for k, name in (("pdsch_gpu_run_us", "PDSCH GPU run (CUDA events, us)"), ("pdsch_gpu_setup_us", "PDSCH GPU setup (us)"), ("comp_gpu_us", "compression GPU exec (us)"), ("chan_to_comp_gap_us", "channel->compression gap (us)")):
        vals = []
        for lab in ("a", "b"):
            v = col([r for r in runs[lab] if int(r["pos"]) == pos], k)
            vals.append("%6.0f /%6.0f /%6.0f" % (q(v, 50), q(v, 99), q(v, 100)) if len(v) else "        -")
        print("%-38s %26s   %26s" % (name, vals[0], vals[1]))
    for lab in ("a", "b"):
        rs = [r for r in runs[lab] if int(r["pos"]) == pos]
        c = col(rs, "comp_wait_end")
        if len(c):
            print("   (%s) compression done after the TX start deadline: %.3f %% of slots (%d of %d); after -400 us: %.3f %%" % (
                lab, 100 * (c > START_TX).mean(), (c > START_TX).sum(), len(c), 100 * (c > -400).mean()))
print("\n=== derived intervals at positions 6/7 (p50 / p99), us ===")
for pos in (6, 7):
    for lab in ("a", "b"):
        rs = [r for r in runs[lab] if int(r["pos"]) == pos and r["pdsch_setup_start"] and r["comp_wait_end"] and r["pdsch_run_end"]]
        enq = col(rs, "enq_us"); ss = col(rs, "pdsch_setup_start"); re_ = col(rs, "pdsch_run_end"); ce = col(rs, "comp_wait_end"); cs = col(rs, "comp_cpu_start")
        print("pos %d (%s): enqueue->setup start %.0f/%.0f | setup start->launched %.0f/%.0f | launched->compression CPU start %.0f/%.0f | compression CPU start->done %.0f/%.0f | launched->done (GPU window) %.0f/%.0f" % (
            pos, lab, q(ss - enq, 50), q(ss - enq, 99), q(re_ - ss, 50), q(re_ - ss, 99), q(cs - re_, 50), q(cs - re_, 99), q(ce - cs, 50), q(ce - cs, 99), q(ce - re_, 50), q(ce - re_, 99)))
print("\n=== UL context: PUSCH GPU window and order kernel (us) ===")
for lab in ("a", "b"):
    for pos in (4, 5):
        rs = [r for r in runs[lab] if int(r["pos"]) == pos]
        ps, pe, pr, orr = col(rs, "pusch_setup_start"), col(rs, "pusch_gpu_end"), col(rs, "pusch_gpu_run_us"), col(rs, "order_run_us")
        print("(%s) pos %d: PUSCH setup start %.0f / GPU end %.0f (p50, rel t0 of that slot) / GPU run %.0f us p50, %.0f p99 | order run %.0f us p50" % (lab, pos, q(ps, 50), q(pe, 50), q(pr, 50), q(pr, 99), q(orr, 50)))
