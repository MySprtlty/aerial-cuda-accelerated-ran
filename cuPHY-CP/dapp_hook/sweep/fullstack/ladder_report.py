#!/usr/bin/env python3
"""ladder_report.py <name> [<name> ...]   -> one block per full-stack run of ladder_runs.sh (files in prof/run61)

Per run: UL PUSCH GPU completion after the slot's air time t0 by TDD position (the "how late does the uplink decode
get" number), order-kernel run time, DL PDSCH run end, the L1 overload strings (pool exhaustion, recovery, work
cancel, order timeout, ERROR.indication codes), the L1's own per-second UL/DL Mbps and CRC counters and testMAC's
view (UL/DL Mbps, ERR, INV). Statistics skip the first second (CONTEXT 9). numpy only.
"""
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
O = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "prof", "run61"))


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def stages(name):
    p = os.path.join(O, "phy_%s.csv" % name)
    if not os.path.exists(p):
        return None
    cols = {}
    with open(p) as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return None
    t0 = np.array([float(x["t0_ns"]) for x in rows])
    keep = t0 - t0.min() >= 1e9
    out = {"slots": int(keep.sum()), "span_s": float((t0.max() - t0.min()) / 1e9)}

    def col(key, pos):
        v = [float(x[key]) for x, k in zip(rows, keep) if k and x["pos"] == str(pos) and x.get(key) not in ("", None)]
        return np.array(v)
    for pos in sorted({int(x["pos"]) for x in rows if x.get("pusch_gpu_end") not in ("", None)}):
        v = col("pusch_gpu_end", pos)
        o = col("order_run_us", pos)
        g = col("pusch_gpu_run_us", pos)
        out["ul%d" % pos] = (len(v), pct(v, 50), pct(v, 99), float(v.max()) if len(v) else float("nan"), pct(o, 50), pct(o, 99), pct(g, 50), pct(g, 99))
    for pos in (0, 6, 7):
        v = col("pdsch_run_end", pos)
        c = col("comp_wait_end", pos)
        if len(v):
            out["dl%d" % pos] = (len(v), pct(v, 50), pct(v, 99), pct(c, 50), pct(c, 99))
    return out


def warn_counts(name):
    p = os.path.join(O, "phy_warn_%s.log" % name)
    keys = ["No available", "L1 in recovery", "recovery", "not responsive", "FATAL", "Attempting PUSCH pipeline termination", "Order kernel timeout",
            "Successive Non-availability", "exit handler", "pipeline failed"]
    c = {k: 0 for k in keys}
    err = {}
    n = 0
    if not os.path.exists(p):
        return None, None, 0
    with open(p, errors="replace") as f:
        for line in f:
            n += 1
            for k in keys:
                if k in line:
                    c[k] += 1
            m = re.search(r"err_code=(0x[0-9a-fA-F]+)", line)
            if m and "Err.ind" in line:
                err[m.group(1)] = err.get(m.group(1), 0) + 1
    return c, err, n


def console(name):
    out = {}
    p3 = os.path.join(O, "run3_%s.log" % name)
    if os.path.exists(p3):
        ul, dl, inv, errs = [], [], [], []
        for line in open(p3, errors="replace"):
            if "[MAC.FAPI] Cell" in line and "Mbps" in line:
                m = re.search(r"DL\s+([\d.]+) Mbps.*UL\s+([\d.]+) Mbps.*ERR\s+(\d+) \| INV\s+(\d+)", line)
                if m:
                    dl.append(float(m.group(1))); ul.append(float(m.group(2))); errs.append(int(m.group(3))); inv.append(int(m.group(4)))
        if ul:
            h = len(ul) // 2       # second half of the run: steady state
            out["mac"] = (float(np.median(ul[h:])), float(np.min(ul[h:])), float(np.median(dl[h:])), int(np.max(errs[h:])), float(np.median(inv[h:])), int(np.max(inv[h:])))
        out["mac_finished"] = "Finished running" in open(p3, errors="replace").read()
    p2 = os.path.join(O, "run2_%s.log" % name)
    if os.path.exists(p2):
        ul, dl, crc, cum = [], [], [], 0
        for line in open(p2, errors="replace"):
            if "[SCF.PHY] Cell" in line and "Mbps" in line:
                m = re.search(r"DL\s+([\d.]+) Mbps.*UL\s+([\d.]+) Mbps\s+\d+ Slots CRC\s+(\d+) \(\s*(\d+)\)", line)
                if m:
                    dl.append(float(m.group(1))); ul.append(float(m.group(2))); crc.append(int(m.group(3))); cum = max(cum, int(m.group(4)))
        if ul:
            h = len(ul) // 2
            out["l1"] = (float(np.median(ul[h:])), float(np.min(ul[h:])), float(np.median(dl[h:])), int(np.max(crc[h:])), cum)
    return out


def main():
    for name in sys.argv[1:]:
        print("=== %s" % name)
        st = stages(name)
        if st is None:
            print("  no per-slot csv (L1 did not come up or no traffic)")
        else:
            print("  slots %d over %.0f s (first second skipped)" % (st["slots"], st["span_s"]))
            for k in sorted(k for k in st if k.startswith("ul")):
                n, p50, p99, mx, o50, o99, g50, g99 = st[k]
                print("  UL pos %s: PUSCH GPU end after t0  p50 %7.0f  p99 %7.0f  max %7.0f us   (order kernel run p50 %5.0f p99 %5.0f, PUSCH GPU run p50 %5.0f p99 %5.0f; n=%d)" % (
                    k[2:], p50, p99, mx, o50, o99, g50, g99, n))
            for k in sorted(k for k in st if k.startswith("dl")):
                n, p50, p99, c50, c99 = st[k]
                print("  DL pos %s: PDSCH run end p50 %7.0f p99 %7.0f us, compression end p50 %7.0f p99 %7.0f (relative to t0; n=%d)" % (k[2:], p50, p99, c50, c99, n))
        c, err, n = warn_counts(name)
        if c is None:
            print("  no phy_warn file")
        else:
            print("  L1 WRN/ERR lines %d | pool 'No available' %d | recovery %d | not responsive %d | FATAL %d | PUSCH work cancel %d | order timeout %d | exit %d | Err.ind %s" % (
                n, c["No available"], c["recovery"], c["not responsive"], c["FATAL"], c["Attempting PUSCH pipeline termination"], c["Order kernel timeout"], c["exit handler"],
                " ".join("%s x%d" % kv for kv in sorted(err.items())) or "-"))
        co = console(name)
        if "l1" in co:
            print("  L1 counters (2nd half, per cell per s): UL %.2f Mbps (min %.2f)  DL %.2f Mbps  CRC errors/s max %d, cumulative %d" % co["l1"])
        if "mac" in co:
            print("  testMAC (2nd half): UL %.2f Mbps (min %.2f)  DL %.2f Mbps  ERR max %d  INV median %.0f max %d  finished=%s" % (co["mac"] + (co.get("mac_finished"),)))


if __name__ == "__main__":
    main()
