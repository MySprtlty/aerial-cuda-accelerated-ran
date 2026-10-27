#!/usr/bin/env python3
"""b3_split.py --dir D --name NAME [--first-s 10] [--window-us 294] [--sym-us 35.7]
Split the RU emulator's late DL U-plane slots into "L1 transmit delay" and "NIC / RU receive delay" by joining, per
late (slot, cell) row of ru_<name>_late.csv (tag 503 INFO), the L1's own DL U-plane transmit task end of that slot
(phy_<name>.csv utx_end_max, us after the slot air time t0, max over cells) with the RU's acceptance deadline of the
first late symbol (T0 + sym * sym_us - window_us). Both clocks are CLOCK_REALTIME on the same PTP-synced host.
  L1-late : the L1 finished handing the slot's U-plane packets to the NIC after the deadline of the first late symbol
  RU-late : the L1 finished before that deadline, the packets still arrived late (NIC queueing, wire, RU rx path)
Also prints utx_end_max p50 / p99.9 / max of ALL DL slots per TDD position (control) for the same run. numpy only."""
import argparse
import bisect
import csv
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--first-s", type=float, default=10.0); ap.add_argument("--window-us", type=float, default=294.0)
    ap.add_argument("--sym-us", type=float, default=35.7)
    a = ap.parse_args()
    phy = list(csv.DictReader(open(os.path.join(a.dir, "phy_%s.csv" % a.name))))
    t0min = min(int(r["t0_ns"]) for r in phy)
    dl = sorted((int(r["t0_ns"]), float(r["utx_end_max"]), int(r["pos"])) for r in phy if r.get("utx_end_max") and int(r["t0_ns"]) - t0min >= a.first_s * 1e9)
    keys = [d[0] for d in dl]
    print("## B3 split %s: DL slots with U-plane tx timing %d" % (a.name, len(dl)))
    print("| pos | DL slots | L1 U-plane tx end after t0, us: p50 / p99 / p99.9 / max | share tx end > sym0 deadline (-%.0f us) |" % a.window_us)
    print("|---|---|---|---|")
    for p in range(10):
        v = np.array([d[1] for d in dl if d[2] == p])
        if len(v):
            print("| %d | %d | %.0f / %.0f / %.0f / %.0f | %.3f %% |" % (p, len(v), np.percentile(v, 50), np.percentile(v, 99), np.percentile(v, 99.9), v.max(), 100.0 * (v > -a.window_us).mean()))
    lp = os.path.join(a.dir, "ru_%s_late.csv" % a.name)
    if not os.path.exists(lp):
        print("(no ru_<name>_late.csv: run without tag 503 INFO)"); return 0
    late = [r for r in csv.DictReader(open(lp)) if "max_ns" in r and int(r["t0_ns"]) - t0min >= a.first_s * 1e9]
    cls = {}
    for r in late:
        T0 = int(r["t0_ns"]); i = bisect.bisect_left(keys, T0 - 200_000)
        # nearest phy slot within 200 us
        cand = [d for d in dl[max(0, i - 2): i + 3] if abs(d[0] - T0) < 200_000]
        if not cand:
            cls.setdefault(int(r["slot"]) % 10, []).append(("nomatch", 0.0, 0.0)); continue
        d = min(cand, key=lambda x: abs(x[0] - T0))
        syms = [int(r.get("sym%d" % k, 0) or 0) for k in range(14)]
        first = next((k for k, v in enumerate(syms) if v > 0), 0)
        deadline_us = first * a.sym_us - a.window_us                     # RU acceptance deadline of the first late symbol, rel. to T0
        tx_end_us = d[1] + (d[0] - T0) / 1e3                               # L1 tx end relative to the RU's T0
        lateness = int(r["max_ns"]) / 1e3 + a.window_us
        cls.setdefault(int(r["slot"]) % 10, []).append(("L1" if tx_end_us > deadline_us else "RU", tx_end_us - deadline_us, lateness))
    print("\n| pos | late rows | L1-late (tx end after the first late symbol's deadline) | RU-late (tx end in time) | no phy match | L1 tx end minus deadline, us: p50 / max | lateness us p50 / max |")
    print("|---|---|---|---|---|---|---|")
    tot = {"L1": 0, "RU": 0, "nomatch": 0}
    for p in sorted(cls):
        v = cls[p]; n1 = sum(1 for x in v if x[0] == "L1"); n2 = sum(1 for x in v if x[0] == "RU"); n0 = sum(1 for x in v if x[0] == "nomatch")
        tot["L1"] += n1; tot["RU"] += n2; tot["nomatch"] += n0
        m = np.array([x[1] for x in v if x[0] != "nomatch"]); l = np.array([x[2] for x in v if x[0] != "nomatch"])
        print("| %d | %d | %d | %d | %d | %s | %s |" % (p, len(v), n1, n2, n0, ("%.0f / %.0f" % (np.percentile(m, 50), m.max())) if len(m) else "-", ("%.0f / %.0f" % (np.percentile(l, 50), l.max())) if len(l) else "-"))
    print("| all | %d | %d | %d | %d | | |" % (len(late), tot["L1"], tot["RU"], tot["nomatch"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
