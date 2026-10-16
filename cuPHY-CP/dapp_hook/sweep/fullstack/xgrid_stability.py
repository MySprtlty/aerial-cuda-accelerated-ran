#!/usr/bin/env python3
"""xgrid_stability.py --csv xgrid_<prefix>.csv --prefix P --x X
Block stability (rule of 2026-10-17 04:20): UNSTABLE only when the closing R0's late DL U-plane PACKET fraction is >= 2 x the
start R0 mean AND >= 1e-5 (absolute), or its B_abs overruns >= 2 x the start mean and > 5. Prints one line."""
import argparse
import csv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--prefix", required=True); ap.add_argument("--x", required=True)
    a = ap.parse_args()
    rows = {r["name"]: r for r in csv.DictReader(open(a.csv))}
    def g(n, k):
        try:
            return float(rows[n][k])
        except (KeyError, ValueError, TypeError):
            return float("nan")
    p, x = a.prefix, a.x
    s_frac = (g("%s_R0a_%sc" % (p, x), "late_pkt_frac") + g("%s_R0b_%sc" % (p, x), "late_pkt_frac")) / 2
    s_b = (g("%s_R0a_%sc" % (p, x), "b_abs_all") + g("%s_R0b_%sc" % (p, x), "b_abs_all")) / 2
    z_frac, z_b = g("%s_R0z_%sc" % (p, x), "late_pkt_frac"), g("%s_R0z_%sc" % (p, x), "b_abs_all")
    unstable = (z_frac >= 2 * s_frac and z_frac >= 1e-5) or (z_b >= 2 * max(s_b, 1) and z_b > 5)
    print("== block %s stability: start R0 late pkt frac %.2e / B_abs %.1f -> closing R0 %.2e / %.0f -> %s" % (p, s_frac, s_b, z_frac, z_b, "UNSTABLE" if unstable else "stable"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
