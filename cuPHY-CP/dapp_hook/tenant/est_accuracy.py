#!/usr/bin/env python3
"""est_accuracy.py [names...]: how well the per-slot estimate (est_ms sent to the RA at request time) predicted the server time
(ra_<name>.csv from kpi_run.sh). Per model: n, mean est, mean actual, mean error, p90 |error|, share of under-estimates > 20 %."""
import csv, glob, os, statistics, sys
D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "prof", "tenant")
names = sys.argv[1:] or sorted(os.path.basename(p)[3:-4] for p in glob.glob(os.path.join(D, "ra_*.csv")))
for n in names:
    f = os.path.join(D, "ra_%s.csv" % n)
    if not os.path.exists(f): continue
    rows = [r for r in csv.DictReader(open(f)) if r["status"] == "success" and r.get("est_ms") not in (None, "", "None")]
    if not rows: print("%-14s no rows with estimates" % n); continue
    print("== %s (%d requests with an estimate)" % (n, len(rows)))
    for m in ("26n", "26s", "26m", "26l", "26x"):
        rs = [r for r in rows if r["model"] == m]
        if not rs: continue
        est = [float(r["est_ms"]) for r in rs]; act = [float(r["server_ms"]) for r in rs]; err = [a - e for a, e in zip(act, est)]
        ae = sorted(abs(x) for x in err)
        print("   %s n %4d  est mean %6.1f  actual mean %6.1f  error mean %+6.1f ms  |err| p90 %5.1f  under-est >20%%: %4.1f%%" % (
            m, len(rs), statistics.mean(est), statistics.mean(act), statistics.mean(err), ae[int(0.9 * (len(ae) - 1))], 100.0 * sum(1 for a, e in zip(act, est) if a > 1.2 * e) / len(rs)))
