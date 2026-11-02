#!/usr/bin/env python3
"""kpi_table.py [names...] [--r 105] [--dir prof/tenant]: KPI table of kpi_run.sh results against the MIG baseline.
Baseline server time = 3g MIG 26x tiled (measured 2026-10-16: service p50 47.1 ms at 10 req/s; sweep 56 SM 40.3 + decode 6.0 = 46.3).
E2E = R (UE upload + core/network, --r ms, not measured yet) + server time. KPI1: ours E2E <= 0.9 x baseline E2E;
KPI2: request-weighted mAP >= 0.9 x 57.5 = 51.75."""
import argparse
import csv
import glob
import os
import re
import statistics

MAP = {"26n": 40.9, "26s": 48.6, "26m": 53.1, "26l": 55.0, "26x": 57.5}
ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="*"); ap.add_argument("--r", type=float, default=105.0); ap.add_argument("--dir", default=None); ap.add_argument("--base-server", type=float, default=46.3)
a = ap.parse_args()
D = a.dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "prof", "tenant")
names = a.names or sorted(os.path.basename(p)[3:-4] for p in glob.glob(os.path.join(D, "ra_*.csv")))
base_e2e = a.r + a.base_server
print("baseline: 3g MIG 26x server %.1f ms, R %.0f ms -> E2E %.1f ms (KPI1 target <= %.1f, KPI2 mAP >= 51.75)" % (a.base_server, a.r, base_e2e, 0.9 * base_e2e))
print("%-14s %5s %4s %5s | %7s %7s %7s %7s | %6s %6s | %7s %6s | %5s %5s | %s" % ("run", "req/s", "n", "fail", "srv_mean", "srv_p50", "srv_p90", "e2e_cli", "E2E(R)", "vs_B", "mAP", "vs_B", "KPI1", "KPI2", "models (sm_mean)"))
for n in names:
    f = os.path.join(D, "ra_%s.csv" % n)
    if not os.path.exists(f): continue
    rows = list(csv.DictReader(open(f))); ok = [r for r in rows if r["status"] == "success"]
    if not ok: print("%-14s no successful requests (%d rows)" % (n, len(rows))); continue
    srv = [float(r["server_ms"]) for r in ok]; e2e = [float(r["e2e_ms"]) for r in ok]
    dur = (max(float(r["t_arrival"]) for r in rows) - min(float(r["t_arrival"]) for r in rows)) or 1.0; rate = len(rows) / dur
    wmap = sum(MAP[r["model"]] for r in ok) / len(ok)
    E = a.r + statistics.mean(srv)
    by = {}
    for r in ok: by.setdefault(r["model"], []).append(float(r["sm_mean"] or 0))
    models = " ".join("%s:%d(%.0f)" % (m, len(v), statistics.mean(v)) for m, v in sorted(by.items()))
    print("%-14s %5.1f %4d %5d | %7.1f %7.1f %7.1f %7.1f | %6.1f %+5.1f%% | %7.2f %+5.1f%% | %5s %5s | %s" % (
        n, rate, len(rows), len(rows) - len(ok), statistics.mean(srv), statistics.median(srv), sorted(srv)[int(0.9 * (len(srv) - 1))], statistics.mean(e2e),
        E, 100 * (E / base_e2e - 1), wmap, 100 * (wmap / 57.5 - 1), "OK" if E <= 0.9 * base_e2e else "FAIL", "OK" if wmap >= 51.75 else "FAIL", models))
    # cuPHY side (2026-11-02: L1 recovery / FATAL counts from run2_<name>.log; any recovery makes the run invalid)
    l1 = os.path.join(D, "run2_%s.log" % n); rec = fat = -1
    if os.path.exists(l1):
        lt = open(l1, errors="replace").read(); rec = lt.count("L1 in recovery for"); fat = lt.count("FATAL exit")
    s = os.path.join(D, "summary_%s.txt" % n)
    if os.path.exists(s):
        txt = open(s, errors="replace").read()
        pu = re.findall(r"PUSCH completion pos (\d): n\s+\d+ p50\s+(\d+) p99\s+(\d+).*?> 2100 us: (\d+)", txt)
        ru = re.findall(r"DL U LATE SLOTS\s*\|\s*(\d+)", txt)
        err = re.search(r"L1 error lines: (.*)", txt)
        print("%-14s cuPHY: PUSCH p99 %s miss %s | RU late %s | L1 recovery %s FATAL %s%s | %s" % ("", "/".join(p[2] for p in pu), "/".join(p[3] for p in pu), ru[0] if ru else "-", rec, fat, " -> INVALID" if rec > 0 or fat > 0 else "", (err.group(1)[:60] if err else "")))
