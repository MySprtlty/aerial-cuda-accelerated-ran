#!/usr/bin/env python3
"""n4_switch.py --dir D --name RUN [--origin 3.64 --seg-s 60] [--r 105] [--kpi-e2e 153.5 --kpi-map 51.75] [--ramp-ms 200]
Night-3 slow alternation (Average 60 s / Peak 60 s, segment k = floor((t - origin)/seg_s), even = Average): for the ours run,
  detection delay   = first heavy UL slot of each Peak segment (table_policy_RUN.csv: ts_ns of the slot that switched the class)
                      -> classifier switch wall time (t_wall_ns) and -> first request DEQUEUED with the heavy caps (disp_RUN.csv)
  return delay      = the same for Peak -> Average
  ramp cost         = B_abs overruns (safecap_check budgets) in +-ramp-ms around every boundary vs the rest (phy_RUN.csv)
  per segment class = requests, E2E mean / p95, service mean, model shares, weighted mAP; time-average E2E and mAP; KPI with R
Works for static / naive runs too (no table_policy csv -> no delays)."""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import safecap_check as SC   # noqa: E402

MAP = {"26x": 57.5, "26l": 55.0, "26m": 53.1, "26s": 48.6, "26n": 40.9}


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v); return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--origin", type=float, default=3.64); ap.add_argument("--seg-s", type=float, default=60.0)
    ap.add_argument("--r", type=float, default=105.0); ap.add_argument("--kpi-e2e", type=float, default=153.5); ap.add_argument("--kpi-map", type=float, default=51.75)
    ap.add_argument("--ramp-ms", type=float, default=200.0); ap.add_argument("--first-s", type=float, default=30.0)
    a = ap.parse_args(); d, n = a.dir, a.name
    seg_of = lambda t_ns: int(((t_ns / 1e9) - a.origin) // a.seg_s)                    # noqa: E731
    cls_of = lambda t_ns: "average" if seg_of(t_ns) % 2 == 0 else "peak"               # noqa: E731
    dist_to_boundary = lambda t_ns: min(((t_ns / 1e9 - a.origin) % a.seg_s), a.seg_s - ((t_ns / 1e9 - a.origin) % a.seg_s))   # noqa: E731
    rows = SC.load_rows(n, d, a.first_s)
    if not rows:
        print("no phy rows"); return 1
    T0 = min(int(r["t0_ns"]) for r in rows) - int(a.first_s * 1e9)
    # ramp cost
    ramp_rows = [r for r in rows if dist_to_boundary(int(r["t0_ns"])) * 1e3 <= a.ramp_ms]
    rest_rows = [r for r in rows if dist_to_boundary(int(r["t0_ns"])) * 1e3 > a.ramp_ms]
    def babs(rs):
        m = SC.misses(SC.channel_values(rs)); return sum(v[1] for v in m.values()), m["pusch"][1]
    rb, rbp = babs(ramp_rows); ob, obp = babs(rest_rows)
    n_bound = len({seg_of(int(r["t0_ns"])) for r in rows})
    print("## %s: %d slots after %.0f s, %d segments touched; B_abs in +-%.0f ms of boundaries: %d (PUSCH %d) over %d slots | elsewhere: %d (PUSCH %d) over %d slots | total %d" % (
        n, len(rows), a.first_s, n_bound, a.ramp_ms, rb, rbp, len(ramp_rows), ob, obp, len(rest_rows), rb + ob))
    for cls in ("average", "peak"):
        rs = [r for r in rows if cls_of(int(r["t0_ns"])) == cls]
        b, bp = babs(rs); print("   %s segments: %d slots, B_abs %d (PUSCH %d)" % (cls, len(rs), b, bp))
    # tenant
    p = os.path.join(d, "disp_%s.csv" % n)
    if os.path.exists(p):
        reqs = [r for r in csv.DictReader(open(p)) if r["status"] == "0" and int(r["t_arrival"]) >= T0 + int(a.first_s * 1e9)]
        print("\n| class | requests | E2E mean / p95 / p99 ms | service mean | shares 26x/26l/26m/26s/26n | weighted mAP |"); print("|---|---|---|---|---|---|")
        tot_e2e, tot_map, tot_n = 0.0, 0.0, 0
        for cls in ("average", "peak", "all"):
            rs = [r for r in reqs if cls == "all" or cls_of(int(r["t_arrival"])) == cls]
            if not rs:
                continue
            e2e = [(int(r["t_reply"]) - int(r["t_arrival"])) / 1e6 for r in rs]; svc = [(int(r["t_end"]) - int(r["t_start"])) / 1e6 for r in rs if int(r["t_end"]) and int(r["t_start"])]
            sh = {m: sum(1 for r in rs if r["model"] == m) / len(rs) for m in MAP}
            wm = sum(sh[m] * MAP[m] for m in MAP)
            print("| %s | %d | %.1f / %.1f / %.1f | %.1f | %s | %.1f |" % (cls, len(rs), sum(e2e) / len(e2e), pct(e2e, 95), pct(e2e, 99), (sum(svc) / len(svc)) if svc else float("nan"),
                  "/".join("%.2f" % sh[m] for m in MAP), wm))
            if cls == "all":
                tot_e2e, tot_map = sum(e2e) / len(e2e), wm
        e2e_ue = a.r + tot_e2e
        print("KPI (R = %.0f ms): E2E = R + sojourn = %.1f ms (%s %.1f), weighted mAP %.1f (%s %.2f)" % (a.r, e2e_ue, "<=" if e2e_ue <= a.kpi_e2e else ">", a.kpi_e2e, tot_map, ">=" if tot_map >= a.kpi_map else "<", a.kpi_map))
        # detection / return delays
        tp = os.path.join(d, "table_policy_%s.csv" % n)
        if os.path.exists(tp):
            sw = [r for r in csv.DictReader(open(tp))]
            prev = None; events = []
            for r in sw:
                if r["class"] != prev:
                    events.append((int(r["ts_ns"]), int(r["t_wall_ns"]), r["class"], int(r["cap_26x"]))); prev = r["class"]
            print("\n| switch -> | slot ts (s from T0) | classifier wall delay ms | first request dequeued with the new caps: delay ms |"); print("|---|---|---|---|")
            allreq = sorted((int(r["t_dequeue"]), int(r["cap_26x"]), int(r["cap_26s"])) for r in csv.DictReader(open(p)))
            for ts, tw, cls, cx in events[1:]:
                nxt = next((td for td, c26x, c26s in allreq if td >= ts and ((cls == "peak" and c26x == 0) or (cls == "average" and c26x > 0))), None)
                print("| %s | %.2f | %.1f | %s |" % (cls, (ts - T0) / 1e9, (tw - ts) / 1e6, ("%.1f" % ((nxt - ts) / 1e6)) if nxt else "n/a"))
            for cls in ("peak", "average"):
                ds = []
                for ts, tw, c, cx in events[1:]:
                    if c != cls:
                        continue
                    nxt = next((td for td, c26x, c26s in allreq if td >= ts and ((cls == "peak" and c26x == 0) or (cls == "average" and c26x > 0))), None)
                    if nxt:
                        ds.append((nxt - ts) / 1e6)
                if ds:
                    print("-> %s: n %d, mean %.1f ms, max %.1f ms (slot -> first request served with the new caps)" % (cls, len(ds), sum(ds) / len(ds), max(ds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
