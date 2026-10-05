#!/usr/bin/env python3
"""
Request-rate -> latency curves of the open-loop tenant runs (openloop_worker.py via sweep.py).
Reads run directories only; never touches the GPU.

    python3 openloop_report.py --runs runs/c16_openloop_1280 --out results/c16_openloop_1280 \
        [--extra "MIG 3g=runs/c16_openloop_mig_1280"] [--lo c16_L1 --hi c16_L4]

Per run: worker_summary.csv (one row per offered rate: served, backlog, latency mean/p50/p99/max, queue,
service, busy fraction) and spec.yaml (point, tenant, cap). Outputs
    openloop_latency.csv                 one row per (source, load, tenant, cap, rate)
    openloop_<tenant>.png                p50 (left) and p99 (right) total latency vs offered rate; lines = alone /
                                         lowest load / highest load per cap (solid = cap 100, dashed = cap 30);
                                         markers hollow where the stage ended with a backlog (offered > capacity)
    openloop_capacity.csv                per (source, load, tenant, cap): highest offered rate served without backlog
                                         and with p99 <= 2x the unloaded service time
"""
import argparse
import csv
import glob
import os
import re

import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLORS = {"alone": "#6c6a66", "lowest": "#2a78d6", "highest": "#eb6834"}
STYLE = {100: "-", 50: "-.", 30: "--"}


def load_runs(root, source, lo, hi):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        m = re.match(r"(.+?)__(ol\w+?)_cap(\d+)_b(\d+)_p(\d+)$", os.path.basename(d))
        if not m:
            continue
        st = os.path.join(d, "status.yaml")
        if not os.path.exists(st) or yaml.safe_load(open(st)).get("state") != "ok":
            continue
        point, tenant, cap, batch = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        load = "alone" if point == "tenant_alone" else ("lowest" if point == lo else ("highest" if point == hi else point))
        p = os.path.join(d, "worker_summary.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            r = {k: (float(v) if re.match(r"^-?[\d.]+$", v or "") else v) for k, v in r.items()}
            r.update(source=source, load=load, tenant=tenant, cap=cap, batch=batch, run=os.path.basename(d))
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", action="append", default=[], help='"label=runs/dir" overlays (e.g. MIG 3g)')
    ap.add_argument("--lo", default="c16_L1")
    ap.add_argument("--hi", default="c16_L4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load_runs(a.runs, "shared", a.lo, a.hi)
    for e in a.extra:
        label, path = e.split("=", 1)
        rows += load_runs(path, label, a.lo, a.hi)
    if not rows:
        print("no finished open-loop runs under", a.runs)
        return 1
    keys = ["source", "load", "tenant", "cap", "batch", "rate", "arrivals", "served", "achieved_rate", "backlog_end", "lat_mean_ms", "lat_p50_ms",
            "lat_p99_ms", "lat_max_ms", "queue_mean_ms", "queue_p99_ms", "service_p50_ms", "service_p99_ms", "busy_frac", "engine", "run"]
    with open(os.path.join(a.out, "openloop_latency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["source"], r["tenant"], r["cap"], r["load"], r["rate"])):
            w.writerow(r)

    # capacity table
    cap_rows = []
    groups = {}
    for r in rows:
        groups.setdefault((r["source"], r["load"], r["tenant"], r["cap"], r["batch"]), []).append(r)
    for (src, load, ten, cap, b), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["rate"])
        s0 = rs[0]["service_p50_ms"]
        no_backlog = [r["rate"] for r in rs if r["backlog_end"] == 0]
        p99_ok = [r["rate"] for r in rs if r["backlog_end"] == 0 and r["lat_p99_ms"] <= 2 * s0]
        cap_rows.append({"source": src, "load": load, "tenant": ten, "cap": cap, "batch": b, "service_p50_ms_at_min_rate": round(s0, 3),
                         "max_rate_no_backlog": max(no_backlog) if no_backlog else 0, "max_rate_p99_le_2x_service": max(p99_ok) if p99_ok else 0,
                         "rates_tested": "/".join("%g" % r["rate"] for r in rs)})
    with open(os.path.join(a.out, "openloop_capacity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cap_rows[0].keys()))
        w.writeheader()
        w.writerows(cap_rows)

    # figures: one per tenant
    for ten in sorted({r["tenant"] for r in rows}):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURFACE)
        for ax, stat, title in zip(axes, ("lat_p50_ms", "lat_p99_ms"), ("p50", "p99")):
            ax.set_facecolor(SURFACE)
            for (src, load, t, cap, b), rs in sorted(groups.items()):
                if t != ten:
                    continue
                rs = sorted(rs, key=lambda r: r["rate"])
                x = [r["rate"] for r in rs]
                y = [r[stat] for r in rs]
                col = COLORS.get(load, "#333333")
                ls = STYLE.get(cap, ":") if src == "shared" else (0, (1, 1))
                lab = "%s%s%s" % (load if load == "alone" else load + " load", "" if src == "shared" else " [%s]" % src, "" if src != "shared" or cap == 100 else " cap %d" % cap)
                ax.plot(x, y, ls=ls, color=col, lw=1.8 if src == "shared" else 2.2, label=lab, zorder=3)
                full = [(r["rate"], r[stat]) for r in rs if r["backlog_end"] == 0]
                over = [(r["rate"], r[stat]) for r in rs if r["backlog_end"] > 0]
                if full:
                    ax.plot(*zip(*full), "o", color=col, ms=5, zorder=4)
                if over:
                    ax.plot(*zip(*over), "o", mfc=SURFACE, mec=col, ms=6, mew=1.6, zorder=4)
            ax.set_yscale("log")
            ax.set_xlabel("offered request rate (req/s)", color=INK)
            ax.set_ylabel("request latency %s (ms, arrival -> result)" % title, color=INK)
            ax.set_title("%s  %s" % (ten, title), color=INK, fontsize=11)
            ax.grid(True, which="both", color=GRID, lw=0.7)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.tick_params(colors=INK2)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=min(4, len(l)), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
        fig.text(0.01, 0.01, "hollow marker = stage ended with a backlog (offered rate above capacity); FIFO, batch 1, 40 s per rate", color=INK2, fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 1, 0.92))
        fig.savefig(os.path.join(a.out, "openloop_%s.png" % ten), dpi=110)
        plt.close(fig)
    print("wrote %s: openloop_latency.csv (%d rows), openloop_capacity.csv (%d), %d figures" % (a.out, len(rows), len(cap_rows), len({r["tenant"] for r in rows})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
