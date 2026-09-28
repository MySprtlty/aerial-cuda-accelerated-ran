#!/usr/bin/env python3
"""
Tenant (YOLO) inference latency vs the tenant's SM cap, from the worker logs of a cap-grid sweep.
Reads existing run directories only; never touches the GPU.

    python3 tenant_latency.py --runs runs/c16_capgrid --out results/c16_capgrid [--lo c16_L1 --hi c16_L4]

Inputs per run directory
    worker.csv          1-second windows written by yolo_worker.py: n, fps, lat_ms_mean/p50/p99/max
    nsys_trace.sqlite   session start (CLOCK_REALTIME) -> absolute time of the traced bench kernels
    chan_samples.*      first/last traced slot execution = the span in which the bench was certainly running
    spec.yaml           cell count

Only the worker windows that lie inside the traced bench span are used for the runs with cuPHY load
(the worker starts before the bench and keeps running after it; those seconds are the tenant alone).
tenant_alone__* runs use every window.

Outputs
    tenant_latency.csv                      one row per (load, tenant, batch, cap)
    tenant_latency_<tenant>_b<batch>.png    two panels (p50, p99), lines = lowest / highest cuPHY load, dashed = tenant alone
    tenant_latency_p99_overview.png         one panel per (tenant, batch), p99
    tenant_latency_p99_overview_slots.png   the same with the y-axis in slots (--slot-ms, default 0.5 ms = mu 1)

Statistics: raw per-inference samples are not stored. p50 / p99 are the median over the windows of the
per-window p50 / p99; mean is weighted by the inferences per window; max is the largest window max.
"""
import argparse
import csv
import glob
import os
import re
import sqlite3

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# same surface, ink and series colours as plot.py (capgrid figures)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = (("lowest load", "#2a78d6"), ("highest load", "#eb6834"))
ALONE = "tenant alone (no cuPHY)"


def traced_span(d):
    """(first, last) ns of the traced slot executions, relative to the nsys session start."""
    pq = os.path.join(d, "chan_samples.parquet")
    try:
        import pyarrow.parquet as pa
        t = pa.read_table(pq, columns=["t_start_ns", "dur_us"]).to_pandas()
        return int(t.t_start_ns.min()), int((t.t_start_ns + t.dur_us * 1000).max())
    except Exception:
        lo = hi = None
        with open(os.path.join(d, "chan_samples.csv")) as f:
            for r in csv.DictReader(f):
                a = int(r["t_start_ns"])
                b = a + int(float(r["dur_us"]) * 1000)
                lo = a if lo is None or a < lo else lo
                hi = b if hi is None or b > hi else hi
        return lo, hi


def collect(runs):
    out = []
    for d in sorted(glob.glob(os.path.join(runs, "*"))):
        w = os.path.join(d, "worker.csv")
        m = re.match(r"(.+?)__(yolo\w+?)_cap(\d+)_b(\d+)_p(\d+)$", os.path.basename(d))
        if not m or not os.path.isfile(w):
            continue
        load, tenant, cap, batch, prio = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
        if prio != 0:
            continue
        rows = list(csv.DictReader(open(w)))
        if not rows:
            continue
        ts = np.array([int(r["ts_ns"]) for r in rows])
        col = {k: np.array([float(r[k]) for r in rows]) for k in ("n", "fps", "lat_ms_mean", "lat_ms_p50", "lat_ms_p99", "lat_ms_max")}
        sel = np.ones(len(rows), bool)
        cells, span_s = "", ""
        if load != "tenant_alone":
            db = sqlite3.connect("file:%s?mode=ro" % os.path.join(d, "nsys_trace.sqlite"), uri=True)
            epoch = db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
            db.close()
            lo, hi = traced_span(d)
            a, b = epoch + lo, epoch + hi
            prev = np.concatenate(([ts[0] - 1_000_000_000], ts[:-1]))
            sel = (prev >= a) & (ts <= b)
            span_s = round((b - a) / 1e9, 1)
            try:
                cells = yaml.safe_load(open(os.path.join(d, "spec.yaml")))["point"]["cells"]
            except Exception:
                cells = ""
        if sel.sum() < 5:
            print("skip %s: only %d windows inside the traced span" % (os.path.basename(d), sel.sum()))
            continue
        n = col["n"][sel]
        out.append({"load_point": load, "cells": cells, "tenant": tenant, "batch": batch, "cap_pct": cap,
                    "windows_used": int(sel.sum()), "windows_logged": len(rows), "traced_span_s": span_s, "inferences": int(n.sum()),
                    "fps": round(float(col["fps"][sel].mean()), 1), "fps_all_windows": round(float(col["fps"].mean()), 1),
                    "lat_mean_ms": round(float((col["lat_ms_mean"][sel] * n).sum() / n.sum()), 3),
                    "lat_p50_ms": round(float(np.median(col["lat_ms_p50"][sel])), 3),
                    "lat_p99_ms": round(float(np.median(col["lat_ms_p99"][sel])), 3),
                    "lat_p99_ms_worst_window": round(float(col["lat_ms_p99"][sel].max()), 3),
                    "lat_max_ms": round(float(col["lat_ms_max"][sel].max()), 3),
                    "run_id": os.path.basename(d)})
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(4, 112)
    ax.set_xticks(range(10, 101, 10))
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)


def panel(ax, rows, tenant, batch, stat, lo, hi, handles, scale=1.0):
    """Draw one panel; returns nothing. Lines: lowest / highest load, dashed = tenant alone."""
    def series(load):
        rs = sorted((r for r in rows if r["tenant"] == tenant and r["batch"] == batch and r["load_point"] == load), key=lambda r: r["cap_pct"])
        return [r["cap_pct"] for r in rs], [r[stat] / scale for r in rs]      # scale = ms per y unit (slot length for slots)
    ymax = 0.0
    xs, ys = series("tenant_alone")
    if xs:
        ln, = ax.plot(xs, ys, color=INK2, lw=1.4, ls=(0, (4, 3)), zorder=2)
        handles[ALONE] = ln
        ymax = max(ymax, max(ys))
    ends = []
    for (lbl, colr), load in zip(SERIES, (lo, hi)):
        xs, ys = series(load)
        if not xs:
            continue
        ln, = ax.plot(xs, ys, "-o", color=colr, lw=2, ms=6.5, mec=SURFACE, mew=1.5, zorder=3, label=lbl)
        handles[lbl] = ln
        ymax = max(ymax, max(ys))
        ends.append((xs[-1], ys[-1]))
    ytop = ymax * 1.12 if ymax else 1
    dy = [-3, -3]
    if len(ends) == 2 and abs(ends[0][1] - ends[1][1]) < 0.07 * ytop:
        hi_i = 0 if ends[0][1] >= ends[1][1] else 1
        dy[hi_i], dy[1 - hi_i] = 3, -9
    for (ex, ey), off in zip(ends, dy):
        ax.annotate("%.2f" % ey if ey < 10 else "%.1f" % ey, (ex, ey), textcoords="offset points", xytext=(7, off), fontsize=8, color=INK2, zorder=4)
    ax.set_ylim(0, ytop)
    style(ax)


def finish(fig, title, sub, handles, path, top):
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.01, top + 0.05, sub, fontsize=9, color=INK2)
    keys = [k for k in (SERIES[0][0], SERIES[1][0], ALONE) if k in handles]
    leg = fig.legend([handles[k] for k in keys], keys, loc="upper left", bbox_to_anchor=(0.005, top + 0.045), ncol=3, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(rect=(0, 0, 1, top - 0.01))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", default=None, help="load point used as the lowest cuPHY load (default: the one with the lower tenant latency)")
    ap.add_argument("--hi", default=None)
    ap.add_argument("--slot-ms", type=float, default=0.5, help="slot length for the slot-unit figure (mu 1 = 0.5 ms)")
    a = ap.parse_args()
    rows = collect(a.runs)
    if not rows:
        raise SystemExit("no tenant runs with worker.csv under %s" % a.runs)
    os.makedirs(a.out, exist_ok=True)
    loads = sorted({r["load_point"] for r in rows if r["load_point"] != "tenant_alone"})
    lo, hi = a.lo, a.hi
    if not lo or not hi:
        if len(loads) != 2:
            raise SystemExit("load points %s: pass --lo and --hi" % loads)
        mean = {l: np.mean([r["lat_p50_ms"] for r in rows if r["load_point"] == l]) for l in loads}
        lo, hi = sorted(loads, key=lambda l: mean[l])
    name = {lo: "lowest", hi: "highest", "tenant_alone": "tenant_alone"}
    for r in rows:
        r["lat_p50_slots"] = round(r["lat_p50_ms"] / a.slot_ms, 2)
        r["lat_p99_slots"] = round(r["lat_p99_ms"] / a.slot_ms, 2)
    table = [dict(load=name.get(r["load_point"], r["load_point"]), **r) for r in sorted(rows, key=lambda r: (r["tenant"], r["batch"], r["load_point"] == "tenant_alone", r["load_point"] != lo, r["cap_pct"]))]
    with open(os.path.join(a.out, "tenant_latency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print("wrote", os.path.join(a.out, "tenant_latency.csv"), len(table), "rows; lowest =", lo, "highest =", hi)

    cells = "/".join(str(c) for c in sorted({r["cells"] for r in rows if r["cells"] != ""}))
    combos = sorted({(r["tenant"], r["batch"]) for r in rows})
    note = "lowest vs highest cuPHY load; dashed = the tenant alone on the GPU; value labels = latency at the largest cap"
    same_y = "; same y-scale in both panels"
    stat_note = "p50 / p99 = median over 1 s windows of the window p50 / p99 (no per-inference samples stored)"

    for tn, batch in combos:
        fig, axes = plt.subplots(1, 2, figsize=(4.9 * 2, 3.5 + 1.25), squeeze=False, facecolor=SURFACE)
        handles = {}
        for j, (stat, ttl) in enumerate((("lat_p50_ms", "p50"), ("lat_p99_ms", "p99"))):
            ax = axes[0][j]
            panel(ax, rows, tn, batch, stat, lo, hi, handles)
            ax.set_title(ttl, loc="left", fontsize=11, color=INK, fontweight="bold")
            ax.set_xlabel("tenant SM cap (% of GPU)", fontsize=8.5, color=INK2)
            if j == 0:
                ax.set_ylabel("latency per inference call, batch %d (ms)" % batch, fontsize=8.5, color=INK2)
        ytop = max(ax.get_ylim()[1] for ax in axes[0])      # same y-scale in both panels so p50 and p99 compare directly
        for ax in axes[0]:
            ax.set_ylim(0, ytop)
        finish(fig, "%s, batch %d: inference latency vs tenant SM cap (%s cells)" % (tn.upper(), batch, cells),
               note + same_y + "\n" + stat_note, handles, os.path.join(a.out, "tenant_latency_%s_b%d.png" % (tn, batch)), 0.80)

    ncol = 2
    nrow = int(np.ceil(len(combos) / ncol))
    for scale, ylab, unit_note, fname in (
            (1.0, "p99 latency per inference call (ms)", "", "tenant_latency_p99_overview.png"),
            (a.slot_ms, "p99 latency per inference call (slots)", "; 1 slot = %g ms" % a.slot_ms, "tenant_latency_p99_overview_slots.png")):
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.2 * nrow + 1.25), squeeze=False, facecolor=SURFACE)
        handles = {}
        for i, (tn, batch) in enumerate(combos):
            ax = axes[i // ncol][i % ncol]
            panel(ax, rows, tn, batch, "lat_p99_ms", lo, hi, handles, scale)
            if scale != 1.0:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.set_title("%s, batch %d" % (tn.upper(), batch), loc="left", fontsize=11, color=INK, fontweight="bold")
            if i % ncol == 0:
                ax.set_ylabel(ylab, fontsize=8.5, color=INK2)
            if i // ncol == nrow - 1:
                ax.set_xlabel("tenant SM cap (% of GPU)", fontsize=8.5, color=INK2)
        for j in range(len(combos), nrow * ncol):
            axes[j // ncol][j % ncol].set_visible(False)
        finish(fig, "Tenant p99 inference latency vs tenant SM cap (%s cells)" % cells,
               note + "; y-scale differs per panel" + unit_note + "\n" + stat_note, handles, os.path.join(a.out, fname), 0.86)

if __name__ == "__main__":
    main()
