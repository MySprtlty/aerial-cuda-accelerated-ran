#!/usr/bin/env python3
"""
plot: figures + CSV tables from one or more run manifests.

    python3 plot.py --runs runs/pilot [--runs runs/sweep_...] --out runs/pilot/plots [--deadline PUSCH=2000 ...]

Reads every <runs>/<run_id>/{spec.yaml,summary.json,chan_samples.csv} and writes
  aggregate.csv            one row per (run, channel): stats from nsys samples + bench stdout, tenant fps, dcgm
  deadline_table.csv       per (run, channel): D_c and the fraction of slot executions above it
  <CH>_<axis>.png          exec time (mean, p99, p99.99, max) vs sweep value; lines = cap, panels = tenant x batch (isolation included)
  <CH>_vs_cap.png          exec time vs cap; lines = min / mid / max of every axis; panels = cells
  grid_mcs_snr_<stat>.png  PUSCH heat map (MCS x SNR) for mean and p99
  grid_cb_scatter.png      PUSCH CB grid: sum vs max per-UE (1 UE concentrated vs 4 UE spread)
  tenant_fps_vs_cap.png    tenant fps vs cap, in % of the tenant-alone fps at the same cap/batch when available
Deadlines default to estimator/rules.yaml deadline_us (provisional) unless overridden.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = os.path.join(HERE, "..", "estimator", "rules.yaml")
SLOT_US = 500.0   # mu=1
# Per-channel deadlines = the latency constraints hard-coded in NVIDIA's cell-capacity check for the DDDSUUDDDD, non-mMIMO
# test bench: testBenches/perf/measure/TDD/check_cell_capacity.py:60-101 ("hard coded latency constraints"). A cell count
# passes only if 100 % of the samples of every channel are <= its constraint (same file, :186-191). The measured quantity
# is END - START of one slot execution for UL and DL alike (parser/standard.py:143-194, parser/common.py unpack: itm[1]-itm[0]),
# i.e. exactly chan_samples.dur_us.
#   PUSCH1/PUSCH2 1500 (:63,:66)   PUCCH1/PUCCH2 1500 (:64,:67)   SRS1/SRS2 500 (:68-69)   SSB 375 (:70)
#   PDSCH 375, 500 with DLBFW (:86-89)   PDCCH 375, also with CSI-RS (:91-94)
#   PRACH 1500 in "Sequential" mode, 1250 in "Parallel", 2000 otherwise (:96-101). The mode is "Sequential" only with
#   --pusch_cascaded = bench flag -B (parser/standard.py:56-59, configure.py:64-65); this harness runs without -B -> 1250.
BENCH_BUDGET_US = {"PUSCH": 1500, "PUCCH": 1500, "PRACH": 1250, "SRS": 500, "PDSCH": 375, "PDCCH": 375, "SSB": 375, "CSIRS": 375}
STATS = ["mean", "p99", "p99_99", "max"]
# point-name -> (channel, axis, value). Defaults give the x value of the "default" point on every axis.
AXES = {
    "PUSCH": {"ue": 4, "prb": 136, "layers": 2, "mcs": 16, "snr": 20},
    "PDSCH": {"ue": 8, "prb": 136, "layers": 2, "mcs": 16},
    "PDCCH": {"dci": 4, "al": 4},
    "PUCCH": {"n": 4},
    "CELLS": {"cells": 4},
}
NAME_RE = re.compile(r"^(pusch|pdsch|pdcch|pucch|prach|srs|cells|skew_cell0|grid)_?(.*)$")


def point_axis(pname, spec):
    """Returns (axis_channel, axis, x_value, series_label) for a point name."""
    if pname == "default":
        return None
    m = NAME_RE.match(pname)
    if not m:
        return None
    head, rest = m.group(1), m.group(2)
    if head == "cells":
        return ("CELLS", "cells", float(rest), "")
    if head == "skew_cell0":
        return ("CELLS", "skew_cell0", float(rest), "cell0 heavy")
    if head == "grid":
        return None
    parts = rest.split("_")
    ch = head.upper()
    if ch == "PUCCH":                       # pucch_f2_11bit_n8 -> axis n, series "F2 11bit"
        n = float(parts[-1].lstrip("n"))
        return (ch, "n", n, "%s %s" % (parts[0].upper(), parts[1]))
    if ch == "PRACH":                       # prach_b4_occ4_ra1 -> axis "cfg", categorical
        return (ch, "cfg", rest, "")
    if ch == "SRS":                         # srs_ue8_p2_s4
        return (ch, "cfg", rest, "")
    axis = parts[0]
    val = parts[1] if len(parts) > 1 else ""
    try:
        return (ch, axis, float(val), "")
    except ValueError:
        return (ch, axis, val, "")


def load_runs(dirs):
    runs = []
    for d in dirs:
        for sp in sorted(glob.glob(os.path.join(d, "*", "spec.yaml"))):
            rd = os.path.dirname(sp)
            st = os.path.join(rd, "status.yaml")
            if not os.path.exists(st) or yaml.safe_load(open(st)).get("state") != "ok":
                continue
            spec = yaml.safe_load(open(sp))
            summ = json.load(open(os.path.join(rd, "summary.json"))) if os.path.exists(os.path.join(rd, "summary.json")) else {}
            runs.append({"dir": rd, "spec": spec, "summary": summ, "manifest": os.path.basename(d)})
    return runs


def tenant_key(spec):
    t = spec.get("tenant") or {}
    if t.get("name", "none") == "none":
        return ("none", 100, 1, 0)
    return (t["name"], int(t.get("cap_pct", 100)), int(t.get("batch", 1)), int(t.get("priority", 0)))


def aggregate(runs, deadlines):
    rows = []
    for r in runs:
        spec, s = r["spec"], r["summary"]
        tn, cap, batch, prio = tenant_key(spec)
        pa = point_axis(spec.get("point_name", ""), spec)
        base = {"manifest": r["manifest"], "run_id": spec["run_id"], "point": spec.get("point_name", ""), "cells": spec["point"]["cells"],
                "tenant": tn, "cap": cap, "batch": batch, "priority": prio, "runner": spec.get("runner", ""),
                "axis_channel": pa[0] if pa else "", "axis": pa[1] if pa else "", "x": pa[2] if pa else "", "series": pa[3] if pa else "",
                "tenant_fps": (s.get("tenant") or {}).get("fps_mean", ""), "tenant_lat_p50": (s.get("tenant") or {}).get("lat_ms_p50", ""),
                "tenant_lat_p99": (s.get("tenant") or {}).get("lat_ms_p99", ""),
                "dcgm_smact": ((s.get("dcgm") or {}).get("SMACT") or {}).get("mean", ""), "dcgm_power": ((s.get("dcgm") or {}).get("POWER") or {}).get("mean", ""),
                "pattern_avg_us": s.get("pattern_avg_us", "")}
        samples = None
        csvp = os.path.join(r["dir"], "chan_samples.csv")
        if os.path.exists(csvp):
            samples = defaultdict(list)
            for row in csv.DictReader(open(csvp)):
                samples[row["channel"]].append(float(row["dur_us"]))
        for ch, c in (s.get("channels") or {}).items():
            n = c.get("nsys") or {}
            row = dict(base, channel=ch, stdout_mean_us=c.get("stdout_mean_us", ""), n=n.get("n", ""), mean=n.get("mean", ""),
                       p50=n.get("p50", ""), p99=n.get("p99", ""), p99_99=n.get("p99_99", ""), max=n.get("max", ""))
            d = deadlines.get(ch)
            if d and samples and samples.get(ch):
                v = np.array(samples[ch])
                row["deadline_us"] = d
                row["over_deadline_frac"] = float((v > d).mean())
                row["budget_used_p99_99"] = float(np.percentile(v, 99.99) / d)
                row["headroom_p99_99_us"] = float(d - np.percentile(v, 99.99))
            for k in ("mean", "p99", "p99_99", "max"):
                if fnum(row.get(k)) is not None:
                    row[k + "_slots"] = round(fnum(row[k]) / SLOT_US, 3)
            rows.append(row)
        if not (s.get("channels") or {}):
            rows.append(dict(base, channel=""))
    return rows


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def plot_axis(rows, out, deadlines):
    """exec time vs sweep value, lines = cap, panels = tenant x batch; one figure per (channel measured, axis)."""
    # which channels are measured for an axis: all channels present; the axis belongs to axis_channel but the other channels react too
    axes = sorted(set((r["axis_channel"], r["axis"]) for r in rows if r["axis"]))
    for ach, axis in axes:
        for ch in sorted(set(r["channel"] for r in rows if r["channel"])):
            sel = [r for r in rows if r["channel"] == ch and ((r["axis_channel"] == ach and r["axis"] == axis) or r["point"] == "default")]
            if not any(r["point"] != "default" for r in sel):
                continue
            panels = sorted(set((r["tenant"], r["batch"]) for r in sel), key=lambda t: (t[0] != "none", t))
            fig, axs = plt.subplots(len(STATS), max(1, len(panels)), figsize=(4.2 * max(1, len(panels)), 2.8 * len(STATS)), squeeze=False, sharex=True)
            default_x = AXES.get(ach, {}).get(axis)
            for pi, (tn, batch) in enumerate(panels):
                for si, stat in enumerate(STATS):
                    ax = axs[si][pi]
                    caps = sorted(set(r["cap"] for r in sel if r["tenant"] == tn and r["batch"] == batch), reverse=True)
                    for cap in caps:
                        pts = defaultdict(list)
                        for r in sel:
                            if r["tenant"] != tn or r["batch"] != batch or r["cap"] != cap or r["priority"] != 0:
                                continue
                            x = fnum(r["x"]) if r["point"] != "default" else default_x
                            if x is None or fnum(r[stat]) is None:
                                continue
                            pts[(r["series"], x)].append(fnum(r[stat]))
                        series = sorted(set(k[0] for k in pts))
                        for sname in series:
                            xs = sorted(k[1] for k in pts if k[0] == sname)
                            ys = [np.mean(pts[(sname, x)]) for x in xs]
                            lbl = ("cap %d" % cap if tn != "none" else "isolation") + (" " + sname if sname else "")
                            ax.plot(xs, ys, marker="o", ms=3, label=lbl)
                    if deadlines.get(ch):
                        ax.axhline(deadlines[ch], color="grey", ls="--", lw=0.8)
                    ax.set_ylabel("%s %s (us)" % (ch, stat.replace("_", ".")))
                    ax.grid(alpha=0.3)
                    if si == 0:
                        ax.set_title("%s batch %d" % (tn, batch) if tn != "none" else "no tenant")
                    if si == len(STATS) - 1:
                        ax.set_xlabel("%s %s" % (ach, axis))
                    if si == 0 and pi == 0:
                        ax.legend(fontsize=7)
            fig.suptitle("%s execution time vs %s %s" % (ch, ach, axis))
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(os.path.join(out, "%s_vs_%s_%s.png" % (ch, ach.lower(), axis)), dpi=110)
            plt.close(fig)


def plot_vs_cap(rows, out):
    """exec time vs cap; lines = min/mid/max value of every axis (+ default), panels = cells."""
    for ch in sorted(set(r["channel"] for r in rows if r["channel"])):
        sel = [r for r in rows if r["channel"] == ch and r["priority"] == 0]
        tenants = sorted(set(r["tenant"] for r in sel if r["tenant"] != "none"))
        if not tenants:
            continue
        cells = sorted(set(r["cells"] for r in sel))
        fig, axs = plt.subplots(len(tenants), len(cells), figsize=(4.2 * len(cells), 3.0 * len(tenants)), squeeze=False)
        for ti, tn in enumerate(tenants):
            for ci, nc in enumerate(cells):
                ax = axs[ti][ci]
                groups = defaultdict(dict)   # label -> cap -> value (mean stat)
                for r in sel:
                    if r["cells"] != nc or r["tenant"] not in (tn, "none") or r["batch"] != 1:
                        continue
                    lbl = "default" if r["point"] == "default" else "%s %s=%s" % (r["axis_channel"], r["axis"], r["x"])
                    cap = r["cap"] if r["tenant"] != "none" else 999
                    if fnum(r["mean"]) is not None:
                        groups[lbl][cap] = fnum(r["mean"])
                # keep default + per axis the min/mid/max x
                keep = {"default"}
                by_axis = defaultdict(list)
                for r in sel:
                    if r["point"] != "default" and r["cells"] == nc and fnum(r["x"]) is not None:
                        by_axis[(r["axis_channel"], r["axis"])].append((fnum(r["x"]), "%s %s=%s" % (r["axis_channel"], r["axis"], r["x"])))
                for k, v in by_axis.items():
                    v = sorted(set(v))
                    for x, lbl in (v[0], v[len(v) // 2], v[-1]):
                        keep.add(lbl)
                for lbl in sorted(keep):
                    d = groups.get(lbl, {})
                    xs = sorted(c for c in d if c != 999)
                    if not xs:
                        continue
                    ax.plot(xs, [d[c] for c in xs], marker="o", ms=3, label=lbl)
                    if 999 in d:
                        ax.axhline(d[999], ls=":", lw=0.8, color=ax.lines[-1].get_color())
                ax.set_title("%s, %d cells" % (tn, nc))
                ax.set_xlabel("tenant SM cap (%)")
                ax.set_ylabel("%s mean (us)" % ch)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=6)
        fig.suptitle("%s execution time vs tenant cap (dotted = isolation)" % ch)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(out, "%s_vs_cap.png" % ch), dpi=110)
        plt.close(fig)


def plot_grids(runs, out):
    mcs_snr = defaultdict(dict)
    cb = []
    for r in runs:
        meta = (r["spec"].get("point") or {}).get("meta") or {}
        s = ((r["summary"].get("channels") or {}).get("PUSCH") or {}).get("nsys") or {}
        if meta.get("_grid") == "mcs_snr" and s:
            mcs_snr[(int(meta["_mcs"]), int(meta["_snr"]))] = s
        elif meta.get("_grid") == "cb_count" and s:
            cb.append((int(meta["_cb"]), meta["_dist"], s, r))
    if mcs_snr:
        mcss = sorted(set(k[0] for k in mcs_snr))
        snrs = sorted(set(k[1] for k in mcs_snr))
        for stat in ("mean", "p99"):
            z = np.full((len(mcss), len(snrs)), np.nan)
            for (mm, ss), s in mcs_snr.items():
                z[mcss.index(mm), snrs.index(ss)] = s.get(stat, np.nan)
            fig, ax = plt.subplots(figsize=(6, 7))
            im = ax.imshow(z, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(snrs)))
            ax.set_xticklabels(snrs)
            ax.set_yticks(range(len(mcss)))
            ax.set_yticklabels(mcss)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel("MCS")
            fig.colorbar(im, label="PUSCH %s (us)" % stat)
            ax.set_title("PUSCH 1 UE 136 PRB 2L: %s exec time" % stat)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(os.path.join(out, "grid_mcs_snr_%s.png" % stat), dpi=110)
            plt.close(fig)
    if cb:
        fig, ax = plt.subplots(figsize=(6, 4))
        for dist in sorted(set(c[1] for c in cb)):
            pts = sorted((c[0], c[2]["mean"], c[2]["p99"]) for c in cb if c[1] == dist)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label="%s mean" % dist)
            ax.plot([p[0] for p in pts], [p[2] for p in pts], marker="x", ls="--", label="%s p99" % dist)
        ax.set_xlabel("total code blocks")
        ax.set_ylabel("PUSCH exec time (us)")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("PUSCH time vs CB count: one UE vs four UEs")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(out, "grid_cb_scatter.png"), dpi=110)
        plt.close(fig)


def plot_tenant(rows, out):
    sel = [r for r in rows if r["tenant"] != "none" and fnum(r["tenant_fps"]) is not None]
    if not sel:
        return
    alone = {}   # tenant-alone reference per (tenant, cap, batch); priority does not change the reference
    for r in sel:
        if r["runner"] == "none":
            alone[(r["tenant"], r["cap"], r["batch"])] = fnum(r["tenant_fps"])
    fig, ax = plt.subplots(figsize=(6, 4))
    seen = set()
    for r in sorted(sel, key=lambda r: (r["tenant"], r["batch"], r["cap"])):
        if r["runner"] == "none" or r["point"] != "default":
            continue
        key = (r["tenant"], r["batch"], r["priority"], r["cap"])
        if key in seen:
            continue
        seen.add(key)
    for tn, batch, prio in sorted(set((r["tenant"], r["batch"], r["priority"]) for r in sel if r["runner"] != "none" and r["point"] == "default")):
        pts = sorted((r["cap"], fnum(r["tenant_fps"]), alone.get((tn, r["cap"], batch))) for r in sel
                     if r["tenant"] == tn and r["batch"] == batch and r["priority"] == prio and r["runner"] != "none" and r["point"] == "default")
        pts = [p for p in pts if p[2]]          # only caps with a tenant-alone reference
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [100.0 * p[1] / p[2] for p in pts]
        ax.plot(xs, ys, marker="o", label="%s b%d%s" % (tn, batch, " prio1" if prio else ""))
    ax.set_xlabel("tenant SM cap (%)")
    ax.set_ylabel("fps under cuPHY load (% of tenant-alone fps at the same cap)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("tenant throughput under cuPHY load vs cap")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, "tenant_fps_vs_cap.png"), dpi=110)
    plt.close(fig)


def plot_minmax(rows, out, deadlines, lo="c8_min", hi="c8_max"):
    """The lightest vs the heaviest load point: per-channel pipeline latency.
    minmax_latency_isolation.png  bars = mean (no tenant), whisker to p99.99, short line = budget
    minmax_latency_tenants.png    p99 relative to isolation vs tenant cap, rows = min/max point, cols = tenant
    minmax_latency.csv            every number behind both figures
    Batch 1 only (the cap grid with batch 8 has its own figures and csv: plot_capgrid)."""
    sel = [r for r in rows if r["point"] in (lo, hi) and r["channel"] and r["priority"] == 0 and (r["tenant"] == "none" or r["batch"] == 1)]
    if not sel:
        return
    order = [c for c in ("PUSCH", "PUCCH", "PRACH", "SRS", "PDSCH", "PDCCH", "SSB", "CSIRS") if any(r["channel"] == c for r in sel)]
    table = []
    iso = {}
    for pt in (lo, hi):
        for ch in order:
            rs = [r for r in sel if r["point"] == pt and r["channel"] == ch and r["tenant"] == "none"]
            if rs:
                iso[(pt, ch)] = {k: float(np.mean([fnum(r[k]) for r in rs if fnum(r[k]) is not None])) for k in ("mean", "p99", "p99_99", "max")}
    for r in sel:
        b = iso.get((r["point"], r["channel"]), {})
        d = deadlines.get(r["channel"])
        table.append({"point": r["point"], "cells": r["cells"], "channel": r["channel"], "tenant": r["tenant"], "cap": r["cap"], "batch": r["batch"],
                      "mean_us": r["mean"], "p99_us": r["p99"], "p99_99_us": r["p99_99"], "max_us": r["max"], "n": r["n"],
                      "budget_us": d or "", "p99_99_over_budget": (fnum(r["p99_99"]) / d) if d and fnum(r["p99_99"]) else "",
                      "p99_vs_isolation": (fnum(r["p99"]) / b["p99"]) if b.get("p99") and fnum(r["p99"]) else "",
                      "mean_vs_isolation": (fnum(r["mean"]) / b["mean"]) if b.get("mean") and fnum(r["mean"]) else "",
                      "tenant_fps": r["tenant_fps"], "run_id": r["run_id"]})
    write_csv(os.path.join(out, "minmax_latency.csv"), table)

    # (1) isolation bars
    if iso:
        fig, ax = plt.subplots(figsize=(1.6 * len(order) + 2.5, 4.6))
        x = np.arange(len(order))
        w = 0.36
        for k, (pt, lbl, col) in enumerate(((lo, "lowest load", "#4C9F70"), (hi, "highest load", "#C0504D"))):
            means = [iso.get((pt, ch), {}).get("mean", np.nan) for ch in order]
            tails = [iso.get((pt, ch), {}).get("p99_99", np.nan) for ch in order]
            pos = x + (k - 0.5) * w
            ax.bar(pos, means, w, label="%s: mean" % lbl, color=col)
            ax.vlines(pos, means, tails, color="black", lw=1.2)
            ax.scatter(pos, tails, marker="_", color="black", s=60, zorder=3, label="p99.99" if k == 0 else None)
            for xx, mv, tv in zip(pos, means, tails):
                if not np.isnan(mv):
                    ax.text(xx, tv * 1.08, "%.0f" % mv, ha="center", va="bottom", fontsize=7)
        for i, ch in enumerate(order):
            if deadlines.get(ch):
                ax.hlines(deadlines[ch], i - 0.45, i + 0.45, color="grey", ls="--", lw=1.0, label="budget" if i == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_ylabel("pipeline latency per slot execution (us, log)")
        cells = sorted(set(str(r["cells"]) for r in sel))
        ax.set_title("%s cells, no tenant: per-channel pipeline latency at the lowest vs highest load" % "/".join(cells))
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "minmax_latency_isolation.png"), dpi=120)
        plt.close(fig)

    # (2) tenants: p99 relative to isolation vs cap
    tenants = sorted(set(r["tenant"] for r in sel if r["tenant"] != "none"))
    if tenants and iso:
        fig, axs = plt.subplots(2, len(tenants), figsize=(4.6 * len(tenants), 6.4), squeeze=False, sharex=True)
        for ri, pt in enumerate((lo, hi)):
            for ci, tn in enumerate(tenants):
                ax = axs[ri][ci]
                for ch in order:
                    pts = sorted((r["cap"], fnum(r["p99"]) / iso[(pt, ch)]["p99"]) for r in sel
                                 if r["point"] == pt and r["tenant"] == tn and r["channel"] == ch and r["batch"] == 1 and (pt, ch) in iso and fnum(r["p99"]))
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, label=ch)
                ax.axhline(1.0, color="grey", ls=":", lw=0.8)
                ax.set_title("%s, %s" % ("lowest load" if pt == lo else "highest load", tn))
                ax.set_ylabel("p99 latency / isolation p99")
                ax.grid(alpha=0.3)
                if ri == 1:
                    ax.set_xlabel("tenant SM cap (%)")
                if ri == 0 and ci == 0:
                    ax.legend(fontsize=7, ncol=2)
        fig.suptitle("slow-down of each channel pipeline under a YOLO tenant (batch 1)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(out, "minmax_latency_tenants.png"), dpi=120)
        plt.close(fig)


# ---- cap grid: lowest vs highest load, one figure per (tenant model, batch) ------------------------------------
# Colours: categorical slots 1-2 of the validated reference palette (blue / orange); checked with the dataviz
# validator on the light surface (CVD dE 24.7, normal-vision dE 33.6, contrast >= 3:1). Text stays in ink tokens.
CG_SURFACE, CG_INK, CG_INK2, CG_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
CG_SERIES = (("lowest load", "#2a78d6"), ("highest load", "#eb6834"))


def plot_capgrid(rows, out, deadlines, lo="c8_min", hi="c8_max"):
    """Per-channel p99 pipeline latency vs the tenant's SM cap, lowest vs highest cuPHY load in the same panel.
    capgrid_latency.csv                 every (load, channel, tenant, batch, cap) incl. the no-tenant baseline and tenant-alone fps
    capgrid_p99_<tenant>_b<batch>.png   small multiples (one panel per channel), 2 lines = lowest / highest load,
                                        dashed = the same load without a tenant"""
    sel = [r for r in rows if r["point"] in (lo, hi) and r["channel"] and r["priority"] == 0]
    if not sel:
        return
    order = [c for c in ("PUSCH", "PUCCH", "PRACH", "SRS", "PDSCH", "PDCCH", "SSB", "CSIRS") if any(r["channel"] == c for r in sel)]
    iso = {}
    for pt in (lo, hi):
        for ch in order:
            rs = [r for r in sel if r["point"] == pt and r["channel"] == ch and r["tenant"] == "none"]
            if rs:
                iso[(pt, ch)] = {k: float(np.mean([fnum(r[k]) for r in rs if fnum(r[k]) is not None])) for k in ("mean", "p99", "p99_99")}
    alone = {}
    for r in rows:
        if str(r.get("run_id", "")).startswith("tenant_alone") and fnum(r.get("tenant_fps")) is not None:
            alone[(r["tenant"], r["batch"], r["cap"])] = fnum(r["tenant_fps"])
    table = []
    for r in sorted(sel, key=lambda r: (r["tenant"], r["batch"], r["point"] != lo, order.index(r["channel"]), -r["cap"])):
        b = iso.get((r["point"], r["channel"]), {})
        d = deadlines.get(r["channel"])
        a = alone.get((r["tenant"], r["batch"], r["cap"]))
        fps = fnum(r["tenant_fps"])
        table.append({"load": "lowest" if r["point"] == lo else "highest", "point": r["point"], "cells": r["cells"], "channel": r["channel"],
                      "tenant": r["tenant"], "batch": r["batch"] if r["tenant"] != "none" else "", "cap_pct": r["cap"] if r["tenant"] != "none" else "",
                      "mean_us": r["mean"], "p50_us": r["p50"], "p99_us": r["p99"], "p99_99_us": r["p99_99"], "max_us": r["max"], "n": r["n"],
                      "budget_us": d or "", "p99_budget_pct": round(100 * fnum(r["p99"]) / d, 1) if d and fnum(r["p99"]) else "",
                      "p99_99_budget_pct": round(100 * fnum(r["p99_99"]) / d, 1) if d and fnum(r["p99_99"]) else "",
                      "iso_mean_us": b.get("mean", ""), "iso_p99_us": b.get("p99", ""),
                      "p99_vs_isolation": round(fnum(r["p99"]) / b["p99"], 3) if b.get("p99") and fnum(r["p99"]) else "",
                      "mean_vs_isolation": round(fnum(r["mean"]) / b["mean"], 3) if b.get("mean") and fnum(r["mean"]) else "",
                      "tenant_fps": r["tenant_fps"], "tenant_alone_fps": a if a is not None else "",
                      "tenant_fps_pct_of_alone": round(100 * fps / a, 1) if a and fps is not None else "", "run_id": r["run_id"]})
    write_csv(os.path.join(out, "capgrid_latency.csv"), table)

    combos = sorted({(r["tenant"], r["batch"]) for r in sel if r["tenant"] != "none"})
    ncells = "/".join(str(c) for c in sorted({r["cells"] for r in sel}))
    for tn, batch in combos:
        # two variants: deadline always on the axis (default file name) and a y-axis zoomed to the data (_zoom)
        for always, suffix in ((True, ""), (False, "_zoom")):
            ncol = 3
            nrow = int(np.ceil(len(order) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow + 1.0), squeeze=False, facecolor=CG_SURFACE)
            handles = {}
            for i, ch in enumerate(order):
                ax = axes[i // ncol][i % ncol]
                ax.set_facecolor(CG_SURFACE)
                ymax = 0.0
                pmax = 0.0
                ends = []
                for (lbl, col), pt in zip(CG_SERIES, (lo, hi)):
                    rs = sorted((r for r in sel if r["tenant"] == tn and r["batch"] == batch and r["point"] == pt and r["channel"] == ch and fnum(r["p99"]) is not None),
                                key=lambda r: r["cap"])
                    if not rs:
                        continue
                    xs, ys = [r["cap"] for r in rs], [fnum(r["p99"]) for r in rs]
                    ln, = ax.plot(xs, ys, "-o", color=col, lw=2, ms=6.5, mec=CG_SURFACE, mew=1.5, zorder=3, label=lbl)
                    handles[lbl] = ln
                    ymax = max(ymax, max(ys))
                    pmax = max(pmax, max(ys))
                    ends.append((xs[-1], ys[-1]))
                    b = iso.get((pt, ch))
                    if b:
                        h = ax.axhline(b["p99"], color=col, lw=1.2, ls=(0, (4, 3)), alpha=0.85, zorder=2)
                        handles["same load, no tenant"] = plt.Line2D([], [], color=CG_INK2, lw=1.2, ls=(0, (4, 3)))
                        ymax = max(ymax, b["p99"])
                d = deadlines.get(ch)
                if d and (always or ymax >= 0.75 * d):   # _zoom: only where the data gets close to the deadline
                    ax.axhline(d, color=CG_INK2, lw=1.4, zorder=1)
                    ax.annotate("deadline %d us" % d, (4.5, d), textcoords="offset points", xytext=(0, 3), fontsize=8, color=CG_INK2)
                    handles["deadline (NVIDIA check_cell_capacity.py constraint)"] = plt.Line2D([], [], color=CG_INK2, lw=1.4)
                    ymax = max(ymax, d)
                ytop = ymax * 1.18 if ymax else 1
                # end-of-line value labels; pushed apart when the two lines end close together
                dy = [-3, -3]
                if len(ends) == 2 and abs(ends[0][1] - ends[1][1]) < 0.06 * ytop:
                    hi_i = 0 if ends[0][1] >= ends[1][1] else 1
                    dy[hi_i], dy[1 - hi_i] = 3, -9
                for (ex, ey), off in zip(ends, dy):
                    ax.annotate("%.0f" % ey, (ex, ey), textcoords="offset points", xytext=(7, off), fontsize=8, color=CG_INK2, zorder=4)
                ax.set_ylim(0, ytop)
                ax.set_xlim(4, 112)
                ax.set_xticks(range(10, 101, 10))
                ax.tick_params(colors=CG_INK2, labelsize=8, length=0)
                ax.grid(True, axis="y", color=CG_GRID, lw=1)
                ax.set_axisbelow(True)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.set_title(ch, loc="left", fontsize=11, color=CG_INK, fontweight="bold")
                if d:
                    ax.text(0.99, 1.03, "worst p99 = %.0f %% of the %d us deadline" % (100 * pmax / d, d), transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=CG_INK2)
                if i % ncol == 0:
                    ax.set_ylabel("p99 latency per slot execution (us)", fontsize=8.5, color=CG_INK2)
                if i // ncol == nrow - 1:
                    ax.set_xlabel("tenant SM cap (% of GPU)", fontsize=8.5, color=CG_INK2)
            for j in range(len(order), nrow * ncol):
                axes[j // ncol][j % ncol].set_visible(False)
            fig.suptitle("%s, batch %d: p99 channel pipeline latency vs tenant SM cap (%s cells)" % (tn.upper(), batch, ncells),
                         x=0.01, y=0.99, ha="left", fontsize=12.5, color=CG_INK, fontweight="bold")
            fig.text(0.01, 0.938, ("lowest vs highest cuPHY load in each panel; one panel per channel, y-scale differs per panel; value labels = p99 at the largest cap"
                                   if always else
                                   "y-axis zoomed to the data (deadline drawn only where p99 comes within 75 % of it); value labels = p99 at the largest cap"),
                     fontsize=9, color=CG_INK2)
            keys = [k for k in ("lowest load", "highest load", "same load, no tenant", "deadline (NVIDIA check_cell_capacity.py constraint)") if k in handles]
            leg = fig.legend([handles[k] for k in keys], keys, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=4, frameon=False, fontsize=9.5)
            for t in leg.get_texts():
                t.set_color(CG_INK)
            fig.tight_layout(rect=(0, 0, 1, 0.885))
            fig.savefig(os.path.join(out, "capgrid_p99_%s_b%d%s.png" % (tn, batch, suffix)), dpi=130, facecolor=CG_SURFACE)
            plt.close(fig)


def plot_tradeoff(rows, out, deadlines, lo="c8_min", hi="c8_max", stat="p99_99"):
    """Tenant throughput vs cuPHY budget use as the tenant's SM cap goes from its smallest to its largest value.
    One panel per (tenant, batch); one curve per cuPHY load; every point is one cap. y = the channel closest to its
    budget (stat of the slot execution time / budget, %). The largest cap whose curve has stayed under the budget is
    ringed; when it differs between the loads, the throughput a load-blind (static) cap gives up is annotated.
    tradeoff_<stat>.png, tradeoff.csv (every point), tradeoff_summary.csv (safe caps and the static-vs-load-aware gap)"""
    sel = [r for r in rows if r["point"] in (lo, hi) and r["channel"] and r["priority"] == 0 and r["tenant"] != "none"
           and fnum(r.get(stat)) is not None and deadlines.get(r["channel"])]
    if not sel:
        return
    pts = {}
    for r in sel:
        k = (r["tenant"], r["batch"], "lowest" if r["point"] == lo else "highest", r["cap"])
        u = 100.0 * fnum(r[stat]) / deadlines[r["channel"]]
        u99 = 100.0 * fnum(r["p99"]) / deadlines[r["channel"]] if fnum(r.get("p99")) is not None else None
        e = pts.setdefault(k, {"tenant": k[0], "batch": k[1], "load": k[2], "cap_pct": k[3], "tenant_fps": fnum(r["tenant_fps"]),
                               "worst_util_pct": -1.0, "worst_channel": "", "worst_util_p99_pct": -1.0, "worst_channel_p99": "", "run_id": r["run_id"]})
        if u > e["worst_util_pct"]:
            e["worst_util_pct"], e["worst_channel"] = round(u, 1), r["channel"]
        if u99 is not None and u99 > e["worst_util_p99_pct"]:
            e["worst_util_p99_pct"], e["worst_channel_p99"] = round(u99, 1), r["channel"]
    table = sorted(pts.values(), key=lambda e: (e["tenant"], e["batch"], e["load"] != "lowest", e["cap_pct"]))
    for e in table:
        e["stat"] = stat
        e["within_budget"] = int(e["worst_util_pct"] <= 100.0)
    write_csv(os.path.join(out, "tradeoff.csv"), table)

    combos = sorted({(e["tenant"], e["batch"]) for e in table})
    ncol = 2
    nrow = int(np.ceil(len(combos) / ncol))
    ytop = max(115.0, max(e["worst_util_pct"] for e in table) * 1.12)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.3 * ncol, 3.9 * nrow + 1.1), squeeze=False, sharey=True, facecolor=CG_SURFACE)
    summary, handles = [], {}
    for i, (tn, batch) in enumerate(combos):
        ax = axes[i // ncol][i % ncol]
        ax.set_facecolor(CG_SURFACE)
        ax.axhspan(100, ytop, color="#efede7", zorder=0)
        ax.axhline(100, color=CG_INK2, lw=1.2, zorder=1)
        curves, safe, unsafe_by_load = {}, {}, {}
        for (lbl, col), load in zip(CG_SERIES, ("lowest", "highest")):
            es = [e for e in table if e["tenant"] == tn and e["batch"] == batch and e["load"] == load and e["tenant_fps"] is not None]
            if not es:
                continue
            curves[load] = es
            xs, ys = [e["tenant_fps"] for e in es], [e["worst_util_pct"] for e in es]
            ln, = ax.plot(xs, ys, "-o", color=col, lw=2, ms=6, mec=CG_SURFACE, mew=1.4, zorder=3, label=lbl)
            handles[lbl] = ln
            # largest cap within the budget. The curve is not monotonic: a very small cap stretches every tenant
            # kernel (a 40 us kernel at cap 100 runs ~10x longer at cap 10) and that long, non-preemptible
            # occupancy hurts the cuPHY tails more than a larger cap does, so low caps can fail while larger ones pass.
            within = [e for e in es if e["worst_util_pct"] <= 100.0]
            ok = max(within, key=lambda e: e["cap_pct"]) if within else None
            safe[load] = ok
            unsafe_low = [e["cap_pct"] for e in es if ok is not None and e["cap_pct"] < ok["cap_pct"] and e["worst_util_pct"] > 100.0]
            if unsafe_low:
                ax.annotate("cap %s over budget" % "/".join(str(c) for c in unsafe_low), (es[0]["tenant_fps"], es[0]["worst_util_pct"]),
                            textcoords="offset points", xytext=(8, 8), fontsize=8, color=CG_INK2, zorder=5)
            unsafe_by_load[load] = unsafe_low
            if ok is not None:
                ax.plot([ok["tenant_fps"]], [ok["worst_util_pct"]], "o", ms=11, mfc="none", mec=CG_INK, mew=1.6, zorder=4)
            below = load == "lowest"
            for e in {id(x): x for x in (es[0], es[-1], ok) if x is not None}.values():
                over = e["worst_util_pct"] > 100.0 and below     # the low-load curve shoots up at the right edge: label beside it
                ax.annotate("cap %d" % e["cap_pct"], (e["tenant_fps"], e["worst_util_pct"]), textcoords="offset points",
                            xytext=(-9, -3) if over else (0, -15 if below else 10), ha="right" if over else "center", fontsize=8, color=CG_INK2, zorder=5)
        xmax = max(e["tenant_fps"] for es in curves.values() for e in es)
        ax.set_xlim(0, xmax * 1.12)
        ax.set_ylim(0, ytop)
        ax.text(xmax * 1.11, 101.5, "budget", ha="right", va="bottom", fontsize=8, color=CG_INK2)
        ax.text(xmax * 0.012, ytop * 0.985, "over budget", ha="left", va="top", fontsize=8, color=CG_INK2)
        ax.tick_params(colors=CG_INK2, labelsize=8.5, length=0)
        ax.grid(True, axis="y", color=CG_GRID, lw=1)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title("%s, batch %d" % (tn.upper(), batch), loc="left", fontsize=11, color=CG_INK, fontweight="bold")
        caps = {ld: (safe[ld]["cap_pct"] if safe.get(ld) else 0) for ld in curves}
        note, row = "", {"tenant": tn, "batch": batch, "stat": stat, "safe_cap_lowest": caps.get("lowest", ""), "safe_cap_highest": caps.get("highest", ""),
                         "unsafe_low_caps_lowest": "/".join(str(c) for c in unsafe_by_load.get("lowest", [])),
                         "unsafe_low_caps_highest": "/".join(str(c) for c in unsafe_by_load.get("highest", []))}
        if len(caps) == 2 and not all(caps.values()):
            dead = [ld for ld in caps if not caps[ld]]
            note = "no cap within budget at %s load (tenant must be gated)" % "/".join(dead)
        if len(caps) == 2 and all(caps.values()):
            static = min(caps.values())
            gain_load = max(caps, key=lambda ld: caps[ld])
            f_static = next(e["tenant_fps"] for e in curves[gain_load] if e["cap_pct"] == static)
            e_aware = safe[gain_load]
            row.update(static_safe_cap=static, load_with_headroom=gain_load, fps_static_cap=round(f_static, 1), fps_load_aware=round(e_aware["tenant_fps"], 1),
                       gain_pct=round(100.0 * (e_aware["tenant_fps"] / f_static - 1.0), 1))
            if caps["lowest"] != caps["highest"]:
                e_static = next(e for e in curves[gain_load] if e["cap_pct"] == static)
                yb = min(e_static["worst_util_pct"], e_aware["worst_util_pct"]) - 17
                ax.annotate("", xy=(e_aware["tenant_fps"], yb), xytext=(f_static, yb), arrowprops=dict(arrowstyle="->", color=CG_INK, lw=1.3), zorder=5)
                ax.text((f_static + e_aware["tenant_fps"]) / 2.0, yb - 3, "+%.0f %% throughput at %s load\nif the cap follows the load (%d -> %d)" % (
                    row["gain_pct"], gain_load, static, e_aware["cap_pct"]), ha="center", va="top", fontsize=8.5, color=CG_INK, zorder=5)
                note = "load-blind safe cap = %d" % static
            else:
                note = "same safe cap (%d) at both loads" % static
        summary.append(row)
        ax.text(0.995, 1.035, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=CG_INK2)
        if i % ncol == 0:
            ax.set_ylabel("channel closest to its budget:\n%s latency / budget (%%)" % stat.replace("_", "."), fontsize=9, color=CG_INK2)
        if i // ncol == nrow - 1 or i + ncol >= len(combos):
            ax.set_xlabel("YOLO throughput next to cuPHY (img/s)", fontsize=9, color=CG_INK2)
    for j in range(len(combos), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    write_csv(os.path.join(out, "tradeoff_summary.csv"), summary)
    ncells = "/".join(str(c) for c in sorted({r["cells"] for r in sel}))
    fig.suptitle("What a larger SM cap buys the tenant and costs cuPHY (%s cells, cap 10 -> 100 %%)" % ncells, x=0.01, y=0.99, ha="left", fontsize=12.5, color=CG_INK, fontweight="bold")
    fig.text(0.01, 0.945, "each point = one cap; ring = largest cap that has stayed within the budget; y-scale shared across panels", fontsize=9, color=CG_INK2)
    keys = [k for k in ("lowest load", "highest load") if k in handles]
    leg = fig.legend([handles[k] for k in keys], ["lowest cuPHY load", "highest cuPHY load"][:len(keys)], loc="upper left", bbox_to_anchor=(0.005, 0.938), ncol=2, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(CG_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    fig.savefig(os.path.join(out, "tradeoff_%s.png" % stat), dpi=130, facecolor=CG_SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--deadline", action="append", default=[], help="CH=us, overrides the selected source")
    ap.add_argument("--deadline-source", choices=["bench", "rules"], default="bench",
                    help="bench = NVIDIA test-bench budgets (check_cell_capacity.py), rules = estimator/rules.yaml (provisional)")
    ap.add_argument("--lo", default="", help="point name of the lowest-load point (default: c8_min, else the no-tenant point with the smallest PUSCH mean)")
    ap.add_argument("--hi", default="", help="point name of the highest-load point (default: c8_max, else the largest PUSCH mean)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.runs[0], "plots")
    os.makedirs(out, exist_ok=True)
    deadlines = dict(BENCH_BUDGET_US)
    if a.deadline_source == "rules" and os.path.exists(RULES):
        deadlines = {k: float(v) for k, v in (yaml.safe_load(open(RULES)).get("deadline_us") or {}).items()}
    for d in a.deadline:
        k, v = d.split("=")
        deadlines[k] = float(v)
    runs = load_runs(a.runs)
    rows = aggregate(runs, deadlines)
    write_csv(os.path.join(out, "aggregate.csv"), rows)
    write_csv(os.path.join(out, "deadline_table.csv"), [{k: r.get(k, "") for k in ("run_id", "channel", "tenant", "cap", "batch", "priority", "deadline_us", "over_deadline_frac", "budget_used_p99_99", "headroom_p99_99_us", "p99_99", "max", "p99_99_slots", "max_slots")}
                                                         for r in rows if r.get("deadline_us")])
    plot_axis(rows, out, deadlines)
    plot_vs_cap(rows, out)
    plot_grids(runs, out)
    plot_tenant(rows, out)
    lo, hi = a.lo, a.hi
    if not (lo and hi):
        pts = sorted({r["point"] for r in rows if r["tenant"] == "none" and r["channel"] == "PUSCH" and not str(r["run_id"]).startswith("tenant_alone")})
        if "c8_min" in pts and "c8_max" in pts:
            lo, hi = lo or "c8_min", hi or "c8_max"
        elif pts:
            m = {pt: np.mean([fnum(r["mean"]) for r in rows if r["point"] == pt and r["tenant"] == "none" and r["channel"] == "PUSCH"]) for pt in pts}
            lo, hi = lo or min(m, key=m.get), hi or max(m, key=m.get)
    if lo and hi:
        print("lowest/highest load points: %s / %s" % (lo, hi))
        plot_minmax(rows, out, deadlines, lo=lo, hi=hi)
        plot_capgrid(rows, out, deadlines, lo=lo, hi=hi)
        plot_tradeoff(rows, out, deadlines, lo=lo, hi=hi, stat="p99")
        plot_tradeoff(rows, out, deadlines, lo=lo, hi=hi, stat="p99_99")      # last: tradeoff.csv / tradeoff_summary.csv describe the p99.99 criterion
    print("plots -> %s (%d runs, %d rows)" % (out, len(runs), len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
