#!/usr/bin/env python3
"""
Report of a MIG sweep (profile c16_mig): cuPHY in one MIG instance, the tenant in another.
Reads finished run directories and the saved table of the shared-GPU (MPS) cap grid; no GPU.

    python3 mig_report.py --runs runs/c16_mig --out results/c16_mig --ref results/c16_capgrid/capgrid_latency.csv

Outputs
    mig_validation.csv   one row per run: is the p99 measurable? (exit codes, context->channel mapping, sample completeness,
                         bench stdout mean vs traced mean)
    mig_latency.csv      one row per (load, tenant, batch, channel): n, mean, p50, p99, p99.99, max, slots, % of the deadline,
                         and the same configuration on the shared GPU (MPS, tenant cap 100) for comparison
    mig_p99_channels.png rows = cuPHY load, columns = channel, x = tenant configuration, two series = MIG / shared GPU
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SCHEMES = (("MIG: cuPHY in 4g (64 SMs), tenant in 3g (60 SMs)", "#1baf7a", "mig"),
           ("shared GPU (MPS, 132 SMs), tenant cap 100 %", "#4a3aa7", "mps"))
ORDER = ("PUSCH", "PUCCH", "PRACH", "SRS", "PDSCH", "PDCCH", "SSB", "CSIRS")
# NVIDIA cell-capacity constraints (testBenches/perf/measure/TDD/check_cell_capacity.py:60-101), PRACH = Parallel mode (no -B)
DEADLINE_US = {"PUSCH": 1500, "PUCCH": 1500, "PRACH": 1250, "SRS": 500, "PDSCH": 375, "PDCCH": 375, "SSB": 375, "CSIRS": 375}
TENANTS = (("none", ""), ("yolo26n", 1), ("yolo26n", 8), ("yolo26x", 1), ("yolo26x", 8))
TLABEL = {("none", ""): "none", ("yolo26n", 1): "26n\nb1", ("yolo26n", 8): "26n\nb8", ("yolo26x", 1): "26x\nb1", ("yolo26x", 8): "26x\nb8"}


def read(path, default=""):
    try:
        return open(path).read().strip()
    except Exception:
        return default


def load_runs(runs):
    out = []
    for d in sorted(glob.glob(os.path.join(runs, "*"))):
        rid = os.path.basename(d)
        m = re.match(r"(.+?)__(?:none(\d+)|(yolo\w+?)_cap(\d+)_b(\d+)_p(\d+))$", rid)
        if not m or not os.path.exists(os.path.join(d, "summary.json")) or m.group(1) == "tenant_alone":
            continue
        s = json.load(open(os.path.join(d, "summary.json")))
        st = yaml.safe_load(open(os.path.join(d, "status.yaml"))) if os.path.exists(os.path.join(d, "status.yaml")) else {}
        by = yaml.safe_load(open(os.path.join(d, "bench.yaml")))
        per_pat = {}
        for sl in by.get("slots", []):
            for ch in sl:
                per_pat[ch] = per_pat.get(ch, 0) + 1
        out.append({"run_id": rid, "dir": d, "point": m.group(1), "tenant": m.group(3) or "none", "batch": int(m.group(5)) if m.group(5) else "",
                    "rep": int(m.group(2)) if m.group(2) else 0, "summary": s, "status": st, "per_pattern": per_pat, "cells": by.get("cells")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="", help="capgrid_latency.csv of the shared-GPU sweep with the same points")
    ap.add_argument("--lo", default="c16_L1")
    ap.add_argument("--hi", default="c16_L4")
    ap.add_argument("--slot-us", type=float, default=500.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.runs)
    if not runs:
        raise SystemExit("no finished runs under %s" % a.runs)
    load = {a.lo: "lowest", a.hi: "highest"}

    # ---- validation: can the p99 be measured in every run? ----
    val = []
    for r in runs:
        s, n = r["summary"], r["summary"].get("nsys", {})
        P = n.get("n_patterns") or 0
        row = {"run_id": r["run_id"], "state": r["status"].get("state", "?"), "bench_exit_file": read(os.path.join(r["dir"], "bench.exit")),
               "nsys_exit": read(os.path.join(r["dir"], "nsys.exit")), "mapping_source": n.get("mapping_source"),
               "mapping_mismatch": json.dumps(n.get("mapping_mismatch")), "patterns_traced": P, "pattern_period_us": n.get("pattern_period_us")}
        # collect.py drops the first seconds of the trace, so the reference is the pattern count after that cut:
        # the channel with the most complete patterns, cross-checked against (window - cut) / pattern period
        per = {ch: s["channels"][ch]["nsys"]["n"] / float(r["per_pattern"][ch]) for ch in ORDER
               if s["channels"].get(ch, {}).get("nsys") and ch in r["per_pattern"]}
        Pref = max(per.values()) if per else 0.0
        cut_s = (n.get("t_cut_ns", 0) - n.get("t_first_ns", 0)) / 1e9
        expect = (n.get("window_s", 0) - cut_s) * 1e6 / n["pattern_period_us"] if n.get("pattern_period_us") else 0.0
        row["patterns_after_cut"] = int(round(Pref))
        row["patterns_expected_after_cut"] = int(round(expect))
        worst = min(1.0, Pref / expect) if expect else 0.0
        for ch in ORDER:
            c = s["channels"].get(ch, {}).get("nsys")
            if not c or ch not in r["per_pattern"]:
                continue
            comp = per[ch] / Pref if Pref else 0.0
            worst = min(worst, comp)
            row["n_" + ch] = c["n"]
            sm = s["channels"][ch].get("stdout_mean_us")
            row["stdout_vs_trace_mean_" + ch] = round(sm / c["mean"], 3) if sm and c["mean"] else ""
        row["sample_completeness_min"] = round(worst, 4)
        row["p99_measurable"] = int(row["state"] == "ok" and row["nsys_exit"] == "0" and n.get("mapping_source") == "names"
                                    and not n.get("mapping_mismatch") and worst >= 0.99)
        val.append(row)
    keys = []
    for row in val:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(a.out, "mig_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(val)
    print("validation: %d runs, %d with a measurable p99" % (len(val), sum(v["p99_measurable"] for v in val)))

    # ---- shared-GPU reference (same points, tenant cap 100 or no tenant) ----
    ref = {}
    if a.ref and os.path.exists(a.ref):
        acc = {}
        for r in csv.DictReader(open(a.ref)):
            if r["tenant"] != "none" and r["cap_pct"] != "100":
                continue
            k = (r["load"], r["tenant"], int(r["batch"]) if r["batch"] else "", r["channel"])
            acc.setdefault(k, []).append(r)
        for k, rs in acc.items():
            ref[k] = {s: float(np.mean([float(x[s + "_us"]) for x in rs])) for s in ("mean", "p50", "p99", "p99_99", "max")}

    # ---- latency table (repeats of the no-tenant run are averaged) ----
    acc = {}
    for r in runs:
        if r["point"] not in load:
            continue
        for ch in ORDER:
            c = r["summary"]["channels"].get(ch, {}).get("nsys")
            if c:
                acc.setdefault((load[r["point"]], r["point"], r["tenant"], r["batch"], ch), []).append((c, r))
    table = []
    for (ld, pt, tn, b, ch), cs in sorted(acc.items(), key=lambda kv: (kv[0][0] != "lowest", [t[0] for t in TENANTS].index(kv[0][2]), str(kv[0][3]), ORDER.index(kv[0][4]))):
        st = {s: float(np.mean([c[s] for c, _ in cs])) for s in ("mean", "p50", "p99", "p99_99", "max")}
        d = DEADLINE_US[ch]
        rf = ref.get((ld, tn, b, ch), {})
        table.append({"load": ld, "point": pt, "cells": cs[0][1]["cells"], "tenant": tn, "batch": b, "channel": ch, "runs": len(cs), "n": sum(c["n"] for c, _ in cs),
                      "mean_us": round(st["mean"], 1), "p50_us": round(st["p50"], 1), "p99_us": round(st["p99"], 1), "p99_99_us": round(st["p99_99"], 1),
                      "max_us": round(max(c["max"] for c, _ in cs), 1), "p99_slots": round(st["p99"] / a.slot_us, 3), "deadline_us": d,
                      "p99_pct_of_deadline": round(100 * st["p99"] / d, 1), "p99_99_pct_of_deadline": round(100 * st["p99_99"] / d, 1),
                      "within_deadline_p99": int(st["p99"] <= d),
                      "shared_gpu_p99_us": round(rf["p99"], 1) if rf else "", "shared_gpu_p99_pct_of_deadline": round(100 * rf["p99"] / d, 1) if rf else "",
                      "mig_vs_shared_p99": round(st["p99"] / rf["p99"], 3) if rf else ""})
    with open(os.path.join(a.out, "mig_latency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print("wrote", os.path.join(a.out, "mig_latency.csv"), len(table), "rows")

    # ---- figure ----
    chans = [c for c in ORDER if any(t["channel"] == c for t in table)]
    loads = [l for l in ("lowest", "highest") if any(t["load"] == l for t in table)]
    S = a.slot_us
    ytop = {}
    for c in chans:
        vals = [t["p99_us"] for t in table if t["channel"] == c] + [v["p99"] for k, v in ref.items() if k[3] == c] + [DEADLINE_US[c]]
        ytop[c] = max(vals) / S * 1.15
    fig, axes = plt.subplots(len(loads), len(chans), figsize=(3.05 * len(chans) + 0.5, 2.6 * len(loads) + 1.5), squeeze=False, facecolor=SURFACE)
    handles = {}
    xs = list(range(len(TENANTS)))
    for i, ld in enumerate(loads):
        for j, c in enumerate(chans):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            d = DEADLINE_US[c] / S
            ax.axhline(d, color=INK2, lw=1.3, zorder=1)
            handles["deadline (NVIDIA check_cell_capacity.py constraint)"] = plt.Line2D([], [], color=INK2, lw=1.3)
            ends, drawn = [], []
            for lbl, col, key in SCHEMES:
                ys = []
                for tn, b in TENANTS:
                    if key == "mig":
                        v = [t["p99_us"] for t in table if t["load"] == ld and t["channel"] == c and t["tenant"] == tn and t["batch"] == b]
                        ys.append(v[0] / S if v else np.nan)
                    else:
                        rf = ref.get((ld, tn, b, c))
                        ys.append(rf["p99"] / S if rf else np.nan)
                if np.all(np.isnan(ys)):
                    continue
                ln, = ax.plot(xs, ys, "-o", color=col, lw=1.8, ms=5.5, mec=SURFACE, mew=1.2, zorder=3)
                handles[lbl] = ln
                ok = [k for k in range(len(ys)) if not np.isnan(ys[k])]
                ends.append((xs[ok[-1]], ys[ok[-1]]))
                drawn += [ys[k] for k in ok]
            # end-of-line value labels, pushed apart when the two lines end close together
            dy = [-3] * len(ends)
            if len(ends) == 2 and abs(ends[0][1] - ends[1][1]) < 0.09 * ytop[c]:
                hi_i = 0 if ends[0][1] >= ends[1][1] else 1
                dy[hi_i], dy[1 - hi_i] = 3, -9
            for (ex, ey), off in zip(ends, dy):
                ax.annotate(("%.2f" % ey) if ey < 10 else ("%.1f" % ey), (ex, ey), textcoords="offset points", xytext=(6, off), fontsize=7.5, color=INK2, zorder=4)
            if i == 0:
                # deadline label on the side of the line that has no data next to it; none when both sides are taken (the legend names the line)
                above = any(d < v < d + 0.12 * ytop[c] for v in drawn)
                below = any(d - 0.12 * ytop[c] < v <= d for v in drawn)
                if not above:
                    ax.annotate("deadline %g slots" % d, (-0.4, d), textcoords="offset points", xytext=(0, 3), fontsize=7.5, color=INK2)
                elif not below:
                    ax.annotate("deadline %g slots" % d, (-0.4, d), textcoords="offset points", xytext=(0, -10), fontsize=7.5, color=INK2)
            ax.set_ylim(0, ytop[c])
            ax.set_xlim(-0.5, len(TENANTS) - 0.1)
            ax.set_xticks(xs)
            ax.set_xticklabels([TLABEL[t] for t in TENANTS] if i == len(loads) - 1 else [])
            ax.tick_params(colors=INK2, labelsize=7.5, length=0)
            ax.grid(True, axis="y", color=GRID, lw=1)
            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(c, loc="left", fontsize=11, color=INK, fontweight="bold")
            if i == len(loads) - 1:
                ax.set_xlabel("tenant running next to cuPHY", fontsize=8, color=INK2)
            if j == 0:
                ax.set_ylabel("p99 latency (slots)", fontsize=8, color=INK2)
                ax.annotate("%s cuPHY load" % ld, xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90, va="center", ha="center", fontsize=10, color=INK, fontweight="bold")
    cells = "/".join(sorted({str(t["cells"]) for t in table}))
    H = fig.get_figheight()
    fig.suptitle("cuPHY p99 channel latency: MIG partition vs shared GPU (%s cells)" % cells, x=0.008, y=1 - 0.10 / H, ha="left", va="top", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.008, 1 - 0.62 / H, "rows = cuPHY load, columns = channel (one y-scale per column); x = tenant configuration; y in slots, 1 slot = %g us; value labels = last point" % S,
             fontsize=9, color=INK2)
    keys = [s[0] for s in SCHEMES if s[0] in handles] + [k for k in handles if k.startswith("deadline")]
    leg = fig.legend([handles[k] for k in keys], keys, loc="upper left", bbox_to_anchor=(0.003, 1 - 0.66 / H), ncol=3, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(rect=(0.012, 0, 1, 1 - 1.30 / H), h_pad=1.0, w_pad=0.8)
    out = os.path.join(a.out, "mig_p99_channels.png")
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
