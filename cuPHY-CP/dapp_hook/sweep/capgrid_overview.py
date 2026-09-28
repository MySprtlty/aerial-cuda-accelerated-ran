#!/usr/bin/env python3
"""
One-page overview of a cap-grid sweep: cuPHY p99 channel latency vs the tenant's SM cap for every
(tenant, batch) at once, y-axis in slots. Reads <dir>/capgrid_latency.csv (written by plot.py); no GPU, no run dirs.

    python3 capgrid_overview.py --dir results/c16_capgrid [--slot-us 500] [--stat p99_us]

Layout: rows = tenant model and batch, columns = channel. Every column has ONE y-scale, so the rows of a
channel compare directly. Lines = lowest / highest cuPHY load, dashed = the same load without a tenant,
solid dark line = the channel deadline (NVIDIA check_cell_capacity.py constraint).
Output: capgrid_<stat>_overview_slots.png
"""
import argparse
import csv
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# same surface, ink and series colours as plot.py (capgrid figures)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = (("lowest load", "#2a78d6", "lowest"), ("highest load", "#eb6834", "highest"))
ORDER = ("PUSCH", "PUCCH", "PRACH", "SRS", "PDSCH", "PDCCH", "SSB", "CSIRS")


def fmt(v):
    return ("%.2f" % v) if v < 10 else ("%.1f" % v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--slot-us", type=float, default=500.0, help="slot length in microseconds (mu 1 = 500)")
    ap.add_argument("--stat", default="p99_us", choices=("p99_us", "p99_99_us", "mean_us", "max_us"))
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(a.dir, "capgrid_latency.csv"))) if r.get(a.stat) not in ("", None)]
    S = a.slot_us
    chans = [c for c in ORDER if any(r["channel"] == c for r in rows)]
    combos = sorted({(r["tenant"], int(r["batch"])) for r in rows if r["tenant"] != "none"})
    cells = "/".join(sorted({r["cells"] for r in rows}))
    deadline = {c: next(float(r["budget_us"]) for r in rows if r["channel"] == c and r["budget_us"]) / S for c in chans}
    iso = {}
    for load in ("lowest", "highest"):
        for c in chans:
            v = [float(r[a.stat]) for r in rows if r["tenant"] == "none" and r["load"] == load and r["channel"] == c]
            if v:
                iso[(load, c)] = float(np.mean(v)) / S

    def series(tn, b, load, c):
        rs = sorted((r for r in rows if r["tenant"] == tn and r["batch"] == str(b) and r["load"] == load and r["channel"] == c), key=lambda r: int(r["cap_pct"]))
        return [int(r["cap_pct"]) for r in rs], [float(r[a.stat]) / S for r in rs]

    # one y-scale per channel column
    ytop = {}
    for c in chans:
        m = max([deadline[c]] + [v for (l, cc), v in iso.items() if cc == c] +
                [y for tn, b in combos for load in ("lowest", "highest") for y in series(tn, b, load, c)[1]])
        ytop[c] = m * 1.15

    nrow, ncol = len(combos), len(chans)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol + 0.5, 2.35 * nrow + 1.45), squeeze=False, facecolor=SURFACE)
    handles = {}
    for i, (tn, b) in enumerate(combos):
        for j, c in enumerate(chans):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            d = deadline[c]
            ax.axhline(d, color=INK2, lw=1.3, zorder=1)
            handles["deadline (NVIDIA check_cell_capacity.py constraint)"] = plt.Line2D([], [], color=INK2, lw=1.3)
            if i == 0:
                ax.annotate("deadline %s slots" % ("%g" % d), (4.5, d), textcoords="offset points", xytext=(0, 3), fontsize=7.5, color=INK2)
            ends = []
            for lbl, col, load in SERIES:
                if (load, c) in iso:
                    ax.axhline(iso[(load, c)], color=col, lw=1.1, ls=(0, (4, 3)), alpha=0.85, zorder=2)
                    handles["same load, no tenant"] = plt.Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 3)))
                xs, ys = series(tn, b, load, c)
                if not xs:
                    continue
                ln, = ax.plot(xs, ys, "-o", color=col, lw=1.8, ms=5, mec=SURFACE, mew=1.2, zorder=3)
                handles[lbl] = ln
                ends.append((xs[-1], ys[-1]))
            dy = [-3, -3]
            if len(ends) == 2 and abs(ends[0][1] - ends[1][1]) < 0.09 * ytop[c]:
                hi_i = 0 if ends[0][1] >= ends[1][1] else 1
                dy[hi_i], dy[1 - hi_i] = 3, -9
            for (ex, ey), off in zip(ends, dy):
                ax.annotate(fmt(ey), (ex, ey), textcoords="offset points", xytext=(6, off), fontsize=7.5, color=INK2, zorder=4)
            ax.set_ylim(0, ytop[c])
            ax.set_xlim(4, 118)
            ax.set_xticks(range(10, 101, 10))
            ax.set_xticklabels([str(x) if x % 20 == 0 or x == 10 else "" for x in range(10, 101, 10)])
            ax.tick_params(colors=INK2, labelsize=7.5, length=0)
            ax.grid(True, axis="y", color=GRID, lw=1)
            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(c, loc="left", fontsize=11, color=INK, fontweight="bold")
            if i == nrow - 1:
                ax.set_xlabel("tenant SM cap (% of GPU)", fontsize=8, color=INK2)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel("%s latency (slots)" % a.stat.replace("_us", "").replace("p99_99", "p99.99"), fontsize=8, color=INK2)
                ax.annotate("%s, batch %d" % (tn.upper(), b), xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90, va="center", ha="center",
                            fontsize=10, color=INK, fontweight="bold")
    stat_name = a.stat.replace("_us", "").replace("p99_99", "p99.99")
    fig.suptitle("cuPHY %s channel latency vs tenant SM cap, every tenant configuration (%s cells)" % (stat_name, cells),
                 x=0.008, y=0.992, ha="left", fontsize=13, color=INK, fontweight="bold")
    top = 1 - 1.30 / fig.get_figheight()
    fig.text(0.008, 1 - 0.62 / fig.get_figheight(),
             "rows = tenant model and batch, columns = cuPHY channel (one y-scale per column); y in slots, 1 slot = %g us; value labels = latency at the largest cap" % S,
             fontsize=9, color=INK2)
    keys = [k for k in ("lowest load", "highest load", "same load, no tenant", "deadline (NVIDIA check_cell_capacity.py constraint)") if k in handles]
    leg = fig.legend([handles[k] for k in keys], keys, loc="upper left", bbox_to_anchor=(0.003, 1 - 0.66 / fig.get_figheight()), ncol=4, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(rect=(0.012, 0, 1, top), h_pad=1.0, w_pad=0.8)
    out = os.path.join(a.dir, "capgrid_%s_overview_slots.png" % a.stat.replace("_us", ""))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
