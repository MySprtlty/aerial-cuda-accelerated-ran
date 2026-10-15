#!/usr/bin/env python3
"""safecap_report.py <prefix> [--first-s 10] [--dir prof/run61] [--plot out.png]
Tables of a SafeCap sweep (safecap_runs.sh): runs <prefix>_R0_16c (no tenant), <prefix>_R1_16c (idle tenant) and
<prefix>_<model>_c<cap>_16c.

  1. per (run, channel): GPU run duration p50 / p99 / p99.99 / max (deadline set A, NVIDIA check_cell_capacity.py
     constraints: PUSCH 1500, PUCCH 1500, PRACH 1500, SRS 500, PDSCH 375, PDCCH 375 us), completion after the channel's
     start reference p99.99 / max (set B, rules.yaml D_c: UL from the slot's last symbol, DL from the tick), misses A / B
     and their increase over R0, L1 overload strings, RU DL U-plane late slots.
  2. SafeCap per model = the largest cap with no miss increase over R0 (A) and p99.99 within the deadline (A).
  3. per cap: channel delay increase over R0 (p50 / p99 of the completion after t0, us).
  4. tenant: service time (dispatcher csv: decode + GPU + post = t_start -> t_end), E2E, throughput; slowdown vs the
     alone sweep is left to the reader (profiles).
  --plot: one TDD period of R0 (59c, 80 slots): per slot index the p50 GPU run duration of every channel (paper figure).
Statistics skip the first --first-s seconds. numpy only (matplotlib for --plot).
"""
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import safecap_check as SC   # noqa: E402

CH = ("pusch", "pucch", "prach", "srs", "pdsch", "pdcch")


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def run_stats(name, d, first_s):
    rows = SC.load_rows(name, d, first_s)
    vals = SC.channel_values(rows)
    miss = SC.misses(vals)
    out = {"n_slots": len(rows), "overload": SC.overload(name, d), "ru_late": SC.ru_late(name, d), "ch": {}}
    for ch in CH:
        dur = np.array(vals[ch]["dur"]); done = np.array(vals[ch]["done"])
        out["ch"][ch] = {"n": len(dur), "dur_p50": pct(dur, 50), "dur_p99": pct(dur, 99), "dur_p9999": pct(dur, 99.99),
                         "dur_max": float(dur.max()) if len(dur) else float("nan"),
                         "done_p50": pct(done, 50), "done_p99": pct(done, 99), "done_p9999": pct(done, 99.99),
                         "done_max": float(done.max()) if len(done) else float("nan"),
                         "miss_a": miss[ch][0], "miss_b": miss[ch][1]}
    return out, rows


def tenant_stats(name, d, first_s):
    p = os.path.join(d, "disp_%s.csv" % name)
    if not os.path.exists(p):
        return None
    rows = [r for r in csv.DictReader(open(p)) if r["status"] == "0"]
    if not rows:
        return None
    t0 = min(int(r["t_arrival"]) for r in rows)
    rows = [r for r in rows if int(r["t_arrival"]) - t0 >= first_s * 1e9]
    if not rows:
        return None
    svc = np.array([(int(r["t_end"]) - int(r["t_start"])) / 1e6 for r in rows if int(r["t_end"])])
    e2e = np.array([(int(r["t_reply"]) - int(r["t_arrival"])) / 1e6 for r in rows])
    span = (max(int(r["t_reply"]) for r in rows) - min(int(r["t_arrival"]) for r in rows)) / 1e9
    return {"n": len(rows), "svc_mean": svc.mean(), "svc_p99": pct(svc, 99), "e2e_mean": e2e.mean(), "e2e_p99": pct(e2e, 99),
            "req_s": len(rows) / span if span > 0 else 0.0, "model": rows[-1]["model"]}


def main():
    a = sys.argv[1:]
    first_s, d, plot = 10.0, SC.DEFAULT_DIR, ""
    if "--first-s" in a:
        i = a.index("--first-s"); first_s = float(a[i + 1]); del a[i:i + 2]
    if "--dir" in a:
        i = a.index("--dir"); d = a[i + 1]; del a[i:i + 2]
    if "--plot" in a:
        i = a.index("--plot"); plot = a[i + 1]; del a[i:i + 2]
    prefix = a[0]
    names = sorted(os.path.basename(p)[4:-4] for p in glob.glob(os.path.join(d, "phy_%s_*_16c.csv" % prefix)))
    order = [n for n in names if n.endswith("_R0_16c")] + [n for n in names if n.endswith("_R1_16c")] + \
            sorted([n for n in names if "_R0_" not in n and "_R1_" not in n],
                   key=lambda n: (("26x", "26s", "26n").index(re.search(r"_(26[xsn])_c", n).group(1)), int(re.search(r"_c(\d+)_", n).group(1))))
    stats = {n: run_stats(n, d, first_s) for n in order}
    r0 = order[0] if order and order[0].endswith("_R0_16c") else None
    S0 = stats[r0][0] if r0 else None

    print("## 1. per run and channel (A = GPU run duration us vs NVIDIA constraint; B = completion from the channel start vs rules.yaml D_c)")
    print("| run | channel | n | A p50 / p99 / p99.99 / max | A misses (+vs R0) | B p99.99 / max | B misses (+vs R0) |")
    print("|---|---|---|---|---|---|---|")
    for n in order:
        s = stats[n][0]
        for ch in CH:
            c = s["ch"][ch]
            if not c["n"]:
                continue
            d_a = c["miss_a"] - (S0["ch"][ch]["miss_a"] if S0 else 0)
            d_b = c["miss_b"] - (S0["ch"][ch]["miss_b"] if S0 else 0)
            bcol = ("%.0f / %.0f" % (c["done_p9999"], c["done_max"])) if c["done_p9999"] == c["done_p9999"] else "- (no run-end stamp)"
            print("| %s | %s | %d | %.0f / %.0f / %.0f / %.0f | %d (%+d) | %s | %d (%+d) |" % (
                n.replace(prefix + "_", "").replace("_16c", ""), ch, c["n"], c["dur_p50"], c["dur_p99"], c["dur_p9999"], c["dur_max"],
                c["miss_a"], d_a, bcol, c["miss_b"], d_b))
    print("\n| run | slots | L1 overload strings | RU DL U-plane late packets (all cells) |")
    print("|---|---|---|---|")
    for n in order:
        s = stats[n][0]
        print("| %s | %d | %d | %d |" % (n.replace(prefix + "_", "").replace("_16c", ""), s["n_slots"], s["overload"], s["ru_late"]))

    print("\n## 2. SafeCap per model")
    print("   criterion B: no miss increase over R0 on any channel (rules.yaml completion budgets), no L1 overload increase, RU DL late")
    print("   slots not above 2x R0; criterion A: the same with the NVIDIA duration constraints (note: at 16 cells the later UL slot's PUSCH")
    print("   GPU run exceeds 1500 us in R0 already, so A saturates for PUSCH and only the other channels can flag an increase).")
    for crit, key in (("B", "miss_b"), ("A", "miss_a")):
        for m in ("26x", "26s", "26n"):
            safe, tried = None, []
            for n in order:
                mm = re.search(r"_(26[xsn])_c(\d+)_", n)
                if not mm or mm.group(1) != m:
                    continue
                cap = int(mm.group(2)); s = stats[n][0]
                ok = all(s["ch"][ch][key] <= (S0["ch"][ch][key] if S0 else 0) for ch in CH) and \
                     s["overload"] <= (S0["overload"] if S0 else 0) and \
                     (s["ru_late"] < 0 or S0 is None or S0["ru_late"] < 0 or s["ru_late"] <= 1.2 * S0["ru_late"] + 100)
                tried.append("%d:%s" % (cap, "ok" if ok else "miss"))
                if ok:
                    safe = cap
            print("- %s %s: SafeCap = %s  (caps %s)" % (crit, m, safe if safe is not None else "none", " ".join(tried) or "-"))

    if S0:
        print("\n## 3. completion delay increase over R0 (us, completion after t0: UL GPU end, DL CPU run end + GPU run)")
        print("| run | " + " | ".join("%s p50 / p99" % ch for ch in CH) + " |")
        print("|---|" + "---|" * len(CH))
        for n in order[1:]:
            s = stats[n][0]
            cells = []
            for ch in CH:
                c, c0 = s["ch"][ch], S0["ch"][ch]
                ok = c["n"] and c0["n"] and c["done_p50"] == c["done_p50"] and c0["done_p50"] == c0["done_p50"]
                cells.append("%+.0f / %+.0f" % (c["done_p50"] - c0["done_p50"], c["done_p99"] - c0["done_p99"]) if ok else "-")
            print("| %s | %s |" % (n.replace(prefix + "_", "").replace("_16c", ""), " | ".join(cells)))

    print("\n## 4. tenant (dispatcher csv; service = dapp_sched t_start -> t_end = decode + GPU + post)")
    print("| run | model | n | service mean / p99 ms | E2E mean / p99 ms | req/s |")
    print("|---|---|---|---|---|---|")
    for n in order:
        t = tenant_stats(n, d, first_s)
        if t:
            print("| %s | %s | %d | %.1f / %.1f | %.1f / %.1f | %.2f |" % (n.replace(prefix + "_", "").replace("_16c", ""), t["model"], t["n"],
                                                                      t["svc_mean"], t["svc_p99"], t["e2e_mean"], t["e2e_p99"], t["req_s"]))

    if plot and r0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = stats[r0][1]
        period = 80
        idx = lambda r: (int(r["sfn"]) * 20 + int(r["slot"])) % period   # noqa: E731
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for ch, col in (("pusch", "tab:red"), ("pucch", "tab:orange"), ("prach", "tab:brown"), ("srs", "tab:pink"), ("pdsch", "tab:blue"), ("pdcch", "tab:green")):
            xs, ys = [], []
            for k in range(period):
                v = [SC.fl(r.get(ch + "_gpu_run_us")) for r in rows if idx(r) == k]
                v = [x for x in v if x is not None]
                if v:
                    xs.append(k); ys.append(float(np.median(v)))
            if xs:
                ax.plot(xs, ys, "o-", ms=3, lw=1, color=col, label=ch.upper())
        ax.set_xlabel("slot index in the 80-slot 59c period (DDDSUUDDDD x 8 frames)")
        ax.set_ylabel("GPU run duration, p50 over the run (us)")
        ax.set_title("%s: cuPHY channel GPU time per slot, 16 cells, no tenant" % r0)
        ax.grid(alpha=0.3); ax.legend(ncol=6, fontsize=8)
        fig.tight_layout(); fig.savefig(plot, dpi=120)
        print("\nplot:", plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
