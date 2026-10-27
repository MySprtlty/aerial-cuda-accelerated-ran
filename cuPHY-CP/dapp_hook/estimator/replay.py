#!/usr/bin/env python3
"""
replay: feed recorded ring dumps through the estimator, all profiles decided side by side
(one Runner per dump, like live.py), and write

  <out>/decisions_<dump>_<profile>.csv   one row per slot (dapp_est/decision_log.py)
  <out>/summary.csv                      one row per (dump, profile): cap distribution, gate-closed %, latencies
  <out>/<dump>_<profile>.png             cap / gate / latency-vs-deadline over time (needs matplotlib)

Example:
  python3 replay.py --dump ../../../prof/ring_59c_8C.bin --profile profiles/yolov8n_b1.yaml \
                    --profile profiles/yolov8m_b8.yaml --out-dir out --plot

--ctrl NAME also writes the control block (ABI 2, one entry per profile) while replaying, and
--realtime paces the replay by the ring timestamps and stamps the block with the wall clock, so a
live reader (dapp_sched --ctrl, e2e/dispatcher.py --policy-source live) can be driven from a
recorded ring without an L1.
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dapp_est import config as C          # noqa: E402
from dapp_est import control_block as CB  # noqa: E402
from dapp_est import decision_log as L    # noqa: E402
from dapp_est import pipeline as P        # noqa: E402
from dapp_est import ringio as R          # noqa: E402

SUMMARY_FIELDS = ["dump", "profile", "n_cells", "slots", "decisions", "gate_closed_pct", "gate_open_pct",
                  "cap_mean", "cap_dist", "lat_PUSCH_p50", "lat_PUSCH_max", "lat_PDSCH_p50", "lat_PDSCH_max",
                  "D_PUSCH", "D_PDSCH", "stale", "proc_us_p50", "proc_us_p99", "proc_us_max", "wall_s"]


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def summarize(decs, rules, dump, profile, n_cells, wall_s, stale):
    n = len(decs)
    caps = Counter(d["cap_pct"] for d in decs)
    closed = sum(1 for d in decs if d["gate_slots"] == 0)
    lp = [d["lat_PUSCH"] for d in decs if d["lat_PUSCH"] > 0]
    ld = [d["lat_PDSCH"] for d in decs if d["lat_PDSCH"] > 0]
    pr = [d["proc_us"] for d in decs]
    return {
        "dump": dump, "profile": profile, "n_cells": n_cells, "slots": n, "decisions": n,
        "gate_closed_pct": round(100.0 * closed / n, 2) if n else 0,
        "gate_open_pct": round(100.0 * (n - closed) / n, 2) if n else 0,
        "cap_mean": round(sum(c * k for c, k in caps.items()) / n, 1) if n else 0,
        "cap_dist": " ".join("%d:%.1f%%" % (c, 100.0 * k / n) for c, k in sorted(caps.items())),
        "lat_PUSCH_p50": pct(lp, 50), "lat_PUSCH_max": max(lp) if lp else 0,
        "lat_PDSCH_p50": pct(ld, 50), "lat_PDSCH_max": max(ld) if ld else 0,
        "D_PUSCH": rules.deadline_us["PUSCH"], "D_PDSCH": rules.deadline_us["PDSCH"],
        "stale": stale,
        "proc_us_p50": pct(pr, 50), "proc_us_p99": pct(pr, 99), "proc_us_max": max(pr) if pr else 0,
        "wall_s": round(wall_s, 1),
    }


def plot(decs, rules, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available: no plot; run with ~/.venvs/dapp_yolo/bin/python for figures)")
        return False
    t0 = decs[0]["ts_ns"]
    t = [(d["ts_ns"] - t0) / 1e9 for d in decs]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].step(t, [d["cap_pct"] for d in decs], where="post", lw=0.8)
    ax[0].set_ylabel("cap_pct")
    ax[0].set_ylim(-5, 105)
    ax[0].set_title(title)
    ax[1].step(t, [d["gate_slots"] for d in decs], where="post", lw=0.8, color="tab:green")
    ax[1].set_ylabel("gate_slots")
    for ch, col in (("PUSCH", "tab:red"), ("PDSCH", "tab:blue")):
        ax[2].plot(t, [d["lat_%s" % ch] or float("nan") for d in decs], ".", ms=1.5, color=col, label="lat %s" % ch)
        ax[2].axhline(rules.deadline_us[ch], color=col, ls="--", lw=0.8, label="D_%s" % ch)
    ax[2].set_ylabel("us")
    ax[2].set_xlabel("s")
    ax[2].legend(loc="upper right", fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def dump_name(spec):
    if "=" in spec and not os.path.exists(spec):
        name, path = spec.split("=", 1)
        return name, path
    return os.path.splitext(os.path.basename(spec))[0], spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", action="append", required=True, help="ring dump (raw stream or shm image); name=path allowed")
    ap.add_argument("--profile", action="append", required=True, help="tenant profile yaml")
    ap.add_argument("--rules", default=C.DEFAULT_RULES)
    ap.add_argument("--cells", default=C.DEFAULT_CELLS)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--max-slots", type=int, default=0)
    ap.add_argument("--ctrl", default="", help="also write this control block (e.g. /aerial_dapp_ctrl) while replaying")
    ap.add_argument("--realtime", action="store_true", help="pace the replay by the ring timestamps, wall-clock stamps in the block")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    rules = C.Rules(a.rules)
    cells = C.Cells(a.cells)
    profiles = C.load_profiles(a.profile)
    if len(profiles) > CB.MAX_MODELS:
        ap.error("at most %d profiles" % CB.MAX_MODELS)
    ctrl = CB.ControlBlock(a.ctrl) if a.ctrl else None
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for spec in a.dump:
        name, path = dump_name(spec)
        rf = R.RecordFile(path)
        mu = rf.header.mu if rf.header else 1
        n_cells_hdr = rf.header.num_cells if rf.header else 0
        logs = {p.name: L.CsvLog(os.path.join(a.out_dir, "decisions_%s_%s.csv" % (name, p.name))) for p in profiles}
        runner = P.Runner(rules, cells, profiles, mu=mu, ctrl=ctrl, ctrl_wall_clock=a.realtime)
        t = time.perf_counter()
        decs = []
        t_wall0, ts0 = time.monotonic(), None
        for rec in rf:
            if a.realtime:
                if ts0 is None:
                    ts0 = rec.ts_ns
                ahead = (rec.ts_ns - ts0) / 1e9 - (time.monotonic() - t_wall0)
                if ahead > 0.0005:
                    time.sleep(ahead)
            for d in runner.feed(rec):
                logs[d["profile"]].write(d)
                decs.append(d)
            if a.max_slots and len(decs) >= a.max_slots * len(profiles):
                break
        else:
            for d in runner.flush():
                logs[d["profile"]].write(d)
                decs.append(d)
        wall = time.perf_counter() - t
        for lg in logs.values():
            lg.close()
        for prof, st in zip(profiles, runner.states):
            pd = [d for d in decs if d["profile"] == prof.name]
            n_cells = max((d["n_cells"] for d in pd), default=n_cells_hdr)
            s = summarize(pd, rules, name, prof.name, n_cells, wall, st.n_stale)
            rows.append(s)
            if not a.quiet:
                print("== %s x %s: %d records -> %d decisions in %.1fs | cells=%d gate closed %.1f%% | cap %s | "
                      "lat PUSCH p50/max %.0f/%.0f (D %.0f) PDSCH %.0f/%.0f (D %.0f) | proc_us p50/p99 %.0f/%.0f"
                      % (name, prof.name, runner.n_records, len(pd), wall, n_cells, s["gate_closed_pct"], s["cap_dist"],
                         s["lat_PUSCH_p50"], s["lat_PUSCH_max"], s["D_PUSCH"], s["lat_PDSCH_p50"], s["lat_PDSCH_max"],
                         s["D_PDSCH"], s["proc_us_p50"], s["proc_us_p99"]))
                reasons = Counter(d["reason"].split(" (")[0] for d in pd)
                print("   reasons: " + ", ".join("%s=%d" % kv for kv in reasons.most_common()))
            if a.plot and pd:
                plot(pd, rules, os.path.join(a.out_dir, "%s_%s.png" % (name, prof.name)), "%s / %s" % (name, prof.name))
    with open(os.path.join(a.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    # comparison table (markdown)
    print("\n| dump | profile | cells | slots | gate closed % | cap dist | PUSCH p50/max us | PDSCH p50/max us | proc p50/p99 us |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in rows:
        print("| %s | %s | %d | %d | %.1f | %s | %.0f/%.0f | %.0f/%.0f | %.0f/%.0f |" % (
            s["dump"], s["profile"], s["n_cells"], s["slots"], s["gate_closed_pct"], s["cap_dist"],
            s["lat_PUSCH_p50"], s["lat_PUSCH_max"], s["lat_PDSCH_p50"], s["lat_PDSCH_max"], s["proc_us_p50"], s["proc_us_p99"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
