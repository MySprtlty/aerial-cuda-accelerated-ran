#!/usr/bin/env python3
"""
evaluate: join a decision log with labels on (sfn, slot, cell, channel) and report,
per channel: deadline-miss rate, reliability (= 1 - miss rate), tightness
(= D_c - measured latency, mean over gate-open slots), plus the cap
distribution and the gate-closed ratio of the decision log.

  python3 evaluate.py --decisions out/decisions_ring_59c_8C_yolov8n_b1.csv \
        [--ind ind_sent.csv] [--viol violations.csv | --nvlog phy.log] [--chan-done chan_done.csv] \
        [--rules rules.yaml] [--min-slots 1000000] [--json report.json]

Channels without any label are reported as "unmeasured" (never as 0 misses).
A channel with fewer observed slots than --min-slots is flagged
"statistically unconfirmed". Latency labels (IND_SENT / CHAN_DONE) are
compared against D_c measured from the channel's start reference:
UL channels from T0 + ul_start_offset_us, DL channels from the tick
(T0 - slot_advance slots); both offsets are provisional (rules.yaml label:).
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dapp_est import config as C            # noqa: E402
from dapp_est import decision_log as L      # noqa: E402
from dapp_est import labels as LB           # noqa: E402
from dapp_est import work_formulas as W     # noqa: E402

CHANNELS = W.CHANNELS
UNMEASURED = "unmeasured"
UNCONFIRMED = "statistically unconfirmed"
DEFAULT_MIN_SLOTS = 1000000


def start_ns_of(row, channel, rules):
    """Reference time from which D_c counts, for this decision row."""
    t0 = int(row["t0_ns"])
    lab = rules.raw.get("label", {})
    if channel in LB.UL_CHANNELS:
        return t0 + int(float(lab.get("ul_start_offset_us", 500)) * 1000)
    return t0 - int(float(lab.get("dl_start_offset_us", 1500)) * 1000)


def evaluate(decisions, labels, rules, min_slots):
    per = {}
    n_dec = 0
    caps = Counter()
    closed = 0
    for row in decisions:
        n_dec += 1
        cap = int(row["cap_pct"])
        gate = int(row["gate_slots"])
        caps[cap] += 1
        if gate == 0:
            closed += 1
        sfn, slot = int(row["sfn"]), int(row["slot"])
        n_cells = int(row["n_cells"])
        for ch in CHANNELS:
            if int(row["%s_n" % ch]) == 0:
                continue
            st = per.setdefault(ch, {"active": 0, "observed": 0, "miss": 0, "lat": [], "lat_open": [], "gate_open": 0})
            st["active"] += 1
            if gate > 0:
                st["gate_open"] += 1
            D = rules.deadline_us[ch]
            observed = False
            missed = False
            lat_vals = []
            for cell in range(max(n_cells, 1)):
                key = (sfn, slot, cell, ch)
                if key in labels.done:
                    s_ns, d_ns = labels.done[key]
                    lat_vals.append((d_ns - s_ns) / 1000.0)
                elif key in labels.ind:
                    lat_vals.append((labels.ind[key] - start_ns_of(row, ch, rules)) / 1000.0)
            if lat_vals:
                observed = True
                worst = max(lat_vals)
                st["lat"].append(worst)
                if gate > 0:
                    st["lat_open"].append(worst)
                if worst > D:
                    missed = True
            if labels.violations_for(sfn, slot, ch):
                observed = True
                missed = True
            if observed:
                st["observed"] += 1
                if missed:
                    st["miss"] += 1
    report = {"decisions": n_dec, "gate_closed_ratio": (closed / n_dec) if n_dec else None,
              "cap_distribution": {str(c): k / n_dec for c, k in sorted(caps.items())} if n_dec else {},
              "unattributed_violations": len(labels.unattributed), "channels": {}}
    for ch in CHANNELS:
        st = per.get(ch)
        if st is None:
            continue
        r = {"active_slots": st["active"], "observed_slots": st["observed"], "gate_open_slots": st["gate_open"],
             "D_us": rules.deadline_us[ch]}
        if st["observed"] == 0:
            r["status"] = UNMEASURED
            r["miss_rate"] = None
            r["reliability"] = None
            r["tightness_us"] = None
        else:
            r["miss_rate"] = st["miss"] / st["observed"]
            r["reliability"] = 1.0 - r["miss_rate"]
            lo = st["lat_open"]
            r["tightness_us"] = (rules.deadline_us[ch] - sum(lo) / len(lo)) if lo else None
            r["status"] = "ok" if st["observed"] >= min_slots else UNCONFIRMED
        report["channels"][ch] = r
    return report


def print_report(report):
    print("decisions: %d | gate closed: %s | cap distribution: %s | unattributed violations: %d" % (
        report["decisions"],
        "%.2f%%" % (100 * report["gate_closed_ratio"]) if report["gate_closed_ratio"] is not None else "n/a",
        " ".join("%s:%.1f%%" % (c, 100 * f) for c, f in report["cap_distribution"].items()),
        report["unattributed_violations"]))
    print("| channel | active slots | observed | miss rate | reliability | tightness us | status |")
    print("|---|---|---|---|---|---|---|")
    for ch, r in report["channels"].items():
        mr = "%.4f" % r["miss_rate"] if r["miss_rate"] is not None else "-"
        rl = "%.4f" % r["reliability"] if r["reliability"] is not None else "-"
        tg = "%.0f" % r["tightness_us"] if r["tightness_us"] is not None else "-"
        print("| %s | %d | %d | %s | %s | %s | %s |" % (ch, r["active_slots"], r["observed_slots"], mr, rl, tg, r["status"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--ind", default="", help="IND_SENT csv: sfn,slot,cell,channel,ts_ns")
    ap.add_argument("--viol", default="", help="SLOT_VIOLATION csv: sfn,slot,cell,reason,ts_ns,side")
    ap.add_argument("--nvlog", action="append", default=[], help="Aerial log file(s) to parse for violations")
    ap.add_argument("--chan-done", default="", help="CHAN_DONE csv: sfn,slot,cell,channel,start_ns,done_ns,source")
    ap.add_argument("--rules", default=C.DEFAULT_RULES)
    ap.add_argument("--min-slots", type=int, default=DEFAULT_MIN_SLOTS)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rules = C.Rules(a.rules)
    labels = LB.LabelSet()
    if a.ind:
        labels.add_ind(LB.load_ind_sent(a.ind))
    if a.viol:
        labels.add_violations(LB.load_violations(a.viol))
    for p in a.nvlog:
        labels.add_violations(LB.parse_nvlog(p))
    if a.chan_done:
        labels.add_done(LB.load_chan_done(a.chan_done))
    report = evaluate(L.read_log(a.decisions), labels, rules, a.min_slots)
    print_report(report)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
