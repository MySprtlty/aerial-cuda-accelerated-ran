#!/usr/bin/env python3
"""
table_policy: the table-based policy (H2). Classifies the live load from the hook ring (dapp_est.load_class: PUSCH PRB
per UL slot, sliding window, hysteresis heavy-immediately / light-after-2-s) and publishes the SafeCap table row of the
class in the control block (ABI 2: one {name, cap_pct, gate_slots} per model; cap 0 = model not allowed). The dispatcher
(e2e/dispatcher.py --policy-source live) picks the model inside the allowed set (rule target or fixed).

  live:    python3 table_policy.py --table safecap_table.yaml [--ring /aerial_dapp_ring] [--ctrl /aerial_dapp_ctrl] [--log x.csv]
  replay:  python3 table_policy.py --table safecap_table.yaml --replay prof/ring_xxx.bin [--truth-period 10.24 --truth-peak 5.83,10.16
           --truth-origin 3.64] [--log x.csv]   -> detection delays and misjudged time against the pattern's known phases

The replay truth: Peak when ((ts - origin) mod period) in [peak_from, peak_to) (phase seconds from SFN 0, the alternating
pattern's Peak frames 583-1015 = 5.83-10.16 s), else Average; ramps count for the class they move toward in the pattern
(the Bresenham ramp is split half/half here, see --truth-ramp).
"""
import argparse
import csv
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from dapp_est import control_block as CB      # noqa: E402
from dapp_est import load_class as LC         # noqa: E402
from dapp_est import ringio as R              # noqa: E402


def publish(ctrl, table, cls, rec):
    """ControlBlock.write(slot_id, sfn, slot, ts_ns, n_decisions, models); wall clock when no record is at hand."""
    if rec is None:
        ctrl.write(0, 0, 0, time.time_ns(), 0, table.entries(cls))
    else:
        ctrl.write(getattr(rec, "slot_id", 0), rec.sfn, rec.slot, rec.ts_ns, 1, table.entries(cls))


def replay(a, table):
    rf = R.RecordFile(a.replay)
    clf = LC.Classifier(table)
    truth_peak = [float(x) for x in a.truth_peak.split(",")]
    def truth(ts):
        ph = ((ts / 1e9) - a.truth_origin) % a.truth_period
        return "peak" if truth_peak[0] <= ph < truth_peak[1] else "average"
    n = 0
    for rec in rf:
        if rec.type == R.REC_UL_TTI:
            clf.feed(rec); n += 1
    clf.finish()
    h = clf.history
    if not h:
        print("no UL_TTI records in", a.replay); return 1
    # per-sample judgement
    mis = 0; mis_peak_as_avg = 0; mis_avg_as_peak = 0
    trans = []       # (truth switch ts, class, detect delay s)
    prev_truth = None; pending = None
    rows = []
    for ts, feat, cls in h:
        tr = truth(ts)
        if tr != cls:
            mis += 1
            if tr == "peak":
                mis_peak_as_avg += 1
            else:
                mis_avg_as_peak += 1
        if prev_truth is not None and tr != prev_truth:
            if pending is not None:
                trans.append((pending[0], pending[1], None))       # never detected before the next truth switch
            pending = (ts, tr)
        if pending is not None and cls == pending[1]:
            trans.append((pending[0], pending[1], (ts - pending[0]) / 1e9)); pending = None
        prev_truth = tr
        rows.append((ts, feat, cls, tr))
    if pending is not None:
        trans.append((pending[0], pending[1], None))
    span = (h[-1][0] - h[0][0]) / 1e9
    slot_s = span / max(1, len(h) - 1)
    print("replay %s: %d UL slots with PUSCH over %.1f s, window %d UL slots, thresholds average <= %.0f / peak >= %.0f PRB per UL slot, light hold %.1f s"
          % (os.path.basename(a.replay), len(h), span, table.window_slots, table.light_max, table.heavy_min, table.light_hold_s))
    feats = sorted(f for _, f, _ in h)
    print("  feature min / p5 / p50 / p95 / max: %.0f / %.0f / %.0f / %.0f / %.0f" % (feats[0], feats[len(feats) // 20], feats[len(feats) // 2], feats[-len(feats) // 20], feats[-1]))
    print("  class switches: %d; misjudged UL slots: %d (%.2f %% of %.1f s = %.2f s): Peak judged Average %d (%.2f s), Average judged Peak %d (%.2f s)"
          % (clf.n_switch, mis, 100.0 * mis / len(h), span, mis * slot_s, mis_peak_as_avg, mis_peak_as_avg * slot_s, mis_avg_as_peak, mis_avg_as_peak * slot_s))
    for ts, cls, dly in trans:
        print("  truth -> %-7s at +%.2f s: detected after %s" % (cls, (ts - h[0][0]) / 1e9, ("%.2f s" % dly) if dly is not None else "NOT DETECTED"))
    det = [d for _, c, d in trans if d is not None]
    for c in ("peak", "average"):
        v = [d for _, cc, d in trans if cc == c and d is not None]
        if v:
            print("  detection delay -> %s: mean %.2f s, max %.2f s, n %d" % (c, sum(v) / len(v), max(v), len(v)))
    if a.log:
        with open(a.log, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["ts_ns", "feature_prb_per_ul_slot", "class", "truth"]); w.writerows(rows)
        print("  log:", a.log)
    return 0


def live(a, table):
    ring = R.LiveRing(a.ring)
    ring.seek_to_head()
    ctrl = CB.ControlBlock(a.ctrl)
    clf = LC.Classifier(table)
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1)); signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    lf = open(a.log, "w", newline="") if a.log else None
    lw = csv.writer(lf) if lf else None
    if lw:
        lw.writerow(["t_wall_ns", "ts_ns", "sfn", "slot", "feature", "class"] + ["cap_%s" % m for m in table.models])
    publish(ctrl, table, clf.cls, None)
    print("table_policy live: ring %s, ctrl %s, table %s, start class %s -> %s" % (a.ring, ctrl.path, table.path, clf.cls, table.entries(clf.cls)), flush=True)
    last_cls, last_pub, t_end = clf.cls, time.monotonic(), (time.monotonic() + a.seconds if a.seconds > 0 else None)
    n_ul = 0
    while not stop:
        if t_end and time.monotonic() >= t_end:
            break
        rec = ring.next()
        if rec is None:
            time.sleep(a.idle_us / 1e6); continue
        if rec == "RESTARTED":
            clf = LC.Classifier(table); publish(ctrl, table, clf.cls, None); last_cls = clf.cls
            print("table_policy: ring restarted, classifier reset", flush=True); continue
        if rec.type != R.REC_UL_TTI:
            continue
        n_ul += 1
        cls = clf.feed(rec)
        now = time.monotonic()
        if cls != last_cls or now - last_pub >= a.period_ms / 1e3:
            publish(ctrl, table, cls, rec); last_pub = now
            if cls != last_cls:
                print("%s: %s -> %s (feature %.0f PRB/UL slot, sfn %d slot %d) caps %s" % (time.strftime("%H:%M:%S"), last_cls, cls,
                      clf.history[-1][1] if clf.history else -1, rec.sfn, rec.slot, [e[1] for e in table.entries(cls)]), flush=True)
            last_cls = cls
            if lw:
                e = table.entries(cls)
                lw.writerow([time.time_ns(), rec.ts_ns, rec.sfn, rec.slot, "%.0f" % (clf.history[-1][1] if clf.history else -1), cls] + [c for (_, c, _) in e])
    print("table_policy: done, %d UL_TTI records, %d class switches, final %s" % (n_ul, clf.n_switch, clf.cls))
    if lf:
        lf.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True); ap.add_argument("--ring", default="/aerial_dapp_ring"); ap.add_argument("--ctrl", default=CB.DEFAULT_NAME)
    ap.add_argument("--log", default=""); ap.add_argument("--seconds", type=float, default=0); ap.add_argument("--period-ms", type=float, default=5.0)
    ap.add_argument("--idle-us", type=float, default=50.0)
    ap.add_argument("--replay", default=""); ap.add_argument("--truth-period", type=float, default=10.24); ap.add_argument("--truth-origin", type=float, default=3.64)
    ap.add_argument("--truth-peak", default="5.83,10.16", help="Peak phase [from,to) seconds in the period")
    a = ap.parse_args()
    table = LC.Table(a.table)
    return replay(a, table) if a.replay else live(a, table)


if __name__ == "__main__":
    sys.exit(main())
