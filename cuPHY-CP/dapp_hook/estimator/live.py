#!/usr/bin/env python3
"""
live: poll the L1's shm ring, decide at every SLOT_END and publish
{gate_slots, cap_pct, slot_id} in the control block. Pure Python (no numpy).

  python3 live.py --profile profiles/yolov8n_b1.yaml [--ring /aerial_dapp_ring] [--ctrl /aerial_dapp_ctrl]
                  [--log out/decisions_live.csv] [--timing out/timing_live.csv] [--seconds N]

Per slot it logs decode_us (ring records -> context), decide_us (rules a-g),
log_us (decision-record CSV, debug only) and total_us = decode + decide; it
warns when total_us exceeds rules.yaml live.budget_us.
"""
import argparse
import csv
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dapp_est import config as C            # noqa: E402
from dapp_est import control_block as CB    # noqa: E402
from dapp_est import decision_log as L      # noqa: E402
from dapp_est import pipeline as P          # noqa: E402
from dapp_est import ringio as R            # noqa: E402

TIMING_FIELDS = ["ts_ns", "sfn", "slot", "n_records", "decode_us", "decide_us", "log_us", "total_us", "over_budget"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ring", default="/aerial_dapp_ring")
    ap.add_argument("--ctrl", default=CB.DEFAULT_NAME)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--rules", default=C.DEFAULT_RULES)
    ap.add_argument("--cells", default=C.DEFAULT_CELLS)
    ap.add_argument("--log", default="")
    ap.add_argument("--timing", default="")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until SIGINT/SIGTERM)")
    ap.add_argument("--from-start", action="store_true", help="consume what is already in the ring instead of starting at head")
    ap.add_argument("--spin", action="store_true", help="busy-poll instead of sleeping when idle")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    rules = C.Rules(a.rules)
    cells = C.Cells(a.cells)
    profile = C.Profile(a.profile)
    ring = R.LiveRing(a.ring)
    if not a.from_start:
        ring.seek_to_head()
    ctrl = CB.ControlBlock(a.ctrl)
    log = L.CsvLog(a.log) if a.log else None
    tf = open(a.timing, "w", newline="") if a.timing else None
    tw = csv.DictWriter(tf, fieldnames=TIMING_FIELDS) if tf else None
    if tw:
        tw.writeheader()
    runner = P.Runner(rules, cells, profile, mu=ring.header.mu, ctrl=ctrl, log=log)
    budget = rules.live_budget_us
    idle = rules.live_idle_sleep_us / 1e6

    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    t_end = time.monotonic() + a.seconds if a.seconds > 0 else None
    print("live: ring %s gen=%d mu=%d slot_advance=%d cells=%d | profile %s | ctrl %s | budget %.0f us"
          % (a.ring, ring.generation, ring.header.mu, ring.header.slot_advance, ring.header.num_cells,
             profile.name, ctrl.path, budget))

    slot_ns = 0                # time spent in next()+feed since the last decision
    slot_recs = 0
    totals = []
    n_over = 0
    last_warn = 0.0
    restarts = 0
    while not stop:
        if t_end and time.monotonic() >= t_end:
            break
        t0 = time.perf_counter_ns()
        rec = ring.next()
        if rec is None:
            if a.spin:
                continue
            time.sleep(idle)
            continue
        if rec == "RESTARTED":
            restarts += 1
            runner.reset()
            print("live: ring restarted (generation %d), state reset" % ring.generation)
            continue
        decs = runner.feed(rec)
        slot_ns += time.perf_counter_ns() - t0
        slot_recs += 1
        if not decs:
            continue
        for d in decs:
            decide_us = d["proc_us"]
            log_us = runner.last_log_us
            total_us = slot_ns / 1000.0 - log_us
            over = total_us > budget
            totals.append(total_us)
            if over:
                n_over += 1
                now = time.monotonic()
                if now - last_warn > 1.0:
                    print("live: WARNING slot %d.%d took %.0f us (decode %.0f + decide %.0f) > budget %.0f us [%d records]"
                          % (d["sfn"], d["slot"], total_us, total_us - decide_us, decide_us, budget, slot_recs))
                    last_warn = now
            if tw:
                tw.writerow({"ts_ns": d["ts_ns"], "sfn": d["sfn"], "slot": d["slot"], "n_records": slot_recs,
                             "decode_us": round(total_us - decide_us, 1), "decide_us": round(decide_us, 1),
                             "log_us": round(log_us, 1), "total_us": round(total_us, 1), "over_budget": int(over)})
            if not a.quiet and runner.n_decisions % 2000 == 0:
                print("live: %d decisions | last %d.%d gate=%d cap=%d | %.0f us/slot | lost=%d"
                      % (runner.n_decisions, d["sfn"], d["slot"], d["gate_slots"], d["cap_pct"], total_us, ring.lost))
        slot_ns = 0
        slot_recs = 0

    if log:
        log.close()
    if tf:
        tf.close()
    s = sorted(totals)
    if s:
        print("live: done. %d decisions, %d ring records, lost=%d, restarts=%d | per-slot total us p50=%.0f p99=%.0f max=%.0f | over budget: %d (%.2f%%)"
              % (len(s), runner.n_records, ring.lost, restarts, s[len(s) // 2], s[int(len(s) * 0.99)], s[-1],
                 n_over, 100.0 * n_over / len(s)))
    else:
        print("live: done, no decisions (no traffic?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
