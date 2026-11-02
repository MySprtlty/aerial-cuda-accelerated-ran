#!/usr/bin/env python3
"""min_sm.py: per-slot minimum SM partition of every cuPHY channel from the hook ring (sm_model.yaml, dapp_est/sm_model.py).

  replay:  python3 min_sm.py --dump ../../../prof/ring_16C_sl_pk_pu100.bin [--model sm_model.yaml] [--out out/min_sm_<name>.csv]
  live:    python3 min_sm.py --live [--ring /aerial_dapp_ring] [--seconds 60] [--out out/min_sm_live.csv]

One row per closed slot context: <ch>_s (partition: s_min for dynamic channels, s_fixed for fixed ones), <ch>_t (predicted GPU
duration at that partition), <ch>_ok (budget met), the PUSCH burst bookkeeping (start / queue / end relative to t0) and the
footprint / tenant remainder of the layout in sm_model.yaml. The summary prints, per TDD position, the p50 / p99 / max of the
dynamic channels' s_min, the share of infeasible slots and the tenant SM distribution."""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from dapp_est import ringio as R, slot_context as SCX, sm_model as M   # noqa: E402


def pct(v, p):
    if not v:
        return float("nan")
    v = sorted(v); i = min(len(v) - 1, int(round((len(v) - 1) * p / 100.0)))
    return v[i]


def summarize(rows, model):
    dyn = [c for c, ch in model.channels.items() if ch.cls == "dynamic"]
    by_pos = defaultdict(list)
    for r in rows:
        by_pos[r["pos"]].append(r)
    print("== %d slots" % len(rows))
    for c in dyn:
        print("-- %s s_min per TDD position (p50 / p99 / max), predicted duration at s_min p99, infeasible slots" % c)
        for p in sorted(by_pos):
            rs = [r for r in by_pos[p] if r.get(c + "_t", 0) > 0]
            if not rs:
                continue
            sv = [r[c + "_s"] for r in rs]; tv = [r[c + "_t"] for r in rs]; bad = sum(1 for r in rs if not r[c + "_ok"])
            print("   pos %d: n %6d | s_min %3d / %3d / %3d | T@s_min p99 %6.0f us | infeasible %d" % (p, len(rs), pct(sv, 50), pct(sv, 99), max(sv), pct(tv, 99), bad))
    fixed = [(c, ch) for c, ch in model.channels.items() if ch.cls == "fixed"]
    print("-- fixed channels: partition, predicted duration p99, budget misses")
    for c, ch in fixed:
        rs = [r for r in rows if r.get(c + "_t", 0) > 0]
        if rs:
            print("   %-6s s_fixed %3d (%s) | T p99 %6.0f us | misses %d / %d" % (c, ch.s_fixed, ch.context, pct([r[c + "_t"] for r in rs], 99), sum(1 for r in rs if not r[c + "_ok"]), len(rs)))
    q = [r.get("pusch_queue_us", 0.0) for r in rows if r.get("pusch_queue_us") is not None and r.get("PUSCH_t", 0) > 0]
    if q:
        print("-- PUSCH burst queueing wait: p50 %.0f p99 %.0f max %.0f us over %d U slots" % (pct(q, 50), pct(q, 99), max(q), len(q)))
    t = [r["tenant_sm"] for r in rows]
    print("-- tenant SMs (sm_total %d - footprint - headroom %d): p50 %d p1 %d min %d | footprint p99 %d" % (
        model.sm_total, model.headroom, pct(t, 50), pct(t, 1), min(t), pct([r["footprint_sm"] for r in rows], 99)))


def run_replay(a, model):
    rows = []
    prev_ul = None
    for ctx in SCX.contexts_from_file(a.dump):
        if not ctx.complete:
            continue
        r = model.plan(ctx, prev_ul)
        if r.get("PUSCH_t", 0) > 0:
            prev_ul = (r["pusch_end_rel_us"], ctx.t0_ns)
        rows.append(r)
    return rows


def run_live(a, model):
    ring = R.LiveRing(a.ring)
    b = SCX.ContextBuilder()
    rows = []; prev_ul = None; t_end = time.time() + a.seconds; last_print = time.time(); n_dec = 0; dec_us = []
    ring.seek_to_head()
    while time.time() < t_end:
        rec = ring.next()
        if rec is None:
            time.sleep(0.00005); continue
        if rec == "RESTARTED":
            b = SCX.ContextBuilder(); prev_ul = None; continue
        for ctx in b.feed(rec):
            if not ctx.complete:
                continue
            t0 = time.perf_counter()
            r = model.plan(ctx, prev_ul)
            dec_us.append((time.perf_counter() - t0) * 1e6)
            if r.get("PUSCH_t", 0) > 0:
                prev_ul = (r["pusch_end_rel_us"], ctx.t0_ns)
            rows.append(r); n_dec += 1
        if time.time() - last_print >= 1.0:
            last_print = time.time()
            last = rows[-2000:]
            print("%s slots %d | PUSCH s_min p99 %s | PDSCH s_min p99 %s | tenant p1 %s | plan p99 %.0f us | lost %d" % (
                time.strftime("%H:%M:%S"), n_dec, pct([r["PUSCH_s"] for r in last if r.get("PUSCH_t", 0) > 0], 99),
                pct([r["PDSCH_s"] for r in last if r.get("PDSCH_t", 0) > 0], 99), pct([r["tenant_sm"] for r in last], 1), pct(dec_us[-2000:], 99), ring.lost), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump"); ap.add_argument("--live", action="store_true"); ap.add_argument("--ring", default="/aerial_dapp_ring")
    ap.add_argument("--seconds", type=float, default=60); ap.add_argument("--model", default=os.path.join(HERE, "sm_model.yaml")); ap.add_argument("--out", default="")
    a = ap.parse_args()
    model = M.SmModel(a.model)
    if model.provisional:
        print("note: %s is provisional (not refit from the SM ladder yet)" % a.model)
    rows = run_live(a, model) if a.live else run_replay(a, model)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=M.plan_fields(model), extrasaction="ignore"); w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
        print("written %s" % a.out)
    summarize(rows, model)


if __name__ == "__main__":
    main()
