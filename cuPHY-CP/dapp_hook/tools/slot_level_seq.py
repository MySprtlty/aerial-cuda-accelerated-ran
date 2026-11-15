#!/usr/bin/env python3
# slot_level_seq.py (2026-11-12, branch feature/slot_level) - per-slot L1 partition sequence for the slot-level gain prediction (tools/slot_level_sim.py).
# Replays a recorded hook ring (FAPI work of every slot, all cells) through the SAME rule as the in-process sizer (include/dapp_hook/dapp_sizer.hpp
# sizer_pick: smallest configured profile whose predicted PUSCH completion / PDSCH duration meets the limit; burst queue of the previous U slot),
# optionally with the PUSCH/PDSCH work scaled (--scale 1.25 = 10 cells from an 8-cell ring: an EXTRAPOLATION, start_us/fixed_us stay as fitted).
# CPU only. Stage 1 caches the per-slot work (<out>.work.json) so other scales / profile sets need no second pass over the ring.
#   slot_level_seq.py --dump prof/ring_16C_sl_pk_pu100.bin --out prof/tenant/slot_level/seq/pk8 [--scale 1.0] [--pusch 20,28,..] [--pdsch 16,24,..]
#   slot_level_seq.py --work prof/tenant/slot_level/seq/pk8.work.json --scale 1.25 --out prof/tenant/slot_level/seq/pk10x
# Output <out>.seq.csv: i,sfn,slot,pos,ul_any,dl_any,ul_sm,dl_sm,pusch_ok,pdsch_ok,pusch_t_us,pdsch_t_us  (ul_sm/dl_sm 0 = no PUSCH / PDSCH work)
import argparse, collections, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "estimator"))

def load_work(dump):
    from dapp_est import slot_context as SCX, work_formulas as W
    rows = []
    for ctx in SCX.contexts_from_file(dump):
        if not ctx.complete: continue
        t = ctx.channel_totals()
        rows.append({"sfn": ctx.sfn, "slot": ctx.slot, "t0": ctx.t0_ns, "pusch": {k: v for k, v in t["PUSCH"].items() if v}, "pdsch": {k: v for k, v in t["PDSCH"].items() if v},
                     "ul": int(any(any(t[c].values()) for c in W.UL_CHANNELS)), "dl": int(any(any(t[c].values()) for c in W.DL_CHANNELS)), "cells": len(ctx.cells)})
    return rows

def pick(ch, completion, work, sms, extra, slot):
    """sizer_pick: (index, timed value, ok)"""
    limit = (ch.complete_by_us - ch.margin_us) if completion else (ch.budget_us - ch.margin_us)
    f = (lambda s: ch.start_for(slot) + extra + ch.work_us(work, s)) if completion else (lambda s: ch.fixed_us + extra + ch.work_us(work, s))
    for i, s in enumerate(sms):
        t = f(s)
        if t <= limit: return i, t, True
    return len(sms) - 1, f(sms[-1]), False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump"); ap.add_argument("--work"); ap.add_argument("--out", required=True); ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--model", default=os.path.join(HERE, "..", "estimator", "sm_model_fit.yaml"))
    ap.add_argument("--pusch", default="20,28,36,44,52,60,68,76,84,92,100,108"); ap.add_argument("--pdsch", default="16,24,32,40,48,56")
    a = ap.parse_args()
    if a.dump:
        rows = load_work(a.dump); json.dump(rows, open(a.out + ".work.json", "w"))
    else: rows = json.load(open(a.work))
    from dapp_est import sm_model as M
    m = M.SmModel(a.model); pu = m.channels["PUSCH"]; pd = m.channels["PDSCH"]
    pus = [int(v) for v in a.pusch.split(",")]; pds = [int(v) for v in a.pdsch.split(",")]
    prev_end = 0; st = collections.defaultdict(collections.Counter); bad = collections.Counter(); cells = collections.Counter(r["cells"] for r in rows)
    with open(a.out + ".seq.csv", "w") as f:
        f.write("i,sfn,slot,pos,ul_any,dl_any,ul_sm,dl_sm,pusch_ok,pdsch_ok,pusch_t_us,pdsch_t_us\n")
        for i, r in enumerate(rows):
            ul_sm = dl_sm = 0; uok = dok = 1; ut = dt = 0.0; slot = r["slot"]; pos = slot % 10
            if r["pusch"]:
                w = {k: v * a.scale for k, v in r["pusch"].items()}; start = pu.start_for(slot); q = 0.0
                if prev_end and r["t0"]:
                    sa = r["t0"] + start * 1e3
                    if prev_end > sa: q = (prev_end - sa) / 1e3
                k, ut, ok = pick(pu, True, w, pus, q, slot); ul_sm = pus[k]; uok = int(ok)
                if r["t0"]: prev_end = r["t0"] + ut * 1e3
                if not ok: bad["PUSCH pos %d" % pos] += 1
            if r["pdsch"]:
                w = {k: v * a.scale for k, v in r["pdsch"].items()}
                k, dt, ok = pick(pd, False, w, pds, 0.0, slot); dl_sm = pds[k]; dok = int(ok)
                if not ok: bad["PDSCH pos %d" % pos] += 1
            st[pos][(ul_sm, dl_sm, r["ul"], r["dl"])] += 1
            f.write("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.1f,%.1f\n" % (i, r["sfn"], slot, pos, r["ul"], r["dl"], ul_sm, dl_sm, uok, dok, ut, dt))
    print("%s: %d slots, cells %s, scale %.3f, PUSCH profiles %s, PDSCH profiles %s" % (a.out, len(rows), dict(cells.most_common(2)), a.scale, pus, pds))
    for pos in sorted(st):
        n = sum(st[pos].values()); print("  pos %d: " % pos + "  ".join("PUSCH %d PDSCH %d ul%d dl%d: %.1f%%" % (k[0], k[1], k[2], k[3], 100.0 * v / n) for k, v in st[pos].most_common(3)))
    print("  over the limit even at the largest profile:", dict(bad) if bad else "none")

if __name__ == "__main__":
    main()
