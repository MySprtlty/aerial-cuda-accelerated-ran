#!/usr/bin/env python3
"""h2_refit.py --dump prof/ring_16C_E4_C2_16c.bin --phy prof/run62/phy_E4_C2_16c.csv [--rules estimator/rules.yaml]
    [--cells estimator/cells.yaml] [--max-records N] [--out-rules prof/tmp/tmp_rules_16c.yaml]
16-cell refit of the estimator's work weights (H2, 2026-10-16):
  1. slot contexts from the hook-ring dump (SlotContext.add_pdu over PDU records) -> per (sfn, slot): per-channel per-kernel
     work sums over cells (the estimator's work units) and the TDD position; the previous UL slot's PUSCH work is added as
     a feature of the following DL slots
  2. labels from the phy csv (parse_phylog.py), joined by (sfn, slot) and the record / tick time (same CLOCK_REALTIME):
     completion after the slot air time t0 (us): PUSCH = pusch_gpu_end, PUCCH / PRACH = *_gpu_end, PDSCH = pdsch_run_end +
     pdsch_gpu_run_us, PDCCH = pdcch_setup_start + pdcch_gpu_run_us; and the GPU run durations (*_gpu_run_us)
  3. per channel: (a) residual of the CURRENT rules (base_times with rules.w, a duration) against the measured GPU run
     duration, (b) non-negative least squares of the measured duration on the work units -> new w_ck (per channel, one
     model over all TDD positions, plus per-position intercepts), (c) linear model of the COMPLETION time per TDD position
     (work units + previous-UL-slot PUSCH work) -> residual p50 / p99 / max
  4. writes --out-rules: rules.yaml with weights_us_per_work replaced by the refit (everything else unchanged)
numpy only (NNLS by projected gradient)."""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from dapp_est import ringio as R, slot_context as SCX, config as C, estimator as E   # noqa: E402

CH = ("PUSCH", "PUCCH", "PRACH", "PDSCH", "PDCCH")
LABEL = {"PUSCH": ("pusch_gpu_end", "pusch_gpu_run_us"), "PUCCH": ("pucch_gpu_end", "pucch_gpu_run_us"), "PRACH": ("prach_gpu_end", "prach_gpu_run_us"),
         "PDSCH": (None, "pdsch_gpu_run_us"), "PDCCH": (None, "pdcch_gpu_run_us")}


def nnls(A, y, iters=20000):
    w = np.zeros(A.shape[1]); lr = 1.0 / (np.linalg.norm(A, 2) ** 2 + 1e-9)
    for _ in range(iters):
        g = A.T @ (A @ w - y); w = np.maximum(0.0, w - lr * g)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--phy", required=True)
    ap.add_argument("--rules", default=os.path.join(HERE, "rules.yaml")); ap.add_argument("--cells", default=os.path.join(HERE, "cells.yaml"))
    ap.add_argument("--max-records", type=int, default=6000000); ap.add_argument("--out-rules", default="")
    a = ap.parse_args()
    rules = C.Rules(a.rules); cells = C.Cells(a.cells)
    # 1. contexts
    with open(a.dump, "rb") as f:
        buf = f.read(a.max_records * R.REC_SIZE + 4096)
    off = R.HDR_SIZE if buf[:8] == R.MAGIC else 0
    ctxs, order = {}, []
    for i in range(a.max_records):
        o = off + i * R.REC_SIZE
        if o + R.REC_SIZE > len(buf):
            break
        rec = R.decode(buf, o)
        if rec is None:
            continue
        key = (rec.sfn, rec.slot, rec.ts_ns // 5_000_000_000)      # the same (sfn, slot) recurs every 10.24 s
        c = ctxs.get(key)
        if c is None:
            c = SCX.SlotContext(rec.sfn, rec.slot); c.ts_first_ns = rec.ts_ns; ctxs[key] = c; order.append(key)
        if rec.type in (R.REC_UL_PDU, R.REC_DL_PDU):
            c.add_pdu(rec)
        elif rec.type in (R.REC_UL_TTI, R.REC_DL_TTI):
            c.add_summary(rec)
        elif rec.type == R.REC_SLOT_END:
            c.add_slot_end(rec)
    print("contexts: %d slots from %d records" % (len(ctxs), i))
    # work units per channel/kernel (sum over cells, scaled as the estimator does) + current-rules base times
    feats = {}
    for key in order:
        c = ctxs[key]
        fw = {}
        for ch in CH:
            d = {}
            for cell, cw in c.cell_work.items():
                wk = cw.get(ch)
                if wk:
                    s = cells.scale_of(cell)
                    for k, v in wk.items():
                        d[k] = d.get(k, 0.0) + v * s
            fw[ch] = d
        tot, crit = E.base_times(c, rules, cells)
        feats[key] = (fw, tot, c.ts_first_ns)
    # 2. labels
    rows = list(csv.DictReader(open(a.phy)))
    by = {}
    for r in rows:
        by.setdefault((int(r["sfn"]), int(r["slot"])), []).append(r)
    def match(key):
        sfn, slot, _ = key; cands = by.get((sfn, slot), [])
        ts = feats[key][2]
        best = min(cands, key=lambda r: abs(int(r["t0_ns"]) - ts), default=None)
        return best if best is not None and abs(int(best["t0_ns"]) - ts) < 20_000_000 else None
    joined = [(key, match(key)) for key in order]
    joined = [(k, r) for k, r in joined if r is not None]
    print("joined with the phy csv: %d slots" % len(joined))
    fl = lambda v: float(v) if v not in ("", None) else None   # noqa: E731
    out_w = {}
    print("\n| channel | n | current rules: pred - measured duration us p50 / p99 / max | refit duration residual p50 / p99 / max | completion model residual p50 / p99 / max (per TDD position) |")
    print("|---|---|---|---|---|")
    for ch in CH:
        kern = sorted({k for key, _ in joined for k in feats[key][0][ch]})
        X, ydur, ycomp, pos, pred_old, prevul = [], [], [], [], [], []
        last_pusch_work = 0.0
        for key, r in joined:
            fw, tot, _ = feats[key]
            p = int(r["pos"])
            if fw["PUSCH"]:
                last_pusch_work = sum(fw["PUSCH"].values())
            if not fw[ch]:
                continue
            dur = fl(r.get(LABEL[ch][1]))
            if dur is None:
                continue
            if LABEL[ch][0]:
                comp = fl(r.get(LABEL[ch][0]))
            elif ch == "PDSCH":
                comp = (fl(r.get("pdsch_run_end")) or 0) + dur
            else:
                comp = (fl(r.get("pdcch_setup_start")) or 0) + dur
            X.append([fw[ch].get(k, 0.0) for k in kern]); ydur.append(dur); ycomp.append(comp); pos.append(p); pred_old.append(tot[ch]); prevul.append(last_pusch_work)
        if not X:
            print("| %s | 0 | - | - | - |" % ch); continue
        X = np.array(X); ydur = np.array(ydur); ycomp = np.array(ycomp); pos = np.array(pos); pred_old = np.array(pred_old); prevul = np.array(prevul)
        res_old = pred_old - ydur
        # (b) duration refit: fixed intercept + nonneg weights
        A = np.hstack([np.ones((len(X), 1)), X]); w = nnls(A, ydur); res_new = A @ w - ydur
        out_w[ch] = dict(zip(kern, w[1:].tolist())); out_w[ch]["fixed_us"] = float(w[0])
        # (c) completion per TDD position
        res_c = []
        for p in sorted(set(pos)):
            m = pos == p
            Ap = np.hstack([np.ones((m.sum(), 1)), X[m], prevul[m, None]])
            wp, *_ = np.linalg.lstsq(Ap, ycomp[m], rcond=None); res_c.append(Ap @ wp - ycomp[m])
        res_c = np.concatenate(res_c)
        f = lambda v: "%.0f / %.0f / %.0f" % (np.percentile(np.abs(v), 50), np.percentile(np.abs(v), 99), np.abs(v).max())   # noqa: E731
        print("| %s | %d | %s | %s | %s |" % (ch, len(X), f(res_old), f(res_new), f(res_c)))
        print("   %s: measured duration p50 %.0f us; current rules p50 %.0f; refit w = %s" % (ch, np.median(ydur), np.median(pred_old), {k: ("%.3g" % v) for k, v in out_w[ch].items()}))
    if a.out_rules:
        import re
        s = open(a.rules).read()
        for ch, w in out_w.items():
            for k, v in w.items():
                if k == "fixed_us":
                    continue
                s = re.sub(r"(\n  %s:\n(?:    [a-z0-9]+:.*\n)*?    %s:\s*)[0-9.e+-]+" % (ch, k), r"\g<1>%.4g" % v, s)
        open(a.out_rules, "w").write(s); print("wrote", a.out_rules)
    return 0


if __name__ == "__main__":
    sys.exit(main())
