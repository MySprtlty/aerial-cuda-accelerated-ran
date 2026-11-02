#!/usr/bin/env python3
"""sm_refit.py --dir prof/sm_ladder --runs sl_pk_pu100 sl_pk_pu80 ... [--model sm_model.yaml] [--out sm_model_fit.yaml]
              [--max-records N] [--runlog prof/sm_ladder_runs.log]
Fit the SM sizing model (sm_model.yaml) from the SM ladder runs (fullstack_run.sh GREEN=1 SM_OVERRIDE=... RECORD=1):
  partition per run : console_<run>.log  "<CH> green context with SM count of N."  (PUSCH, PUCCH, PRACH, PDSCH, DL Ctrl, UL order, GPU comm)
  work per slot     : prof/ring_16C_<run>.bin (+.hdr), dapp_est slot contexts (work units per channel, all cells)
  duration per slot : phy_<run>.csv (parse_phylog.py)  {pusch,pucch,prach,pdsch,pdcch}_gpu_run_us, joined by (sfn, slot) and time
Per channel: non-negative least squares of the duration on [1, work_k, work_k * s_ref / s] over all runs -> fixed_us, w_ins_k,
w_par_k; margin_us = p99 of (measured - predicted). A channel seen at a single partition gets an insensitive-only fit (w_par 0).
Also reported: measured duration p50 / p99 per (channel, partition) against the budget (-> s_fixed / s_min choice), the PUSCH GPU
start offset per U position (-> ul_start_offset_us), and, from --runlog, the L1 error lines of every run."""
import argparse
import csv
import os
import re
import sys

import numpy as np
import yaml
from scipy.optimize import nnls, lsq_linear

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from dapp_est import ringio as R, slot_context as SCX, work_formulas as W, sm_model as M   # noqa: E402

CTX_OF = {"PUSCH": "PUSCH", "PDSCH": "PDSCH", "PDCCH": "DL_CTRL", "SSB": "DL_CTRL", "CSIRS": "DL_CTRL", "PUCCH": "PUCCH", "PRACH": "PRACH", "SRS": "SRS"}
CONSOLE_CTX = {"PUSCH": "PUSCH", "PUCCH": "PUCCH", "PRACH": "PRACH", "PDSCH": "PDSCH", "DL Ctrl": "DL_CTRL", "UL order": "ORDER", "GPU comm": "COMMS", "SRS": "SRS"}
DUR_COL = {"PUSCH": "pusch_gpu_run_us", "PUCCH": "pucch_gpu_run_us", "PRACH": "prach_gpu_run_us", "PDSCH": "pdsch_gpu_run_us", "PDCCH": "pdcch_gpu_run_us"}


def completion(ch, row):
    """Completion after t0 (us) of the channel in this slot row (parse_phylog.py columns)."""
    if ch == "PUSCH":
        return fl(row.get("pusch_gpu_end"))
    if ch == "PUCCH":
        return fl(row.get("pucch_gpu_end"))
    if ch == "PRACH":
        return fl(row.get("prach_gpu_end"))
    if ch == "PDSCH":
        v = fl(row.get("comp_wait_end"))          # compression done (DL deadline = fronthaul TX start, t0 - 470 us in the lab timing)
        if v is None:
            e, d = fl(row.get("pdsch_run_end")), fl(row.get("pdsch_gpu_run_us"))
            v = (e + d) if (e is not None and d is not None) else None
        return v
    if ch == "PDCCH":
        e, d = fl(row.get("pdcch_setup_start")), fl(row.get("pdcch_gpu_run_us"))
        return (e + d) if (e is not None and d is not None) else None
    return None


def partitions(console):
    """Partition per green context from the L1 console (run2_<run>.log, fullstack_run.sh); console_<run>.log as fallback."""
    out = {}
    cand = [os.path.join(os.path.dirname(console), os.path.basename(console).replace("console_", "run2_")), console]
    src = next((c for c in cand if os.path.exists(c)), console)
    for ln in open(src, errors="replace"):
        m = re.search(r"(PUSCH|PUCCH|PRACH|PDSCH|DL Ctrl|UL order|GPU comm|SRS) green context with SM count of (\d+)", ln)
        if m:
            out[CONSOLE_CTX[m.group(1)]] = int(m.group(2))
    return out


def contexts(dump, max_records):
    with open(dump, "rb") as f:
        buf = f.read(max_records * R.REC_SIZE + 4096)
    off = R.HDR_SIZE if buf[:8] == R.MAGIC else 0
    ctxs, order = {}, []
    i = 0
    while True:
        o = off + i * R.REC_SIZE
        if o + R.REC_SIZE > len(buf) or i >= max_records:
            break
        rec = R.decode(buf, o); i += 1
        if rec is None:
            continue
        key = (rec.sfn, rec.slot, rec.ts_ns // 5_000_000_000)
        c = ctxs.get(key)
        if c is None:
            c = SCX.SlotContext(rec.sfn, rec.slot); c.ts_first_ns = rec.ts_ns; ctxs[key] = c; order.append(key)
        if rec.type in (R.REC_UL_PDU, R.REC_DL_PDU):
            c.add_pdu(rec)
        elif rec.type in (R.REC_UL_TTI, R.REC_DL_TTI):
            c.add_summary(rec)
        elif rec.type == R.REC_SLOT_END:
            c.add_slot_end(rec)
    return [ctxs[k] for k in order]


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--model", default=os.path.join(HERE, "sm_model.yaml")); ap.add_argument("--out", default=os.path.join(HERE, "sm_model_fit.yaml"))
    ap.add_argument("--max-records", type=int, default=4000000); ap.add_argument("--runlog", default="")
    ap.add_argument("--ring-prefix", default="ring_16C_"); ap.add_argument("--ring-dir", default="")
    a = ap.parse_args()
    model = M.SmModel(a.model)
    repo = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # estimator -> dapp_hook -> cuPHY-CP -> repo root
    rows_by_ch = {c: [] for c in DUR_COL}        # (run, s, work dict, duration, pos, start_rel)
    run_info = {}
    for run in a.runs:
        console = os.path.join(a.dir, "console_%s.log" % run); phy = os.path.join(a.dir, "phy_%s.csv" % run)
        dump = os.path.join(a.ring_dir or os.path.join(repo, "prof"), a.ring_prefix + run + ".bin")
        if not (os.path.exists(console) and os.path.exists(phy) and os.path.exists(dump)):
            print("skip %s: missing %s" % (run, [p for p in (console, phy, dump) if not os.path.exists(p)])); continue
        part = partitions(console)
        ctxs = contexts(dump, a.max_records)
        by = {}
        for r in csv.DictReader(open(phy)):
            by.setdefault((int(r["sfn"]), int(r["slot"])), []).append(r)
        joined = 0
        for c in ctxs:
            cands = by.get((c.sfn, c.slot), [])
            best = min(cands, key=lambda r: abs(int(r["t0_ns"]) - c.ts_first_ns), default=None)
            if best is None or abs(int(best["t0_ns"]) - c.ts_first_ns) > 20_000_000:
                continue
            joined += 1
            tot = c.channel_totals()
            for ch, col in DUR_COL.items():
                work = tot.get(ch) or {}
                if not any(work.values()):
                    continue
                d = fl(best.get(col))
                if d is None or d <= 0:
                    continue
                s = part.get(CTX_OF[ch])
                if not s:
                    continue
                start_rel = None
                if ch == "PUSCH":
                    e = fl(best.get("pusch_gpu_end"))
                    start_rel = (e - d) if e is not None else None
                rows_by_ch[ch].append((run, s, work, d, int(best["pos"]), start_rel, completion(ch, best), int(best["slot"]) % 20))
        run_info[run] = (part, len(ctxs), joined)
        print("%-14s partitions %s | contexts %d joined %d" % (run, part, len(ctxs), joined))
    out = dict(model.raw)
    out["provisional"] = False
    out["channels"] = dict(model.raw["channels"])
    print("\n== fit per channel (duration_us = fixed + sum_k work_k * (w_ins_k + w_par_k * s_ref / s))")
    for ch, rows in rows_by_ch.items():
        if not rows:
            print("%s: no rows" % ch); continue
        kern = list(W.KERNELS[ch]); s_ref = model.channels[ch].s_ref
        svals = sorted({r[1] for r in rows})
        Xw = np.array([[r[2].get(k, 0.0) for k in kern] for r in rows], dtype=float)
        s = np.array([r[1] for r in rows], dtype=float); y = np.array([r[3] for r in rows]); pos = np.array([r[4] for r in rows]); sif = np.array([r[7] for r in rows])
        if len(svals) > 1:
            A = np.hstack([np.ones((len(rows), 1)), Xw, Xw * (s_ref / s)[:, None]])
            w, _ = nnls(A, y); fixed = w[0]; w_ins = w[1:1 + len(kern)]; w_par = w[1 + len(kern):]
        else:
            A = np.hstack([np.ones((len(rows), 1)), Xw]); w, _ = nnls(A, y); fixed = w[0]; w_ins = w[1:]; w_par = np.zeros(len(kern))
        pred = A @ w; res = y - pred
        margin = float(np.percentile(res, 99))
        print("%s: n=%d partitions %s | fixed %.1f us | w_ins %s | w_par %s | residual p50 %.1f p99 %.1f max %.1f us (measured - pred)" % (
            ch, len(rows), svals, fixed, {k: "%.3g" % v for k, v in zip(kern, w_ins)}, {k: "%.3g" % v for k, v in zip(kern, w_par)},
            np.percentile(np.abs(res), 50), np.percentile(np.abs(res), 99), np.abs(res).max()))
        for sv in svals:
            m = s == sv
            line = "    s=%3d: measured p50 %6.1f p99 %6.1f max %6.1f | predicted p50 %6.1f | n %d" % (sv, np.percentile(y[m], 50), np.percentile(y[m], 99), y[m].max(), np.percentile(pred[m], 50), m.sum())
            line += " | budget %d %s" % (model.channels[ch].budget_us, "OK" if np.percentile(y[m], 99) + max(0.0, margin) <= model.channels[ch].budget_us else "MISS")
            print(line)
        # completion model: end = start[pos] + sum_k work_k (w_ins + w_par s_ref/s); intercepts free, weights >= 0
        comp = np.array([r[6] if r[6] is not None else np.nan for r in rows]); okc = ~np.isnan(comp) & (np.abs(np.nan_to_num(comp, nan=1e9)) < 10000)
        start_us, w_ins_c, w_par_c, margin_c = {}, None, None, None
        if okc.sum() > 10:
            P = sorted(set(sif[okc].tolist())); oh = np.array([[1.0 if pp == q else 0.0 for q in P] for pp in sif[okc]])   # intercept per slot-in-frame
            if len(svals) > 1:
                Ac = np.hstack([oh, Xw[okc], (Xw * (s_ref / s)[:, None])[okc]]); nk = len(kern); nw = 2 * nk
            else:
                Ac = np.hstack([oh, Xw[okc]]); nk = len(kern); nw = nk
            lo = np.array([-np.inf] * len(P) + [0.0] * nw); hi = np.array([np.inf] * (len(P) + nw))
            fit = lsq_linear(Ac, comp[okc], bounds=(lo, hi), lsmr_tol="auto"); wc = fit.x
            start_us = {int(q): round(float(wc[i]), 1) for i, q in enumerate(P)}
            w_ins_c = wc[len(P):len(P) + nk]; w_par_c = wc[len(P) + nk:len(P) + nw] if len(svals) > 1 else np.zeros(nk)
            resc = comp[okc] - Ac @ wc; margin_c = float(np.percentile(resc, 99))
            cb = model.channels[ch].complete_by_us
            print("    completion fit: start_us %s | w_ins %s | w_par %s | residual p50 %.1f p99 %.1f max %.1f us" % (
                start_us, {k: "%.3g" % v for k, v in zip(kern, w_ins_c)}, {k: "%.3g" % v for k, v in zip(kern, w_par_c)},
                np.percentile(np.abs(resc), 50), np.percentile(np.abs(resc), 99), np.abs(resc).max()))
            for sv in svals:
                for q in P:
                    m2 = okc & (s == sv) & (sif == q)
                    if m2.sum():
                        print("      s=%3d slot %2d: completion p50 %7.0f p99 %7.0f max %7.0f us | complete_by %s %s" % (
                            sv, q, np.percentile(comp[m2], 50), np.percentile(comp[m2], 99), comp[m2].max(), cb,
                            "" if cb is None else ("OK" if np.percentile(comp[m2], 99) <= cb else "MISS")))
        if ch == "PUSCH":
            for p in sorted(set(pos)):
                st = [r[5] for r in rows if r[4] == p and r[5] is not None]
                if st:
                    print("    U position %d: PUSCH GPU start after t0 p50 %.0f p90 %.0f us (n %d)" % (p, np.percentile(st, 50), np.percentile(st, 90), len(st)))
        d = dict(out["channels"][ch]); d["fixed_us"] = round(float(fixed), 2)
        kind = d.get("budget_kind") or model.budget_kind
        if w_ins_c is not None and kind == "completion":      # the fit that min_sm uses for this channel; the other kept as *_dur / *_comp
            d["start_us"] = start_us; d["margin_us"] = round(max(0.0, margin_c), 1)
            d["w_ins"] = {k: float("%.4g" % v) for k, v in zip(kern, w_ins_c)}; d["w_par"] = {k: float("%.4g" % v) for k, v in zip(kern, w_par_c)}
            d["w_ins_dur"] = {k: float("%.4g" % v) for k, v in zip(kern, w_ins)}; d["w_par_dur"] = {k: float("%.4g" % v) for k, v in zip(kern, w_par)}
            d["margin_dur_us"] = round(max(0.0, margin), 1)
        else:
            d["margin_us"] = round(max(0.0, margin), 1)
            d["w_ins"] = {k: float("%.4g" % v) for k, v in zip(kern, w_ins)}; d["w_par"] = {k: float("%.4g" % v) for k, v in zip(kern, w_par)}
            if w_ins_c is not None:
                d["start_us"] = start_us; d["margin_comp_us"] = round(max(0.0, margin_c), 1)
                d["w_ins_comp"] = {k: float("%.4g" % v) for k, v in zip(kern, w_ins_c)}; d["w_par_comp"] = {k: float("%.4g" % v) for k, v in zip(kern, w_par_c)}
        d["s_ref"] = s_ref; d["fit_runs"] = sorted({r[0] for r in rows}); d["fit_partitions"] = svals
        out["channels"][ch] = d
    if a.runlog and os.path.exists(a.runlog):
        print("\n== per-run L1 error lines / ring recording (from %s)" % a.runlog)
        cur = None
        for ln in open(a.runlog, errors="replace"):
            m = re.match(r"##### (\S+)", ln)
            if m:
                cur = m.group(1)
            elif cur in run_info and ("L1:" in ln or "gaps" in ln or "recorded" in ln):
                print("   %s %s" % (cur, ln.strip()[:150]))
    with open(a.out, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print("\nwritten %s (channels refit: %s)" % (a.out, [c for c, r in rows_by_ch.items() if r]))


if __name__ == "__main__":
    main()
