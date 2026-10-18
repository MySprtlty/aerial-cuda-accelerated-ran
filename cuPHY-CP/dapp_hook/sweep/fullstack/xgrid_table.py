#!/usr/bin/env python3
"""xgrid_table.py (2026-10-16 night: demo cell-count X grid)
  budgets --dir D --r0 A,B [--out budgets.json]      per-TDD-position PUSCH completion budget = p99.99 x 1.05 of the two start R0
                                                     runs (their measurement windows), RU late reference = mean late rows / window
  eval    --dir D --name N --budgets budgets.json --csv grid.csv [--model M --cap C --kind R0|cell|rate|naive]
                                                     metrics of one run in its measurement window, PASS/FAIL, one row appended
  table   --csv grid.csv                             markdown block table of the rows
  rules   --avg a.csv[,..] --peak p.csv[,..] --out prof/tmp/tmp_rules_<X>c.yaml [--template safecap_table.yaml]
                                                     load-classifier thresholds: heavy >= midpoint(Average p95, Peak p5), light <= 0.9 x midpoint for 0.5 s
Measurement window: W = [T0 + 35 s, T0 + 95 s], T0 = first slot in phy_<name>.csv (100 s traffic; the tenant client starts
~T0 + 5 s, so >= 30 s warm-up before W). Columns: B_rel overruns per PUSCH position (completion after t0 - 500 us > budget),
B_abs (safecap_check.B absolute budgets, PUSCH and all channels), Err.ind / slot drop / L1 CRC / MAC ERR in W, RU late
(slot, cell) rows in W (tag 503 INFO) and their ratio to the R0 reference, tenant service p50 / p95 (dapp_sched start->end),
sojourn mean / p95 (arrival -> reply), served requests, SM clock and PTP cores from the run's precheck/ptp files. PRB window
distribution of an R0 (p5 / p50 / p95 of PUSCH PRB per UL slot over 60-UL-slot windows) from ringprof_<name>.csv when present.
Pass (rule of 2026-10-17 01:00) = B_abs (all channels, absolute budgets per TDD position) <= 5 per 60 s AND Err.ind = drop = CRC = 0 AND late DL
U-plane packet fraction <= 5e-5 (late packets / window's DL U-plane packets, RU table total x window share of DL slots). B_rel
(R0 p99.99 x 1.05) and late x R0 stay as reference columns; late (slot, cell) rows / (DL slots x cells) is the second unit."""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import safecap_check as SC   # noqa: E402

W_START, W_LEN = 35.0, 60.0
UL_START_US = 500.0


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def window(d, name):
    p = os.path.join(d, "phy_%s.csv" % name)
    if not os.path.exists(p):                      # tenant-only run (no L1): the window starts 30 s after the first request
        q = os.path.join(d, "disp_%s.csv" % name)
        if not os.path.exists(q):
            return None, []
        t0 = min(int(r["t_arrival"]) for r in csv.DictReader(open(q))) - int(5e9)
        return (t0 + int(W_START * 1e9), t0 + int((W_START + W_LEN) * 1e9)), []
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None, []
    t0 = min(int(r["t0_ns"]) for r in rows)
    lo, hi = t0 + int(W_START * 1e9), t0 + int((W_START + W_LEN) * 1e9)
    return (lo, hi), [r for r in rows if lo <= int(r["t0_ns"]) < hi]


def pusch_done_by_pos(rows):
    out = {}
    for r in rows:
        v = SC.fl(r.get("pusch_gpu_end"))
        if v is not None:
            out.setdefault(int(r["pos"]), []).append(v - UL_START_US)
    return out


def in_window_stamp(t_s, win):
    """phy/run logs carry UTC HH:MM:SS stamps; W is CLOCK_REALTIME ns (UTC epoch)."""
    lo = (win[0] / 1e9) % 86400; hi = (win[1] / 1e9) % 86400
    if hi < lo:
        return t_s >= lo or t_s < hi
    return lo <= t_s < hi


def count_lines(path, keys, win):
    n = {k: 0 for k in keys}
    try:
        with open(path, errors="replace") as f:
            for ln in f:
                t = SC.stamp_s(ln)
                if t is None or not in_window_stamp(t, win):
                    continue
                for k in keys:
                    if k in ln:
                        n[k] += 1
    except FileNotFoundError:
        pass
    return n


def l1_crc(d, name, win):
    p = os.path.join(d, "l1_persec_%s.log" % name)
    if not os.path.exists(p):
        p = os.path.join(d, "run2_%s.log" % name)
    rx = re.compile(r"Cell\s+(\d+) \| DL .* UL .* Slots CRC\s+(\d+) \(\s*(\d+)\)")
    mx, first, last = 0, {}, {}
    try:
        with open(p, errors="replace") as f:
            for ln in f:
                m = rx.search(ln); t = SC.stamp_s(ln)
                if not m or t is None or not in_window_stamp(t, win):
                    continue
                c = int(m.group(1)); mx = max(mx, int(m.group(2))); first.setdefault(c, int(m.group(3))); last[c] = int(m.group(3))
    except FileNotFoundError:
        return None, None
    return mx, sum(last[c] - first[c] for c in last)


def mac_err(d, name, win):
    p = os.path.join(d, "mac_persec_%s.log" % name)
    if not os.path.exists(p):
        p = os.path.join(d, "run3_%s.log" % name)
    rx = re.compile(r"Cell\s+(\d+) \| DL .* ERR\s+(\d+) \| INV\s+(\d+)")
    s = 0; seen = False
    try:
        with open(p, errors="replace") as f:
            for ln in f:
                m = rx.search(ln); t = SC.stamp_s(ln)
                if not m or t is None or not in_window_stamp(t, win):
                    continue
                s += int(m.group(2)); seen = True
    except FileNotFoundError:
        return None
    return s if seen else None


def ru_late_rows(d, name, win):
    p = os.path.join(d, "ru_%s_late.csv" % name)
    if not os.path.exists(p):
        return None, None, {}
    n = pk = 0; pos = {}; total = 0
    for r in csv.DictReader(open(p)):
        total += 1
        t = int(r["t0_ns"])
        if win[0] <= t < win[1]:
            n += 1; pk += int(r["late"]); k = int(r["slot"]) % 10; pos[k] = pos.get(k, 0) + 1
    if total == 0:
        return None, None, {}          # no per-slot late lines at all (tag 503 INFO off): unknown, not zero
    return n, pk, pos


def ru_table_pkts(d, name):
    """total DL U-plane packets of the whole run (EARLY + ON TIME + LATE, all cells) from the RU end-of-run table."""
    tot = 0; seen = False
    for q in (os.path.join(d, "ru_%s.log" % name), os.path.join(d, "ru_warn_%s.log" % name), os.path.join(d, "run1_%s.log" % name)):   # run1 = RU stdout (table too)
        if not os.path.exists(q):
            continue
        with open(q, errors="replace") as f:
            for ln in f:
                m = re.search(r"\[RU\] \| DL U (EARLY|ON TIME|LATE) PACKETS\s+\|(.*)", ln)
                if m:
                    tot += sum(int(x) for x in re.findall(r"\d+", m.group(2))); seen = True
        if seen:
            break
    return tot if seen else None


def tenant(d, name, win):
    p = os.path.join(d, "disp_%s.csv" % name)
    if not os.path.exists(p):
        return {}
    rows = [r for r in csv.DictReader(open(p)) if win[0] <= int(r["t_arrival"]) < win[1]]
    ok = [r for r in rows if r["status"] == "0" and int(r["t_end"]) and int(r["t_start"])]
    svc = np.array([(int(r["t_end"]) - int(r["t_start"])) / 1e6 for r in ok])
    soj = np.array([(int(r["t_reply"]) - int(r["t_arrival"])) / 1e6 for r in rows if int(r["t_reply"])])
    models = {}
    for r in rows:
        models[r["model"]] = models.get(r["model"], 0) + 1
    return {"req": len(rows), "served": len(ok), "err": len(rows) - len(ok), "hold": sum(int(r.get("n_hold", 0) or 0) for r in rows),
            "svc_p50": pct(svc, 50), "svc_p95": pct(svc, 95), "soj_mean": float(soj.mean()) if len(soj) else float("nan"),
            "soj_p95": pct(soj, 95), "soj_max": float(soj.max()) if len(soj) else float("nan"), "over60": int((soj > 60.0).sum()) if len(soj) else 0,
            "models": " ".join("%s:%d" % kv for kv in sorted(models.items()))}


def host_state(d, name):
    clk = ptp = "-"
    p = os.path.join(d, "precheck_%s.txt" % name)
    if os.path.exists(p):
        lines = open(p, errors="replace").read().splitlines()
        for i, ln in enumerate(lines):
            if ln.startswith("-- GPU state") and i + 2 < len(lines):
                f = lines[i + 2].split(",")
                if len(f) >= 3:
                    clk = "sm %s app %s" % (f[1].strip(), f[3].strip() if len(f) > 3 else "?")
    p = os.path.join(d, "ptp_%s.txt" % name)
    if os.path.exists(p):
        ptp = " ".join(re.sub(r"^\d+ ", "", ln.strip()).replace(" last_cpu=", "@") for ln in open(p) if ln.strip() and "phc2sys" in ln and "aerial" not in ln)[:40] or "-"
        ptp = ",".join(sorted({re.search(r"allowed=(\S+)", ln).group(1) for ln in open(p) if "allowed=" in ln and ("ptp4l" in ln or "phc2sys" in ln)}))
    return clk, ptp


def prb_profile(d, name):
    p = os.path.join(d, "ringprof_%s.csv" % name)
    if not os.path.exists(p):
        return None
    v = []
    for r in csv.DictReader(open(p)):
        try:
            v.append(float(r["pusch_prb_per_s"]) / 600.0)        # 600 UL slots per second in DDDDDDSUUU at mu 1
        except (KeyError, ValueError):
            pass
    v = np.array(v)
    if not len(v):
        return None
    return {"prb_p5": pct(v, 5), "prb_p50": pct(v, 50), "prb_p95": pct(v, 95), "prb_n": len(v)}


def budgets(d, r0names):
    done = {}; late = []
    for n in r0names:
        win, rows = window(d, n)
        if win is None:
            continue
        for k, v in pusch_done_by_pos(rows).items():
            done.setdefault(k, []).extend(v)
        l, _, _ = ru_late_rows(d, n, win)
        if l is not None:
            late.append(l)
    b = {str(k): round(pct(v, 99.99) * 1.05, 1) for k, v in done.items()}
    return {"r0": list(r0names), "pusch_budget_us": b, "late_ref": (sum(late) / len(late)) if late else None, "late_r0": late,
            "pusch_p50": {str(k): round(pct(v, 50), 1) for k, v in done.items()}}


def evaluate(d, name, bud, kind, model, cap):
    win, rows = window(d, name)
    if win is None:
        return {"name": name, "kind": kind, "model": model, "cap": cap, "verdict": "no data"}
    if not rows and kind == "tenant-only":
        ten = tenant(d, name, win); clk, ptp = host_state(d, name)
        return {"name": name, "kind": kind, "model": model, "cap": cap, "slots": 0, "svc_p50": round(ten.get("svc_p50", float("nan")), 1), "svc_p95": round(ten.get("svc_p95", float("nan")), 1),
                "soj_mean": round(ten.get("soj_mean", float("nan")), 1), "soj_p95": round(ten.get("soj_p95", float("nan")), 1), "soj_max": round(ten.get("soj_max", float("nan")), 1),
                "over60": ten.get("over60"), "served": ten.get("served"), "req": ten.get("req"), "ten_err": ten.get("err"), "hold": ten.get("hold"), "models": ten.get("models"),
                "clock": clk, "ptp": ptp, "verdict": "n/a"}
    done = pusch_done_by_pos(rows)
    brel = {k: sum(1 for x in v if x > bud["pusch_budget_us"].get(str(k), 1e9)) for k, v in done.items()}
    babs_p = {k: sum(1 for x in v if x > SC.B["pusch"]) for k, v in done.items()}
    allm = SC.misses(SC.channel_values(rows))
    babs_all = sum(m[1] for m in allm.values())
    drops = count_lines(os.path.join(d, "phy_warn_%s.log" % name), ("Send Err.ind", "Dropping the slot command", "dropped due to latency", "SFN mismatch"), win)
    crc_max, crc_cum = l1_crc(d, name, win)
    merr = mac_err(d, name, win)
    late_n, late_pk, late_pos = ru_late_rows(d, name, win)
    ten = tenant(d, name, win)
    clk, ptp = host_state(d, name)
    ref = bud.get("late_ref")
    late_ratio = (late_n / ref) if (ref and late_n is not None) else None          # reference column only (x R0)
    # absolute late fractions (2026-10-17 01:00 rule): late DL U-plane PACKETS / all DL U-plane packets of the window
    # (RU table total scaled by the window's share of DL slots) and, as the second unit, late (slot, cell) rows / DL slots x cells
    dl_win = sum(1 for r in rows if r.get("pdsch_gpu_run_us"))
    allrows = list(csv.DictReader(open(os.path.join(d, "phy_%s.csv" % name)))); dl_all = sum(1 for r in allrows if r.get("pdsch_gpu_run_us"))
    cells = len(set(int(r["cell"]) for r in csv.DictReader(open(os.path.join(d, "ru_%s_late.csv" % name))))) if os.path.exists(os.path.join(d, "ru_%s_late.csv" % name)) else 0
    try:
        cells = max(cells, int(re.search(r"_(\d+)c$", name).group(1)))
    except AttributeError:
        pass
    tot_pk = ru_table_pkts(d, name)
    pk_den = (tot_pk * dl_win / dl_all) if (tot_pk and dl_all) else None
    late_pkt_frac = (late_pk / pk_den) if (pk_den and late_pk is not None) else None
    late_row_frac = (late_n / (dl_win * cells)) if (dl_win and cells and late_n is not None) else None
    babs_total = babs_all
    ok_b = babs_total <= 5
    ok_e = drops["Send Err.ind"] == 0 and (drops["Dropping the slot command"] + drops["dropped due to latency"] + drops["SFN mismatch"]) == 0 and (crc_cum or 0) == 0
    ok_l = (late_pkt_frac is None) or (late_pkt_frac <= 5e-5)        # screening 60 s: <= 5e-5 of the DL U-plane packets; unknown -> rule 1 only
    verdict = "PASS" if (ok_b and ok_e and ok_l) else "FAIL(%s)" % "".join(x for x, f in (("B", not ok_b), ("E", not ok_e), ("L", not ok_l)) if f)
    if kind in ("cell", "rate", "naive", "mig-iso") and not ten.get("served"):
        verdict = "INVALID(no requests served)"             # tenant configured but nothing was served: not a measurement of that pair
    row = {"name": name, "kind": kind, "model": model, "cap": cap, "slots": len(rows),
           "b_rel": sum(brel.values()), "b_rel_pos": " ".join("%d:%d" % kv for kv in sorted(brel.items())),
           "b_abs_pusch": sum(babs_p.values()), "b_abs_all": babs_all,
           "pusch_p50": " ".join("%d:%.0f" % (k, pct(v, 50)) for k, v in sorted(done.items())),
           "pusch_p9999": " ".join("%d:%.0f" % (k, pct(v, 99.99)) for k, v in sorted(done.items())),
           "errind": drops["Send Err.ind"], "drop": drops["Dropping the slot command"] + drops["dropped due to latency"] + drops["SFN mismatch"],
           "crc_max_s": crc_max, "crc": crc_cum, "mac_err": merr,
           "late": late_n, "late_pkts": late_pk, "late_pos": " ".join("%d:%d" % kv for kv in sorted(late_pos.items())),
           "late_ratio": round(late_ratio, 2) if late_ratio is not None else None,
           "late_pkt_frac": ("%.2e" % late_pkt_frac) if late_pkt_frac is not None else None, "late_row_frac": ("%.2e" % late_row_frac) if late_row_frac is not None else None,
           "pkt_den": int(pk_den) if pk_den else None, "dl_slots_win": dl_win, "cells": cells,
           "svc_p50": round(ten.get("svc_p50", float("nan")), 1) if ten else None, "svc_p95": round(ten.get("svc_p95", float("nan")), 1) if ten else None,
           "soj_mean": round(ten.get("soj_mean", float("nan")), 1) if ten else None, "soj_p95": round(ten.get("soj_p95", float("nan")), 1) if ten else None,
           "soj_max": round(ten.get("soj_max", float("nan")), 1) if ten else None, "over60": ten.get("over60") if ten else None,
           "served": ten.get("served") if ten else None, "req": ten.get("req") if ten else None, "ten_err": ten.get("err") if ten else None,
           "hold": ten.get("hold") if ten else None, "models": ten.get("models") if ten else None,
           "clock": clk, "ptp": ptp, "verdict": verdict}
    prof = prb_profile(d, name)
    if prof:
        row.update({k: round(v, 0) if isinstance(v, float) else v for k, v in prof.items()})
    return row


def append_csv(path, row):
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    rows = [r for r in rows if r["name"] != row["name"]] + [row]
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


def f(v):
    return "-" if v in (None, "", "None", "nan") else str(v)


def table(path):
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    print("| run | kind | model@cap | verdict | B_abs all / PUSCH (<=5) | Err.ind / drop / CRC / MAC ERR | late pkt frac (<=5e-5) / row frac | RU late rows (x R0, ref) | B_rel (ref) | late pos | PUSCH done p50 / p99.99 us | svc p50 / p95 | sojourn mean / p95 / max | >60 ms | served | SM clk | PTP |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %s@%s | %s | %s / %s | %s / %s / %s / %s | %s / %s | %s (%s) | %s | %s | %s / %s | %s / %s | %s / %s / %s | %s | %s | %s | %s |" % (
            r["name"], r["kind"], f(r.get("model")), f(r.get("cap")), r["verdict"], f(r.get("b_abs_all")), f(r.get("b_abs_pusch")),
            f(r.get("errind")), f(r.get("drop")), f(r.get("crc")), f(r.get("mac_err")), f(r.get("late_pkt_frac")), f(r.get("late_row_frac")), f(r.get("late")), f(r.get("late_ratio")), f(r.get("b_rel")), f(r.get("late_pos")),
            f(r.get("pusch_p50")), f(r.get("pusch_p9999")), f(r.get("svc_p50")), f(r.get("svc_p95")), f(r.get("soj_mean")), f(r.get("soj_p95")), f(r.get("soj_max")),
            f(r.get("over60")), f(r.get("served")), f(r.get("clock")), f(r.get("ptp"))))
    prof = [r for r in rows if r.get("prb_p50") not in (None, "", "None")]
    if prof:
        print("\nR0 PUSCH PRB per UL slot (60-UL-slot windows): " + "; ".join("%s p5 %s / p50 %s / p95 %s" % (r["name"], r["prb_p5"], r["prb_p50"], r["prb_p95"]) for r in prof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["budgets", "eval", "table", "rules"])
    ap.add_argument("--avg", default="", help="rules: comma list of Average R0 ringprof csvs"); ap.add_argument("--peak", default="", help="rules: Peak R0 ringprof csvs")
    ap.add_argument("--template", default="", help="rules: SafeCap table yaml whose classes block is replaced")
    ap.add_argument("--dir"); ap.add_argument("--name"); ap.add_argument("--r0"); ap.add_argument("--out"); ap.add_argument("--budgets")
    ap.add_argument("--csv"); ap.add_argument("--kind", default="cell"); ap.add_argument("--model", default=""); ap.add_argument("--cap", default="")
    a = ap.parse_args()
    if a.phase == "budgets":
        b = budgets(a.dir, a.r0.split(","))
        if a.out:
            json.dump(b, open(a.out, "w"), indent=1)
        print(json.dumps(b)); return 0
    if a.phase == "eval":
        bud = json.load(open(a.budgets)) if a.budgets and os.path.exists(a.budgets) else {"pusch_budget_us": {}, "late_ref": None}
        row = evaluate(a.dir, a.name, bud, a.kind, a.model, a.cap)
        if a.csv:
            append_csv(a.csv, row)
        print(json.dumps(row)); print("VERDICT %s" % row["verdict"]); return 0
    if a.phase == "rules":
        def prb(paths):
            v = []
            for q in paths.split(","):
                for r in csv.DictReader(open(q)):
                    try:
                        v.append(float(r["pusch_prb_per_s"]) / 600.0)
                    except (KeyError, ValueError):
                        pass
            return np.array(v)
        av, pk = prb(a.avg), prb(a.peak)
        mid = (pct(av, 95) + pct(pk, 5)) / 2.0
        txt = ("# load classifier thresholds for X cells (xgrid_table.py rules, %s): feature = PUSCH PRB per UL slot summed over cells,\n"
               "# 60-UL-slot windows of the Average / Peak R0 ring profiles. Average p95 %.0f, Peak p5 %.0f -> midpoint %.0f;\n"
               "# heavy at once at >= midpoint, light after 0.5 s at <= 90 %% of the midpoint.\n"
               "window_ul_slots: 60\nlight_hold_s: 0.5\nclasses:\n  average: {pusch_prb_per_ul_slot_max: %d}\n  peak:    {pusch_prb_per_ul_slot_min: %d}\n"
               % (a.out or "", pct(av, 95), pct(pk, 5), mid, int(round(0.9 * mid)), int(round(mid))))
        if a.template and os.path.exists(a.template):
            body = open(a.template).read()
            i = body.find("classes:"); j = body.find("\n", body.find("peak:", i)) + 1 if i >= 0 else -1
            body = (body[:i] + txt.split("classes:\n")[1].join(["classes:\n", ""]) + body[j:]) if i >= 0 and j > 0 else body + txt
            txt = txt.split("window_ul_slots")[0] + body
        if a.out:
            open(a.out, "w").write(txt)
        print(txt); print("average n=%d p5/p50/p95 %.0f/%.0f/%.0f | peak n=%d p5/p50/p95 %.0f/%.0f/%.0f" % (len(av), pct(av, 5), pct(av, 50), pct(av, 95), len(pk), pct(pk, 5), pct(pk, 50), pct(pk, 95)))
        return 0
    table(a.csv); return 0


if __name__ == "__main__":
    sys.exit(main())
