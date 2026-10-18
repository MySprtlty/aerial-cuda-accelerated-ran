#!/usr/bin/env python3
"""tenmin_table.py --dir D --name RUN [--warm 30] [--window 600] [--b-limit 12] [--late-ref 5e-5]
Ten-minute operating run, per-MINUTE aggregation (night 3, 2026-10-18). Window W = [T0 + warm, T0 + warm + window), T0 = first slot
of phy_RUN.csv (traffic start). Per minute: slots, UL slots, B_abs overruns (safecap_check.B absolute budgets: PUSCH per TDD position
7/8/9 and the other channels), Err.ind / slot drops (phy_warn log stamps), L1 CRC (l1_persec log), RU late rows / packets (ru_RUN_late.csv,
tag 503), tenant (disp_RUN.csv: served, service p50/p95, sojourn mean/p95/p99/max, > 60 ms). Totals, late packet fraction (late packets in
W / (RU table total packets x W share of the traffic time)), FATAL/ERR signatures with times, early end (crash) detection and the verdict:
PASS = B_abs total <= b-limit AND Err.ind = drops = CRC = 0 (late fraction reported against --late-ref, not a gate)."""
import argparse
import csv
import datetime
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import safecap_check as SC   # noqa: E402


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v); return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def sod_utc(t_ns):
    d = datetime.datetime.utcfromtimestamp(t_ns / 1e9); return d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6


def minute_of_stamp(stamp, t_lo_sod, window, nmin):
    """log stamp (seconds of day, UTC = container clock) -> minute index in W or None"""
    if stamp is None:
        return None
    dt = stamp - t_lo_sod
    if dt < -43200:
        dt += 86400
    if dt < 0 or dt >= window:
        return None
    return min(nmin - 1, int(dt // 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--warm", type=float, default=30.0); ap.add_argument("--window", type=float, default=600.0)
    ap.add_argument("--b-limit", type=int, default=12); ap.add_argument("--late-ref", type=float, default=5e-5)
    ap.add_argument("--csv", default="", help="append the summary row to this csv")
    a = ap.parse_args(); d, n = a.dir, a.name
    nmin = int(round(a.window / 60))
    p = os.path.join(d, "phy_%s.csv" % n)
    if not os.path.exists(p):
        print("no phy csv for", n); return 1
    rows = list(csv.DictReader(open(p)))
    T0 = min(int(r["t0_ns"]) for r in rows); t_last = max(int(r["t0_ns"]) for r in rows)
    lo = T0 + int(a.warm * 1e9); hi = lo + int(a.window * 1e9)
    traffic_s = (t_last - T0) / 1e9
    early_end = t_last < hi - int(2e9)
    def minute_of(t_ns):
        if t_ns < lo or t_ns >= hi:
            return None
        return min(nmin - 1, int((t_ns - lo) // int(60e9)))
    per = [dict(slots=0, ul=0, b_pusch={7: 0, 8: 0, 9: 0}, b_other=0, b_other_ch={}, errind=0, drops=0, crc=0, late_rows=0, late_pkts=0, late_slots=set(),
                served=0, svc=[], soj=[]) for _ in range(nmin)]
    bym = {}
    for r in rows:
        m = minute_of(int(r["t0_ns"]))
        if m is None:
            continue
        bym.setdefault(m, []).append(r); per[m]["slots"] += 1; per[m]["ul"] += 1 if r.get("ul") == "1" else 0
    for m, rs in bym.items():
        vals = SC.channel_values(rs); mis = SC.misses(vals)
        for ch, (ma, mb, cnt) in mis.items():
            if ch == "pusch":
                continue
            per[m]["b_other"] += mb
            if mb:
                per[m]["b_other_ch"][ch] = mb
        for r in rs:
            pos = r.get("pos"); e = SC.fl(r.get("pusch_gpu_end"))
            if e is not None and pos in ("7", "8", "9") and e - SC.UL_START_US > SC.B["pusch"]:
                per[m]["b_pusch"][int(pos)] += 1
    t_lo_sod = sod_utc(lo)
    fatal, sigs = [], {}
    pw = os.path.join(d, "phy_warn_%s.log" % n)
    if os.path.exists(pw):
        with open(pw, errors="replace") as f:
            for ln in f:
                st = SC.stamp_s(ln); m = minute_of_stamp(st, t_lo_sod, a.window, nmin)
                if " FATAL " in ln or "Order kernel" in ln and "launch" in ln:
                    fatal.append(ln.strip()[:200])
                if m is None:
                    continue
                if "Send Err.ind" in ln:
                    per[m]["errind"] += 1
                if "Dropping the slot command" in ln or "dropped due to latency" in ln or "SFN mismatch" in ln:
                    per[m]["drops"] += 1
                if " ERR " in ln or " FATAL " in ln:
                    key = re.sub(r"\d+", "N", ln.split("] ", 1)[-1])[:90]; sigs[key] = sigs.get(key, 0) + 1
    rx = re.compile(r"Cell\s+(\d+) \| DL .* UL .* Slots CRC\s+(\d+) \(\s*(\d+)\)")
    for q in (os.path.join(d, "l1_persec_%s.log" % n), os.path.join(d, "run2_%s.log" % n)):
        if os.path.exists(q):
            with open(q, errors="replace") as f:
                for ln in f:
                    mm = rx.search(ln)
                    if mm:
                        m = minute_of_stamp(SC.stamp_s(ln), t_lo_sod, a.window, nmin)
                        if m is not None:
                            per[m]["crc"] += int(mm.group(2))
            break
    q = os.path.join(d, "ru_%s_late.csv" % n)
    if os.path.exists(q):
        for r in csv.DictReader(open(q)):
            m = minute_of(int(r["t0_ns"]))
            if m is None:
                continue
            per[m]["late_rows"] += 1; per[m]["late_pkts"] += int(r["late"]); per[m]["late_slots"].add((r["frame"], r["slot"]))
    q = os.path.join(d, "disp_%s.csv" % n)
    has_t = os.path.exists(q)
    if has_t:
        for r in csv.DictReader(open(q)):
            if r.get("status") != "0":
                continue
            m = minute_of(int(r["t_arrival"]))
            if m is None:
                continue
            per[m]["served"] += 1; per[m]["soj"].append((int(r["t_reply"]) - int(r["t_arrival"])) / 1e6)
            if int(r["t_end"]) and int(r["t_start"]):
                per[m]["svc"].append((int(r["t_end"]) - int(r["t_start"])) / 1e6)
    # RU late packet fraction over W: late pkts / (table total pkts x W / traffic time)
    import xgrid_table as XG
    tot_pkts = XG.ru_table_pkts(d, n)
    late_w = sum(x["late_pkts"] for x in per)
    frac = (late_w / (tot_pkts * min(a.window, max(1e-9, traffic_s - a.warm)) / traffic_s)) if tot_pkts else None
    b_tot = sum(sum(x["b_pusch"].values()) + x["b_other"] for x in per)
    e_tot, d_tot, c_tot = (sum(x[k] for x in per) for k in ("errind", "drops", "crc"))
    verdict = "PASS" if (b_tot <= a.b_limit and e_tot == 0 and d_tot == 0 and c_tot == 0 and not early_end) else "FAIL"
    why = []
    if b_tot > a.b_limit: why.append("B_abs %d > %d" % (b_tot, a.b_limit))
    if e_tot: why.append("Err.ind %d" % e_tot)
    if d_tot: why.append("drops %d" % d_tot)
    if c_tot: why.append("CRC %d" % c_tot)
    if early_end: why.append("EARLY END at T0+%.0f s (crash?)" % traffic_s)
    print("## %s: T0 %s UTC, traffic %.0f s, W = [T0+%.0f, T0+%.0f) -> %s %s" % (n, datetime.datetime.utcfromtimestamp(T0 / 1e9).strftime("%H:%M:%S"), traffic_s, a.warm, a.warm + a.window, verdict, "; ".join(why)))
    hdr = "| min | slots | UL | B_abs PUSCH p7/p8/p9 | B_abs other | Err.ind | drops | CRC | RU late rows / slots / pkts |"
    if has_t:
        hdr += " served | svc p50 / p95 | sojourn mean / p95 / p99 / max | > 60 ms |"
    print(hdr); print("|---" * hdr.count("|") + "|" if False else "|" + "---|" * (hdr.count("|") - 1))
    for i, x in enumerate(per):
        line = "| %d | %d | %d | %d / %d / %d | %d%s | %d | %d | %d | %d / %d / %d |" % (
            i + 1, x["slots"], x["ul"], x["b_pusch"][7], x["b_pusch"][8], x["b_pusch"][9], x["b_other"],
            (" (" + ",".join("%s %d" % kv for kv in x["b_other_ch"].items()) + ")") if x["b_other_ch"] else "", x["errind"], x["drops"], x["crc"], x["late_rows"], len(x["late_slots"]), x["late_pkts"])
        if has_t:
            s, j = x["svc"], x["soj"]
            line += " %d | %.1f / %.1f | %.1f / %.1f / %.1f / %.1f | %d |" % (x["served"], pct(s, 50), pct(s, 95), (sum(j) / len(j)) if j else float("nan"), pct(j, 95), pct(j, 99), max(j) if j else float("nan"), sum(1 for v in j if v > 60))
        print(line)
    bp = [sum(x["b_pusch"].values()) + x["b_other"] for x in per]
    tot = "| total | %d | %d | %d / %d / %d | %d | %d | %d | %d | %d / %d / %d |" % (
        sum(x["slots"] for x in per), sum(x["ul"] for x in per), sum(x["b_pusch"][7] for x in per), sum(x["b_pusch"][8] for x in per), sum(x["b_pusch"][9] for x in per),
        sum(x["b_other"] for x in per), e_tot, d_tot, c_tot, sum(x["late_rows"] for x in per), sum(len(x["late_slots"]) for x in per), late_w)
    if has_t:
        s = [v for x in per for v in x["svc"]]; j = [v for x in per for v in x["soj"]]
        tot += " %d | %.1f / %.1f | %.1f / %.1f / %.1f / %.1f | %d |" % (sum(x["served"] for x in per), pct(s, 50), pct(s, 95), (sum(j) / len(j)) if j else float("nan"), pct(j, 95), pct(j, 99), max(j) if j else float("nan"), sum(1 for v in j if v > 60))
    print(tot)
    print("B_abs total %d (max minute %d), late packet fraction %s (ref %.0e; RU table total pkts %s), FATAL/Order-launch lines %d" % (
        b_tot, max(bp) if bp else 0, ("%.2e" % frac) if frac is not None else "n/a", a.late_ref, tot_pkts, len(fatal)))
    for ln in fatal[:6]:
        print("  FATAL:", ln)
    if sigs:
        print("  ERR signatures in W: " + "; ".join("%s x%d" % (k, v) for k, v in sorted(sigs.items(), key=lambda kv: -kv[1])[:5]))
    if a.csv:
        row = {"name": n, "verdict": verdict, "why": "; ".join(why), "b_abs_total": b_tot, "b_abs_max_min": max(bp) if bp else 0, "errind": e_tot, "drops": d_tot, "crc": c_tot,
               "late_frac": frac, "late_pkts": late_w, "traffic_s": round(traffic_s, 1), "fatal": len(fatal)}
        if has_t:
            s = [v for x in per for v in x["svc"]]; j = [v for x in per for v in x["soj"]]
            row.update(served=sum(x["served"] for x in per), svc_p50=round(pct(s, 50), 1), svc_p95=round(pct(s, 95), 1), soj_mean=round(sum(j) / len(j), 1) if j else None,
                       soj_p95=round(pct(j, 95), 1), soj_p99=round(pct(j, 99), 1), soj_max=round(max(j), 1) if j else None, over60=sum(1 for v in j if v > 60))
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "verdict", "why", "b_abs_total", "b_abs_max_min", "errind", "drops", "crc", "late_frac", "late_pkts", "traffic_s", "fatal",
                                              "served", "svc_p50", "svc_p95", "soj_mean", "soj_p95", "soj_p99", "soj_max", "over60"])
            if new:
                w.writeheader()
            w.writerow(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
