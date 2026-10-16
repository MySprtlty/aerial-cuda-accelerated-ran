#!/usr/bin/env python3
"""g3_table.py --dir D --prefix P --phase eval|rerun|table   (DDDDDDSUUU SafeCap sweep, see g3_sweep.sh)
R0 references: Peak = mean RU late slots of G1_P1 / G2_P2 (and *_r2); Average = mean of G1_A1 / G2_A2. The late limit is
1.25 x the Peak mean for BOTH loads (absolute). B limit: absolute 5 misses per 60 s run (fixed decision) and, as a second
column, R0's own misses + 5. rerun: one repeat of a failed run when B misses <= 20 and late <= 3 x limit; 26x cap 24 (SM 32)
in Peak when 26x fails at cap 43 in Peak."""
import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import safecap_check as SC   # noqa: E402

R0 = {"peak": ("G1_P1_16c", "G2_P2_16c", "G3_P3_16c", "G3b_P4_16c"), "average": ("G1_A1_16c", "G2_A2_16c", "G3b_A3_16c")}


def r0_late(d, load):
    v = [SC.ru_late_slots(n, d) for n in R0[load] if os.path.exists(os.path.join(d, "phy_%s.csv" % n))]
    v = [x for x in v if x > 0]
    return (sum(v) / len(v) if v else -1), v


def r0_b(d, load, first_s=10.0):
    v = [SC.b_misses(n, d, first_s) for n in R0[load] if os.path.exists(os.path.join(d, "phy_%s.csv" % n))]
    return (sum(v) / len(v) if v else 0), v


def runs(d, prefix):
    out = []
    for p in sorted(os.listdir(d)):
        m = re.match(r"phy_%s_([SP])_(26[nsx])_c(\d+)_(peak|average)(_r2)?_16c\.csv$" % prefix, p)
        if m:
            out.append({"name": p[4:-4], "kind": m.group(1), "model": m.group(2), "cap": int(m.group(3)), "load": m.group(4), "rerun": bool(m.group(5))})
    return out


def tenant(d, name):
    p = os.path.join(d, "disp_%s.csv" % name)
    if not os.path.exists(p):
        return {}
    rows = [r for r in csv.DictReader(open(p)) if r.get("status") == "0"]
    if not rows:
        return {}
    t0 = min(int(r["t_arrival"]) for r in rows); rows = [r for r in rows if int(r["t_arrival"]) - t0 >= 1e9]
    svc = sorted((int(r["t_end"]) - int(r["t_start"])) / 1e6 for r in rows if int(r["t_end"]) and int(r["t_start"]))
    e2e = sorted((int(r["t_reply"]) - int(r["t_arrival"])) / 1e6 for r in rows)
    span = (max(int(r["t_reply"]) for r in rows) - min(int(r["t_arrival"]) for r in rows)) / 1e9
    q = lambda v, p: v[min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1))))] if v else float("nan")   # noqa: E731
    return {"t_n": len(rows), "t_svc_mean": sum(svc) / len(svc) if svc else float("nan"), "t_svc_p50": q(svc, 50), "t_svc_p99": q(svc, 99),
            "t_e2e_mean": sum(e2e) / len(e2e), "t_e2e_p50": q(e2e, 50), "t_e2e_p99": q(e2e, 99), "t_req_s": len(rows) / span if span else 0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--prefix", default="G3"); ap.add_argument("--phase", default="eval")
    a = ap.parse_args(); d = a.dir
    peak_ref, pv = r0_late(d, "peak"); avg_ref, av = r0_late(d, "average")
    late_limit = 1.25 * peak_ref if peak_ref > 0 else None
    b_r0 = {l: r0_b(d, l) for l in ("peak", "average")}
    print("R0: Peak late slots %s mean %.0f -> absolute late limit %.0f (both loads); Average late %s; R0 B misses Peak %s Average %s" % (
        pv, peak_ref, late_limit or -1, av, b_r0["peak"][1], b_r0["average"][1]))
    res = []
    for r in runs(d, a.prefix):
        r0name = R0[r["load"]][0]
        ok, why, n = SC.check(r0name, r["name"], d, 10.0, 1.0, peak_ref if peak_ref > 0 else None, b_max=5)
        ok_rel = n["b_miss"] <= b_r0[r["load"]][0] + 5 and n["errind"] == 0 and n["drops"] == 0 and n["crc"] == 0 and (late_limit is None or 0 <= n["ru_late_slots"] <= late_limit)
        res.append(dict(r, passed=ok, passed_rel=ok_rel, why="; ".join(why), **n, **tenant(d, r["name"])))
    keys = sorted({k for r in res for k in r})
    with open(os.path.join(d, "g3_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in res]
    print("| run | pass (B<=5) | pass (B<=R0+5) | B misses (R0) | Err.ind / drops / CRC | RU late slots (limit) | service mean / p50 / p99 ms | E2E mean / p50 / p99 | req/s | why |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in res:
        print("| %s | %s | %s | %d (%.0f) | %d / %d / %d | %d (%.0f) | %s | %s | %s | %s |" % (
            r["name"], "PASS" if r["passed"] else "FAIL", "pass" if r["passed_rel"] else "fail", r["b_miss"], b_r0[r["load"]][0], r["errind"], r["drops"], r["crc"],
            r["ru_late_slots"], late_limit or -1, ("%.1f / %.1f / %.1f" % (r["t_svc_mean"], r["t_svc_p50"], r["t_svc_p99"])) if "t_svc_mean" in r else "-",
            ("%.1f / %.1f / %.1f" % (r["t_e2e_mean"], r["t_e2e_p50"], r["t_e2e_p99"])) if "t_e2e_mean" in r else "-", ("%.2f" % r["t_req_s"]) if "t_req_s" in r else "-", r["why"]))
    if a.phase == "rerun":
        lst = [r for r in res if r["kind"] in ("S", "P") and not r["passed"] and not r["rerun"] and not any(x["name"] == r["name"] + "_r2" for x in res)
               and r["b_miss"] <= 20 and (late_limit is None or r["ru_late_slots"] <= 3 * late_limit)]
        with open(os.path.join(d, "ladder_%s_rerun.txt" % a.prefix), "w") as f:
            for r in lst:
                arr = "poisson" if r["kind"] == "P" else "closed"; extra = " xclass=104 xmodels=26x" if r["cap"] == 80 else ""
                f.write("%s_%s_%s_c%d_%s_r2 pattern=ddddddsuuu_%s slots=120000 tags=311,223 keep_raw=2 tenant=dispatcher model=%s cap=%d arrivals=%s rate=10 tdelay=28 tsecs=45%s\n" % (
                    a.prefix, r["kind"], r["model"], r["cap"], r["load"], r["load"], r["model"], r["cap"], arr, extra))
            p43 = [r for r in res if r["kind"] == "S" and r["model"] == "26x" and r["cap"] == 43 and r["load"] == "peak"]
            if p43 and not any(r["passed"] for r in p43) and not any(r["cap"] == 24 for r in res):
                f.write("%s_S_26x_c24_peak pattern=ddddddsuuu_peak slots=120000 tags=311,223 keep_raw=2 tenant=dispatcher model=26x cap=24 arrivals=closed rate=10 tdelay=28 tsecs=45 xclass=32 xmodels=26x\n" % a.prefix)
        print("rerun list: %d run(s)" % sum(1 for _ in open(os.path.join(d, "ladder_%s_rerun.txt" % a.prefix))))
    if a.phase == "table":
        for crit, key in (("B <= 5 (fixed)", "passed"), ("B <= R0 + 5 (relative)", "passed_rel")):
            print("\n## SafeCap [%s]: model x load -> largest passing cap; service mean / p99 ms; monotonic?" % crit)
            print("| model | load | caps (marks) | SafeCap | service mean / p99 | monotonic? |"); print("|---|---|---|---|---|---|")
            for load in ("peak", "average"):
                for m in ("26n", "26s", "26x"):
                    rs = [r for r in res if r["kind"] == "S" and r["model"] == m and r["load"] == load]
                    if not rs:
                        continue
                    bycap = {}
                    for r in rs:
                        bycap.setdefault(r["cap"], []).append(r)
                    marks, safe, best = [], None, None
                    for cap in sorted(bycap):
                        ok = any(r[key] for r in bycap[cap]); marks.append("%d:%s" % (cap, "ok" if ok else "fail"))
                        if ok:
                            safe = cap; best = next(r for r in bycap[cap] if r[key])
                    oks = [any(r[key] for r in bycap[c]) for c in sorted(bycap)]
                    mono = all(not (not oks[i] and any(oks[i + 1:])) for i in range(len(oks)))
                    print("| %s | %s | %s | %s | %s | %s |" % (m, load, " ".join(marks), safe if safe is not None else "none",
                          ("%.1f / %.1f" % (best["t_svc_mean"], best["t_svc_p99"])) if best and "t_svc_mean" in best else "-", "yes" if mono else "NO"))
        print("\n## cap -> B misses / RU late slots (closed loop)")
        for load in ("peak", "average"):
            print("| %s | R0 B %s late %.0f | " % (load, b_r0[load][1], peak_ref if load == "peak" else avg_ref) + " | ".join("%s c%d: B %d / late %d" % (r["model"], r["cap"], r["b_miss"], r["ru_late_slots"]) for r in sorted((r for r in res if r["load"] == load and r["kind"] == "S"), key=lambda r: (r["model"], r["cap"], r["rerun"]))) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
