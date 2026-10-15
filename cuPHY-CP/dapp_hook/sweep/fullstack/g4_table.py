#!/usr/bin/env python3
"""g4_table.py --dir D --prefix P --phase eval|rerun|table
eval : pass/fail of every tenant run of the sweep (safecap_check.check) -> g4_results.csv and a table
rerun: writes ladder_<P>_rerun.txt with one re-run (suffix _r2) per failed run (+ 26x cap 24 Peak if 26x fails at 43)
table: final table: model x load -> max passing cap (a cap passes when at least one of its runs passes) with service
       time, cap -> RU late slots curve, monotonicity flags. R0 reference late slots: Peak = mean(A1a/2, A1b/2, R0_59c),
       Average = R0_60c (prof/run62 A1a/A1b were 120 s runs)."""
import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import safecap_check as SC   # noqa: E402

CAPS = (24, 43, 61, 100)


def r0_ref(d, prefix, pat):
    """mean of every R0 run of this load (R0_<pat>, R0_<pat>_r2, ...) and, for Peak, of the two 120 s A1 runs halved."""
    owns = [SC.ru_late_slots(n[4:-4], d) for n in sorted(os.listdir(d)) if re.match(r"phy_%s_R0_%s(_r\d+)?_16c\.csv$" % (prefix, pat), n)]
    owns = [x for x in owns if x > 0]
    prev = []
    if pat == "59c":
        prev = [SC.ru_late_slots(n, os.path.join(d, "..", "run62")) / 2.0 for n in ("E1_A1a_16c", "E1_A1b_16c")]
        prev = [x for x in prev if x > 0]
    vals = prev + owns
    return (sum(vals) / len(vals) if vals else -1), (owns[0] if owns else -1), prev + owns[1:]


def runs(d, prefix):
    out = []
    for p in sorted(os.listdir(d)):
        m = re.match(r"phy_%s_([SP])_(26[nsx])_c(\d+)_(59c|60c)(_r2)?_16c\.csv$" % prefix, p)
        if m:
            out.append({"name": p[4:-4], "kind": m.group(1), "model": m.group(2), "cap": int(m.group(3)), "pat": m.group(4), "rerun": bool(m.group(5))})
    return out


def tenant(d, name):
    p = os.path.join(d, "disp_%s.csv" % name)
    if not os.path.exists(p):
        return None
    rows = [r for r in csv.DictReader(open(p)) if r.get("status") == "0"]
    if not rows:
        return None
    t0 = min(int(r["t_arrival"]) for r in rows); rows = [r for r in rows if int(r["t_arrival"]) - t0 >= 1e9]
    svc = sorted((int(r["t_end"]) - int(r["t_start"])) / 1e6 for r in rows if int(r["t_end"]) and int(r["t_start"]))
    e2e = sorted((int(r["t_reply"]) - int(r["t_arrival"])) / 1e6 for r in rows)
    span = (max(int(r["t_reply"]) for r in rows) - min(int(r["t_arrival"]) for r in rows)) / 1e9
    q = lambda v, p: v[min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1))))] if v else float("nan")   # noqa: E731
    return {"n": len(rows), "svc_mean": sum(svc) / len(svc) if svc else float("nan"), "svc_p50": q(svc, 50), "svc_p99": q(svc, 99), "e2e_mean": sum(e2e) / len(e2e), "e2e_p99": q(e2e, 99), "req_s": len(rows) / span if span else 0}


def evaluate(d, prefix):
    refs = {pat: r0_ref(d, prefix, pat) for pat in ("59c", "60c")}
    res = []
    for r in runs(d, prefix):
        ref = refs[r["pat"]][0]; r0name = "%s_R0_%s_16c" % (prefix, r["pat"])
        ok, why, n = SC.check(r0name, r["name"], d, 10.0, 1.0, ref if ref > 0 else None)
        t = tenant(d, r["name"]) or {}
        res.append(dict(r, passed=ok, why="; ".join(why), **n, **{"t_" + k: v for k, v in t.items()}))
    return refs, res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--prefix", default="G4"); ap.add_argument("--phase", default="eval")
    a = ap.parse_args(); d = a.dir
    refs, res = evaluate(d, a.prefix)
    print("R0 reference late slots: " + ", ".join("%s: %.0f (own R0 %d, prior %s)" % (p, v[0], v[1], ["%.0f" % x for x in v[2]]) for p, v in refs.items()))
    keys = sorted({k for r in res for k in r})
    with open(os.path.join(d, "g4_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in res]
    print("| run | pass | B miss (R0) | Err.ind / drops / CRC | RU late slots (ref) | service mean / p50 / p99 ms | E2E mean / p99 | req/s | why |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in res:
        print("| %s | %s | %d (%d) | %d / %d / %d | %d (%.0f) | %s | %s | %s | %s |" % (
            r["name"], "PASS" if r["passed"] else "FAIL", r["b_miss"], r["b_miss_r0"], r["errind"], r["drops"], r["crc"], r["ru_late_slots"], r["ru_late_slots_r0"] or -1,
            ("%.1f / %.1f / %.1f" % (r["t_svc_mean"], r["t_svc_p50"], r["t_svc_p99"])) if "t_svc_mean" in r else "-",
            ("%.1f / %.1f" % (r["t_e2e_mean"], r["t_e2e_p99"])) if "t_e2e_mean" in r else "-", ("%.2f" % r["t_req_s"]) if "t_req_s" in r else "-", r["why"]))
    if a.phase == "rerun":
        lst = []
        for r in res:
            if r["kind"] in ("S", "P") and not r["passed"] and not r["rerun"] and not any(x["name"] == r["name"] + "_r2" for x in res):
                lst.append(r)
        with open(os.path.join(d, "ladder_%s_rerun.txt" % a.prefix), "w") as f:
            for r in lst:
                arr = "poisson" if r["kind"] == "P" else "closed"
                f.write("%s_%s_%s_c%d_%s_r2 pattern=%s slots=120000 tags=311,223 tenant=dispatcher model=%s cap=%d arrivals=%s rate=10 tdelay=28 tsecs=45\n" % (
                    a.prefix, r["kind"], r["model"], r["cap"], r["pat"], r["pat"], r["model"], r["cap"], arr))
            peak43 = [r for r in res if r["kind"] == "S" and r["model"] == "26x" and r["cap"] == 43 and r["pat"] == "59c"]
            if peak43 and not any(r["passed"] for r in peak43) and not any(r["cap"] == 24 for r in res):
                f.write("%s_S_26x_c24_59c pattern=59c slots=120000 tags=311,223 tenant=dispatcher model=26x cap=24 arrivals=closed rate=10 tdelay=28 tsecs=45 xclass=32 xmodels=26x\n" % a.prefix)
        print("rerun list: %d run(s) -> ladder_%s_rerun.txt" % (sum(1 for _ in open(os.path.join(d, "ladder_%s_rerun.txt" % a.prefix))), a.prefix))
    if a.phase == "table":
        print("\n## SafeCap: model x load -> largest cap with a passing run (closed loop); service mean / p99 ms at that cap")
        print("| model | load | caps tried (pass marks) | SafeCap | service mean / p99 | monotonic? |")
        print("|---|---|---|---|---|---|")
        for pat in ("59c", "60c"):
            for m in ("26n", "26s", "26x"):
                rs = [r for r in res if r["kind"] == "S" and r["model"] == m and r["pat"] == pat]
                if not rs:
                    continue
                bycap = {}
                for r in rs:
                    bycap.setdefault(r["cap"], []).append(r)
                marks, safe, best = [], None, None
                for cap in sorted(bycap):
                    ok = any(r["passed"] for r in bycap[cap]); marks.append("%d:%s" % (cap, "ok" if ok else "fail"))
                    if ok:
                        safe = cap; best = next(r for r in bycap[cap] if r["passed"])
                oks = [any(r["passed"] for r in bycap[c]) for c in sorted(bycap)]
                mono = all(not (not oks[i] and any(oks[i + 1:])) for i in range(len(oks)))
                print("| %s | %s | %s | %s | %s | %s |" % (m, pat, " ".join(marks), safe if safe is not None else "none",
                      ("%.1f / %.1f" % (best["t_svc_mean"], best["t_svc_p99"])) if best and "t_svc_mean" in best else "-", "yes" if mono else "NO"))
        print("\n## cap -> RU late slots (closed loop; R0 reference first)")
        for pat in ("59c", "60c"):
            print("| %s | R0 ref %.0f | " % (pat, refs[pat][0]) + " | ".join("%s c%d: %s" % (r["model"], r["cap"], r["ru_late_slots"]) for r in sorted((r for r in res if r["pat"] == pat and r["kind"] == "S"), key=lambda r: (r["model"], r["cap"], r["rerun"]))) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
