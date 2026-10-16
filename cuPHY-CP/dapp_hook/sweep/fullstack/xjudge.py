#!/usr/bin/env python3
"""xjudge.py --dir D --x X [--r 105] [--map "26n=40.9,26s=48.6,26m=53.1,26l=55.0,26x=57.5"] [--target-map 51.75]
Per-X judgment (night prompt §6, amended 2026-10-17 04:20) from xgrid_X<X>P.csv / xgrid_X<X>A.csv, budgets_X<X>{P,A}.json and
xgrid_MIG.csv (4g cuPHY rows re-scored here with the full-GPU budgets of the same config).
  Baseline 1 : 4g cuPHY protection (Peak / Average) + 3g 26x sojourn at 10 req/s (concurrent run if it differs > 10 %) -> E2E_B1
  Baseline 2 : naive 26x@100; static-safe = pairs that PASS in both loads, duty-weighted sojourn <= target with max mAP (else min
               sojourn); pairs passing one load but unmeasured in the other are listed as '미완' (recomputed when the fill cells land)
  Ours       : (light, heavy) chosen jointly per duty (50:50, 60:40 light:heavy): among light = Average-passing pairs and heavy =
               Peak-passing pairs, the combination with duty-weighted sojourn <= target and the highest weighted mAP (tie: lower sojourn);
               none feasible -> lowest sojourn. ours_alt (reference): the feasible pair with the lowest sojourn whose weighted mAP is
               not below the static-safe choice's mAP.
  target sojourn = 0.9 x (R + sojourn_B1) - R;  KPI1: R + ours sojourn <= 0.9 x E2E_B1;  KPI2: weighted mAP >= target-map
  ramp table (추정): per light candidate, Peak B_abs overrun rate of the same pair per UL slot (b_abs_all / (60 s x 600 UL slot/s)) x
               exposure UL slots x switches per 10 min (2.56 s period: 234, 60 s period: 10), plus heavy's own Peak overruns x 10.
               Exposure = in-flight light request (service p50 -> /5 ms x 3 UL slots) and, in the second figure, + classifier
               detection lag (--detect-ms, default 600 = 60-UL-slot window 100 ms + hold 500 ms). Reference: 5/60 s = 50 per 10 min.
  spots      : kind 'spot' rows (8 req/s) listed as measured, not used for the selection."""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import xgrid_table as XT   # noqa: E402


def load_grid(d, x, load):
    p = os.path.join(d, "xgrid_X%d%s.csv" % (x, "P" if load == "peak" else "A"))
    rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    cells, spots = {}, {}
    for r in rows:
        if not (r.get("model") and r.get("cap")):
            continue
        key = (r["model"], int(float(r["cap"])))
        if r["kind"] in ("cell", "rate", "naive") and not r["verdict"].startswith("INVALID"):
            cells[key] = r
        elif r["kind"] == "spot":
            spots[key] = r
    return cells, spots, [r for r in rows if r["kind"] == "R0"]


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def ok(r):
    return r is not None and r.get("verdict") == "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--x", type=int, required=True); ap.add_argument("--r", type=float, default=105.0)
    ap.add_argument("--map", default="26n=40.9,26s=48.6,26m=53.1,26l=55.0,26x=57.5"); ap.add_argument("--target-map", type=float, default=51.75); ap.add_argument("--detect-ms", type=float, default=600.0)
    a = ap.parse_args()
    d, X, R, TM = a.dir, a.x, a.r, a.target_map
    MAP = {k: float(v) for k, v in (kv.split("=") for kv in a.map.split(","))}
    pk, pk_spots, pk_r0 = load_grid(d, X, "peak"); av, av_spots, av_r0 = load_grid(d, X, "average")
    mig = {r["name"]: r for r in csv.DictReader(open(os.path.join(d, "xgrid_MIG.csv")))} if os.path.exists(os.path.join(d, "xgrid_MIG.csv")) else {}
    out = {"X": X}
    for l, sfx in (("peak", "P"), ("average", "A")):
        bp = os.path.join(d, "budgets_X%d%s.json" % (X, sfx)); bud = json.load(open(bp)) if os.path.exists(bp) else None
        name = "MIG_mig4g_%d%s_%dc" % (X, sfx, X)
        if bud and os.path.exists(os.path.join(d, "phy_%s.csv" % name)):
            row = XT.evaluate(d, name, bud, "mig-cuphy", "", "")
            out["mig_prot_" + l] = "%s (B_abs %s, late pkt frac %s, Err %s)" % (row["verdict"], row["b_abs_all"], row.get("late_pkt_frac"), row["errind"])
        else:
            out["mig_prot_" + l] = "미측정" if name not in mig else "예산 없음"
    alone, iso = mig.get("mig3g_26x_r10"), mig.get("MIG_migiso_%dP_%dc" % (X, X))
    soj_b1 = num(alone, "soj_mean") if alone else float("nan"); src = "3g 단독"
    if iso and alone and abs(num(iso, "soj_mean") - soj_b1) > 0.1 * soj_b1:
        soj_b1 = num(iso, "soj_mean"); src = "동시 런(단독과 10 % 이상 차이)"
    elif iso:
        src += " (동시 %.1f, 10 %% 안)" % num(iso, "soj_mean")
    e2e_b1 = R + soj_b1; target = 0.9 * e2e_b1 - R
    out.update(b1=src, soj_b1=soj_b1, e2e_b1=e2e_b1, target_soj=target)
    nv = lambda r: ("%s (체류 %.1f)" % (r["verdict"], num(r, "soj_mean"))) if r else "미측정"      # noqa: E731
    out["naive"] = "Peak %s / Average %s" % (nv(pk.get(("26x", 100))), nv(av.get(("26x", 100))))
    # static-safe
    both = [mc for mc in pk if mc in av and ok(pk[mc]) and ok(av[mc])]
    incomplete = [mc for mc in set(pk) | set(av) if (ok(pk.get(mc)) and mc not in av) or (ok(av.get(mc)) and mc not in pk)]
    def dsoj(mc, wl):
        return wl * num(av[mc], "soj_mean") + (1 - wl) * num(pk[mc], "soj_mean")
    ss = {}
    for wl, tag in ((0.5, "50"), (0.6, "60")):
        fit = [mc for mc in both if dsoj(mc, wl) <= target]
        c = max(fit, key=lambda mc: (MAP[mc[0]], -dsoj(mc, wl))) if fit else (min(both, key=lambda mc: dsoj(mc, wl)) if both else None)
        if c:
            s_e2e = R + dsoj(c, wl); kpi = (s_e2e <= 0.9 * e2e_b1, MAP[c[0]] >= TM)
            ss[tag] = (c, dsoj(c, wl), MAP[c[0]], s_e2e, kpi)
            out["static_safe_" + tag] = "%s@%d 체류 %.1f E2E %.1f mAP %.1f | KPI1 %s KPI2 %s" % (c[0], c[1], dsoj(c, wl), s_e2e, MAP[c[0]], "OK" if kpi[0] else "FAIL", "OK" if kpi[1] else "FAIL")
        else:
            out["static_safe_" + tag] = "없음(양 부하 통과 쌍 없음)"
    out["static_safe_incomplete"] = ", ".join("%s@%d" % mc for mc in sorted(incomplete)) or "-"
    # ours: joint (light, heavy)
    L = [mc for mc in av if ok(av[mc])]; H = [mc for mc in pk if ok(pk[mc])]
    for wl, tag in ((0.5, "50"), (0.6, "60")):
        combos = [(l, h) for l in L for h in H]
        f = lambda l, h: wl * num(av[l], "soj_mean") + (1 - wl) * num(pk[h], "soj_mean")     # noqa: E731
        m = lambda l, h: wl * MAP[l[0]] + (1 - wl) * MAP[h[0]]                               # noqa: E731
        fit = [c for c in combos if f(*c) <= target]
        best = max(fit, key=lambda c: (m(*c), -f(*c))) if fit else (min(combos, key=lambda c: f(*c)) if combos else None)
        if best:
            l, h = best; soj = f(l, h); mp = m(l, h)
            out["ours_" + tag] = "light %s@%d (%.1f) + heavy %s@%d (%.1f) -> 체류 %.1f E2E %.1f mAP %.1f | KPI1 %s KPI2 %s%s" % (
                l[0], l[1], num(av[l], "soj_mean"), h[0], h[1], num(pk[h], "soj_mean"), soj, R + soj, mp, "OK" if R + soj <= 0.9 * e2e_b1 else "FAIL", "OK" if mp >= TM else "FAIL", "" if fit else " (목표 불충족: 최소 체류 쌍)")
            if tag in ss:
                out["gain_" + tag] = "static-safe 대비 mAP %+.1f, 체류 %+.1f ms" % (mp - ss[tag][2], soj - ss[tag][1])
                # 참고: static-safe 이상의 mAP를 지키면서 체류가 가장 짧은 쌍 (KPI1 여유를 택하는 대안)
                alt = [c for c in fit if m(*c) >= ss[tag][2] - 1e-9]
                if alt:
                    al, ah = min(alt, key=lambda c: (f(*c), -m(*c))); asoj = f(al, ah)
                    out["ours_alt_" + tag] = "light %s@%d + heavy %s@%d -> 체류 %.1f E2E %.1f mAP %.1f (static-safe 대비 체류 %+.1f ms, mAP %+.1f)" % (
                        al[0], al[1], ah[0], ah[1], asoj, R + asoj, m(al, ah), asoj - ss[tag][1], m(al, ah) - ss[tag][2])
            out["_pick_" + tag] = (l, h)
        else:
            out["ours_" + tag] = "없음"
    # ramp table per light candidate (two switch periods, exposure with / without detection lag)
    hp = out.get("_pick_50")
    heavy_norm = (num(pk[hp[1]], "b_abs_all") * 10) if hp else 0.0
    ramp_rows = []
    for l in sorted(L, key=lambda mc: -MAP[mc[0]]):
        lp = pk.get(l)
        if lp is None:
            ramp_rows.append((l, None)); continue
        r_ul = num(lp, "b_abs_all") / (60.0 * 600.0)
        e_inf = num(av[l], "svc_p50") / 5.0 * 3.0
        e_det = e_inf + a.detect_ms / 5.0 * 3.0
        ramp_rows.append((l, (lp["b_abs_all"], r_ul, e_inf, e_det, [(n, r_ul * e_inf * n + heavy_norm, r_ul * e_det * n + heavy_norm) for n in (234, 10)])))
    out["ramp_heavy"] = "heavy %s 정상 Peak 초과 %.0f /10 min" % (("%s@%d" % hp[1]) if hp else "-", heavy_norm)
    spots = ["Peak %s@%d: 체류 %.1f/p95 %.1f (svc %.1f, %s)" % (k[0], k[1], num(r, "soj_mean"), num(r, "soj_p95"), num(r, "svc_p50"), r["verdict"]) for k, r in sorted(pk_spots.items())] + \
            ["Average %s@%d: 체류 %.1f/p95 %.1f (svc %.1f, %s)" % (k[0], k[1], num(r, "soj_mean"), num(r, "soj_p95"), num(r, "svc_p50"), r["verdict"]) for k, r in sorted(av_spots.items())]
    out["spots_8rps"] = "; ".join(spots) or "-"
    print("## X=%d 판정 (R = %.0f ms, 체류 목표 %.1f ms, mAP 목표 %.2f)" % (X, R, target, TM))
    print("| 항목 | 값 |"); print("|---|---|")
    for k in ("mig_prot_peak", "mig_prot_average", "b1", "soj_b1", "e2e_b1", "naive", "static_safe_50", "static_safe_60", "static_safe_incomplete",
              "ours_50", "gain_50", "ours_alt_50", "ours_60", "gain_60", "ours_alt_60", "spots_8rps"):
        v = out.get(k, "-"); print("| %s | %s |" % (k, ("%.1f" % v) if isinstance(v, float) else v))
    print("\n### ramp 노출 추정 (10분당 B_abs 초과 건수, 기준 50 = 5/60 s x 10; %s; 노출 = light 처리 중 요청 / + 감지 지연 %.0f ms)" % (out["ramp_heavy"], a.detect_ms))
    print("| light 후보 | Peak B_abs /60 s | 노출 UL slot (in-flight / +감지) | 2.56 s 교대 x234 | 60 s 교대 x10 |"); print("|---|---|---|---|---|")
    for l, info in ramp_rows:
        if info is None:
            print("| %s@%d | Peak 미측정 | - | - | - |" % l); continue
        babs, r_ul, e_inf, e_det, per = info
        cells = ["%.0f / %.0f %s" % (ti, td, "OK" if td <= 50 else ("OK(in-flight만)" if ti <= 50 else "초과")) for n, ti, td in per]
        print("| %s@%d | %s | %.0f / %.0f | %s | %s |" % (l[0], l[1], babs, e_inf, e_det, cells[0], cells[1]))
    print("\nPeak 통과 쌍: %s | Average 통과 쌍: %s" % (", ".join("%s@%d" % mc for mc in sorted(H)) or "-", ", ".join("%s@%d" % mc for mc in sorted(L)) or "-"))
    print("R0 late pkt frac (start/start/close): Peak %s | Average %s" % (" / ".join(str(r.get("late_pkt_frac")) for r in pk_r0), " / ".join(str(r.get("late_pkt_frac")) for r in av_r0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
