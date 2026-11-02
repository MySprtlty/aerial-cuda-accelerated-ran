#!/usr/bin/env python3
"""kpi_html.py [--r 105] [names...]: the KPI table of kpi_run.sh results as an HTML fragment (for the report page)."""
import argparse, csv, glob, os, re, statistics
MAP = {"26n": 40.9, "26s": 48.6, "26m": 53.1, "26l": 55.0, "26x": 57.5}
ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="*"); ap.add_argument("--r", type=float, default=105.0); ap.add_argument("--base-server", type=float, default=46.3)
a = ap.parse_args()
D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "prof", "tenant")
names = a.names or sorted(os.path.basename(p)[3:-4] for p in glob.glob(os.path.join(D, "ra_*.csv")))
base_e2e = a.r + a.base_server
LABEL = {"k1_26l_av4": "1셀 Average · 26l 고정 · 4 req/s", "k1_26x_av4": "1셀 Average · 26x 고정 · 4 req/s", "k1_map_av8": "1셀 Average · map 정책(deadline 40) · 8 req/s",
         "k1_dl_av8": "1셀 Average · deadline 30 정책 · 8 req/s", "k1_map_pk8": "1셀 Peak · map 정책 · 8 req/s", "k1_26x_av12": "1셀 Average · 26x 고정 · 12 req/s",
         "k4_26x_av4": "4셀 Average · 26x 고정 · 4 req/s", "k4_map_av8": "4셀 Average · map 정책 · 8 req/s", "b_noinf": "이분법: 디코드만(추론 생략)", "b_nodec": "이분법: 추론만(디코드 생략)", "c56_26x_av4": "구멍 회피 설정(PUSCH 96·PDSCH 40·class 56) 26x", "a1_26x_av4": "TRT 보조 스트림 수정 26x", "a2_26x_av4": "CUDA 그래프 26x", "a2_26n_av4": "CUDA 그래프 26n(크기 시험)", "dbg_26x_av4": "디버그 26x", "dbg2_26x_av4": "디버그2 26x",
         "u1_26n_av4": "합집합 배치 guard 1 (class 56) · 26n · 4 req/s", "u1_26s_av4": "합집합 배치 guard 1 (class 56) · 26s", "u1_26m_av4": "합집합 배치 guard 1 (class 56) · 26m", "u1_26l_av4": "합집합 배치 guard 1 (class 56) · 26l", "u1_26x_av4": "합집합 배치 guard 1 (class 56) · 26x",
         "u0_26m_av4": "합집합 배치 guard 0 (class 80) · 26m", "u0_26l_av4": "합집합 배치 guard 0 (class 80) · 26l", "u0_26x_av4": "합집합 배치 guard 0 (class 80) · 26x", "u1_map_av8": "합집합 guard 1 · map 정책(deadline 40) · 8 req/s", "u0_map_av8": "합집합 guard 0 · map 정책(deadline 40) · 8 req/s",
         "u2_26n_av4": "조각 발사 · guard 1 (class 56) · 26n", "u2_26s_av4": "조각 발사 · guard 1 (class 56) · 26s", "u2_26m_av4": "조각 발사 · guard 1 (class 56) · 26m", "u2_26l_av4": "조각 발사 · guard 1 (class 56) · 26l", "u2_26x_av4": "조각 발사 · guard 1 (class 56) · 26x",
         "u3_26m_av4": "조각 발사 · guard 0 (class 80) · 26m", "u3_26l_av4": "조각 발사 · guard 0 (class 80) · 26l", "u3_26x_av4": "조각 발사 · guard 0 (class 80) · 26x", "u2_map_av8": "조각 발사 · guard 1 · map 정책(deadline 40) · 8 req/s", "u3_map_av8": "조각 발사 · guard 0 · map 정책(deadline 40) · 8 req/s",
         "s1_26m_av4": "진단: 보조 스트림 0 · class 56만 · 26m(MPS 없음)", "f26m_av4": "진단: nsys 트레이스 · 26m", "n26m_av4": "진단: nsys capture-range(실패) · 26m",
         "v_26x_av8": "검증: 3노드 조각 · 80 SM · 26x · 8 req/s", "v_26l_av4_g1": "검증: 3노드 조각 · guard 1 · 26l", "v_map_av8_dl40": "검증: 3노드 조각 · 80 SM · map(deadline 40) · 8 req/s", "v_26n_av4": "검증: 3노드 조각 · 80 SM · 26n", "v_26s_av4": "검증: 3노드 조각 · 80 SM · 26s", "v_26m_av4": "검증: 3노드 조각 · 80 SM · 26m", "v_26l_av4": "검증: 3노드 조각 · 80 SM · 26l", "v_26x_av4": "검증: 3노드 조각 · 80 SM · 26x", "v_map_av8_dl32": "검증: 3노드 조각 · 80 SM · map(deadline 32) · 8 req/s",
         "w_26x_av8": "6노드·선행 2·80 SM: 26x · 8 req/s", "w_26m_av4": "6노드·선행 2·80 SM: 26m", "w_26l_av4": "6노드·선행 2·80 SM: 26l", "w_26x_av4": "6노드·선행 2·80 SM: 26x", "w_map_av8_dl36": "6노드·선행 2·80 SM: map(deadline 36) · 8 req/s",
         "x_26x_av8": "최종(3노드·선행 2·80 SM): 26x · 8 req/s", "x_26m_av4": "최종(3노드·선행 2·80 SM): 26m", "x_26l_av4": "최종(3노드·선행 2·80 SM): 26l", "x_26x_av4": "최종(3노드·선행 2·80 SM): 26x", "x_map_av8_dl36": "최종(3노드·선행 2·80 SM): map(deadline 36) · 8 req/s"}
out = ['<div class="tw"><table>', '<tr><th>런</th><th class="n">L1 recovery / FATAL</th><th class="n">req/s</th><th class="n">n / 실패</th><th class="n">서버 평균 / p50 / p90 (ms)</th><th class="n">E2E(R=%.0f)</th><th class="n">baseline 대비</th><th class="n">mAP</th><th>KPI1 / KPI2</th><th>모델 (평균 SM)</th><th class="n">PUSCH p99 / miss</th><th class="n">RU late</th></tr>' % a.r]
out.append('<tr><td>baseline: MIG 3g 26x (실측 10-16)</td><td class="n">–</td><td class="n">–</td><td class="n">–</td><td class="n">46.3</td><td class="n">%.1f</td><td class="n">0</td><td class="n">57.5</td><td>–</td><td>26x (60)</td><td class="n">–</td><td class="n">–</td></tr>' % base_e2e)
for n in names:
    f = os.path.join(D, "ra_%s.csv" % n)
    if not os.path.exists(f): continue
    rows = list(csv.DictReader(open(f))); ok = [r for r in rows if r["status"] == "success"]
    if not ok: out.append('<tr><td>%s</td><td colspan="11">성공 요청 없음 (%d)</td></tr>' % (LABEL.get(n, n), len(rows))); continue
    srv = [float(r["server_ms"]) for r in ok]
    dur = (max(float(r["t_arrival"]) for r in rows) - min(float(r["t_arrival"]) for r in rows)) or 1.0; rate = len(rows) / dur
    wmap = sum(MAP[r["model"]] for r in ok) / len(ok); E = a.r + statistics.mean(srv)
    by = {}
    for r in ok: by.setdefault(r["model"], []).append(float(r["sm_mean"] or 0))
    models = ", ".join("%s %d (%.0f)" % (m, len(v), statistics.mean(v)) for m, v in sorted(by.items()))
    k1 = E <= 0.9 * base_e2e; k2 = wmap >= 51.75
    pu = ru = "–"
    l1 = os.path.join(D, "run2_%s.log" % n); rec = fat = -1
    if os.path.exists(l1):
        lt = open(l1, errors="replace").read(); rec = lt.count("L1 in recovery for"); fat = lt.count("FATAL exit")
    invalid = rec > 0 or fat > 0
    s = os.path.join(D, "summary_%s.txt" % n)
    if os.path.exists(s):
        txt = open(s, errors="replace").read()
        p = re.findall(r"PUSCH completion pos (\d): n\s+\d+ p50\s+(\d+) p99\s+(\d+).*?> 2100 us: (\d+)", txt); r_ = re.findall(r"DL U LATE SLOTS\s*\|\s*(\d+)", txt)
        if p: pu = "%s / %s" % ("·".join(x[2] for x in p), "·".join(x[3] for x in p))
        if r_: ru = r_[0]
    out.append('<tr><td>%s</td><td class="n">%s</td><td class="n">%.1f</td><td class="n">%d / %d</td><td class="n">%.1f / %.1f / %.1f</td><td class="n">%.1f</td><td class="n">%+.1f %%</td><td class="n">%.1f (%+.1f %%)</td><td><span class="%s">%s</span> / <span class="%s">%s</span></td><td>%s</td><td class="n">%s</td><td class="n">%s</td></tr>' % (
        LABEL.get(n, n), ('<span class="bad">%d / %d 무효</span>' % (rec, fat)) if invalid else ('0 / 0' if rec == 0 else '–'), rate, len(rows), len(rows) - len(ok), statistics.mean(srv), statistics.median(srv), sorted(srv)[int(0.9 * (len(srv) - 1))], E, 100 * (E / base_e2e - 1), wmap, 100 * (wmap / 57.5 - 1),
        "ok" if k1 else "bad", "OK" if k1 else "FAIL", "ok" if k2 else "bad", "OK" if k2 else "FAIL", models, pu, ru))
out.append("</table></div>")
print("\n".join(out))
