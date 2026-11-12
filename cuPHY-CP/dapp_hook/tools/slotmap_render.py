#!/usr/bin/env python3
"""slotmap_render.py - the GPU slot-map video from the L1 tenant's per-slot CSV (runs/<name>/dapp_tenant_slots.csv, 2026-11-11).

Two lanes share one time axis (slots of 0.5 ms flow left to right, the 132 SMs stack bottom to top):
  Ours      recorded: free = gap_P - gap_N - 8 is what the sizer offered YOLO in that slot, run_cls the SMs the in-flight inference held,
            RAN reserved = 132 - free (PDSCH + PUSCH + PUCCH/PRACH + Order, drawn as one "signal processing" block).
  Baseline  computed from the SAME RAN reservation and the SAME request starts with a MIG wall (72 SMs RAN, 60 SMs AI): behind the wall
            72 - RAN is stranded every slot; the AI side serves the same requests at --baseline-factor x the recorded duration (FIFO).
Metrics: Stranded SMs (1 s mean per slot), Reclaimed for AI (= free - 60), SM-slot utilization ((RAN + AI running) / 132), cumulative SM-slots stranded.

  slotmap_render.py runs/<name>/dapp_tenant_slots.csv -o runs/<name>/slotmap.mp4 [--from-sec 0] [--sec 10] [--slowdown 20] [--fps 60]
                    [--view 80] [--baseline-factor 1.3] [--wall 72 --ai-wall 60] [--size 1920x1080] [--lang ko|en] [--png-dir DIR]
Playback is 1/slowdown of real time (10 real seconds at --slowdown 20 = 200 s of video). Needs Pillow (+ numpy, cv2 for .mp4; without cv2
the frames go to --png-dir as PNGs). Older CSVs without gap_P/run_cls fall back to the plan columns (RAN approximate, AI from `class`).
"""
import argparse, csv, gzip, math, os, sys, time
from PIL import Image, ImageDraw, ImageFont

TDD = "DDDDDDSUUU"; NSM = 132; ORDER = 8
C = dict(bg=(10, 14, 19), panel=(17, 22, 29), panel2=(23, 30, 39), line=(35, 44, 55), fg=(232, 238, 244), muted=(136, 148, 161), faint=(91, 102, 115),
         base=(163, 173, 185), ours=(79, 209, 197), ran=(63, 110, 166), ai=(217, 179, 106), ai_idle=(58, 50, 34), stranded=(15, 19, 24), hatch=(42, 51, 64),
         wall=(255, 107, 107), good=(61, 220, 151))
T = {  # ko / en strings
    "title": ("GPU Slot Map", "GPU Slot Map"),
    "sub": ("슬롯(0.5 ms)마다 GPU 132개 코어를 누가 쓰는가 · 재생 1/{sd} 배속 · {run}", "who uses the 132 GPU cores in every 0.5 ms slot · playback 1/{sd} · {run}"),
    "base": ("Baseline · MIG 고정 분할", "Baseline · static MIG split"), "base_sub": ("5G에 72 SM, AI에 60 SM. 벽 너머는 못 씁니다.", "72 SMs for 5G, 60 for AI; nothing crosses the wall."),
    "ours": ("Ours · 슬롯별 green context", "Ours · per-slot green contexts"), "ours_sub": ("5G가 쓰는 만큼만 떼어 주고 나머지를 AI가 받습니다.", "5G takes what it needs; AI gets the rest."),
    "stranded": ("Stranded SMs · 고립", "Stranded SMs"), "per_slot": ("1초 평균, 슬롯당", "1 s mean per slot"), "ai_avail": ("AI에 쓸 수 있는 SM", "SMs available to AI"),
    "fixed": ("고정", "fixed"), "reclaimed": ("Reclaimed for AI · 회수", "Reclaimed for AI"), "vs60": ("Baseline 60 대비", "vs 60 in Baseline"),
    "util": ("SM-slot utilization", "SM-slot utilization"), "util_sub": ("(5G 사용 + AI 추론) / 132", "(5G in use + AI running) / 132"),
    "odo": ("시작 후 버린 SM·slot", "SM-slots stranded since start"), "queue": ("AI 대기열", "AI queue"), "clock": ("기록 시각", "recorded at"),
    "leg_ran": ("신호 처리 (PDSCH·PUSCH·PUCCH/PRACH·Order)", "signal processing (PDSCH, PUSCH, PUCCH/PRACH, Order)"), "leg_ai": ("AI 추론 중", "AI inference running"),
    "leg_idle": ("AI 몫, 요청 없음", "AI share, no request"), "leg_str": ("Stranded · 아무도 못 쓰는 SM", "Stranded: SMs nobody can use"),
    "wall": ("MIG wall 72 | 60", "MIG wall 72 | 60"), "ago": ("← {ms} ms 전", "← {ms} ms ago"), "now": ("지금 →", "now →"),
    "foot": ("Ours = L1 테넌트가 슬롯마다 기록한 실측 플랜. Baseline = 같은 5G 예약·같은 요청에 MIG 72/60 벽을 씌운 계산값(추론 시간 ×{bf}).",
             "Ours = the plan the L1 tenant recorded every slot. Baseline = the same 5G reservation and requests behind a 72/60 MIG wall (inference x{bf})."),
}

def tr(k, lang, **kw): return T[k][0 if lang == "ko" else 1].format(**kw)

def font(size, mono=False, bold=False):
    cands = (["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"] if mono else
             ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for p in cands:
        if os.path.exists(p):
            if p.endswith(".ttc"):
                for idx in range(6):
                    try:
                        f = ImageFont.truetype(p, size, index=idx)
                        if "KR" in f.getname()[0]: return f
                    except Exception: break
                return ImageFont.truetype(p, size, index=0)
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def load(path, from_sec, sec):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as fh:
        rd = csv.reader(fh); hdr = next(rd); ix = {h: i for i, h in enumerate(hdr)}
        new = "gap_P" in ix and "run_cls" in ix
        if not new: print("note: old CSV format (no gap_P/run_cls): RAN from the plan columns, AI from `class` (segment-enqueue slots only)", file=sys.stderr)
        iS, iT = ix["abs_slot"], ix.get("t_ns"); iP, iN = ix.get("gap_P"), ix.get("gap_N"); iR = ix.get("run_cls", ix["class"]); iQ, iM = ix.get("queue"), ix.get("model")
        iUL, iDL, iPOS = ix["plan_ul_sm"], ix["plan_dl_sm"], ix["pos"]
        rows = []; s0 = None; t0 = None
        for r in rd:
            if len(r) < len(hdr) or not r[iS].lstrip("-").isdigit(): continue
            s = int(r[iS]); s0 = s if s0 is None else s0
            if iT is not None and r[iT]:
                tn = int(r[iT]); t0 = tn if t0 is None else t0; rel = (tn - t0) / 1e9
            else: tn = 0; rel = (s - s0) / 2000.0
            if rel < from_sec: continue
            if rel >= from_sec + sec: break
            if new and int(r[iP]) > 0 and int(r[iN]) >= 0:
                # usable YOLO positions of the union layout: [N, 64) and [72, min(P, 120)); the 12-SM group [120-131] is never part of a class
                # (so (P132,N24) = 88, (P112,N24) = 80, (P104,N24) = 72, (P132,N40) = 72); its 12 SMs count as stranded when P = 132
                P_, N_ = int(r[iP]), int(r[iN]); free = max(0, 64 - N_) + max(0, min(P_, 120) - 72)
            else:
                dl = int(r[iDL]); ul = int(r[iUL]); dl = dl if dl > 0 else 24; ul = ul if ul > 0 else 0
                free = NSM - (dl + ORDER + ul)
            free = max(0, min(NSM - 24 - ORDER, free))
            if new and int(r[iP]) > 0 and int(r[iN]) >= 0: ran = int(r[iN]) + ORDER + (NSM - int(r[iP]))     # what the 5G side really reserves
            else: ran = NSM - free
            ran = max(0, min(NSM - free, ran))
            run = int(r[iR]) if r[iR] else 0; q = int(r[iQ]) if iQ is not None and r[iQ] else 0; m = r[iM] if iM is not None else ""
            rows.append((s, tn, int(r[iPOS]), free, run, q, m, ran))       # NSM - ran - free = SMs no class can take (the 12-SM group when P = 132)
    return rows, new

def baseline(rows, factor, ai_wall):
    """same request starts, FIFO single server at --baseline-factor x the recorded duration -> busy flag + queue depth per slot"""
    n = len(rows); starts = []
    for i in range(n):
        run, q, m = rows[i][4], rows[i][5], rows[i][6]; prun, pq, pm = (rows[i - 1][4], rows[i - 1][5], rows[i - 1][6]) if i else (0, 0, "")
        if run > 0 and (prun == 0 or m != pm or q < pq): starts.append(i)
    busy = [0] * n; queue = [0] * n; free_at = 0; pending = []
    durs = []
    for k, st in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else n
        d = 0
        for j in range(st, end):
            if rows[j][4] == 0: break
            d += 1
        durs.append(max(1, int(round(d * factor))))
    jobs = list(zip(starts, durs)); jp = 0; srv_end = -1; waiting = []
    for i in range(n):
        while jp < len(jobs) and jobs[jp][0] <= i: waiting.append(jobs[jp]); jp += 1
        if srv_end < i and waiting: _, d = waiting.pop(0); srv_end = i + d - 1
        busy[i] = 1 if srv_end >= i else 0; queue[i] = len(waiting)
    return busy, queue, len(starts)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv"); ap.add_argument("-o", "--out", default="slotmap.mp4"); ap.add_argument("--from-sec", type=float, default=0.0); ap.add_argument("--sec", type=float, default=10.0)
    ap.add_argument("--slowdown", type=float, default=20.0); ap.add_argument("--fps", type=int, default=60); ap.add_argument("--view", type=int, default=80)
    ap.add_argument("--baseline-factor", type=float, default=1.3); ap.add_argument("--wall", type=int, default=72); ap.add_argument("--ai-wall", type=int, default=60)
    ap.add_argument("--size", default="1920x1080"); ap.add_argument("--lang", default="ko"); ap.add_argument("--png-dir"); ap.add_argument("--max-frames", type=int, default=18000)
    ap.add_argument("--run", default=""); a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x")); lang = a.lang; run = a.run or os.path.basename(os.path.dirname(os.path.abspath(a.csv)))
    t_load = time.time(); rows, new = load(a.csv, a.from_sec, a.sec)
    if len(rows) < a.view + 1: sys.exit(f"only {len(rows)} slots in the window (from {a.from_sec} s, {a.sec} s) - nothing to render")
    bbusy, bqueue, nreq = baseline(rows, a.baseline_factor, a.ai_wall)
    n = len(rows); print(f"{n} slots ({n / 2000:.1f} s), {nreq} request starts, loaded in {time.time() - t_load:.1f} s", file=sys.stderr)
    # per-slot figures + prefix sums for 1 s means
    import itertools
    so = [(NSM - r[7] - r[3]) + (max(0, r[3] - r[4]) if r[4] > 0 else 0) for r in rows]   # ours stranded: the unusable group + free the chosen class left
    sb = [max(0, a.wall - r[7]) for r in rows]                                           # baseline stranded behind the RAN wall
    uo = [min(NSM, r[7] + r[4]) for r in rows]                                           # ours used (RAN + AI running)
    ub = [min(NSM, min(a.wall, r[7]) + (a.ai_wall if bbusy[i] else 0)) for i, r in enumerate(rows)]
    fo = [r[3] for r in rows]
    P = {k: list(itertools.accumulate(v, initial=0)) for k, v in (("so", so), ("sb", sb), ("uo", uo), ("ub", ub), ("fo", fo))}
    mean = lambda k, i: (P[k][i + 1] - P[k][max(0, i + 1 - 2000)]) / min(i + 1, 2000)
    # static layer
    F = dict(h1=font(40, bold=True), sub=font(20), lane=font(26, bold=True), lanesub=font(18), k=font(15), big=font(46, mono=True, bold=True), unit=font(18), n=font(16),
             mono=font(16, mono=True), legend=font(17), foot=font(16), letter=font(16, mono=True, bold=True))
    gx = [(40, 940), (W - 940, W - 40)]; gy0, gy1 = 222, 650; gw = gx[0][1] - gx[0][0]; gh = gy1 - gy0
    base_img = Image.new("RGB", (W, H), C["bg"]); d = ImageDraw.Draw(base_img)
    d.text((40, 28), tr("title", lang), font=F["h1"], fill=C["fg"]); d.text((40, 80), tr("sub", lang, sd=f"{a.slowdown:g}", run=run), font=F["sub"], fill=C["muted"])
    for li, (x0, x1) in enumerate(gx):
        d.rounded_rectangle((x0 - 20, 120, x1 + 20, 960), radius=16, fill=C["panel"], outline=C["ours"] if li else C["line"])
        d.text((x0, 132), tr("ours" if li else "base", lang), font=F["lane"], fill=C["ours"] if li else C["base"])
        d.text((x0, 166), tr("ours_sub" if li else "base_sub", lang), font=F["lanesub"], fill=C["muted"])
        d.rectangle((x0, gy0, x1, gy1), fill=(5, 7, 10))
        d.text((x0, gy1 + 6), tr("ago", lang, ms=a.view // 2), font=F["n"], fill=C["faint"]); d.text((x1 - 70, gy1 + 6), tr("now", lang), font=F["n"], fill=C["faint"])
        for c in range(3): d.line((x0 + c * gw / 3, 700, x0 + c * gw / 3, 830), fill=C["line"]) if c else None
        d.line((x0 - 20, 690, x1 + 20, 690), fill=C["line"]); d.line((x0 - 20, 840, x1 + 20, 840), fill=C["line"])
        ks = (tr("stranded", lang), tr("ai_avail", lang) if not li else tr("reclaimed", lang), tr("util", lang))
        for c, k in enumerate(ks): d.text((x0 + c * gw / 3 + (12 if c else 0), 704), k.upper(), font=F["k"], fill=C["faint"])
        d.text((x0, 856), tr("odo", lang).upper(), font=F["k"], fill=C["faint"]); d.text((x0 + gw / 2, 856), (tr("queue", lang) + " · " + tr("clock", lang)).upper(), font=F["k"], fill=C["faint"])
    lx = 40
    for col, key, hatch in ((C["ran"], "leg_ran", False), (C["ai"], "leg_ai", False), (C["ai_idle"], "leg_idle", False), (C["stranded"], "leg_str", True)):
        d.rectangle((lx, 986, lx + 18, 1004), fill=col, outline=C["line"])
        if hatch:
            for k in range(-18, 18, 5): d.line((lx + k, 1004, lx + k + 18, 986), fill=C["hatch"])
        d.text((lx + 26, 984), tr(key, lang), font=F["legend"], fill=C["muted"]); lx += 26 + int(d.textlength(tr(key, lang), font=F["legend"])) + 34
    d.text((40, 1030), tr("foot", lang, bf=f"{a.baseline_factor:g}"), font=F["foot"], fill=C["faint"])
    # hatch tile the size of a grid
    hatch_img = Image.new("RGB", (gw, gh), C["stranded"]); hd = ImageDraw.Draw(hatch_img)
    for k in range(-gh, gw + gh, 7): hd.line((k, gh, k + gh, 0), fill=C["hatch"])
    # writer
    writer = None; png_dir = a.png_dir
    if not png_dir:
        try:
            import cv2, numpy as np
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
            if not writer.isOpened(): raise RuntimeError("cv2.VideoWriter could not open " + a.out)
        except Exception as e:
            print("note:", e, "-> PNG frames", file=sys.stderr); png_dir = os.path.splitext(a.out)[0] + "_png"
    if png_dir: os.makedirs(png_dir, exist_ok=True)
    spf = 2000.0 / (a.slowdown * a.fps); frames = min(a.max_frames, int((n - a.view) / spf)); colw = gw / a.view; sm_h = gh / NSM
    print(f"{frames} frames at {a.fps} fps = {frames / a.fps:.0f} s of video, {spf:.2f} slots per frame", file=sys.stderr)
    t_r = time.time()
    for f in range(frames):
        i = a.view + int(f * spf); img = base_img.copy(); d = ImageDraw.Draw(img)
        for li, (x0, x1) in enumerate(gx):
            for k in range(a.view):
                j = i - a.view + 1 + k; r = rows[j]; x = x0 + k * colw + 0.6; w = colw - 1.2; y = gy1
                ran = r[7]; leftover = NSM - r[7] - r[3]
                def put(nsm, col):
                    nonlocal y
                    if nsm <= 0: return
                    h = nsm * sm_h; y -= h; d.rectangle((x, y, x + w, y + h - 0.6), fill=col)
                def put_hatch(nsm):
                    nonlocal y
                    if nsm <= 0: return
                    h = nsm * sm_h; y -= h; box = (int(x), int(y), int(x + w), int(y + h - 0.6))
                    if box[2] > box[0] and box[3] > box[1]: img.paste(hatch_img.crop((box[0] - x0, box[1] - gy0, box[2] - x0, box[3] - gy0)), (box[0], box[1]))
                if li == 0:
                    rb = min(a.wall, ran); put(rb, C["ran"]); put_hatch(a.wall - rb); put(a.ai_wall, C["ai"] if bbusy[j] else C["ai_idle"])
                else:
                    put(ran, C["ran"])
                    if r[4] > 0: put(r[4], C["ai"]); put_hatch(r[3] - r[4] + leftover)
                    else: put(r[3], C["ai_idle"]); put_hatch(leftover)
                if k % 2 == 0 or colw > 14: d.text((x + w / 2 - 5, gy0 - 20), TDD[r[2]], font=F["letter"], fill=C["faint"])
            if li == 0:
                yw = gy1 - a.wall * sm_h
                for xx in range(int(x0), int(x1), 12): d.line((xx, yw, xx + 6, yw), fill=C["wall"], width=2)
                d.text((x0 + 8, yw - 24), tr("wall", lang), font=F["mono"], fill=C["wall"])
            # HUD
            if li == 0: vals = (f"{mean('sb', i):.0f}", f"{a.ai_wall}", f"{100 * mean('ub', i) / NSM:.0f}"); units = ("SM", "SM", "%"); subs = (tr("per_slot", lang), tr("fixed", lang), tr("util_sub", lang)); col = C["base"]
            else: rec = mean("fo", i) - a.ai_wall; vals = (f"{mean('so', i):.0f}", f"{rec:+.0f}", f"{100 * mean('uo', i) / NSM:.0f}"); units = ("SM", "SM", "%"); subs = (tr("per_slot", lang), tr("vs60", lang), tr("util_sub", lang)); col = C["ours"]
            for c in range(3):
                cx = x0 + c * gw / 3 + (12 if c else 0); d.text((cx, 730), vals[c], font=F["big"], fill=col if c == 0 else C["fg"])
                d.text((cx + d.textlength(vals[c], font=F["big"]) + 8, 752), units[c], font=F["unit"], fill=C["muted"]); d.text((cx, 790), subs[c], font=F["n"], fill=C["muted"])
            odo = P["sb" if li == 0 else "so"][i + 1]; d.text((x0, 880), f"{odo:,}", font=F["big"], fill=col)
            q = bqueue[i] if li == 0 else rows[i][5]; clock = time.strftime("%H:%M:%S", time.localtime(rows[i][1] / 1e9)) + f".{int(rows[i][1] / 1e6) % 1000:03d}" if rows[i][1] else f"slot {rows[i][0]}"
            d.text((x0 + gw / 2, 880), f"{q}", font=F["big"], fill=C["fg"]); d.text((x0 + gw / 2 + 80, 896), clock, font=F["mono"], fill=C["muted"])
        if writer is not None: writer.write(np.asarray(img)[:, :, ::-1].copy())
        else: img.save(os.path.join(png_dir, f"f{f:06d}.png"))
        if f % 300 == 0 and f: print(f"  frame {f}/{frames} ({time.time() - t_r:.0f} s)", file=sys.stderr)
    if writer is not None: writer.release()
    print(f"done: {a.out if writer is not None else png_dir} | 1 s means at the end: stranded base {mean('sb', n - 1):.1f} ours {mean('so', n - 1):.1f} SM, "
          f"AI available ours {mean('fo', n - 1):.1f} SM, util base {100 * mean('ub', n - 1) / NSM:.0f} % ours {100 * mean('uo', n - 1) / NSM:.0f} % | "
          f"stranded SM-slots total base {P['sb'][n]:,} ours {P['so'][n]:,}", file=sys.stderr)

if __name__ == "__main__":
    main()
