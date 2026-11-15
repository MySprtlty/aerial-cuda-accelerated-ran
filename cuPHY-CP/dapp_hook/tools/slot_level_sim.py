#!/usr/bin/env python3
# slot_level_sim.py (2026-11-12, branch feature/slot_level) - CPU replay of YOLO inference latency under three ways of giving it SMs:
#   mig    Baseline: a fixed slice (MIG 3g = 60 SMs), independent of the L1 load
#   req    request level (today's tenant, dapp_tenant.cu choose_part): ONE class per request = the largest class that fits EVERY slot of its
#          horizon (horizon = slots spanned at the slowest loaded class + guard); no class fits -> retry at the next slot
#   piece  slot level: every 3-node graph piece runs on the largest class that fits the slot(s) that piece spans (the 132-SM engine on green
#          streams, prof/tenant/xclass); the gap is below the smallest class -> pause to a slot boundary; --wait-ahead K also considers
#          starting the piece at one of the next K slot boundaries when that finishes it earlier
# Inputs: per-piece times of the 132-SM engine per SM class (xpiece.py JSON; additive within 1-3 %, prof/tenant/slot_level), and a per-slot
# partition sequence: tools/slot_level_seq.py CSV (sizer replay of a hook ring) or --pattern (free SMs per TDD position, e.g. the measured
# 1-cell stack). Free SMs of wall slot w follow the tenant's busy rule (busy_of): PUSCH of slots w-1..w-ul_back (w-1: smallest profile only),
# PDSCH of w+1..w+dl_fwd, free = P - N - order.  All numbers are PREDICTIONS (no L1 contention, no launch jitter beyond --guard-us).
#   slot_level_sim.py --piece-dir prof/tenant/slot_level/piece --seq prof/tenant/slot_level/seq/av8.seq.csv [--rate 0] [--gran 8|2]
#   slot_level_sim.py --piece-dir ... --pattern 80,80,80,80,100,100,100,100,80,80
#   slot_level_sim.py --piece-dir ... --mix l2_10x:4,av10x:6,l3_10x:3,av10x:5 --seq-dir prof/tenant/slot_level/seq
import argparse, bisect, csv, json, math, os, random
SLOT = 500.0   # us

def load_seq(path, a):
    r = list(csv.DictReader(open(path)))
    ids = [int(x["sfn"]) * 20 + int(x["slot"]) for x in r]
    # longest run of consecutive slots, cut to whole frames (the sequence is replayed cyclically)
    best = (0, 0); s = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or (ids[i] - ids[i - 1]) % 20480 != 1:
            if i - s > best[1] - best[0]: best = (s, i)
            s = i
    s, e = best
    while s < e and int(r[s]["slot"]) != 0: s += 1
    e = s + ((e - s) // 20) * 20; r = r[s:e]; n = len(r)
    ula = [int(x["ul_any"]) for x in r]; dla = [int(x["dl_any"]) for x in r]; uls = [int(x["ul_sm"]) for x in r]; dls = [int(x["dl_sm"]) for x in r]
    free = []
    for w in range(n):
        ul = 0; dl = 0
        for b in range(1, a.ul_back + 1):
            k = (w - b) % n
            if ula[k]: ul = max(ul, a.pusch_front if (b == 1 or not uls[k]) else uls[k])
        for f in range(1, a.dl_fwd + 1):
            k = (w + f) % n
            if dla[k]: dl = max(dl, dls[k] if dls[k] else a.pdsch_front)
        free.append(((a.nsm - ul) if ul else a.nsm) - (dl if dl else a.pdsch_front) - a.order)
    return free

class Model:
    def __init__(self, path, fine=None):
        d = json.load(open(path)); self.np = d["pieces"]; self.t = {int(c): [v * 1000.0 for v in d["piece_ms"][c]] for c in d["piece_ms"]}
        if fine and os.path.exists(fine):
            f = json.load(open(fine))
            for c in f["piece_ms"]:
                if int(c) not in self.t: self.t[int(c)] = [v * 1000.0 for v in f["piece_ms"][c]]
        self.meas = sorted(self.t)
    def pieces(self, c):
        if c in self.t: return self.t[c]
        i = bisect.bisect_left(self.meas, c); lo = self.meas[max(0, i - 1)]; hi = self.meas[min(len(self.meas) - 1, i)]
        if lo == hi: return self.t[lo]
        # between two measured classes: linear in 1/SM per piece (time ~ a + b/SM)
        x = (1.0 / c - 1.0 / hi) / (1.0 / lo - 1.0 / hi); self.t[c] = [h + (l - h) * x for l, h in zip(self.t[lo], self.t[hi])]
        return self.t[c]
    def total(self, c): return sum(self.pieces(c))

def classes_for(a, fine_ok):
    if a.gran == 8: return list(range(a.cls_min, a.cls_max + 1, 8))
    # 2-SM steps (split flag 1) run TensorRT only from --fine-min SMs up (smgran2.py: 26n from 44, 26s/m/l/x from 76); 8-SM steps below
    return sorted(set(list(range(a.cls_min, a.fine_min, 8)) + list(range(max(a.fine_min, a.cls_min), a.cls_max + 1, 2))))

def run_req(t, m, cl, free, a):
    n = len(free); est = max(m.total(c) for c in cl)
    for _ in range(4000):
        w = int(t // SLOT); end = (w + 1) * SLOT; span = a.guard_req_us + est - (end - t); J = min(int(span // SLOT) + 1 if span > 0 else 0, a.horizon_max)
        fmin = min(free[(w + j) % n] for j in range(J + 1)); i = bisect.bisect_right(cl, fmin)
        if i: return t, t + m.total(cl[i - 1]), cl[i - 1], 0
        t = end
    return t, float("inf"), 0, 0

def run_piece(t, m, cl, free, a):
    n = len(free); t0 = t; prev = None; sw = 0; smt = 0.0; k = 0
    while k < m.np:
        best = None
        for ahead in range(a.wait_ahead + 1):
            w = int(t // SLOT); st = t if ahead == 0 else (w + ahead) * SLOT + a.resume_us; ws = int(st // SLOT)
            for c in reversed(cl):
                if c > free[ws % n]: continue
                d = m.pieces(c)[k]; we = int((st + d + a.guard_us) // SLOT)
                if we == ws or all(c <= free[j % n] for j in range(ws + 1, we + 1)):
                    fin = st + d + (a.switch_us if (prev is not None and prev != c) else 0.0)
                    if best is None or fin < best[0] - 1e-9: best = (fin, c, d)
                    break
        if best is None: t = (int(t // SLOT) + a.wait_ahead + 1) * SLOT; prev = None; continue
        if prev is not None and best[1] != prev: sw += 1
        t = best[0]; prev = best[1]; smt += best[1] * best[2]; k += 1
    return t0, t, smt / max(1e-9, (t - t0)), sw

def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p / 100.0 * len(v)))] if v else float("nan")

def simulate(fn, m, cl, free, a, fixed_us=None):
    rnd = random.Random(1); n = len(free); T = n * SLOT; inf = []; resp = []; sm = []; sws = []
    period = 1e6 / a.rate if a.rate > 0 else 0.0; arr = rnd.uniform(0, 10000.0); free_at = 0.0
    for i in range(a.requests):
        st = max(arr, free_at)
        if fixed_us is not None: s, e, c, sw = st, st + fixed_us, a.mig_sm, 0
        else: s, e, c, sw = fn(st, m, cl, free, a)
        if e == float("inf"): return None
        inf.append((e - st) / 1000.0); resp.append((e - arr) / 1000.0); sm.append(c); sws.append(sw); free_at = e
        arr = (arr + period + rnd.uniform(-0.02, 0.02) * period) if period else e + rnd.uniform(0, 300.0)   # closed loop: back to back (+ up to 0.3 ms so the start phase drifts)
    return {"mean": sum(inf) / len(inf), "p50": pct(inf, 50), "p99": pct(inf, 99), "resp_mean": sum(resp) / len(resp), "resp_p99": pct(resp, 99), "sm": sum(sm) / len(sm), "sw": sum(sws) / len(sws)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece-dir", required=True); ap.add_argument("--models", default="n,s,m,l,x"); ap.add_argument("--seq"); ap.add_argument("--pattern"); ap.add_argument("--mix"); ap.add_argument("--seq-dir", default=".")
    ap.add_argument("--rate", type=float, default=0.0, help="req/s, periodic (0 = closed loop, back to back)"); ap.add_argument("--requests", type=int, default=1500)
    ap.add_argument("--gran", type=int, default=8); ap.add_argument("--fine-min", type=int, default=76); ap.add_argument("--cls-min", type=int, default=8); ap.add_argument("--cls-max", type=int, default=104)
    ap.add_argument("--req-classes", default="", help="class list of the request-level policy (default: the same ladder)")
    ap.add_argument("--nsm", type=int, default=132); ap.add_argument("--order", type=int, default=8); ap.add_argument("--ul-back", type=int, default=4); ap.add_argument("--dl-fwd", type=int, default=2)
    ap.add_argument("--pusch-front", type=int, default=20); ap.add_argument("--pdsch-front", type=int, default=16)
    ap.add_argument("--guard-req-us", type=float, default=60.0); ap.add_argument("--horizon-max", type=int, default=200)
    ap.add_argument("--guard-us", type=float, default=30.0); ap.add_argument("--switch-us", type=float, default=5.0); ap.add_argument("--resume-us", type=float, default=20.0); ap.add_argument("--wait-ahead", type=int, default=2)
    ap.add_argument("--mig-sm", type=int, default=60); ap.add_argument("--mig-ms", default="x=17.37", help="measured Baseline times (model=ms,...); others: the 132-SM engine at --mig-sm x the 26x ratio")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.pattern: p = [int(v) for v in a.pattern.split(",")]; free = p * (20 // len(p)) if 20 % len(p) == 0 else p; label = "pattern " + a.pattern
    elif a.mix:
        free = []; label = "mix " + a.mix
        for part in a.mix.split(","):
            nm, sec = part.split(":"); f = load_seq(os.path.join(a.seq_dir, nm + ".seq.csv"), a); k = int(float(sec) * 2000) // 20 * 20
            free += (f * (k // len(f) + 1))[:k]
    else: free = load_seq(a.seq, a); label = os.path.basename(a.seq)
    n = len(free); hist = {}
    for w, f in enumerate(free): hist.setdefault(w % 10, {}); hist[w % 10][f] = hist[w % 10].get(f, 0) + 1
    print("== %s: %d slots (%.1f s), free SMs mean %.1f min %d max %d | rate %s | piece ladder %d-SM steps %d..%d, wait-ahead %d" % (label, n, n / 2000.0, sum(free) / n, min(free), max(free), ("%g req/s" % a.rate) if a.rate else "closed loop", a.gran, a.cls_min, a.cls_max, a.wait_ahead))
    print("   free SMs per TDD position: " + "  ".join("%d: %s" % (p, "/".join("%d(%.0f%%)" % (k, 100.0 * v / sum(hist[p].values())) for k, v in sorted(hist[p].items(), key=lambda x: -x[1])[:3])) for p in sorted(hist)))
    mig = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.mig_ms.split(",") if kv}; out = {"label": label, "rate": a.rate, "models": {}}
    mods = {mm: Model(os.path.join(a.piece_dir, "fhd26%s.json" % mm), os.path.join(a.piece_dir, "fhd26%s_fine.json" % mm) if a.gran == 2 else None) for mm in a.models.split(",")}
    ratio = (mig["x"] * 1000.0 / mods["x"].total(a.mig_sm)) if ("x" in mig and "x" in mods) else 1.0
    print("   %-4s | %-22s | %-30s | %-38s | %-9s | gain of piece vs MIG / vs request" % ("", "Baseline MIG %d SM" % a.mig_sm, "request level (mean/p99, SM)", "slot level (mean/p99, SM, switches)", "132 SM"))
    for mm, m in mods.items():
        cl = classes_for(a, True); rcl = [int(v) for v in a.req_classes.split(",")] if a.req_classes else list(range(a.cls_min, a.cls_max + 1, 8))
        tm = mig[mm] * 1000.0 if mm in mig else m.total(a.mig_sm) * ratio
        b = simulate(None, m, cl, free, a, fixed_us=tm); r = simulate(run_req, m, rcl, free, a); p = simulate(run_piece, m, cl, free, a)
        rs = ("%6.2f /%6.2f ms, %5.1f SM" % (r["mean"], r["p99"], r["sm"])) if r else "never fits (starves)"
        g1 = 100.0 * (p["mean"] / b["mean"] - 1.0); g2 = (100.0 * (p["mean"] / r["mean"] - 1.0)) if r else -100.0
        print("   26%-2s | %6.2f ms%s resp %6.2f | %-30s | %6.2f /%6.2f ms, %5.1f SM, %5.1f sw | %6.2f ms | %+6.1f %% / %+6.1f %%%s" % (mm, b["mean"], " " if mm in mig else "*", b["resp_mean"], rs, p["mean"], p["p99"], p["sm"], p["sw"], m.total(132) / 1000.0, g1, g2,
              ("  | resp mean MIG %.1f req %s piece %.1f" % (b["resp_mean"], ("%.1f" % r["resp_mean"]) if r else "-", p["resp_mean"])) if a.rate else ""))
        out["models"][mm] = {"mig": b, "req": r, "piece": p, "mig_measured": mm in mig, "t132_ms": m.total(132) / 1000.0}
    print("   (* = estimated: the 132-SM engine at %d SMs x %.3f, the measured 26x MIG ratio; mean SM of the slot level = time-weighted)" % (a.mig_sm, ratio))
    if a.json: json.dump(out, open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
