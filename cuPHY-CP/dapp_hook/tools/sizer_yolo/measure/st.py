#!/usr/bin/env python3
# st.py <ovl dump> <dapp_tenant_slots.csv> : (A) PUSCH completion tail by sizer plan x YOLO class in flight, by model, by slot position
# (B) what the sizer could have known at plan time (rules: request in flight / queued at plan time; activity hold of H ms)
# (C) periodicity of the plan sequence (to pair a slot with the same slot of other frames)
import sys, csv, bisect, collections
from array import array
T = array('q'); RC = array('h'); Q = array('h'); LA = array('b'); MD = []
with open(sys.argv[2], newline='', errors='replace') as fh:
    r = csv.reader(fh); h = next(r); ix = {n: i for i, n in enumerate(h)}
    for f in r:
        try: t = int(f[ix['t_ns']]); rc = int(f[ix['run_cls']]); qd = int(f[ix['queue']]); la = 1 if (int(f[ix['part']]) >= 0 and int(f[ix['segments']]) > 0) else 0
        except Exception: continue
        T.append(t); RC.append(rc); Q.append(qd); LA.append(la); MD.append(f[ix['model']] if len(f) > ix['model'] else '')
n = len(T)
# last time a request was visible (in flight or queued) at or before each row
LASTACT = array('q'); la_t = 0
for i in range(n):
    if RC[i] > 0 or Q[i] > 0: la_t = T[i]
    LASTACT.append(la_t)
def q(g, p): g = sorted(g); return g[min(len(g) - 1, int(p * len(g)))] if g else 0
S_ = []  # per PUSCH slot
for l in open(sys.argv[1]):
    f = l.split('\t'); st = int(f[4])
    if st < 0: continue
    a = int(f[0]); slot = int(f[2]); S = int(f[3]); comp = float(f[5]); grun = float(f[6]); end = int(f[7])
    t0 = end - int(comp * 1000); ws = end - int(grun * 1000)
    i0 = bisect.bisect_right(T, ws); i1 = min(n - 1, bisect.bisect_left(T, end))
    rc = 0; md = ''
    for i in range(i0, i1 + 1):
        if RC[i] > rc: rc = RC[i]; md = MD[i]
    S_.append((a, slot, S, st, comp, grun, end, t0, rc, md))
def line(name, g):
    c = [x[4] for x in g]; r_ = [x[5] for x in g]
    return f"  {name:26s} | {len(g):7d} | {q(c,.5):5.0f} / {q(c,.99):5.0f} / {max(c):5.0f} | {100*sum(1 for x in c if x > 1592.5)/len(c):5.2f}% | {q(r_,.5):4.0f} / {q(r_,.99):4.0f} / {max(r_):5.0f}"
hdr = "| slots | completion us p50 / p99 / max | > 1592.5 | GPU run us p50 / p99 / max"
Y = [x for x in S_ if x[3] == 1]
print(f"PUSCH slots {len(S_)}, YOLO inference in flight on separate SMs {len(Y)}")
print("(A1) YOLO in flight, by sizer plan x YOLO class " + hdr)
g = collections.defaultdict(list)
for x in Y: g[(x[2], x[8])].append(x)
for k in sorted(g):
    if len(g[k]) >= 100: print(line(f"plan {k[0]:3d} yolo {k[1]:3d}", g[k]))
print("(A2) YOLO in flight, by YOLO class " + hdr)
g = collections.defaultdict(list)
for x in Y: g[x[8]].append(x)
for k in sorted(g): print(line(f"yolo {k:3d}", g[k]))
print("(A3) YOLO in flight, by model " + hdr)
g = collections.defaultdict(list)
for x in Y: g[x[9]].append(x)
for k in sorted(g): print(line(f"model {k}", g[k]))
print("(A4) by UL slot (frame slot index), no YOLO vs YOLO " + hdr)
g = collections.defaultdict(list)
for x in S_:
    if x[3] in (0, 1): g[(x[1], x[3])].append(x)
for k in sorted(g): print(line(f"slot {k[0]:2d} {'YOLO' if k[1] else 'none'}", g[k]))
# (B) knowability
S_.sort(); oo_first = set()
for x, y in zip(S_, S_[1:]):
    if y[0] - x[0] == 1 and x[6] > y[6]: oo_first.add(x[0])
print(f"(B) what was visible when the slot was planned (YOLO-in-flight slots {len(Y)}, of them over target {sum(1 for x in Y if x[4] > 1592.5)}, earlier slot of an out-of-order pair {sum(1 for x in Y if x[0] in oo_first)})")
for lead_us in (1400, 1000):
    for name, bound in (("lower bound (state at the last CSV row before the plan)", 0), ("upper bound (state at the first CSV row after the plan)", 1)):
        print(f"  plan time = T0 - {lead_us} us, {name}")
        for H in (0, 5, 10, 20, 34, 50, 100):
            cov = cov_over = cov_oo = on = 0; tot_over = tot_oo = 0
            for x in S_:
                tp = x[7] - lead_us * 1000
                j = bisect.bisect_right(T, tp) - 1 + bound
                if j < 0 or j >= n: continue
                vis = (tp - LASTACT[j] <= H * 1_000_000) if H else (RC[j] > 0 or Q[j] > 0)
                if H and bound: vis = vis or RC[j] > 0 or Q[j] > 0
                if vis: on += 1
                if x[3] == 1:
                    ov = x[4] > 1592.5; isoo = x[0] in oo_first
                    tot_over += ov; tot_oo += isoo
                    if vis: cov += 1; cov_over += ov; cov_oo += isoo
            print(f"    rule {'request in flight or queued at plan time' if H == 0 else f'request seen in the last {H:3d} ms':42s}: term ON in {100*on/len(S_):5.1f}% of UL slots | covers {100*cov/len(Y):5.1f}% of YOLO-in-flight slots, {cov_over}/{tot_over} over-target, {cov_oo}/{tot_oo} out-of-order")
# (C) periodicity of the plan sequence
P = {x[0]: x[2] for x in S_}
print("(C) share of UL slots whose plan equals the plan `lag` slots later: " + "  ".join(f"{lag}: {100*sum(1 for a, s in P.items() if P.get(a + lag) == s)/max(1, sum(1 for a in P if (a + lag) in P)):.0f}%" for lag in (20, 40, 80, 160, 320, 640, 1280, 2000, 4000)))
