#!/usr/bin/env python3
# cell.py <ovl dump> <work5c.json> : per (plan vector, UL slot) cell - model prediction (start_us + work_us at the plan) next to the measured
# completion without YOLO and with a YOLO inference in flight on separate SMs. "need(q)" = measured YOLO completion quantile q - model
# prediction = what a YOLO term would have to add (model coordinates) for the prediction to cover that quantile.
import sys, json, collections
sys.path.insert(0, '.'); import work as W
def q(g, p): g = sorted(g); return g[min(len(g) - 1, int(p * len(g)))] if g else 0
WK = {int(k): v for k, v in json.load(open(sys.argv[2])).items()}
LV = {(20,)*6: 0, (20,20,20,20,20,28): 1, (20,20,28,20,20,28): 1, (28,36,44,28,36,44): 2, (28,28,36,20,28,36): 3}   # vector -> frame of the pattern (levels 1,2,4,3)
R = {}
for l in open(sys.argv[1]):
    f = l.split('\t'); a = int(f[0]); R[a] = (int(f[2]), int(f[3]), int(f[4]), float(f[5]), float(f[6]), int(f[7]))
frames = collections.defaultdict(dict)
for a, x in R.items(): frames[a - x[0]][x[0]] = x
UL = (7, 8, 9, 17, 18, 19)
by = collections.defaultdict(lambda: collections.defaultdict(list)); nv = collections.Counter()
for fb, d in frames.items():
    if any(s not in d for s in UL): continue
    vec = tuple(d[s][1] for s in UL); nv[vec] += 1
    for s in UL:
        if d[s][2] in (0, 1): by[vec][(s, d[s][2])].append(d[s])
print("vector | slot plan | model work_us, t_us, margin | none: n, start p50/p99, run p50/p99, compl p50/p99/p99.9/max | YOLO: n, start p50/p99, run p50/p99, compl p50/p90/p99/p99.9/max, >1592.5 | need(p90) need(p99) need(p99.9) | hidden margin none p99")
out = []
for vec in LV:
    fr = LV[vec]; print(vec, "frame", fr, "share %.1f%%" % (100 * nv[vec] / sum(nv.values())))
    for i, s in enumerate(UL):
        w = WK[fr * 20 + s]; sm = vec[i]; wk = W.work_us(w, sm); t = W.M['start'][s] + wk
        g0 = by[vec][(s, 0)]; g1 = by[vec][(s, 1)]
        if len(g0) < 20 or len(g1) < 20: print("   slot %2d plan %3d few samples" % (s, sm)); continue
        c0 = [x[3] for x in g0]; r0 = [x[4] for x in g0]; s0 = [x[3] - x[4] for x in g0]
        c1 = [x[3] for x in g1]; r1 = [x[4] for x in g1]; s1 = [x[3] - x[4] for x in g1]
        ov = 100 * sum(1 for v in c1 if v > 1592.5) / len(c1)
        print("   slot %2d plan %3d | %3.0f %4.0f %3.0f | %5d  %4.0f/%4.0f  %3.0f/%3.0f  %4.0f/%4.0f/%4.0f/%4.0f | %5d  %4.0f/%4.0f  %3.0f/%4.0f  %4.0f/%4.0f/%4.0f/%4.0f/%4.0f  %5.2f%% | %+4.0f %+4.0f %+4.0f | %+4.0f" % (
            s, sm, wk, t, 1592.5 - t, len(g0), q(s0,.5), q(s0,.99), q(r0,.5), q(r0,.99), q(c0,.5), q(c0,.99), q(c0,.999), max(c0),
            len(g1), q(s1,.5), q(s1,.99), q(r1,.5), q(r1,.99), q(c1,.5), q(c1,.9), q(c1,.99), q(c1,.999), max(c1), ov,
            q(c1,.9) - t, q(c1,.99) - t, q(c1,.999) - t, t - q(c0,.99)))
        out.append((vec, s, sm, wk, t, q(c1,.99) - t, q(c1,.999) - t, ov, q(r0,.5), q(r1,.5), q(r1,.99), q(r0,.99)))
json.dump(out, open(sys.argv[3], 'w'))
