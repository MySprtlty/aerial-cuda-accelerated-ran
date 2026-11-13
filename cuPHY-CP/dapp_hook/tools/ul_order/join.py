#!/usr/bin/env python3
# join.py <rows.tsv> <dapp_tenant_slots.csv> : PUSCH slot time vs the sizer's per-slot SM plan and the YOLO state
import sys, csv, collections
rows = {}
for l in open(sys.argv[1]):
    f = l.rstrip('\n').split('\t'); rows[int(f[0])] = (int(f[1]), int(f[2]), int(f[5]), int(f[7]), float(f[8]), int(f[9]), int(f[11]))
plan = {}
with open(sys.argv[2], newline='', errors='replace') as fh:
    r = csv.reader(fh); h = next(r); ix = {n: i for i, n in enumerate(h)}
    for f in r:
        try:
            if int(f[ix['pos']]) < 7: continue
            s = int(f[ix['sfn']]) * 20 + int(f[ix['slot']]); t = int(f[ix['t_ns']])
            a = round((t / 500000 - s) / 20480) * 20480 + s
            plan[a] = (int(f[ix['plan_ul_sm']]), int(f[ix['run_cls']]), int(f[ix['gap_P']]), int(f[ix['gap_N']]), int(f[ix['class']]), f[ix['model']])
        except Exception: pass
def q(g, p): g = sorted(g); return g[min(len(g) - 1, int(p * len(g)))] if g else 0
T0, T1 = int(sys.argv[3]) if len(sys.argv) > 3 else 0, int(sys.argv[4]) if len(sys.argv) > 4 else 1 << 62
tab = collections.defaultdict(list); miss = 0
for a, v in rows.items():
    if not (T0 <= v[3] < T1): continue
    p = plan.get(a)
    if p is None: miss += 1; continue
    if v[2] == 1: tab[(p[0], 'YOLO running' if p[1] > 0 else 'YOLO idle')].append((v[5], v[4]))
print(f"UL slots {len(rows)}, without a sizer row {miss}")
print("PUSCH SM plan | YOLO          | slots   | slot time us p50 / p99 / max | GPU run us p50 / p99")
for k in sorted(tab):
    g = [x[0] for x in tab[k]]; r_ = [x[1] for x in tab[k]]
    print(f"   {k[0]:4d}       | {k[1]:13s} | {len(g):7d} | {q(g,.5):5d} / {q(g,.99):5d} / {max(g):5d}        | {q(r_,.5):5.0f} / {q(r_,.99):5.0f}")
# out-of-order pairs: neighbouring UL slots (same UL burst), later one reports first
inv = collections.Counter(); allp = collections.Counter(); n = 0
ks = sorted(rows)
for i, a in enumerate(ks):
    for j in (i + 1, i + 2):
        if j >= len(ks) or ks[j] - a > 2: break
        b = ks[j]; va, vb = rows[a], rows[b]
        if not (T0 <= va[3] < T1) or va[6] == 0 or vb[6] == 0: continue
        pa, pb = plan.get(a), plan.get(b)
        if pa is None or pb is None: continue
        key = (f"earlier slot {'failed' if va[2] != 1 else 'ok'} / later slot {'failed' if vb[2] != 1 else 'ok'}", 'YOLO running' if (pa[1] > 0 or pb[1] > 0) else 'YOLO idle', f"plan {pa[0]}->{pb[0]}")
        allp[key] += 1
        if vb[6] < va[6]: inv[key] += 1; n += 1
print(f"out-of-order pairs {n}")
for k, c in sorted(inv.items(), key=lambda x: -x[1])[:25]:
    print(f"  {c:6d} of {allp[k]:7d} such pairs ({100*c/allp[k]:5.1f} %)  {k[0]} | {k[1]} | {k[2]}")
