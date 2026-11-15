#!/usr/bin/env python3
# ovl.py <rows.tsv> <dapp_tenant_slots.csv> : PUSCH completion after T0 vs (a) the sizer's SM plan and (b) what the YOLO inference in flight
# during the PUSCH GPU work really occupied. union layout: PUSCH partition S = SM positions [132-S, 131]; a YOLO class c launched in gap (P, N)
# = positions [N, N+c-1]  ->  shared SMs = max(0, N + c + S - 132).  completion = [Max GPU Time] END - T0 (T0 on the 0.5-ms slot grid).
import sys, csv, bisect, collections
from array import array
T = array('q'); TOP = array('h'); LT = array('q'); plan = {}
cur_top = -1; cur_cls = 0; cur_lt = 0; prev_t = 0
with open(sys.argv[2], newline='', errors='replace') as fh:
    r = csv.reader(fh); h = next(r); ix = {n: i for i, n in enumerate(h)}
    iS, iL, iT, iP, iN, iPa, iC, iSe, iR, iU = (ix[k] for k in ('sfn', 'slot', 't_ns', 'P', 'N', 'part', 'class', 'segments', 'run_cls', 'plan_ul_sm'))
    for f in r:
        try:
            t = int(f[iT]); s = int(f[iS]) * 20 + int(f[iL]); rc = int(f[iR]); u = int(f[iU])
            if int(f[iPa]) >= 0 and int(f[iSe]) > 0: cur_cls = int(f[iC]); cur_top = int(f[iN]) + cur_cls - 1; cur_lt = prev_t if prev_t else t
        except Exception: continue
        if u >= 0: plan[round((t / 500000 - s) / 20480) * 20480 + s] = u
        T.append(t); TOP.append(cur_top if (rc > 0 and rc == cur_cls) else -1); LT.append(cur_lt); prev_t = t
def q(g, p): g = sorted(g); return g[min(len(g) - 1, int(p * len(g)))] if g else 0
tab = collections.defaultdict(list); n = fail = noplan = 0; t_first = None
import os
dump = open(os.environ['OVL_DUMP'], 'w') if os.environ.get('OVL_DUMP') else None
cause = collections.Counter(); ovsm = collections.Counter()
for l in open(sys.argv[1]):
    f = l.rstrip('\n').split('\t'); a = int(f[0]); slot = int(f[2]); st = int(f[7]); grun = float(f[8]); end = int(f[10])
    if t_first is None: t_first = st
    if st - t_first < 1_000_000_000: continue
    if int(f[5]) != 1 or int(f[6]) != 1:
        fail += 1
        if dump: dump.write(f"{a}\t{st}\t{slot}\t{plan.get(a, -1)}\t-1\t0\t0\t{end}\n")
        continue
    S = plan.get(a)
    if S is None or S <= 0: noplan += 1; continue
    d = (st + 497000 - slot * 500000) % 10_000_000
    if d > 5_000_000: d -= 10_000_000
    t0 = st + 497000 - d; comp = (end - t0) / 1000.0
    ws = end - int(grun * 1000) - 50000
    i0 = bisect.bisect_right(T, ws); i1 = min(len(T) - 1, bisect.bisect_left(T, end))
    top = -1; lt = 0
    for i in range(i0, i1 + 1):
        if TOP[i] > top: top = TOP[i]; lt = LT[i]
    n += 1
    if top < 0: state = '1 no YOLO in flight'
    else:
        ov = top - (132 - S) + 1
        if ov <= 0: state = '2 YOLO in flight, separate SMs'
        else:
            state = '3 YOLO in flight, SHARED SMs'; ovsm[ov] += 1
            lead = (t0 - lt) / 1000.0     # how long before this slot's T0 the inference was launched
            pv = plan.get(a - 20)
            if lead > 1500: cause['launched before this slot was planned; same slot one frame earlier had ' + ('a smaller plan' if (pv is not None and pv < S) else ('the same or a larger plan' if pv is not None else 'no plan'))] += 1
            elif lead < 1000: cause['launched after this slot was planned'] += 1
            else: cause['launched while this slot was being planned'] += 1
    tab[(S, state)].append((comp, grun)); tab[(0, state)].append((comp, grun))
    if dump: dump.write(f"{a}\t{st}\t{slot}\t{S}\t{int(state[0]) - 1}\t{comp:.1f}\t{grun:.1f}\t{end}\n")
print(f"PUSCH slots {n} (failed {fail}, no plan row {noplan})")
print("plan SM | YOLO during the PUSCH GPU work | slots (share) | completion after T0 us p50 / p99 / max | > 1592.5 us | > 2100 us | GPU run us p50 / p99")
for k in sorted(tab):
    g = [x[0] for x in tab[k]]; r_ = [x[1] for x in tab[k]]; tot = sum(len(tab[(k[0], s)]) for s in ('1 no YOLO in flight', '2 YOLO in flight, separate SMs', '3 YOLO in flight, SHARED SMs'))
    print(f"  {('all' if k[0] == 0 else k[0])!s:>5} | {k[1][2:]:30s} | {len(g):7d} ({100*len(g)/tot:5.1f}%) | {q(g,.5):6.0f} / {q(g,.99):6.0f} / {max(g):6.0f} | {100*sum(1 for x in g if x > 1592.5)/len(g):6.2f}% | {100*sum(1 for x in g if x > 2100)/len(g):6.3f}% | {q(r_,.5):4.0f} / {q(r_,.99):4.0f}")
tot3 = sum(cause.values())
if tot3:
    print("shared-SM slots by how the inference came to sit on PUSCH SMs:")
    for k, c in cause.most_common(): print(f"  {c:7d} ({100*c/tot3:5.1f}%)  {k}")
    print("shared SMs: " + "  ".join(f"{k}: {100*v/tot3:.1f}%" for k, v in sorted(ovsm.items())))
