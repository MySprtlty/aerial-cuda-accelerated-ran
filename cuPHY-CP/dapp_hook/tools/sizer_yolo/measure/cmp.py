#!/usr/bin/env python3
# cmp.py <label=ovl dump> ... : like-for-like summary of several runs (one line per run + per-minute trend), from the ovl dumps
import sys, collections
def q(g, p): g = sorted(g); return g[min(len(g) - 1, int(p * len(g)))] if g else 0
print("run | minutes | UL slots | YOLO in flight (separate SMs) | of them > 1592.5 us | > 1700 | > 2100 | completion p99 / max us | out-of-order pairs | per 10 min | mean plan SM (YOLO slots) | no-YOLO > 1592.5")
trend = {}
for a in sys.argv[1:]:
    lab, fn = a.split('=', 1)
    rows = sorted((int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]), float(f[5]), int(f[7])) for f in (l.split('\t') for l in open(fn)))
    dur = (rows[-1][1] - rows[0][1]) / 60e9
    y = [r for r in rows if r[4] == 1]; n0 = [r for r in rows if r[4] == 0]
    inv = 0; permin = collections.defaultdict(lambda: [0, 0, 0])
    for i in range(len(rows) - 1):
        r, nx = rows[i], rows[i + 1]
        m = int((r[1] - rows[0][1]) / 60e9)
        if r[4] == 1: permin[m][0] += 1; permin[m][1] += r[5] > 1592.5
        if nx[0] != r[0] + 1 or r[4] < 0 or nx[4] < 0: continue
        if r[6] > nx[6]: inv += 1; permin[m][2] += 1
    c = [r[5] for r in y]
    print(f"{lab} | {dur:.2f} | {len(rows)} | {len(y)} ({100*len(y)/len(rows):.1f}%) | {sum(1 for x in c if x > 1592.5)} ({100*sum(1 for x in c if x > 1592.5)/max(1,len(c)):.2f}%) | {100*sum(1 for x in c if x > 1700)/max(1,len(c)):.2f}% | {sum(1 for x in c if x > 2100)} | {q(c,.99):.0f} / {max(c):.0f} | {inv} | {inv*10/dur:.0f} | {sum(r[3] for r in y)/max(1,len(y)):.1f} | {sum(1 for r in n0 if r[5] > 1592.5)}")
    trend[lab] = permin
print("per minute: over-target share of YOLO-in-flight slots % / out-of-order pairs")
for lab, pm in trend.items():
    print(f"   {lab}: " + "  ".join(f"{100*v[1]/max(1,v[0]):.2f}/{v[2]}" for k, v in sorted(pm.items())))
