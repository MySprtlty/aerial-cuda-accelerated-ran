#!/usr/bin/env python3
# inv2.py <ovl dump> : ok/ok neighbouring UL slots finishing out of order - by the earlier slot's plan and by what the pair looked like
import sys, collections
rows = sorted((int(f[0]), int(f[2]), int(f[3]), int(f[4]), float(f[5]), int(f[7])) for f in (l.split('\t') for l in open(sys.argv[1])))
byplan = collections.Counter(); allplan = collections.Counter(); pos = collections.Counter(); st2 = collections.Counter(); late = []; succ = []
ymax = collections.defaultdict(list)
for i in range(len(rows) - 1):
    r, nx = rows[i], rows[i + 1]
    if r[3] >= 1: ymax[r[2]].append(r[4])
    if nx[0] != r[0] + 1 or r[3] < 0 or nx[3] < 0: continue
    if r[3] >= 1: allplan[r[2]] += 1
    if r[5] > nx[5]:
        byplan[r[2]] += 1; pos[f"{r[1] % 10}->{nx[1] % 10}"] += 1; late.append(r[4]); succ.append(nx[4])
        st2['later slot: ' + ('no YOLO' if nx[3] == 0 else 'YOLO in flight')] += 1
def q(g, p): g = sorted(g); return g[min(len(g) - 1, int(p * len(g)))] if g else 0
print("out-of-order pairs by the earlier slot's plan (share of YOLO-in-flight pairs with that plan):", "  ".join(f"{k}: {v} ({100*v/max(1,allplan[k]):.2f}%)" for k, v in sorted(byplan.items())))
print("by slot position:", dict(pos), "|", dict(st2))
print(f"earlier slot completion after T0: p50 {q(late,.5):.0f} min {min(late):.0f} max {max(late):.0f} us | its successor: p50 {q(succ,.5):.0f} max {max(succ):.0f} us")
print("YOLO in flight, completion max / share > 1700 us by plan:", "  ".join(f"{k}: {max(v):.0f} / {100*sum(1 for x in v if x > 1700)/len(v):.2f}%" for k, v in sorted(ymax.items())))
