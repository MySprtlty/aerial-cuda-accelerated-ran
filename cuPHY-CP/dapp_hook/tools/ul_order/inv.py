#!/usr/bin/env python3
# inv.py <rows.tsv> [gnb_unexpected.log] : UL result order between neighbouring UL slots (PUSCH callback start = the moment CRC/RX_DATA go up to L2)
import sys, bisect, datetime, collections
rows = []
for l in open(sys.argv[1]):
    f = l.rstrip('\n').split('\t')
    rows.append((int(f[0]), int(f[1]), int(f[2]), int(f[5]), int(f[7]), float(f[8]), int(f[9]), int(f[11]), int(f[12]), f[4]))
rows.sort()
# abs sfn slot setup_st setup_start gpu_run maxgpu cb_start cb_end thread
def minute(ns): return datetime.datetime.utcfromtimestamp(ns / 1e9).strftime('%H:%M')
def hms(ns): return datetime.datetime.utcfromtimestamp(ns / 1e9).strftime('%H:%M:%S.%f')
per = collections.defaultdict(lambda: dict(n=0, fail=0, inv=0, tight=0, mg=[]))
inv = []
for i, a in enumerate(rows):
    m = per[minute(a[4])]; m['n'] += 1
    if a[3] != 1: m['fail'] += 1; continue
    m['mg'].append(a[6])
    for j in (i + 1, i + 2):
        if j >= len(rows): break
        b = rows[j]
        if b[0] - a[0] > 2: break
        if b[3] != 1 or b[7] == 0 or a[7] == 0: continue
        d = b[7] - a[7]
        if d < 0: m['inv'] += 1; inv.append((b[7], a, b, d))
        elif d < 70000 and b[0] - a[0] == 1: m['tight'] += 1
print("minute  UL slots  setup-failed  out-of-order pairs  near (<70us)  slot time us p50 / p99 / max")
for k in sorted(per):
    m = per[k]; g = sorted(m['mg']); q = lambda p: g[min(len(g) - 1, int(p * len(g)))] if g else 0
    print(f"{k}   {m['n']:7d}   {m['fail']:7d}   {m['inv']:9d}   {m['tight']:9d}      {q(.5):5d} / {q(.99):5d} / {g[-1] if g else 0:6d}")
print("out-of-order pairs total", len(inv))
inv.sort()
for t, a, b, d in inv[:12]:
    print(f"  {hms(t)}  {a[1]}.{a[2]} ({a[9]}, {a[6]} us, GPU run {a[5]:.0f}) reported {-d/1000:.0f} us AFTER {b[1]}.{b[2]} ({b[9]}, {b[6]} us, GPU run {b[5]:.0f})")
if len(sys.argv) > 2:
    day0 = (rows[0][4] // 86400_000_000_000) * 86400_000_000_000
    w = []
    for l in open(sys.argv[2], errors='replace'):
        t = l[1:16]
        try: h, mi, s = t.split(':'); w.append(day0 + int((int(h) * 3600 + int(mi) * 60 + float(s)) * 1e9))
        except Exception: pass
    w.sort()
    # burst starts: a warning with no warning in the 20 ms before it
    starts = [x for i, x in enumerate(w) if i == 0 or x - w[i - 1] > 20_000_000]
    it = [x[0] for x in inv]
    hit = sum(1 for s in starts if (lambda k: k > 0 and s - it[k - 1] < 1_000_000)(bisect.bisect_right(it, s + 100_000)))
    print(f"gNB warnings {len(w)}, quiet-then-warning starts {len(starts)}; starts with an out-of-order pair in the 1 ms before: {hit}")
    foll = sum(1 for x in it if (lambda k: k < len(w) and w[k] - x < 1_000_000)(bisect.bisect_left(w, x - 100_000)))
    print(f"out-of-order pairs followed by a gNB warning within 1 ms: {foll} of {len(it)}")
    # the first warnings of the run and of each minute-long quiet period
    big = [x for i, x in enumerate(w) if i == 0 or x - w[i - 1] > 5_000_000_000]
    for s in big[:12]:
        k = bisect.bisect_right(it, s + 100_000)
        near = f"out-of-order pair {(s - it[k-1])/1000:.0f} us before" if k > 0 and s - it[k - 1] < 5_000_000 else "no out-of-order pair in the 5 ms before"
        print(f"  first warning after >5 s quiet {hms(s)}: {near}")
