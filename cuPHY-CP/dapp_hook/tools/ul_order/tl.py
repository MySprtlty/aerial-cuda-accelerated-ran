#!/usr/bin/env python3
# tl.py <rows.tsv> <l1_console.log> <HH:MM:SS from> <HH:MM:SS to> <bin s> : timeline of generator level, phone-cell UL, out-of-order pairs (both slots ok), failed slots
import sys, re, datetime, collections
t0, t1, B = sys.argv[3], sys.argv[4], int(sys.argv[5])
def sec(s): h, m, x = s.split(':'); return int(h) * 3600 + int(m) * 60 + int(float(x))
a0, a1 = sec(t0), sec(t1)
rows = []
for l in open(sys.argv[1]):
    f = l.rstrip('\n').split('\t'); rows.append((int(f[0]), int(f[5]), int(f[7]), int(f[9]), int(f[11])))
rows.sort()
bins = collections.defaultdict(lambda: dict(inv=0, fail=0, mg=[], g=[], p=[], crc=[]))
def b_of(ns): s = int(ns // 1_000_000_000) % 86400; return (s // B) * B
for i, a in enumerate(rows):
    s = b_of(a[2])
    if not (a0 <= s < a1): continue
    d = bins[s]
    if a[1] != 1: d['fail'] += 1; continue
    d['mg'].append(a[3])
    for j in (i + 1, i + 2):
        if j >= len(rows) or rows[j][0] - a[0] > 2: break
        b = rows[j]
        if b[1] == 1 and b[4] and a[4] and b[4] < a[4]: d['inv'] += 1
pat = re.compile(r'^(\d\d:\d\d:\d\d).*?Cell +(\d+) \|.*?UL +([\d.]+) Mbps +(\d+) Slots CRC +(\d+)')
for l in open(sys.argv[2], errors='replace'):
    m = pat.match(l)
    if not m: continue
    s = sec(m.group(1))
    if not (a0 <= s < a1): continue
    d = bins[(s // B) * B]
    if m.group(2) == '0': d['g'].append(float(m.group(3)))
    elif m.group(2) == '5': d['p'].append(float(m.group(3))); d['crc'].append(int(m.group(5)))
print("UTC       generator cell-0 UL Mbps (min..max) | phone cell UL Mbps (mean), CRC/s | slot time p99/max us | out-of-order (both ok) | failed slots")
for s in sorted(bins):
    d = bins[s]; g = sorted(d['mg']); q = g[min(len(g) - 1, int(.99 * len(g)))] if g else 0
    gl = f"{min(d['g']):6.1f}..{max(d['g']):6.1f}" if d['g'] else "      -       "
    pl = f"{sum(d['p'])/len(d['p']):6.1f}, {sum(d['crc'])/len(d['crc']):4.0f}" if d['p'] else "     -      "
    print(f"{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}   {gl}                   | {pl}                  | {q:5d} / {g[-1] if g else 0:5d}       | {d['inv']:4d}                   | {d['fail']:5d}")
