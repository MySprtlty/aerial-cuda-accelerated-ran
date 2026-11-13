#!/usr/bin/env python3
# slice.py <log> <t0 HH:MM:SS.ffffff> <t1> : print the lines of a time-ordered nvlog file between t0 and t1 (bisect on byte offsets, reads only the window)
import sys, os, re
p, t0, t1 = sys.argv[1], sys.argv[2], sys.argv[3]
pat = re.compile(rb'^(\d\d:\d\d:\d\d\.\d{6})')
f = open(p, 'rb'); size = os.fstat(f.fileno()).st_size
def ts_at(off):
    f.seek(off); f.readline()
    for _ in range(200):
        pos = f.tell(); l = f.readline()
        if not l: return None, size
        m = pat.match(l)
        if m: return m.group(1).decode(), pos
    return None, size
lo, hi = 0, size
while hi - lo > 1 << 16:
    mid = (lo + hi) // 2
    t, _ = ts_at(mid)
    if t is None or t >= t0: hi = mid
    else: lo = mid
f.seek(max(0, lo - (1 << 20))); f.readline()   # 1 MB back: lines of different threads are only nearly ordered
out = sys.stdout.buffer; late = 0
for l in f:
    m = pat.match(l)
    if not m: continue
    t = m.group(1).decode()
    if t < t0: continue
    if t > t1:
        late += 1
        if late > 20000: break
        continue
    out.write(l)
