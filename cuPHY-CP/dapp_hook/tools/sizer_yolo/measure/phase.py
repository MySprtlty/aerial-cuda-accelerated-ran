#!/usr/bin/env python3
# phase.py <label=per-slot csv> ... : where in the 10-ms frame YOLO requests enter service (slot-in-frame of the first in-service row),
# per run. A run-constant phase between the camera period (66.7 ms) and the frame would show as a few sharp peaks that differ per run.
import sys, csv
for a in sys.argv[1:]:
    lab, path = a.split('=', 1)
    starts = [0] * 20; busy = [0] * 20; rows = [0] * 20; prev = ''
    with open(path, newline='') as f:
        r = csv.reader(f); h = next(r); si = h.index('slot'); mi = h.index('model')
        for row in r:
            if len(row) <= mi: continue
            try: s = int(row[si])
            except ValueError: continue
            if not 0 <= s < 20: continue
            m = row[mi]; rows[s] += 1
            if m:
                busy[s] += 1
                if m != prev: starts[s] += 1
            prev = m
    n = sum(starts) or 1
    top = sorted(range(20), key=lambda s: -starts[s])[:6]
    print(f"{lab:>9} | starts {sum(starts):6d} | share of starts by slot-in-frame 0..19 %: " + ' '.join(f"{100*starts[s]/n:4.1f}" for s in range(20)))
    print(f"{'':>9} | top-6 slots hold {100*sum(starts[s] for s in top)/n:4.1f} % | in-service share of rows at UL slots 7 8 9 17 18 19 %: " + ' '.join(f"{100*busy[s]/max(rows[s],1):4.1f}" for s in (7, 8, 9, 17, 18, 19)))
