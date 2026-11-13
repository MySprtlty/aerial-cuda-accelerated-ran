#!/usr/bin/env python3
# runsum.py <run dir under /home/nvidia/_dgist/runs> <from HH:MM:SS> <to HH:MM:SS>  (L1 clock, UTC) - per-second console lines of one generator run
import sys, re
R, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
inw = lambda l: A <= l[:8] <= B
pc = re.compile(r'Cell +(\d+) \| DL +([\d.]+) Mbps +(\d+) Slots \| UL +([\d.]+) Mbps +(\d+) Slots CRC +(\d+)')
pr = re.compile(r'\[RU\] Cell +(\d+) UL.*DL_C_ON +([\d.]+)% DL_U_ON +([\d.]+)% UL_C_ON +([\d.]+)%')
pg = re.compile(r'PUSCH in-order gate: slots (\d+) held (\d+) \(sum (\d+) us, max (\d+) us\) timeout (\d+) out-of-order (\d+) setup-failed (\d+)')
sec = {}; c5 = []; gate = []; err = fat = 0
for l in open(R + '/l1_console.log', errors='replace'):
    if not inw(l): continue
    if 'Send Err.ind' in l: err += 1
    if 'FATAL' in l: fat += 1
    m = pc.search(l)
    if m:
        c = int(m.group(1)); t = l[:8]
        if c < 5:
            s = sec.setdefault(t, [0, 0.0, 0.0, 0, 0]); s[0] += 1; s[1] += float(m.group(2)); s[2] += float(m.group(4)); s[3] += int(m.group(5)); s[4] += int(m.group(6))
        else: c5.append((float(m.group(2)), float(m.group(4)), int(m.group(5)), int(m.group(6))))
        continue
    m = pg.search(l)
    if m: gate.append((l[:15],) + tuple(int(x) for x in m.groups()))
v = [s for s in sec.values() if s[0] == 5]
if v:
    ul = sorted(s[2] for s in v)
    print(f"emulator cells 0-4: {len(v)} s | UL sum Mbps min/median/max {ul[0]:.0f}/{ul[len(ul)//2]:.0f}/{ul[-1]:.0f} | UL slots {sum(s[3] for s in v)} | CRC errors {sum(s[4] for s in v)} | DL sum Mbps mean {sum(s[1] for s in v)/len(v):.0f}")
if c5: print(f"phone cell 5: {len(c5)} s | UL Mbps mean {sum(x[1] for x in c5)/len(c5):.2f} max {max(x[1] for x in c5):.2f} | UL slots {sum(x[2] for x in c5)} | CRC {sum(x[3] for x in c5)}")
print(f"L1 console: Send Err.ind {err} FATAL {fat}")
if gate:
    a, b = gate[0], gate[-1]
    print(f"gate counters {a[0]} -> {b[0]}: slots +{b[1]-a[1]} held +{b[2]-a[2]} (hold sum +{b[3]-a[3]} us, max so far {b[4]} us) timeout +{b[5]-a[5]} out-of-order +{b[6]-a[6]} setup-failed +{b[7]-a[7]} | {len(gate)} lines")
n = 0; lo = [101.0] * 3; bad = [0] * 3; tot = [0.0] * 3
for l in open(R + '/ru_console.log', errors='replace'):
    if not inw(l): continue
    m = pr.search(l)
    if not m: continue
    n += 1
    for i in range(3):
        x = float(m.group(2 + i)); tot[i] += x; lo[i] = min(lo[i], x); bad[i] += x < 100.0
if n: print("RU on-time over %d cell-seconds: " % n + " | ".join(f"{k} mean {tot[i]/n:.3f} % min {lo[i]:.2f} % below-100 {bad[i]}" for i, k in enumerate(('DL-C', 'DL-U', 'UL-C'))))
