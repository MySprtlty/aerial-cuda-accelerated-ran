#!/usr/bin/env python3
# infeas.py <dapp_tenant.log> : per-10-s deltas of the sizer's PUSCH counters (slots, infeasible, per-profile) and of the agent's requests
import sys, re
prev = None
print("epoch       dslots dinfeas  d76 d108  dreq  pauses")
for l in open(sys.argv[1], errors='replace'):
    m = re.match(r'\[(\d+)\].*pauses (\d+).*agent ok (\d+).*PUSCH slots (\d+) infeasible (\d+) prof (.*?) fallback', l)
    if not m: continue
    t, pa, ok, sl, inf = (int(m.group(i)) for i in range(1, 6))
    prof = {int(a): int(b) for a, b in re.findall(r'(\d+):(\d+)/\d+', m.group(6))}
    cur = (t, pa, ok, sl, inf, prof.get(76, 0), prof.get(108, 0))
    if prev: print(f"{t}  {sl-prev[3]:6d} {inf-prev[4]:6d} {cur[5]-prev[5]:5d} {cur[6]-prev[6]:4d} {ok-prev[2]:5d} {pa-prev[1]:6d}")
    prev = cur
