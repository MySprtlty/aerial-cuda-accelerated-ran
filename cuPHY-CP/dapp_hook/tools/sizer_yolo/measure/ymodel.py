#!/usr/bin/env python3
# ymodel.py <dapp_tenant_slots.csv> [vcam_status.json] : YOLO cost and plan mix from the per-slot CSV only (cheap).
#  (1) time a request is in service (decode + inference + waits for a gap; CSV column model != '') per model:
#      mean = busy time of that model / requests of that model (vcam models_sent if given, else spans); span quantiles are approximate
#      (a span ends at a model change, an idle row or a queue decrement).
#  (2) sizer PUSCH plan (plan_ul_sm) of the UL slots by slot-in-frame, without / with a request in service.
import sys, csv, json, collections
def q(g, p): g = sorted(g); return g[min(len(g) - 1, int(p * len(g)))] if g else 0
busy = collections.Counter(); spans = collections.defaultdict(list); plan = collections.defaultdict(collections.Counter)
tot = 0; pm = ''; pq = 0; st = 0; pt = 0; seen = set(); t_first = t_last = 0
with open(sys.argv[1], newline='', errors='replace') as fh:
    r = csv.reader(fh); h = next(r); ix = {n: i for i, n in enumerate(h)}
    iT, iM, iQ, iU, iS, iL, iA = (ix[k] for k in ('t_ns', 'model', 'queue', 'plan_ul_sm', 'sfn', 'slot', 'abs_slot'))
    for f in r:
        try: t = int(f[iT]); m = f[iM] if len(f) > iM else ''; qd = int(f[iQ]); u = int(f[iU]); sl = int(f[iL]); a = int(f[iA])
        except Exception: continue
        if pt:
            d = min(t - pt, 2000000)
            if pm: busy[pm] += d
            if t_first: tot += d
        if m and not t_first: t_first = t
        if m: t_last = t
        if pm and (m != pm or qd < pq): spans[pm].append((pt - st) / 1e6 + 0.5); st = t
        if m and not pm: st = t
        if u > 0 and sl in (7, 8, 9, 17, 18, 19) and (a, sl) not in seen:
            seen.add((a, sl)); plan[(sl, 1 if m else 0)][u] += 1
            if len(seen) > 400000: seen.clear()
        pm = m; pq = qd; pt = t
sent = {}
if len(sys.argv) > 2:
    try: sent = json.load(open(sys.argv[2]))['total']['models_sent']
    except Exception as e: print("no vcam status:", e)
print("(1) request in service, by model | requests (vcam) | spans | mean ms/request (busy/vcam requests) | span p50 / p90 / p99 ms (approx)")
allb = 0; alln = 0
for m in ('26n', '26s', '26m', '26l', '26x'):
    if m not in busy: continue
    n = sent.get(m, len(spans[m])); allb += busy[m]; alln += n
    print(f"   {m} | {sent.get(m, '-')!s:>6} | {len(spans[m]):6d} | {busy[m] / 1e6 / max(1, n):6.2f} | {q(spans[m], .5):5.1f} / {q(spans[m], .9):5.1f} / {q(spans[m], .99):5.1f}")
if alln: print(f"   all | {alln} | mean {allb / 1e6 / alln:.2f} ms/request | GPU has a request in service {100 * allb / max(1, t_last - t_first):.1f} % of the time between the first and the last request")
print("(2) sizer PUSCH plan by UL slot-in-frame: share of slots per plan SM, mean plan | no request  ||  request in service")
for sl in (7, 8, 9, 17, 18, 19):
    out = []
    for y in (0, 1):
        c = plan[(sl, y)]; n = sum(c.values())
        out.append((f"n {n:6d} mean {sum(k * v for k, v in c.items()) / max(1, n):5.1f} : " + " ".join(f"{k}:{100 * v / max(1, n):.1f}" for k, v in sorted(c.items()))))
    print(f"   slot {sl:2d} | {out[0]}  ||  {out[1]}")
for y in (0, 1):
    c = collections.Counter()
    for sl in (7, 8, 9, 17, 18, 19): c.update(plan[(sl, y)])
    n = sum(c.values())
    print(f"   all UL slots, {'request in service' if y else 'no request':18s}: n {n} mean plan {sum(k * v for k, v in c.items()) / max(1, n):.1f} SM | " + " ".join(f"{k}:{100 * v / max(1, n):.1f}%" for k, v in sorted(c.items())))
