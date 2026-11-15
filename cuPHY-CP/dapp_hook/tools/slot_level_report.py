#!/usr/bin/env python3
"""slot_level_report.py --dir DIR base=<name> ours=<name> [req=<name> ...]   (2026-11-13, branch feature/slot_level)

Compares the request CSVs (python/ra_stub.py --csv) of the arms of a tenant/sl_fluct.sh measurement: inference latency (infer_ms = GPU time
of the batch-3 inference, independent of queueing), server latency (wait + decode + inference) and end-to-end latency, overall and per load
phase. The phase of a request is taken from the wall clock exactly as the block-repeat option of the RU emulator / testMAC does it
(frame = (CLOCK_REALTIME - 3.64 s) / 10 ms; phase = (frame / (block frames x repeat)) mod 4), so both arms are cut at the same boundaries.
The first --skip seconds of every request file are left out. The first arm is the reference for the gain column."""
import argparse, csv, math, os, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("arms", nargs="+", help="label=name")
ap.add_argument("--dir", default="prof/tenant/slot_level/fluct"); ap.add_argument("--skip", type=float, default=1.0)
ap.add_argument("--levels", default="1,2,4,3"); ap.add_argument("--block-frames", type=int, default=1); ap.add_argument("--block-repeat", type=int, default=300)
ap.add_argument("--edge", type=float, default=0.0, help="also leave out requests within this many seconds of a phase boundary (per-phase rows only)")
a = ap.parse_args()
levels = [int(x) for x in a.levels.split(",")]; per = a.block_frames * a.block_repeat * 0.01


def pct(v, p):
    v = sorted(v); k = (len(v) - 1) * p / 100.0; lo = int(math.floor(k)); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def load(name):
    rows = list(csv.DictReader(open(os.path.join(a.dir, "ra_%s.csv" % name))))
    t0 = float(rows[0]["t_arrival"]); out = []; bad = {}
    for r in rows:
        if r["status"] != "success": bad[r["status"] + ":" + r["reason"]] = bad.get(r["status"] + ":" + r["reason"], 0) + 1; continue
        if float(r["t_arrival"]) - t0 < a.skip: continue
        t = float(r["t_send"]) + (float(r["wait_ms"]) + float(r["decode_ms"])) / 1000.0     # ~ start of the inference
        x = (t - 3.64) / per; ph = int(math.floor(x)) % len(levels); edge = min(x - math.floor(x), math.ceil(x) - x) * per
        out.append(dict(t=t, ph=ph, edge=edge, infer=float(r["infer_ms"]), server=float(r["server_ms"]), e2e=float(r["e2e_ms"]), wait=float(r["wait_ms"]),
                        sm=float(r["sm_mean"] or 0), det=r["detections"], img=r["image"]))
    return rows, out, bad


def line(label, v):
    return "%-22s n %5d  mean %6.2f  p50 %6.2f  p90 %6.2f  p99 %6.2f  max %6.2f" % (label, len(v), st.mean(v), pct(v, 50), pct(v, 90), pct(v, 99), max(v))


arms = []
for s in a.arms:
    label, name = s.split("=", 1); rows, ok, bad = load(name); arms.append((label, name, rows, ok, bad))
    dur = float(rows[-1]["t_arrival"]) - float(rows[0]["t_arrival"])
    dets = {}
    for r in ok: dets.setdefault(r["img"], set()).add(r["det"])
    print("== %s (%s): %d requests in %.0f s (%.2f req/s), %d ok after the first %.0f s, not ok %s | mean SMs %.1f | detections %s" % (
        label, name, len(rows), dur, len(rows) / max(dur, 1e-9), len(ok), a.skip, bad or "none", st.mean([r["sm"] for r in ok]),
        " ".join("%s:%s" % (k.replace("_q85.jpg", ""), "/".join(sorted(v))) for k, v in sorted(dets.items()))))
    for key, nm in (("infer", "inference ms"), ("server", "server ms"), ("e2e", "end-to-end ms"), ("wait", "queue wait ms")):
        print("   " + line(nm, [r[key] for r in ok]))
ref = arms[0]
print("\n== inference latency per load phase (PUSCH UEs per cell and U slot), gain vs %s" % ref[0])
hdr = "%-10s" % "phase" + "".join(" | %-34s" % ("%s  n / mean / p50 / p99 / SMs" % l) for l, *_ in arms) + "".join(" | %s vs %s mean / p50 / p99" % (l, ref[0]) for l, *_ in arms[1:])
print(hdr)
for ph in list(range(len(levels))) + [-1]:
    cells = []; stats = []
    for label, name, rows, ok, bad in arms:
        v = [r for r in ok if (ph < 0 or (r["ph"] == ph and r["edge"] >= a.edge))]
        if not v: cells.append(" | %-34s" % "-"); stats.append(None); continue
        x = [r["infer"] for r in v]; s = (st.mean(x), pct(x, 50), pct(x, 99), len(x), st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0)
        stats.append(s); cells.append(" | %5d %6.2f %6.2f %6.2f  %5.1f   " % (s[3], s[0], s[1], s[2], st.mean([r["sm"] for r in v])))
    gains = []
    for s in stats[1:]:
        if s is None or stats[0] is None: gains.append(" | -"); continue
        b = stats[0]; se = math.sqrt(s[4] ** 2 + b[4] ** 2) / b[0] * 100.0
        gains.append(" | %+6.1f %% (+-%.1f) / %+6.1f %% / %+6.1f %%" % ((s[0] / b[0] - 1) * 100, 1.96 * se, (s[1] / b[1] - 1) * 100, (s[2] / b[2] - 1) * 100))
    print("%-10s" % ("all" if ph < 0 else "L%d" % levels[ph]) + "".join(cells) + "".join(gains))
print("(+- = 95 % interval of the mean difference from the request-to-request spread; p99 of a phase with n < 1000 is a rough value)")
