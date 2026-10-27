#!/usr/bin/env python3
"""g45_phase.py --dir D --name RUN --pattern alt_long.yaml [--peak-src ...peak.yaml --avg-src ...average.yaml] [--first-s 10]
                [--origin 3.64 --period 10.24] [--map 26x=57.5,26s=48.6,26n=40.9]
Per pattern segment (Average / ramp / Peak frames of the alternating 20,480-slot pattern) of one full-stack run:
  L1 (phy_RUN.csv):      UL slots, PUSCH B misses (rules.yaml budget) and A misses, PUSCH completion p50 / p99.99 (pos 9), PDSCH B misses
  RU (ru_RUN_late.csv):  late (slot, cell) rows, distinct late slots, late packets      - rows after the first --first-s s of traffic
  tenant (disp_RUN.csv): requests, req/s, E2E mean / p50 / p99, service mean, model shares, share-weighted mAP, HOLD / errors
The segment of a slot or request is the pattern frame of its CLOCK_REALTIME stamp: frame = floor(((t - origin) mod period) / 0.01),
classified from the pattern file by its vector ids (Peak-source ids vs Average-source ids; a frame with both = ramp)."""
import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import safecap_check as SC   # noqa: E402

MAP = {"26x": 57.5, "26s": 48.6, "26n": 40.9}


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v); return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def frame_classes(pattern, peak_src, avg_src):
    ids = lambda path: set(re.findall(r"(?:ULMIX|DLMIX)_(\d+)", open(path).read()))   # noqa: E731
    only_p, only_a = ids(peak_src) - ids(avg_src), ids(avg_src) - ids(peak_src)
    txt = open(pattern).read()
    i = txt.find("SCHED:"); blocks = txt[i:].split("- config:")[1:]
    per_frame = {}
    for b in blocks:
        m = re.search(r"^\s*slot:\s*(\d+)", b, re.M)
        if not m:
            continue
        k = int(m.group(1)); got = set(re.findall(r"(?:ULMIX|DLMIX)_(\d+)", b))
        c = ("P" if got & only_p else "") + ("A" if got & only_a else "")
        if c:
            per_frame.setdefault(k // 20, set()).add(c)
    nfr = (max(per_frame) + 1) if per_frame else 1024
    cls = {}
    for f in range(nfr):
        s = per_frame.get(f, set())
        cls[f] = "peak" if s == {"P"} else "average" if s == {"A"} else "ramp" if s else "none"
    # the generator's ramps mix whole frames (Bresenham): mark everything between the first Peak frame and the start of the longest
    # pure-Peak run, and between its end and the last Peak frame, as ramp
    pk = [f for f in range(nfr) if cls[f] == "peak"]
    if pk:
        best, cur = (pk[0], pk[0]), [pk[0], pk[0]]
        for f in pk[1:]:
            if f == cur[1] + 1:
                cur[1] = f
            else:
                cur = [f, f]
            if cur[1] - cur[0] > best[1] - best[0]:
                best = tuple(cur)
        for f in range(pk[0], best[0]):
            cls[f] = "ramp"
        for f in range(best[1] + 1, pk[-1] + 1):
            cls[f] = "ramp"
    return cls, nfr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--name", required=True); ap.add_argument("--pattern", default="")
    tv = os.path.join(HERE, "..", "..", "..", "..", "testVectors", "multi-cell")
    ap.add_argument("--peak-src", default=os.path.join(tv, "launch_pattern_F08_16C_ddddddsuuu_peak.yaml"))
    ap.add_argument("--avg-src", default=os.path.join(tv, "launch_pattern_F08_16C_ddddddsuuu_average.yaml"))
    ap.add_argument("--first-s", type=float, default=10.0); ap.add_argument("--origin", type=float, default=3.64); ap.add_argument("--period", type=float, default=10.24)
    ap.add_argument("--map", default="26x=57.5,26l=55.0,26m=53.1,26s=48.6,26n=40.9")
    ap.add_argument("--segments", default="", help='time segments instead of a pattern file, e.g. "average:0-60,peak:60-120" (seconds inside '
                    '--period; night-3 block-repeat alternation: segment = floor((t - origin) / 60 s) mod 2)')
    ap.add_argument("--ramp-s", type=float, default=0.0, help="with --segments: mark +-ramp-s around every boundary as ramp")
    a = ap.parse_args(); d = a.dir
    mp = {k: float(v) for k, v in (x.split("=") for x in a.map.split(","))}
    if a.segments:
        nfr = int(round(a.period / 0.01)); rng = []
        for part in a.segments.split(","):
            lab, span = part.split(":"); lo_s, hi_s = (float(x) for x in span.split("-")); rng.append((lab, lo_s, hi_s))
        cls = {}
        for f in range(nfr):
            t = f * 0.01 + 0.005
            lab = next((lab for lab, lo_s, hi_s in rng if lo_s <= t < hi_s), "none")
            if a.ramp_s > 0 and any(abs(t - b) < a.ramp_s or abs(t - b + a.period) < a.ramp_s or abs(t - b - a.period) < a.ramp_s for _, b, _ in rng):
                lab = "ramp"
            cls[f] = lab
    else:
        cls, nfr = frame_classes(a.pattern, a.peak_src, a.avg_src)
    segs = tuple(x for x in ("average", "ramp", "peak") if any(cls[f] == x for f in cls))
    frames = {s: sum(1 for f in cls if cls[f] == s) for s in segs}
    print("%s: %d frames per period %.2f s: " % (os.path.basename(a.pattern) if a.pattern else "segments " + a.segments, nfr, a.period) + ", ".join("%s %d (%.1f s)" % (s, frames[s], frames[s] * 0.01) for s in segs))
    runs = []; prev = None
    for f in range(nfr):
        if cls[f] != prev:
            runs.append([f, f, cls[f]]); prev = cls[f]
        else:
            runs[-1][1] = f
    print("  frame runs: " + "; ".join("%d-%d %s" % tuple(r) for r in runs))
    seg_of = lambda t_ns: cls.get(int((((t_ns / 1e9) - a.origin) % a.period) / 0.01) % nfr, "none")   # noqa: E731

    rows = SC.load_rows(a.name, d, a.first_s)
    t_first = min(int(float(r["t0_ns"])) for r in rows) if rows else 0
    print("\n## L1 per segment (%s, %d slots after %.0f s)" % (a.name, len(rows), a.first_s))
    print("| segment | slots | UL slots | PUSCH B misses (per UL slot) | PUSCH A misses | pos-9 PUSCH completion p50 / p99.99 us | PDSCH B misses | PUCCH / PRACH B |")
    print("|---|---|---|---|---|---|---|---|")
    for s in segs:
        rs = [r for r in rows if seg_of(int(float(r["t0_ns"]))) == s]
        if not rs:
            continue
        vals = SC.channel_values(rs); m = SC.misses(vals)
        ul = [r for r in rs if r.get("ul") == "1"]
        p9 = [SC.fl(r.get("pusch_gpu_end")) for r in rs if r.get("pos") == "9"]; p9 = [x - SC.UL_START_US for x in p9 if x is not None]
        print("| %s | %d | %d | %d (%.4f) | %d | %.0f / %.0f | %d | %d / %d |" % (s, len(rs), len(ul), m["pusch"][1], m["pusch"][1] / max(1, m["pusch"][2]), m["pusch"][0],
              pct(p9, 50), pct(p9, 99.99), m["pdsch"][1], m["pucch"][1], m["prach"][1]))

    p = os.path.join(d, "ru_%s_late.csv" % a.name)
    if os.path.exists(p):
        late = [r for r in csv.DictReader(open(p)) if int(r["t0_ns"]) >= t_first]
        print("\n## RU late per segment (rows after the L1 window start; a row = one late (slot, cell))")
        print("| segment | late rows | distinct late slots | late packets | late slots per minute of segment |")
        print("|---|---|---|---|---|")
        for s in segs:
            rs = [r for r in late if seg_of(int(r["t0_ns"])) == s]
            secs = sum(1 for r in rows if seg_of(int(float(r["t0_ns"]))) == s) * 0.0005
            print("| %s | %d | %d | %d | %.0f |" % (s, len(rs), len({(r["frame"], r["slot"]) for r in rs}), sum(int(r["late"]) for r in rs), len({(r["frame"], r["slot"]) for r in rs}) / max(secs, 1e-9) * 60))

    p = os.path.join(d, "disp_%s.csv" % a.name)
    if os.path.exists(p):
        reqs = [r for r in csv.DictReader(open(p)) if int(r["t_arrival"]) >= t_first]
        print("\n## tenant per segment (dispatcher csv, requests arriving after the L1 window start)")
        print("| segment | requests | req/s | E2E mean / p50 / p99 ms | service mean ms | 26x / 26s / 26n shares | weighted mAP | HOLDs / errors | caps seen |")
        print("|---|---|---|---|---|---|---|---|---|")
        for s in segs + ("all",):
            rs = [r for r in reqs if s == "all" or seg_of(int(r["t_arrival"])) == s]
            ok = [r for r in rs if r["status"] == "0"]
            if not rs:
                continue
            e2e = [(int(r["t_reply"]) - int(r["t_arrival"])) / 1e6 for r in ok]
            svc = [(int(r["t_end"]) - int(r["t_start"])) / 1e6 for r in ok if int(r["t_end"]) and int(r["t_start"])]
            secs = (sum(1 for r in rows if seg_of(int(float(r["t0_ns"]))) == s) if s != "all" else len(rows)) * 0.0005
            sh = {m: sum(1 for r in ok if r["model"] == m) / max(1, len(ok)) for m in ("26x", "26s", "26n")}
            caps = sorted({(r["cap_26x"], r["cap_26s"], r["cap_26n"]) for r in rs})
            print("| %s | %d | %.2f | %.1f / %.1f / %.1f | %.1f | %.2f / %.2f / %.2f | %.1f | %d / %d | %s |" % (
                s, len(rs), len(rs) / max(secs, 1e-9), sum(e2e) / max(1, len(e2e)), pct(e2e, 50), pct(e2e, 99), sum(svc) / max(1, len(svc)),
                sh["26x"], sh["26s"], sh["26n"], sum(sh[m] * mp[m] for m in sh), sum(int(r["n_hold"]) for r in rs), sum(1 for r in rs if r["status"] != "0"),
                " ".join("%s/%s/%s" % c for c in caps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
