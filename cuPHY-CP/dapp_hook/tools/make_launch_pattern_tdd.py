#!/usr/bin/env python3
"""make_launch_pattern_tdd.py --tdd DDDSUUDDDD|DDDDDDSUUU|DSUUUDSUUU --out FILE [--src launch_pattern_F08_16C_59c.yaml]
    [--slots 80] [--level L] [--new-base 90000] [--alt-src FILE --alt-every F | --alt-plan "A:0.56,R:0.05,B:0.34,R:0.05"]
    [--s-sources 3,13] [--u-sources 4,5,14,15]

Launch pattern (testMAC + RU emulator) for another TDD pattern, built from an official pattern file as the vector library:
every slot of the official file is a (slot-in-frame, type) sample - D = DL slot, S = special slot (its vectors carry
PDSCH over symbols 1-6 only), U = UL slot. The new schedule assigns, for slot k of the new pattern (frame f = k // 20,
position p = k % 20, type t = tdd[p % len(tdd)]), the official entry of frame f % 4 at a source position of the same type:
  D: positions whose official type is D, in order, cycling (SSB-carrying positions 0-2 first)
  S: --s-sources (default 3,13)          U: --u-sources (default 4,5,14,15), cycling per frame
--level L swaps the UL vectors for their PUSCH load-level copies (make_pusch_load_tvs.py ids). --slots N repeats the
20-slot frames up to N (20480 = 10.24 s, the SFN period). --alt-src/--alt-every take the frames of a second official file
(e.g. 60c = Average) for every other block of F frames (alternating load); --alt-plan instead gives fractions of the frame period
(A = --src frames, B = --alt-src frames, R = linear ramp between them, Bresenham-mixed whole frames). The 2026-10-16 demo patterns:
  peak      --tdd DDDDDDSUUU --src ..._16C_59c.yaml --slots 80            average   --src ..._16C_60c.yaml --slots 80
  alt_long  --tdd DDDDDDSUUU --src ..._16C_60c.yaml --alt-src ..._16C_59c.yaml --slots 20480 --alt-plan "A:0.56,R:0.05,B:0.34,R:0.05"
  (20,480-slot files are 34 MB and are not committed; regenerate them). The header (Cell_Configs, UL_Cell_Configs) is
copied from --src. With --tdd DDDSUUDDDD --slots 80 the output reproduces --src byte for byte (self-check in the tests).
The caller must register the pattern name in testBenches/phase4_test_scripts/valid_perf_patterns.sh. Refuses to overwrite."""
import argparse
import os
import re
import sys


def load(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("SCHED:"))
    header = lines[:i + 1]
    blocks, cur = [], None
    for l in lines[i + 1:]:
        if l.startswith("- config:"):
            if cur:
                blocks.append(cur)
            cur = [l]
        elif cur is not None:
            cur.append(l)
    if cur:
        blocks.append(cur)
    types = []
    for b in blocks:
        kinds = set(re.findall(r"TVnr_(DLMIX|ULMIX)_", "\n".join(b)))
        ids = [int(x) for x in re.findall(r"TVnr_DLMIX_(\d+)_", "\n".join(b))]
        if kinds == {"ULMIX"}:
            types.append("U")
        else:
            types.append("D")
    return header, blocks, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdd", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="testVectors/multi-cell/launch_pattern_F08_16C_59c.yaml")
    ap.add_argument("--slots", type=int, default=80); ap.add_argument("--level", type=int, default=0); ap.add_argument("--new-base", type=int, default=90000)
    ap.add_argument("--alt-src", default=""); ap.add_argument("--alt-every", type=int, default=0)
    ap.add_argument("--alt-plan", default="", help="instead of --alt-every: fractions of the frame period, letters A (src), B (alt-src), R (linear "
                    "ramp A->B or B->A), e.g. A:0.56,R:0.05,B:0.34,R:0.05 = Average 60 %% / Peak 40 %% including the ramps")
    ap.add_argument("--s-sources", default="3,13"); ap.add_argument("--u-sources", default="4,5,14,15")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("exists: " + a.out)
    tdd = a.tdd.upper()
    if set(tdd) - set("DSU"):
        sys.exit("tdd must use D, S, U only")
    header, blocks, types = load(a.src)
    libs = [(blocks, types)]
    if a.alt_src:
        libs.append(load(a.alt_src)[1:])
    nfr = len(blocks) // 20
    s_src = [int(x) for x in a.s_sources.split(",")]; u_src = [int(x) for x in a.u_sources.split(",")]
    d_src = [p for p in range(20) if types[p] == "D" and p not in s_src]
    plan = None
    if a.alt_src and a.alt_plan:
        # per-frame library index over one period of nfr_period frames (the SFN period for --slots 20480: 1024 frames)
        nfr_period = max(1, a.slots // 20)
        segs = [(x.split(":")[0].upper(), float(x.split(":")[1])) for x in a.alt_plan.split(",")]
        plan, prev = [], 0
        for i, (kind, frac) in enumerate(segs):
            n = int(round(frac * nfr_period))
            if kind == "A":
                plan += [0] * n; prev = 0
            elif kind == "B":
                plan += [1] * n; prev = 1
            else:   # ramp from the previous kind to the next kind: the share of B rises or falls linearly (Bresenham)
                nxt = 1 if (i + 1 < len(segs) and segs[i + 1][0] == "B") else 0
                acc = 0.0
                for j in range(n):
                    share = (j + 0.5) / n
                    pb = share if nxt == 1 else 1.0 - share
                    acc += pb
                    plan.append(1 if acc >= 1.0 else 0)
                    if acc >= 1.0:
                        acc -= 1.0
                prev = nxt
        plan = (plan + [plan[-1]] * nfr_period)[:nfr_period]
    out, uidx = list(header), 0
    for k in range(a.slots):
        f, p = k // 20, k % 20
        if plan is not None:
            lib_blocks, lib_types = libs[plan[f % len(plan)]]
        else:
            lib_blocks, lib_types = libs[(f // a.alt_every) % len(libs)] if a.alt_src and a.alt_every else libs[0]
        t = tdd[p % len(tdd)]
        if t == "D":
            dpos = [q for q in range(20) if q not in s_src and q not in u_src and lib_types[q] == "D"]
            # position mapping: the n-th D of the new frame takes the n-th D of the official frame (SSB positions first)
            n = sum(1 for q in range(p) if tdd[q % len(tdd)] == "D")
            src_pos = dpos[n % len(dpos)]
        elif t == "S":
            n = sum(1 for q in range(p) if tdd[q % len(tdd)] == "S")
            src_pos = s_src[n % len(s_src)]
        else:
            src_pos = u_src[uidx % len(u_src)]; uidx += 1
        b = lib_blocks[(f % nfr) * 20 + src_pos]
        for l in b:
            l = re.sub(r"^  slot: \d+\.0$", "  slot: %d.0" % k, l)
            if t == "U" and a.level:
                l = re.sub(r"TVnr_ULMIX_(\d+)_", lambda m: "TVnr_ULMIX_%d_" % (a.new_base + 1000 * a.level + (int(m.group(1)) - 4471)) if 4471 <= int(m.group(1)) <= 4606 else m.group(0), l)
            out.append(l)
    open(a.out, "w").write("\n".join(out) + "\n")
    uniq = len(set(re.findall(r"TVnr_[A-Z]+_\d+_", "\n".join(out))))
    if plan is not None:
        print("alt plan over %d frames: B (alt-src) share %.1f %%, first B frame %d, last B frame %d" % (len(plan), 100.0 * sum(plan) / len(plan), plan.index(1) if 1 in plan else -1, len(plan) - 1 - plan[::-1].index(1) if 1 in plan else -1))
    print("%s: tdd %s, %d slots, %d unique vectors" % (a.out, tdd, a.slots, uniq))
    return 0


if __name__ == "__main__":
    sys.exit(main())
