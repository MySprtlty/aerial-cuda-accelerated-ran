#!/usr/bin/env python3
"""make_launch_pattern_tdd.py --tdd DDDSUUDDDD|DDDDDDSUUU|DSUUUDSUUU --out FILE [--src launch_pattern_F08_16C_59c.yaml]
    [--slots 80] [--level L] [--new-base 90000] [--alt-src FILE --alt-every F] [--s-sources 3,13] [--u-sources 4,5,14,15]

Launch pattern (testMAC + RU emulator) for another TDD pattern, built from an official pattern file as the vector library:
every slot of the official file is a (slot-in-frame, type) sample - D = DL slot, S = special slot (its vectors carry
PDSCH over symbols 1-6 only), U = UL slot. The new schedule assigns, for slot k of the new pattern (frame f = k // 20,
position p = k % 20, type t = tdd[p % len(tdd)]), the official entry of frame f % 4 at a source position of the same type:
  D: positions whose official type is D, in order, cycling (SSB-carrying positions 0-2 first)
  S: --s-sources (default 3,13)          U: --u-sources (default 4,5,14,15), cycling per frame
--level L swaps the UL vectors for their PUSCH load-level copies (make_pusch_load_tvs.py ids). --slots N repeats the
20-slot frames up to N (20480 = 10.24 s, the SFN period). --alt-src/--alt-every take the frames of a second official file
(e.g. 60c = Average) for every other block of F frames (alternating load). The header (Cell_Configs, UL_Cell_Configs) is
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
    out, uidx = list(header), 0
    for k in range(a.slots):
        f, p = k // 20, k % 20
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
    print("%s: tdd %s, %d slots, %d unique vectors" % (a.out, tdd, a.slots, uniq))
    return 0


if __name__ == "__main__":
    sys.exit(main())
