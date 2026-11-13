#!/usr/bin/env python3
"""slot_level_fluct_pattern.py --src BASE.yaml --out OUT.yaml [--levels 1,2,4,3]   (2026-11-13, branch feature/slot_level)

Fluctuating-load launch pattern for the slot-level measurement: every 20-slot frame of an 80-slot DDDDDDSUUU pattern (BASE = the output of
make_launch_pattern_tdd.py --tdd DDDDDDSUUU --src launch_pattern_F08_<N>C_59c.yaml --slots 80) gets its own PUSCH load level: the UL vectors
TVnr_ULMIX_<4471..4606> of frame k are replaced by their level-L copies 90000 + 1000 L + (id - 4471) made by make_pusch_load_tvs.py
(level L = the first L PUSCH UEs per cell and U slot; the 59c Peak vectors carry 6). PUCCH / PRACH PDUs and all DL vectors stay as they are,
so only the PUSCH load changes. Played with the block-repeat option of the f1002 RU emulator / testMAC
(RU_LP_BLOCK_FRAMES=1 RU_LP_BLOCK_REPEAT=<frames per level>, same for TESTMAC_) each frame is one load phase.
Refuses to overwrite OUT and stops when a level vector is missing."""
import argparse
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--levels", default="1,2,4,3", help="PUSCH UEs per cell and U slot for frames 0..3 (0 = keep the Peak vector)")
ap.add_argument("--tv-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "testVectors"))
a = ap.parse_args()
levels = [int(x) for x in a.levels.split(",")]
if os.path.exists(a.out): sys.exit("refusing to overwrite " + a.out)
s = open(a.src).read(); i = s.index("SCHED:"); head, body = s[:i], s[i:]
parts = body.split("- config:")
if len(parts) != 81 or len(levels) != 4: sys.exit("expected an 80-slot pattern and 4 levels (got %d slots, %d levels)" % (len(parts) - 1, len(levels)))
used = set()


def swap(m, L):
    v = int(m.group(1))
    if L == 0: return m.group(0)
    if not 4471 <= v <= 4606: sys.exit("unexpected UL vector id %d" % v)
    n = 90000 + 1000 * L + (v - 4471); used.add(n); return "ULMIX_%d_gNB" % n


for k in range(80):
    parts[1 + k] = re.sub(r"ULMIX_(\d+)_gNB", lambda m: swap(m, levels[k // 20]), parts[1 + k])
missing = [n for n in sorted(used) if not os.path.exists(os.path.join(a.tv_dir, "TVnr_ULMIX_%d_gNB_FAPI_s0.h5" % n))]
if missing: sys.exit("missing level vectors (make_pusch_load_tvs.py --levels ...): %s ..." % missing[:5])
open(a.out, "w").write(head + "- config:".join(parts))
print("%s: frames = PUSCH levels %s, %d level vectors used" % (a.out, levels, len(used)))
