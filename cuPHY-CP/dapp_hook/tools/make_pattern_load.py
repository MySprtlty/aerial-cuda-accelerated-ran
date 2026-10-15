#!/usr/bin/env python3
"""make_pattern_load.py --level L [--src testVectors/multi-cell/launch_pattern_F08_16C_59c.yaml] --out FILE [--new-base 90000]
16-cell 59c pattern with every UL slot's vector replaced by its PUSCH load-level copy (make_pusch_load_tvs.py:
new id = new_base + 1000 * L + (old id - 4471)); DL slots unchanged. Refuses to overwrite."""
import argparse
import os
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--src", default="testVectors/multi-cell/launch_pattern_F08_16C_59c.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--new-base", type=int, default=90000)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("exists: " + a.out)
    s = open(a.src).read()
    n = [0]
    def sub(m):
        old = int(m.group(1))
        if not 4471 <= old <= 4606:
            sys.exit("unexpected UL id %d" % old)
        n[0] += 1
        return "TVnr_ULMIX_%d_" % (a.new_base + 1000 * a.level + (old - 4471))
    # SCHED entries only: the UL_Cell_Configs header keeps the original vectors (cell configuration source)
    i = s.index("SCHED:")
    out = s[:i] + re.sub(r"TVnr_ULMIX_(\d+)_", sub, s[i:])
    open(a.out, "w").write(out)
    print("%s: %d UL references -> level %d" % (a.out, n[0], a.level))
    return 0


if __name__ == "__main__":
    sys.exit(main())
