#!/usr/bin/env python3
"""make_pusch_load_tvs.py [--tv-dir D] [--levels 2,3,4,5] [--base-ids 4471,4511,4551,4591] [--cells 16] [--new-base 90000] [--dry]
PUSCH load levels for the 16-cell 59c pattern: copies of the four UL vector blocks (slot-in-frame 4 / 5 / 14 / 15, one vector
per cell: id = block base + cell) with only the first L PUSCH PDUs kept (Peak = 6). The PUSCH PDUs are the last PDUs of
these vectors, so the copy drops datasets PDU<n> (and PDU<n>_*) for the removed ones and writes nPdu = old - (6 - L).
PUCCH / PRACH PDUs, the expected indications, Cell_Config and the IQ (X_tf*) are untouched: the RU emulator still sends
the full slot IQ, the L1 decodes only the remaining UEs. New id = new_base + 1000 * L + (old id - 4471), file name keeps the
TVnr_ULMIX_<id>_gNB_FAPI_s0.h5 convention. Refuses to overwrite. h5py."""
import argparse
import os
import re
import sys

import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tv-dir", default="/opt/nvidia/cuBB/testVectors")
    ap.add_argument("--levels", default="2,3,4,5")
    ap.add_argument("--base-ids", default="4471,4511,4551,4591")
    ap.add_argument("--cells", type=int, default=16)
    ap.add_argument("--new-base", type=int, default=90000)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",")]
    made = 0
    for base in (int(x) for x in a.base_ids.split(",")):
        for c in range(a.cells):
            src = os.path.join(a.tv_dir, "TVnr_ULMIX_%d_gNB_FAPI_s0.h5" % (base + c))
            with h5py.File(src, "r") as h:
                npdu = int(h["nPdu"][()].flatten()[0])
                types = {int(k[3:]): int(h[k][()][0]["type"]) for k in h.keys() if re.match(r"PDU\d+$", k)}
            pusch = sorted(n for n, t in types.items() if t == 8)
            if len(pusch) != 6 or pusch != list(range(npdu - 5, npdu + 1)):
                sys.exit("%s: PUSCH PDUs are not the last six (%s, nPdu %d)" % (src, pusch, npdu))
            for L in levels:
                nid = a.new_base + 1000 * L + (base + c - 4471)
                dst = os.path.join(a.tv_dir, "TVnr_ULMIX_%d_gNB_FAPI_s0.h5" % nid)
                if os.path.exists(dst):
                    sys.exit("exists: " + dst)
                drop = pusch[L:]
                if a.dry:
                    print("%s -> %s drop PDU %s nPdu %d->%d" % (os.path.basename(src), os.path.basename(dst), drop, npdu, npdu - len(drop))); continue
                with h5py.File(src, "r") as hs, h5py.File(dst, "w") as hd:
                    for k in hs.keys():
                        m = re.match(r"PDU(\d+)(_|$)", k)
                        if m and int(m.group(1)) in drop:
                            continue
                        hs.copy(k, hd)
                    for k, v in hs.attrs.items():
                        hd.attrs[k] = v
                    del hd["nPdu"]
                    hd.create_dataset("nPdu", data=hs["nPdu"][()] * 0 + (npdu - len(drop)))
                made += 1
    print("made %d vectors" % made)
    return 0


if __name__ == "__main__":
    sys.exit(main())
