#!/usr/bin/env python3
"""ring_content_compare.py <ring_A.bin> <ring_B.bin> [--records 2000] [--period 80]
E6 check: compare the FAPI content of two hook-ring dumps (dapp_ring_record.py raw streams or shm images) record by
record, keyed by launch-pattern slot (SLOT_END / PRODUCER_START records are runtime status, skipped by default). Every record's fields except seq, ts_ns, handle and reserved are compared; records
are grouped by key = ((sfn*20 + slot) % period, cell_id, type, pdu_index / pdu_type) and the SET of value tuples per key
must be equal for every key present in both dumps. Only the first --records records of each dump are read, so the two
dumps may start at different SFNs. Exit 0 = identical content, 1 = differences (listed), 2 = no common key."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "estimator"))
from dapp_est import ringio   # noqa: E402

SKIP = ("seq", "ts_ns", "handle", "reserved", "pdu_truncated", "pid", "generation")


def load(path, n, period):
    with open(path, "rb") as f:
        buf = f.read(n * ringio.REC_SIZE + 4096)
    off = ringio.HDR_SIZE if buf[:8] == ringio.MAGIC else 0
    groups = {}
    recs = 0
    for i in range(n):
        o = off + i * ringio.REC_SIZE
        if o + ringio.REC_SIZE > len(buf):
            break
        r = ringio.decode(buf, o)
        if r is None:
            continue
        recs += 1
        d = r._asdict()
        pidx = (d["sfn"] * 20 + d["slot"]) % period
        sub = d.get("pdu_index", d.get("pdu_type", 0))
        key = (pidx, d["cell_id"], d["type"], sub)
        val = tuple((k, v) for k, v in d.items() if k not in SKIP and not k.startswith("ts") and not k.endswith("_ns") and k not in ("sfn", "slot"))
        groups.setdefault(key, set()).add(val)
    return groups, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--records", type=int, default=2000); ap.add_argument("--period", type=int, default=80)
    ap.add_argument("--skip-types", default="5,6", help="record types left out: 5 SLOT_END (L1 runtime enqueue status), 6 PRODUCER_START")
    a = ap.parse_args()
    skip = {int(x) for x in a.skip_types.split(",") if x}
    ga, na = load(a.a, a.records, a.period); gb, nb = load(a.b, a.records, a.period)
    ga = {k: v for k, v in ga.items() if k[2] not in skip}; gb = {k: v for k, v in gb.items() if k[2] not in skip}
    common = sorted(set(ga) & set(gb))
    print("A: %d records, %d keys | B: %d records, %d keys | common keys %d" % (na, len(ga), nb, len(gb), len(common)))
    if not common:
        print("no common (pattern slot, cell, type, pdu) key - dumps too short or different pattern"); return 2
    diff = [k for k in common if ga[k] != gb[k]]
    types = {}
    for k in common:
        types[k[2]] = types.get(k[2], 0) + 1
    print("common keys by record type:", " ".join("%s:%d" % (ringio_name(t), n) for t, n in sorted(types.items())))
    if diff:
        print("DIFFERENT content for %d of %d common keys; first 5:" % (len(diff), len(common)))
        for k in diff[:5]:
            onlya = ga[k] - gb[k]; onlyb = gb[k] - ga[k]
            fa = dict(next(iter(onlya))) if onlya else {}; fb = dict(next(iter(onlyb))) if onlyb else {}
            changed = [f for f in fa if fa.get(f) != fb.get(f)]
            print("  key %s: fields differing %s" % (k, {f: (fa.get(f), fb.get(f)) for f in changed[:6]}))
        return 1
    print("IDENTICAL FAPI content on all %d common keys (timestamps, seq and handles excluded)" % len(common))
    return 0


def ringio_name(t):
    return {1: "UL_PDU", 2: "UL_TTI", 3: "DL_PDU", 4: "DL_TTI", 5: "SLOT_END", 6: "PRODUCER_START"}.get(t, str(t))


if __name__ == "__main__":
    sys.exit(main())
