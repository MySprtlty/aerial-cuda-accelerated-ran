#!/usr/bin/env python3
"""ring_compare.py <ref.bin> <new.bin> [--slots N]: are two recorded hook rings (tools/dapp_ring_record.py output, raw
128-byte records) equivalent?  Used to check the asynchronous export against the inline hook on the same launch
pattern (59c, 16 cells).

1. Integrity of each ring: sequence gaps, record types.
2. Structure of every slot block (records between two SLOT_END records): per cell "PDU records then exactly one TTI
   summary", SLOT_END last, every record carrying the slot's (sfn, slot), SLOT_END.num_cells = cells seen.
3. Content: the launch pattern is periodic, so the records of a slot at pattern position p = (sfn*20 + slot) mod P must be
   byte-identical between the runs once the time fields are masked (record seq/ts, TTI ts_l2_send_ns, SLOT_END
   tick/t0/latency/l2a times). Compared as an ordered sequence, i.e. the order inside the slot is part of the check.
Numpy only; the first 4000 slot blocks (2 s of traffic) of each ring are skipped (start-up)."""
import sys, argparse, collections, hashlib
import numpy as np

REC = 128
T_UL_PDU, T_UL_TTI, T_DL_PDU, T_DL_TTI, T_SLOT_END, T_START = 1, 2, 3, 4, 5, 6
DT = np.dtype([("seq", "<u8"), ("ts", "<u8"), ("type", "<u2"), ("sfn", "<u2"), ("slot", "<u2"), ("cell", "<u2"), ("pay", "u1", REC - 24)])

SKIP = 4000     # slot blocks skipped at the start of the traffic (2 s: start-up slots differ from the steady pattern)

def load(path, nslots):
    a = np.fromfile(path, dtype=DT, count=(nslots + SKIP) * 260 + 1_000_000)
    # whole slot blocks only: begin right after the SKIP-th SLOT_END, end at the nslots-th SLOT_END after it
    ends = np.nonzero(a["type"] == T_SLOT_END)[0]
    if len(ends) < SKIP + nslots + 1: nslots = len(ends) - SKIP - 1
    return a[ends[SKIP - 1] + 1: ends[SKIP + nslots - 1] + 1], nslots

def masked(a):
    """payload bytes with the time-dependent fields zeroed"""
    p = a["pay"].copy()
    ty = a["type"]
    p[np.ix_(ty == T_UL_TTI, range(16, 24))] = 0            # ts_l2_send_ns
    p[np.ix_(ty == T_DL_TTI, range(16, 24))] = 0
    p[np.ix_(ty == T_SLOT_END, range(24, 72))] = 0          # tick_original, t0, l2a_latency, l1_slot_ind_tick, l2a_start, l2a_end
    return p

def blocks(a):
    ends = np.nonzero(a["type"] == T_SLOT_END)[0]
    s = 0
    for e in ends:
        yield s, e + 1
        s = e + 1

def analyse(path, nslots, label):
    a, nslots = load(path, nslots)
    gaps = int((np.diff(a["seq"].astype(np.int64)) != 1).sum())
    types = collections.Counter(a["type"].tolist())
    print("%s: %s  records %d, slot blocks %d, seq gaps %d, types %s" % (label, path, len(a), nslots, gaps, dict(sorted(types.items()))))
    p = masked(a)
    bad = collections.Counter(); sig = {}; order = {}; ncell_bad = 0; per_slot = []
    ty = a["type"]; cell = a["cell"]; sfn = a["sfn"]; slot = a["slot"]
    for s, e in blocks(a):
        ksfn, kslot = int(sfn[e - 1]), int(slot[e - 1])
        if not ((sfn[s:e] == ksfn).all() and (slot[s:e] == kslot).all()): bad["foreign (sfn, slot) inside the block"] += 1
        # per-cell groups: PDUs of one direction, then that direction's TTI summary, same cell throughout
        i = s; cells = set(); ok = True
        while i < e - 1:
            c = int(cell[i]); j = i
            while j < e - 1 and ty[j] in (T_UL_PDU, T_DL_PDU) and cell[j] == c: j += 1
            if j >= e - 1 or ty[j] not in (T_UL_TTI, T_DL_TTI) or cell[j] != c: ok = False; break
            want = T_UL_PDU if ty[j] == T_UL_TTI else T_DL_PDU
            if not (ty[i:j] == want).all(): ok = False; break
            cells.add(c); i = j + 1
        if not ok: bad["PDU/TTI grouping broken"] += 1
        ncell = int(np.frombuffer(a["pay"][e - 1][12:16].tobytes(), "<u4")[0])
        if ncell != len(cells): ncell_bad += 1
        h = hashlib.blake2b(digest_size=12)
        h.update(ty[s:e].tobytes()); h.update(cell[s:e].tobytes()); h.update(p[s:e].tobytes())
        per_slot.append((ksfn, kslot, h.digest(), e - s))
    print("   structure: %s; SLOT_END.num_cells != cells seen in %d blocks" % (dict(bad) if bad else "every block = [cell: PDUs, TTI]... then SLOT_END", ncell_bad))
    return a, per_slot, gaps, bool(bad) or ncell_bad > 0

def period(per_slot):
    for P in (20, 40, 80, 160, 320, 640, 20480):
        seen = {}
        if all(seen.setdefault((s * 20 + sl) % P, d) == d for s, sl, d, n in per_slot): return P, seen
    return None, None

ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("new"); ap.add_argument("--slots", type=int, default=40000)
args = ap.parse_args()
ra, rs, rg, rbad = analyse(args.ref, args.slots, "ref")
na, ns, ng, nbad = analyse(args.new, args.slots, "new")
P, ref_sig = period(rs)
print("ref: content is periodic with P = %s slots" % P)
if P is None:
    print("RESULT: cannot compare content (reference not periodic)"); sys.exit(2)
miss = [(s, sl) for s, sl, d, n in ns if ref_sig.get((s * 20 + sl) % P) != d]
cnt_ref = collections.Counter(n for s, sl, d, n in rs); cnt_new = collections.Counter(n for s, sl, d, n in ns)
print("records per slot block (count: blocks)  ref %s" % dict(sorted(cnt_ref.items())))
print("                                         new %s" % dict(sorted(cnt_new.items())))
print("content + order: %d of %d new slot blocks differ from the reference at the same pattern position%s" % (len(miss), len(ns), (" e.g. " + str(miss[:5])) if miss else ""))
ok = (ng == 0) and (not nbad) and (not miss)
print("RESULT: %s" % ("EQUIVALENT (same records, same order, no gaps)" if ok else "DIFFERENT"))
sys.exit(0 if ok else 1)
