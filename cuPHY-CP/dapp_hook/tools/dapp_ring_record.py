#!/usr/bin/env python3
"""
Record the dApp ring to a file for offline replay.

The ring keeps only the last ring_len records (65536 = 0.3 s at 8 cells with
PDU export), so a replay needs a continuous dump. This tool snapshots the
whole ring body every --interval seconds, keeps the records whose seq is new
since the previous snapshot and <= the head seen before the copy (those are
committed and cannot have been overwritten during the copy), and appends
them in seq order to <out>. The ring header (4096 B) is saved to <out>.hdr so
a replay knows slot_advance, mu and the cell count.

    dapp_ring_record.py --out prof/ring_8C.bin --seconds 75

Output: raw 128-byte records back to back (the ABI of dapp_ring_abi.h),
readable by estimator/dapp_est/ringio.py and by python/dapp_ring.py dtypes.
Gaps (seq jumps) are counted and printed; they mean the recorder fell behind.
"""
import argparse
import mmap
import os
import signal
import struct
import sys
import time

import numpy as np

HDR_SIZE = 4096
REC_SIZE = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", default="/dev/shm/aerial_dapp_ring")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until SIGTERM)")
    ap.add_argument("--interval", type=float, default=0.03, help="snapshot period in seconds")
    ap.add_argument("--wait", type=float, default=120.0, help="wait this long for the ring to appear")
    a = ap.parse_args()

    t_wait = time.time()
    while not os.path.exists(a.ring):
        if time.time() - t_wait > a.wait:
            print("ring %s did not appear" % a.ring)
            return 1
        time.sleep(0.2)
    fd = os.open(a.ring, os.O_RDONLY)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
    magic = bytes(mm[0:8])
    if magic != b"DAPPRNG1":
        print("bad magic", magic)
        return 1
    abi, rec_size, ring_len, hdr_size, generation = struct.unpack_from("<IIIII", mm, 8)
    if rec_size != REC_SIZE or hdr_size != HDR_SIZE:
        print("unexpected layout rec_size=%d hdr_size=%d" % (rec_size, hdr_size))
        return 1
    with open(a.out + ".hdr", "wb") as f:
        f.write(bytes(mm[0:HDR_SIZE]))
    body_off = hdr_size
    body_len = ring_len * rec_size
    rec_dt = np.dtype([("seq", "<u8"), ("rest", "V%d" % (rec_size - 8))])

    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))

    out = open(a.out, "wb")
    last_seq = 0
    written = 0
    gaps = 0
    snapshots = 0
    t0 = time.time()
    next_t = t0
    while not stop["v"]:
        if a.seconds and time.time() - t0 > a.seconds:
            break
        head = struct.unpack_from("<Q", mm, 40)[0]
        if head > last_seq:
            snap = bytes(mm[body_off:body_off + body_len])
            recs = np.frombuffer(snap, dtype=rec_dt)
            seq = recs["seq"]
            sel = (seq > last_seq) & (seq <= head)
            if sel.any():
                idx = np.nonzero(sel)[0]
                order = idx[np.argsort(seq[idx], kind="stable")]
                s_sorted = seq[order]
                # gaps: a jump of more than 1 inside this batch or from last_seq
                d = np.diff(np.concatenate(([last_seq], s_sorted)))
                gaps += int((d > 1).sum())
                out.write(recs[order].tobytes())
                written += len(order)
                last_seq = int(s_sorted[-1])
            snapshots += 1
        next_t += a.interval
        dt = next_t - time.time()
        if dt > 0:
            time.sleep(dt)
        else:
            next_t = time.time()
    out.close()
    el = time.time() - t0
    print("recorded %d records (%.1f MB) in %.1f s, %d snapshots, %d seq gaps, last seq %d, generation %d -> %s" % (
        written, written * rec_size / 1e6, el, snapshots, gaps, last_seq, generation, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
