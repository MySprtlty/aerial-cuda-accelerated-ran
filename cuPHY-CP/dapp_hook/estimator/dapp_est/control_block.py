"""
Shared-memory control block written by the estimator and read by the tenant
runner (dapp_sched). Layout (64 bytes, little endian), also in
include/dapp_hook/dapp_ctrl_abi.h:

  0  char[8]  magic      "DAPPCTL1"
  8  u32      abi        1
  12 u32      size       64
  16 u64      seq        seqlock: odd while the writer is inside an update
  24 u32      gate_slots number of slots (from slot_id) the tenant may start work in; 0 = do not start
  28 u32      cap_pct    SM cap (% of the GPU) the tenant must run under
  32 u32      slot_id    sfn * slots_per_frame + slot of the decision's "now" slot
  36 u16      sfn
  38 u16      slot
  40 u64      ts_ns      ring clock timestamp of the decision
  48 u64      n_decisions
  56 u32      batch
  60 u32      reserved
"""
import mmap
import os
import struct

MAGIC = b"DAPPCTL1"
ABI = 1
SIZE = 64
_FMT = struct.Struct("<8sIIQIIIHHQQII")
DEFAULT_NAME = "/aerial_dapp_ctrl"


class ControlBlock:
    def __init__(self, name=DEFAULT_NAME, create=True):
        path = "/dev/shm" + name if name.startswith("/") else "/dev/shm/" + name
        self.path = path
        self.seq = 0
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        fd = os.open(path, flags, 0o644)
        try:
            if os.fstat(fd).st_size < SIZE:
                if not create:
                    raise ValueError("control block %s too small" % path)
                os.ftruncate(fd, SIZE)
            self.mm = mmap.mmap(fd, SIZE)
        finally:
            os.close(fd)
        if create:
            self.write(0, 0, 0, 0, 0, 0, 0, 0)
        else:
            magic, abi = struct.unpack_from("<8sI", self.mm, 0)
            if magic != MAGIC or abi != ABI:
                raise ValueError("control block %s: bad magic/abi" % path)

    def write(self, gate_slots, cap_pct, slot_id, sfn, slot, ts_ns, n_decisions, batch):
        mm = self.mm
        self.seq += 1                                   # odd: writing
        struct.pack_into("<Q", mm, 16, self.seq)
        _FMT.pack_into(mm, 0, MAGIC, ABI, SIZE, self.seq, int(gate_slots), int(cap_pct), int(slot_id),
                       int(sfn) & 0xFFFF, int(slot) & 0xFFFF, int(ts_ns), int(n_decisions), int(batch), 0)
        self.seq += 1                                   # even: consistent
        struct.pack_into("<Q", mm, 16, self.seq)

    def read(self):
        """Seqlock read; returns a dict or None if the writer was mid-update twice in a row."""
        for _ in range(8):
            s1 = struct.unpack_from("<Q", self.mm, 16)[0]
            if s1 & 1:
                continue
            f = _FMT.unpack_from(self.mm, 0)
            s2 = struct.unpack_from("<Q", self.mm, 16)[0]
            if s1 == s2:
                return {"seq": f[3], "gate_slots": f[4], "cap_pct": f[5], "slot_id": f[6], "sfn": f[7],
                        "slot": f[8], "ts_ns": f[9], "n_decisions": f[10], "batch": f[11]}
        return None

    def close(self):
        self.mm.close()

    def unlink(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
