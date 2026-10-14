"""
Shared-memory control block written by the estimator (one entry per tenant profile) or by the
dispatcher (manual / timetable policy) and read by the tenant runner (dapp_sched --ctrl) and the
dispatcher (live policy). Layout (128 bytes, little endian, ABI 2), also in
include/dapp_hook/dapp_ctrl_abi.h:

  0   char[8]  magic        "DAPPCTL1"
  8   u32      abi          2
  12  u32      size         128
  16  u64      seq          seqlock: odd while the writer is inside an update
  24  u32      slot_id      sfn * slots_per_frame + slot of the decision's "now" slot
  28  u16      sfn
  30  u16      slot
  32  u64      ts_ns        CLOCK_REALTIME ns of the decision (ring clock in live mode)
  40  u64      n_decisions
  48  u32      n_models     valid entries below
  52  u32      reserved
  56  4 x { char[8] name, u32 cap_pct, u32 gate_slots }   one entry per model (RUN model=NAME)
  120 u32[2]   reserved

The estimator answers "what is safe per model" (cap, gate); the dispatcher decides which model runs.
"""
import mmap
import os
import struct

MAGIC = b"DAPPCTL1"
ABI = 2
SIZE = 128
MAX_MODELS = 4
NAME_LEN = 8
_HEAD = struct.Struct("<8sIIQIHHQQII")     # 56 bytes
_ENTRY = struct.Struct("<8sII")            # 16 bytes
_TAIL = struct.Struct("<II")
assert _HEAD.size + MAX_MODELS * _ENTRY.size + _TAIL.size == SIZE
DEFAULT_NAME = "/aerial_dapp_ctrl"


def _name_bytes(name):
    b = str(name).encode()[:NAME_LEN]
    return b + b"\0" * (NAME_LEN - len(b))


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
                    raise ValueError("control block %s too small (ABI 1 block? the writer must be restarted)" % path)
                os.ftruncate(fd, SIZE)
            self.mm = mmap.mmap(fd, SIZE)
        finally:
            os.close(fd)
        if create:
            self.write(0, 0, 0, 0, 0, [])
        else:
            magic, abi = struct.unpack_from("<8sI", self.mm, 0)
            if magic != MAGIC or abi != ABI:
                raise ValueError("control block %s: bad magic/abi (%r, %d; want ABI %d)" % (path, magic, abi, ABI))

    def write(self, slot_id, sfn, slot, ts_ns, n_decisions, models):
        """models: iterable of (name, cap_pct, gate_slots), at most MAX_MODELS; the first entry is index 0."""
        entries = list(models)[:MAX_MODELS]
        mm = self.mm
        self.seq += 1                                   # odd: writing
        struct.pack_into("<Q", mm, 16, self.seq)
        _HEAD.pack_into(mm, 0, MAGIC, ABI, SIZE, self.seq, int(slot_id), int(sfn) & 0xFFFF, int(slot) & 0xFFFF,
                        int(ts_ns), int(n_decisions), len(entries), 0)
        off = _HEAD.size
        for i in range(MAX_MODELS):
            if i < len(entries):
                name, cap, gate = entries[i]
                _ENTRY.pack_into(mm, off, _name_bytes(name), int(cap), int(gate))
            else:
                _ENTRY.pack_into(mm, off, b"\0" * NAME_LEN, 0, 0)
            off += _ENTRY.size
        _TAIL.pack_into(mm, off, 0, 0)
        self.seq += 1                                   # even: consistent
        struct.pack_into("<Q", mm, 16, self.seq)

    def read(self):
        """Seqlock read; returns a dict or None if the writer was mid-update on every attempt."""
        for _ in range(8):
            s1 = struct.unpack_from("<Q", self.mm, 16)[0]
            if s1 & 1:
                continue
            h = _HEAD.unpack_from(self.mm, 0)
            n = min(h[9], MAX_MODELS)
            models = []
            off = _HEAD.size
            for i in range(n):
                name, cap, gate = _ENTRY.unpack_from(self.mm, off)
                models.append({"name": name.rstrip(b"\0").decode(errors="replace"), "cap_pct": cap, "gate_slots": gate})
                off += _ENTRY.size
            s2 = struct.unpack_from("<Q", self.mm, 16)[0]
            if s1 == s2:
                d = {"seq": h[3], "slot_id": h[4], "sfn": h[5], "slot": h[6], "ts_ns": h[7], "n_decisions": h[8],
                     "n_models": n, "models": models}
                # convenience: the first entry's values under the ABI-1 names
                if models:
                    d["cap_pct"] = models[0]["cap_pct"]
                    d["gate_slots"] = models[0]["gate_slots"]
                return d
        return None

    def entry(self, name):
        """The entry of one model from a read(), or None."""
        d = self.read()
        if not d:
            return None
        for m in d["models"]:
            if m["name"] == name:
                return m
        return None

    def close(self):
        self.mm.close()

    def unlink(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
