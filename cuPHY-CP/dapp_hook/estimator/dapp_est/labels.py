"""
Label interface for evaluate.py. Three label kinds, all keyed by (sfn, slot, cell, channel):

  IND_SENT       (sfn, slot, cell, channel, ts_ns)   the L1 sent the channel's indication
                 (CRC/RX_DATA/UCI/RACH/SRS.indication, or the DL slot's C/U-plane send).
                 Source: csv written by an L2-adapter hook (not implemented in cuPHY here).
  SLOT_VIOLATION (sfn, slot, cell, reason, ts_ns, side)   parsed from the Aerial nvlog text:
                 ul_order_timeout, pusch_wait_timeout, work_cancel, error_ind_late, nvlog_error
  CHAN_DONE      (sfn, slot, cell, channel, start_ns, done_ns, source)   GPU/CPU completion of one
                 channel (label_points.md). Adapter only: loads a csv; the producer is future work.

Clock: IND_SENT/CHAN_DONE timestamps must be on the same clock as the ring's
ts_ns/t0_ns (CLOCK_REALTIME ns). nvlog lines only carry wall time of day;
SLOT_VIOLATION.ts_ns is nanoseconds since midnight and is used for ordering only.
"""
import csv
import re
from collections import namedtuple

IndSent = namedtuple("IndSent", "sfn slot cell channel ts_ns")
SlotViolation = namedtuple("SlotViolation", "sfn slot cell reason ts_ns side")
ChanDone = namedtuple("ChanDone", "sfn slot cell channel start_ns done_ns source")

UL_CHANNELS = ("PUSCH", "PUCCH", "PRACH", "SRS")
DL_CHANNELS = ("PDSCH", "PDCCH", "SSB", "CSIRS")

# nvlog line: "HH:MM:SS.ffffff LVL PID TID [TAG] message"
_NVLOG = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d{6})\s+(\w+)\s+(\d+)\s+(\d+)\s+\[([^\]]+)\]\s*(.*)$")
_SFN_SLOT = re.compile(r"SFN\s*(\d+)\.(\d+)")
_SFN_SLOT2 = re.compile(r"sfn\s*[=:]?\s*(\d+)\D{1,6}slot\s*[=:]?\s*(\d+)", re.I)
_CELL = re.compile(r"cell(?:\s*index|\s*id|_id)?\s*[=:]?\s*(\d+)", re.I)

# (reason, regex on the message, side, channel hint)
VIOLATION_PATTERNS = (
    ("ul_order_timeout", re.compile(r"Order kernel timeout error"), "UL", None),
    ("srs_order_timeout", re.compile(r"SRS Order kernel timeout"), "UL", "SRS"),
    ("pusch_wait_timeout", re.compile(r"PUSCH (?:Pre|Post) Early Harq Wait kernel timeout"), "UL", "PUSCH"),
    ("work_cancel", re.compile(r"work[ -]?cancel|WorkCancel|task cancel", re.I), "UL", "PUSCH"),
    ("ulc_task_timeout", re.compile(r"timeout waiting for ULC Tasks"), "UL", None),
    ("error_ind", re.compile(r"Err\.ind for SFN|ERROR\.indication|MSG_LATE|LATE_SLOT|late slot|too late", re.I), "ANY", None),
)
_ERR_CODE = re.compile(r"err_code=(0x[0-9A-Fa-f]+)")
ERROR_LEVELS = ("ERR", "ERROR", "E", "FATAL", "CRIT")
_HINT = {r: h for r, _, _, h in VIOLATION_PATTERNS}


def _ts_of_day_ns(h, m, s, us):
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000000000 + int(us) * 1000


def classify_line(line):
    """Returns a SlotViolation for a violation/error line, else None."""
    m = _NVLOG.match(line.rstrip("\n"))
    if not m:
        return None
    ts = _ts_of_day_ns(m.group(1), m.group(2), m.group(3), m.group(4))
    level = m.group(5)
    msg = m.group(9)
    reason = None
    side = "ANY"
    for r, pat, sd, _ in VIOLATION_PATTERNS:
        if pat.search(msg):
            reason, side = r, sd
            break
    if reason is None:
        if level.upper() in ERROR_LEVELS:
            reason = "nvlog_error"
        else:
            return None
    if reason == "error_ind":
        me = _ERR_CODE.search(msg)
        if me:
            reason = "error_ind_" + me.group(1).lower()
    sfn = slot = -1
    mm = _SFN_SLOT.search(msg) or _SFN_SLOT2.search(msg)
    if mm:
        sfn, slot = int(mm.group(1)), int(mm.group(2))
    cell = -1
    mc = _CELL.search(msg)
    if mc:
        cell = int(mc.group(1))
    return SlotViolation(sfn, slot, cell, reason, ts, side)


def parse_nvlog(path):
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            v = classify_line(line)
            if v is not None:
                out.append(v)
    return out


def load_ind_sent(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append(IndSent(int(r["sfn"]), int(r["slot"]), int(r["cell"]), r["channel"].upper(), int(r["ts_ns"])))
    return out


def load_violations(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append(SlotViolation(int(r["sfn"]), int(r["slot"]), int(r.get("cell", -1) or -1), r["reason"],
                                     int(r.get("ts_ns", 0) or 0), r.get("side", "ANY") or "ANY"))
    return out


def load_chan_done(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append(ChanDone(int(r["sfn"]), int(r["slot"]), int(r["cell"]), r["channel"].upper(),
                                int(r["start_ns"]), int(r["done_ns"]), r.get("source", "")))
    return out


def write_violations(path, viols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sfn", "slot", "cell", "reason", "ts_ns", "side"])
        for v in viols:
            w.writerow(list(v))


class LabelSet:
    """Everything joined on (sfn, slot); per channel lookups for evaluate.py."""

    def __init__(self):
        self.ind = {}          # (sfn, slot, cell, channel) -> ts_ns
        self.done = {}         # (sfn, slot, cell, channel) -> (start_ns, done_ns)
        self.viol = {}         # (sfn, slot) -> [SlotViolation]
        self.unattributed = []  # violations without sfn/slot

    def add_ind(self, items):
        for i in items:
            self.ind[(i.sfn, i.slot, i.cell, i.channel)] = i.ts_ns

    def add_done(self, items):
        for d in items:
            self.done[(d.sfn, d.slot, d.cell, d.channel)] = (d.start_ns, d.done_ns)

    def add_violations(self, items):
        for v in items:
            if v.sfn < 0:
                self.unattributed.append(v)
            else:
                self.viol.setdefault((v.sfn, v.slot), []).append(v)

    def channels_with_labels(self):
        chs = set(k[3] for k in self.ind) | set(k[3] for k in self.done)
        for vs in self.viol.values():
            for v in vs:
                if v.side == "UL":
                    chs.update(UL_CHANNELS)
                elif v.side == "DL":
                    chs.update(DL_CHANNELS)
                else:
                    chs.update(UL_CHANNELS + DL_CHANNELS)
        return chs

    def violations_for(self, sfn, slot, channel):
        out = []
        for v in self.viol.get((sfn, slot), ()):
            hint = _HINT.get(v.reason)
            if hint and hint != channel:
                continue
            if v.side == "UL" and channel not in UL_CHANNELS:
                continue
            if v.side == "DL" and channel not in DL_CHANNELS:
                continue
            out.append(v)
        return out
