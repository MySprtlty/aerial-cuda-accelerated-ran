"""
Decision log record: one row per decision (per slot). Same columns in replay
and live mode so the two can be diffed. Columns:

  ts_ns       ring-clock time of the decision (SLOT_END of the newest confirmed slot)
  sfn, slot   the "now" slot the decision applies to; slot_id = sfn*slots_per_frame+slot
  t0_ns       T0 of the now slot (for joining with labels)
  n_cells, ul, dl
  {ch}_n, {ch}_rnti, {ch}_{kernel}     work summary of the now slot (all cells)
  base_{ch}   rule-a base time (us) of the now slot
  lat_{ch}    rule-b latency estimate (us) of the now slot at the chosen cap
  gate_slots, cap_pct, safe_k, need_slots, reason
  proc_us     wall time of base_times + decide for this slot (decode time excluded)
"""
import csv

from . import work_formulas as W

CHANNELS = W.CHANNELS
KERNELS = W.KERNELS

FIELDS = (["ts_ns", "sfn", "slot", "slot_id", "t0_ns", "n_cells", "ul", "dl"]
          + ["%s_n" % c for c in CHANNELS]
          + ["%s_rnti" % c for c in CHANNELS]
          + ["%s_%s" % (c, k) for c in CHANNELS for k in KERNELS[c]]
          + ["base_%s" % c for c in CHANNELS]
          + ["lat_%s" % c for c in CHANNELS]
          + ["gate_slots", "cap_pct", "safe_k", "need_slots", "reason", "proc_us"])

COMPARE_FIELDS = [f for f in FIELDS if f != "proc_us"]   # replay == live on these


def make_record(ctx, ts_ns, slot_id, base_total, lat, gate_slots, cap_pct, info, proc_us):
    r = {"ts_ns": ts_ns, "sfn": ctx.sfn, "slot": ctx.slot, "slot_id": slot_id, "t0_ns": ctx.t0_ns,
         "n_cells": len(ctx.cells), "ul": int(ctx.is_ul), "dl": int(ctx.is_dl)}
    for c in CHANNELS:
        a = ctx.ch[c]
        r["%s_n" % c] = a.count
        r["%s_rnti" % c] = len(a.rntis)
        for k in KERNELS[c]:
            r["%s_%s" % (c, k)] = a.total[k]
        r["base_%s" % c] = round(base_total[c], 1)
        r["lat_%s" % c] = round(lat[c], 1) if lat else 0
    r["gate_slots"] = gate_slots
    r["cap_pct"] = cap_pct
    r["safe_k"] = info.get("safe_k", 0)
    r["need_slots"] = round(info.get("need_slots", 0.0), 2)
    r["reason"] = info.get("reason", "")
    r["proc_us"] = round(proc_us, 1)
    return r


class CsvLog:
    def __init__(self, path):
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=FIELDS)
        self.w.writeheader()
        self.n = 0

    def write(self, rec):
        self.w.writerow(rec)
        self.n += 1

    def close(self):
        self.f.close()


def read_log(path):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            yield row
