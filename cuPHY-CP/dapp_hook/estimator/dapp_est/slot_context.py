"""
slot_context: fold the ring records of one (sfn, slot) across all cells into
one GPU-level context.

For every channel and kernel group it keeps
  total    = sum over all PDUs of all cells
  critical = max over single PDUs (the longest chain a slot has to wait for)
  count    = number of PDUs, and the number of distinct RNTIs
plus per-cell totals (for the static per-cell scale in the estimator) and the
slot bookkeeping from SLOT_END (t0, enqueued, cells reporting).

A context is closed by the SLOT_END record of the slot (written by the L1
after all cells' TTI requests were processed). UL_TTI/DL_TTI summaries are
kept too so a context can be built even when PDU export is off
(export_pdus: 0) -- then only the summary totals are available and
per-kernel work is estimated from them (flagged summary_only).
"""
from . import ringio as R
from . import work_formulas as W

KERNELS = W.KERNELS
CHANNELS = W.CHANNELS


class ChannelAgg:
    __slots__ = ("total", "critical", "count", "rntis")

    def __init__(self, kernels):
        self.total = {k: 0 for k in kernels}
        self.critical = {k: 0 for k in kernels}
        self.count = 0
        self.rntis = set()

    def add(self, work, rnti):
        for k, v in work.items():
            self.total[k] += v
            if v > self.critical[k]:
                self.critical[k] = v
        self.count += 1
        if rnti:
            self.rntis.add(rnti)


class SlotContext:
    __slots__ = ("sfn", "slot", "ch", "cells", "cell_work", "t0_ns", "tick_ns", "ts_first_ns", "ts_end_ns",
                 "enqueued", "is_ul", "is_dl", "summary_only", "n_records", "complete")

    def __init__(self, sfn, slot):
        self.sfn = sfn
        self.slot = slot
        self.ch = {c: ChannelAgg(KERNELS[c]) for c in CHANNELS}
        self.cells = set()
        self.cell_work = {}        # cell_id -> {channel: {kernel: total}}
        self.t0_ns = 0
        self.tick_ns = 0
        self.ts_first_ns = 0
        self.ts_end_ns = 0
        self.enqueued = None
        self.is_ul = False
        self.is_dl = False
        self.summary_only = False
        self.n_records = 0
        self.complete = False

    def key(self):
        return (self.sfn, self.slot)

    def n_cells(self):
        return len(self.cells)

    def _cell(self, cell, channel):
        d = self.cell_work.setdefault(cell, {})
        return d.setdefault(channel, {k: 0 for k in KERNELS[channel]})

    def add_pdu(self, rec):
        if rec.type == R.REC_UL_PDU:
            channel, work = W.work_ul_pdu(rec)
            rnti = rec.rnti if channel in ("PUSCH", "PUCCH", "SRS") else 0
        else:
            channel, work = W.work_dl_pdu(rec)
            rnti = rec.rnti if channel == "PDSCH" else 0
        if channel is None:
            return
        self.ch[channel].add(work, rnti)
        cw = self._cell(rec.cell_id, channel)
        for k, v in work.items():
            cw[k] += v
        self.cells.add(rec.cell_id)
        if channel in W.UL_CHANNELS:
            self.is_ul = True
        else:
            self.is_dl = True

    def add_summary(self, rec):
        self.cells.add(rec.cell_id)
        if rec.type == R.REC_UL_TTI:
            self.is_ul = self.is_ul or rec.num_pdus > 0
            if not self.n_records:
                self.summary_only = True
        else:
            self.is_dl = self.is_dl or rec.num_pdus > 0
            if not self.n_records:
                self.summary_only = True

    def add_slot_end(self, rec):
        self.t0_ns = rec.t0_ns
        self.tick_ns = rec.tick_original_ns
        self.ts_end_ns = rec.ts_ns
        self.enqueued = bool(rec.enqueued)
        self.is_ul = self.is_ul or bool(rec.is_ul)
        self.is_dl = self.is_dl or bool(rec.is_dl)
        self.complete = True

    def channel_totals(self):
        """{channel: {kernel: total}} for the whole GPU (all cells)."""
        return {c: dict(a.total) for c, a in self.ch.items()}

    def channel_criticals(self):
        return {c: dict(a.critical) for c, a in self.ch.items()}

    def summary(self):
        """Short dict for logs: counts and the dominant work of each channel."""
        out = {"sfn": self.sfn, "slot": self.slot, "n_cells": len(self.cells), "ul": int(self.is_ul), "dl": int(self.is_dl)}
        for c, a in self.ch.items():
            out[c + "_n"] = a.count
            out[c + "_rnti"] = len(a.rntis)
            for k in KERNELS[c]:
                out["%s_%s" % (c, k)] = a.total[k]
                out["%s_%s_max" % (c, k)] = a.critical[k]
        return out


class ContextBuilder:
    """Feed ring records in seq order; get closed SlotContexts back.

    close_on: "slot_end" (default) closes a context when its SLOT_END arrives.
    A context that never sees SLOT_END (slot dropped before enqueue, or L1
    emitted no SLOT_END because the slot had no channels) is flushed as
    incomplete when max_open newer slots have been opened after it.
    """

    def __init__(self, max_open=8):
        self.open = {}          # key -> SlotContext
        self.order = []         # keys in first-seen order
        self.max_open = max_open
        self.n_closed = 0
        self.n_incomplete = 0

    def _get(self, rec):
        key = (rec.sfn, rec.slot)
        ctx = self.open.get(key)
        if ctx is None:
            ctx = SlotContext(rec.sfn, rec.slot)
            ctx.ts_first_ns = rec.ts_ns
            self.open[key] = ctx
            self.order.append(key)
        return ctx

    def feed(self, rec):
        """Returns a list of contexts closed by this record (usually 0 or 1)."""
        out = []
        t = rec.type
        if t == R.REC_UL_PDU or t == R.REC_DL_PDU:
            ctx = self._get(rec)
            ctx.add_pdu(rec)
            ctx.n_records += 1
        elif t == R.REC_UL_TTI or t == R.REC_DL_TTI:
            ctx = self._get(rec)
            ctx.add_summary(rec)
        elif t == R.REC_SLOT_END:
            ctx = self._get(rec)
            ctx.add_slot_end(rec)
            self.open.pop(ctx.key(), None)
            self.order.remove(ctx.key())
            self.n_closed += 1
            out.append(ctx)
        else:
            return out
        # flush stale open contexts
        while len(self.order) > self.max_open:
            k = self.order.pop(0)
            stale = self.open.pop(k)
            self.n_incomplete += 1
            out.append(stale)
        return out

    def flush(self):
        out = [self.open.pop(k) for k in self.order]
        self.n_incomplete += len(out)
        self.order = []
        return out


def contexts_from_file(path, hdr_path=None):
    """Convenience: iterate closed contexts from a recorded file, in slot order."""
    b = ContextBuilder()
    for rec in R.RecordFile(path, hdr_path):
        for ctx in b.feed(rec):
            yield ctx
    for ctx in b.flush():
        yield ctx
