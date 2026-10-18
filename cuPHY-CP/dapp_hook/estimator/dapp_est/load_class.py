"""Load classifier for the table-based policy (H2, 2026-10-16).

Input: the hook ring's UL_TTI / DL_TTI summaries (one per cell per slot). The feature is the PUSCH PRB count summed over
all cells per slot, averaged over a sliding window of `window_slots` UL slots (only slots that carry PUSCH count: the
D/S slots of the TDD pattern contribute nothing, so the feature is "PRB per UL slot"). Two classes, thresholds in the
SafeCap table file:

    classes:
      average: {pusch_prb_per_ul_slot_max: X}      # <= X  -> average   (light)
      peak:    {pusch_prb_per_ul_slot_min: Y}      # >= Y  -> peak      (heavy)

Hysteresis (fixed decision): heavy is taken immediately (the first window at or above Y), light only after the feature
has stayed at or below X for `light_hold_s` seconds (default 2 s). In between (X, Y) the previous class is kept.

The table file also carries, per class and model, the cap and whether the model is allowed (cap 0 = not allowed, which the
dispatcher already treats as "not a candidate"), plus the gate_slots used for every entry:

    gate_slots: 40
    table:
      average: {26x: 61, 26s: 100, 26n: 100}
      peak:    {26x: 0,  26s: 43,  26n: 43}
"""
import collections

import yaml

MODELS = ("26x", "26s", "26n")


class Table:
    def __init__(self, path):
        with open(path) as f:
            y = yaml.safe_load(f)
        self.path = path
        self.classes = y["classes"]
        self.light_max = float(self.classes["average"]["pusch_prb_per_ul_slot_max"])
        self.heavy_min = float(self.classes["peak"]["pusch_prb_per_ul_slot_min"])
        if self.heavy_min < self.light_max:
            raise ValueError("table %s: peak min %.0f < average max %.0f" % (path, self.heavy_min, self.light_max))
        self.gate_slots = int(y.get("gate_slots", 40))
        # models = the union of the names in the table rows (accuracy order 26x > 26l > 26m > 26s > 26n), at most 4 (control block ABI 2)
        order = ("26x", "26l", "26m", "26s", "26n")
        names = {m for v in y["table"].values() for m in v}
        self.models = tuple(m for m in order if m in names) + tuple(sorted(names - set(order)))
        if len(self.models) > 4:
            raise ValueError("table %s: %d models, the control block holds 4" % (path, len(self.models)))
        self.table = {c: {m: int(v.get(m, 0)) for m in self.models} for c, v in y["table"].items()}
        for c in ("average", "peak"):
            if c not in self.table:
                raise ValueError("table %s: no row for class %s" % (path, c))
        self.window_slots = int(y.get("window_ul_slots", 60))
        self.light_hold_s = float(y.get("light_hold_s", 2.0))
        self.start_class = y.get("start_class", "peak")   # before the first full window: the safe side

    def entries(self, cls):
        """(name, cap_pct, gate_slots) tuples for the control block, cap 0 = model not allowed."""
        return [(m, self.table[cls][m], self.gate_slots if self.table[cls][m] > 0 else 0) for m in self.models]


class Classifier:
    """Feed one UL_TTI summary at a time (or a per-slot total via feed_slot); .cls is the current class."""

    def __init__(self, table):
        self.t = table
        self.win = collections.deque(maxlen=table.window_slots)
        self.cls = table.start_class
        self.below_since_ns = None      # first time the feature was <= light_max without interruption
        self.cur_key = None             # (sfn, slot) being accumulated
        self.cur_prb = 0
        self.cur_ts = 0
        self.n_switch = 0
        self.history = []               # (ts_ns, feature, cls) per UL slot (replay / debug)

    def feed(self, rec):
        """rec: ringio UL_TTI record (needs .sfn, .slot, .ts_ns, .tot_pusch_prb). Returns the class after this record."""
        key = (rec.sfn, rec.slot)
        if key != self.cur_key:
            if self.cur_key is not None:
                self._close_slot()
            self.cur_key, self.cur_prb, self.cur_ts = key, 0, rec.ts_ns
        self.cur_prb += rec.tot_pusch_prb
        return self.cls

    def feed_slot(self, ts_ns, prb_total):
        self.cur_key, self.cur_prb, self.cur_ts = ("x", ts_ns), prb_total, ts_ns
        self._close_slot()
        self.cur_key = None
        return self.cls

    def _close_slot(self):
        if self.cur_prb <= 0:
            return                      # a UL slot without PUSCH (PUCCH/PRACH only) is not a load sample
        self.win.append(self.cur_prb)
        feat = sum(self.win) / len(self.win)
        full = len(self.win) == self.win.maxlen
        ts = self.cur_ts
        if full and feat >= self.t.heavy_min:
            if self.cls != "peak":
                self.cls = "peak"; self.n_switch += 1
            self.below_since_ns = None
        elif full and feat <= self.t.light_max:
            if self.below_since_ns is None:
                self.below_since_ns = ts
            if self.cls != "average" and ts - self.below_since_ns >= self.t.light_hold_s * 1e9:
                self.cls = "average"; self.n_switch += 1
        else:
            self.below_since_ns = None
        self.history.append((ts, feat, self.cls))

    def finish(self):
        if self.cur_key is not None:
            self._close_slot(); self.cur_key = None
