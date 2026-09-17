"""
Loaders for rules.yaml, cells.yaml and tenant profiles. Pure Python + PyYAML.
Everything the estimator needs is read once into plain dicts/floats so the
decision path does no YAML or dict-of-dict lookups beyond what is unavoidable.
"""
import os

import yaml

from . import work_formulas as W

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RULES = os.path.join(HERE, "..", "rules.yaml")
DEFAULT_CELLS = os.path.join(HERE, "..", "cells.yaml")


class Rules:
    def __init__(self, path=DEFAULT_RULES):
        with open(path) as f:
            y = yaml.safe_load(f)
        self.path = path
        self.raw = y
        self.provisional = bool(y.get("provisional", True))
        self.slot_us = float(y["slot_us"])
        self.sm_total = int(y["sm_total"])
        self.caps = sorted(int(c) for c in y["caps_pct"])
        self.deadline_us = {c: float(v) for c, v in y["deadline_us"].items()}
        self.fixed_us = {c: float(v) for c, v in y.get("fixed_us", {}).items()}
        self.w = {}
        for c, ks in y["weights_us_per_work"].items():
            self.w[c] = {k: float(v) for k, v in ks.items()}
        for c in W.CHANNELS:
            if c not in self.w:
                raise ValueError("rules.yaml: no weights for channel %s" % c)
            for k in W.KERNELS[c]:
                if k not in self.w[c]:
                    raise ValueError("rules.yaml: no weight for %s.%s" % (c, k))
            if c not in self.deadline_us:
                raise ValueError("rules.yaml: no deadline for %s" % c)
        m = y["margin"]
        self.jitter_us = float(m["jitter_us"])
        g = y["gate"]
        self.known_slots = int(g["known_slots"])
        self.gate_window_s = float(g["window_s"])
        self.max_gate_slots = int(g["max_gate_slots"])
        cp = y["cap"]
        self.cap_window_s = float(cp["window_s"])
        self.cap_percentile = float(cp["percentile"])
        self.raise_after = int(cp["raise_after"])
        s = y.get("safety", {})
        self.stale_after_ms = float(s.get("stale_after_ms", 20))
        self.cell_scale_default = float(y.get("cell_scale_default", 1.0))
        lv = y.get("live", {})
        self.live_budget_us = float(lv.get("budget_us", 50))
        self.live_idle_sleep_us = float(lv.get("idle_sleep_us", 50))
        self.calibrate_min_r2 = float(y.get("calibrate", {}).get("min_r2", 0.8))


class Cells:
    def __init__(self, path=DEFAULT_CELLS):
        with open(path) as f:
            y = yaml.safe_load(f) or {}
        d = y.get("default", {})
        self.default_scale = float(d.get("scale", 1.0))
        self.scale = {}
        for cid, cfg in (y.get("cells") or {}).items():
            cfg = cfg or {}
            if "scale" in cfg:
                sc = float(cfg["scale"])
            elif "bandwidth_mhz" in cfg or "tx_ant" in cfg:
                bw = float(cfg.get("bandwidth_mhz", d.get("bandwidth_mhz", 100)))
                ant = float(cfg.get("tx_ant", d.get("tx_ant", 4)))
                sc = (bw / float(d.get("bandwidth_mhz", 100))) * (ant / float(d.get("tx_ant", 4)))
            else:
                sc = self.default_scale
            self.scale[int(cid)] = sc

    def scale_of(self, cell_id):
        return self.scale.get(cell_id, self.default_scale)


class Profile:
    """One tenant (YOLO variant). dur_ms and sm_fill are interpolated in cap."""

    def __init__(self, path):
        with open(path) as f:
            y = yaml.safe_load(f)
        self.path = path
        self.name = y["name"]
        self.framework = y.get("framework", "?")
        self.input_res = y.get("input_res")
        self.batches = [int(b) for b in y["batches"]]
        self.dur_ms = {int(b): {int(c): float(v) for c, v in d.items()} for b, d in y["dur_ms"].items()}
        self.sm_fill = {int(b): {int(c): float(v) for c, v in d.items()} for b, d in y.get("sm_fill", {}).items()}
        self.kernel_max_us = float(y.get("kernel_max_us", 100))
        self.status = y.get("status", {})

    @staticmethod
    def _interp(table, cap):
        if cap in table:
            return table[cap]
        ks = sorted(table)
        if cap <= ks[0]:
            return table[ks[0]]
        if cap >= ks[-1]:
            return table[ks[-1]]
        for a, b in zip(ks, ks[1:]):
            if a <= cap <= b:
                t = (cap - a) / float(b - a)
                return table[a] + t * (table[b] - table[a])
        return table[ks[-1]]

    def duration_ms(self, batch, cap_pct):
        if cap_pct <= 0:
            return float("inf")
        return self._interp(self.dur_ms[batch], cap_pct)

    def fill(self, batch, cap_pct):
        t = self.sm_fill.get(batch)
        if not t:
            return 1.0 if batch >= 8 else 0.5
        return self._interp(t, cap_pct)


def load_profiles(paths):
    return [Profile(p) for p in paths]
