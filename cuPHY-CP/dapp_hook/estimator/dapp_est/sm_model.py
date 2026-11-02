"""
sm_model: per-channel GPU duration as a function of the SM partition and the smallest partition that meets the
channel's budget (sm_model.yaml). Pure Python, no numpy: usable in the live loop.

    T_c(ctx, s) = fixed_c + sum_k work_ck * (w_ins_ck + w_par_ck * s_ref_c / s)            (duration model)
    end_c(ctx, s) = start_c[pos] + queue + sum_k work_ck * (w_ins_ck + w_par_ck * s_ref_c / s)   (completion model, after t0)
    s_min_c(ctx) = smallest s in {g, 2g, ..., s_max_c} with  T_c + margin_c <= budget_c  (duration)
                                                       or  end_c + margin_c <= complete_by_c  (completion, budget_kind)

work_ck comes from a SlotContext (channel_totals(): all cells of the slot). Channels of class "dynamic" (PUSCH, PDSCH)
get s_min per slot; "fixed" channels share one partition (s_fixed) for every slot and only their duration at that
partition is checked; "off" channels are skipped. The UL burst check queues a U slot's pipeline behind the previous U
slot's when that one is still running at the measured start offset (ul_start_offset_us), so consecutive U slots that
share a partition are not sized as if they ran alone.
"""
import os

import yaml

from . import work_formulas as W

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "sm_model.yaml")


class Channel:
    __slots__ = ("name", "cls", "context", "s_ref", "s_min", "s_max", "s_fixed", "fixed_us", "budget_us", "margin_us", "w_ins", "w_par",
                 "complete_by_us", "start_us", "budget_kind")

    def __init__(self, name, d):
        self.name = name
        self.cls = str(d.get("class", "dynamic"))
        self.context = str(d.get("context", name))
        self.s_ref = float(d.get("s_ref", 100))
        self.s_min = int(d.get("s_min", 8))
        self.s_max = int(d.get("s_max", d.get("s_fixed", 132) or 132))
        self.s_fixed = int(d.get("s_fixed", 0) or 0)
        self.fixed_us = float(d.get("fixed_us", 0.0))
        self.budget_us = float(d.get("budget_us", 1e9))
        self.margin_us = float(d.get("margin_us", 0.0))
        self.w_ins = {k: float(v) for k, v in (d.get("w_ins") or {}).items()}
        self.w_par = {k: float(v) for k, v in (d.get("w_par") or {}).items()}
        self.complete_by_us = float(d["complete_by_us"]) if d.get("complete_by_us") is not None else None
        self.start_us = {int(k): float(v) for k, v in (d.get("start_us") or {}).items()}   # per TDD position: pipeline start (+ waits) after t0
        self.budget_kind = d.get("budget_kind")      # per-channel override of the global budget_kind

    def work_us(self, work, s):
        """SM-dependent compute part only (no fixed_us)."""
        if not work:
            return 0.0
        s = float(max(1, s))
        t = 0.0
        for k, v in work.items():
            if v:
                t += v * (self.w_ins.get(k, 0.0) + self.w_par.get(k, 0.0) * self.s_ref / s)
        return t

    def start_for(self, slot):
        """start_us keyed by slot-in-frame (slot % 20, refit) or by TDD position (slot % 10)."""
        v = self.start_us.get(slot % 20)
        return v if v is not None else self.start_us.get(slot % 10, 0.0)

    def end_us(self, work, s, slot, queue_us=0.0):
        """Predicted completion after t0 (completion model): start_us[slot] + queue + work_us(s)."""
        return self.start_for(slot) + queue_us + self.work_us(work, s)

    def duration_us(self, work, s):
        """GPU run duration (us) of this channel for work {kernel: units} on a partition of s SMs."""
        if not work:
            return 0.0
        s = float(max(1, s))
        t = self.fixed_us
        for k, v in work.items():
            if v:
                t += v * (self.w_ins.get(k, 0.0) + self.w_par.get(k, 0.0) * self.s_ref / s)
        return t


class SmModel:
    def __init__(self, path=DEFAULT_MODEL):
        with open(path) as f:
            y = yaml.safe_load(f)
        self.path = path
        self.raw = y
        self.provisional = bool(y.get("provisional", True))
        self.sm_total = int(y.get("sm_total", 132))
        self.g = int(y.get("granularity", 8))
        self.headroom = int(y.get("headroom_sm", 0))
        self.budget_kind = str(y.get("budget_kind", "duration"))
        self.ul_positions = [int(p) for p in y.get("ul_positions", [7, 8, 9])]
        self.ul_start = {int(k): float(v) for k, v in (y.get("ul_start_offset_us") or {}).items()}
        self.channels = {c: Channel(c, d) for c, d in y["channels"].items()}
        for c in W.CHANNELS:
            if c not in self.channels:
                raise ValueError("sm_model.yaml: no entry for channel %s" % c)
        self.contexts_fixed = {k: int(v) for k, v in (y.get("contexts_fixed") or {}).items()}
        lay = y.get("layout") or {}
        self.ul_set = list(lay.get("ul_set", ["PUSCH", "PUCCH", "PRACH"]))
        self.dl_set = list(lay.get("dl_set", ["PDSCH", "DL_CTRL", "COMMS"]))
        self.separate = list(lay.get("separate", ["ORDER"]))

    # ---- sizing ----
    def quantize_up(self, s):
        g = self.g
        return max(g, ((int(s) + g - 1) // g) * g)

    def _limit(self, ch, slot, extra_us):
        """Returns (limit_us, timer): timer(work, s) is the quantity that must stay <= limit_us. slot = slot-in-frame."""
        kind = ch.budget_kind or self.budget_kind
        if kind == "completion" and ch.complete_by_us is not None and ch.start_us:
            return ch.complete_by_us - ch.margin_us, (lambda work, s: ch.end_us(work, s, slot, extra_us))
        return ch.budget_us - ch.margin_us - extra_us, (lambda work, s: ch.duration_us(work, s))

    def min_sm(self, channel, work, extra_us=0.0, slot=0):
        """(s_min, timed value at s_min, feasible). extra_us = queueing wait of this slot; slot = slot-in-frame (completion model)."""
        ch = self.channels[channel]
        if not work:
            return 0, 0.0, True
        limit, timer = self._limit(ch, slot, extra_us)
        s = self.quantize_up(ch.s_min)
        while s <= ch.s_max:
            t = timer(work, s)
            if t <= limit:
                return s, t, True
            s += self.g
        return ch.s_max, timer(work, ch.s_max), False

    def plan(self, ctx, prev_ul=None):
        """Size one closed SlotContext. prev_ul = (end_rel_us, t0_ns) of the previous U slot's PUSCH (for the burst check).

        Returns a dict: per channel s (partition), t (predicted duration at s), ok; footprint, tenant_sm; and the burst
        bookkeeping of this slot (pusch_start_rel_us, pusch_end_rel_us, queue_us) when it is a U slot.
        """
        totals = ctx.channel_totals()
        out = {"sfn": ctx.sfn, "slot": ctx.slot, "pos": ctx.slot % 10, "t0_ns": ctx.t0_ns}
        queue_us = 0.0
        pos = ctx.slot % 10
        if "PUSCH" in self.channels and totals["PUSCH"] and any(totals["PUSCH"].values()):
            chp = self.channels["PUSCH"]
            start = chp.start_for(ctx.slot) if (chp.budget_kind or self.budget_kind) == "completion" else self.ul_start.get(pos, 300.0)
            if prev_ul is not None and prev_ul[1] and ctx.t0_ns:
                prev_end_abs = prev_ul[1] + prev_ul[0] * 1e3
                start_abs = ctx.t0_ns + start * 1e3
                if prev_end_abs > start_abs:
                    queue_us = (prev_end_abs - start_abs) / 1e3
        for name, ch in self.channels.items():
            work = totals.get(name) or {}
            active = any(work.values())
            if ch.cls == "off" or not active:
                out[name + "_s"] = ch.s_fixed if ch.cls == "fixed" else 0
                out[name + "_t"] = 0.0
                out[name + "_ok"] = 1
                continue
            extra = queue_us if name == "PUSCH" else 0.0
            limit, timer = self._limit(ch, ctx.slot, extra)
            if ch.cls == "fixed":
                t = timer(work, ch.s_fixed)
                out[name + "_s"] = ch.s_fixed
                out[name + "_t"] = t
                out[name + "_ok"] = int(t <= limit)
            else:
                s, t, ok = self.min_sm(name, work, extra, ctx.slot)
                out[name + "_s"] = s
                out[name + "_t"] = t
                out[name + "_ok"] = int(ok)
        # burst bookkeeping (PUSCH)
        if out.get("PUSCH_t", 0.0) > 0:
            chp = self.channels["PUSCH"]
            kind = chp.budget_kind or self.budget_kind
            start = chp.start_for(ctx.slot) if kind == "completion" else self.ul_start.get(pos, 300.0)
            out["pusch_start_rel_us"] = start
            out["pusch_queue_us"] = queue_us
            out["pusch_end_rel_us"] = out["PUSCH_t"] if kind == "completion" else start + queue_us + out["PUSCH_t"]
        # footprint: UL set = max(PUSCH, PUCCH + PRACH riding inside it), DL set = max(PDSCH, DL_CTRL + COMMS), + separate
        ul = max(out.get("PUSCH_s", 0), self._ctx_sm("PUCCH", out) + self._ctx_sm("PRACH", out))
        dl = max(out.get("PDSCH_s", 0), self._ctx_sm("DL_CTRL", out) + self.contexts_fixed.get("COMMS", 0))
        sep = sum(self.contexts_fixed.get(c, 0) for c in self.separate)
        foot = ul + dl + sep + self.headroom
        out["footprint_sm"] = foot
        out["tenant_sm"] = max(0, (self.sm_total - foot) // self.g * self.g)
        return out

    def _ctx_sm(self, context, out):
        """SMs of a fixed green context = max s_fixed of the channels that share it (0 if none active in this slot)."""
        best = 0
        for name, ch in self.channels.items():
            if ch.context == context and ch.cls == "fixed" and out.get(name + "_t", 0.0) > 0:
                best = max(best, ch.s_fixed)
        return best


def plan_fields(model):
    f = ["sfn", "slot", "pos", "t0_ns"]
    for name in model.channels:
        f += [name + "_s", name + "_t", name + "_ok"]
    f += ["pusch_start_rel_us", "pusch_queue_us", "pusch_end_rel_us", "footprint_sm", "tenant_sm"]
    return f
