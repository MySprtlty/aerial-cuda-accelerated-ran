"""
Rule-based estimator. Interfaces are fixed; a learned predictor later replaces
the bodies, not the signatures:

    predict_latency_us(ctx, cap_pct, profile, rules, cells, batch=None) -> {channel: latency_upper_us}
    decide(ctx_now, ctx_future, profile, state, cells, ...) -> (gate_slots, cap_pct, info)

Rules a-g (numbers live in rules.yaml, reasoning in README.md):
  a. base_c = fixed_c + sum_k w_ck * work_ck * cell_scale   (total: all PDUs of all cells;
     critical: the largest single PDU per kernel = the chain the slot can never beat)
  b. sm_avail = 1 - cap/100 * sm_fill ; latency_c = max(base_crit_c, base_total_c / sm_avail)
  c. margin = profile.kernel_max_us + jitter_us  (one tenant kernel that cannot be preempted + timer jitter)
  d. gate: over the confirmed slots (now, +1, +2 ...) count consecutive safe slots k;
     if all are safe and the max context of the last W seconds is safe, k = max_gate_slots;
     gate_slots = k if k * slot_us >= dur_us[batch][cap] else 0
  e. cap: largest cap whose p-percentile context of the last T seconds is safe;
     raising needs N consecutive votes, lowering is immediate
  f. safety: no/incomplete/stale context -> gate 0, cap kept; no profile -> cap 0
  g. per-cell static scale from cells.yaml (applied in base_times)
"""
from collections import deque

from . import work_formulas as W

CHANNELS = W.CHANNELS


class SlidingMax:
    """Max of a time series over the trailing window; amortised O(1) per push (monotonic deque)."""
    __slots__ = ("window_ns", "dq")

    def __init__(self, window_ns):
        self.window_ns = window_ns
        self.dq = deque()            # (ts, value), values strictly decreasing

    def push(self, ts, v):
        dq = self.dq
        while dq and dq[-1][1] <= v:
            dq.pop()
        dq.append((ts, v))

    def get(self, now_ns):
        dq = self.dq
        lo = now_ns - self.window_ns
        while dq and dq[0][0] < lo:
            dq.popleft()
        return dq[0][1] if dq else 0.0


class State:
    """Mutable estimator state carried across decisions (one per tenant)."""

    def __init__(self, rules):
        self.rules = rules
        self.cap_pct = 0
        self.raise_votes = 0
        self.last_ctx_ts_ns = 0
        self.n_pushed = 0
        self.n_decisions = 0
        self.n_gate_closed = 0
        self.n_stale = 0
        wins = sorted({rules.gate_window_s, rules.cap_window_s})
        self.maxes = {}
        for w in wins:
            self.maxes[w] = ({c: SlidingMax(w * 1e9) for c in CHANNELS},
                             {c: SlidingMax(w * 1e9) for c in CHANNELS})
        # slow path, only kept when a percentile below 100 is configured
        self.history = deque() if rules.cap_percentile < 100 else None

    def push(self, ts_ns, base_total, base_crit):
        for mt, mc in self.maxes.values():
            for c in CHANNELS:
                mt[c].push(ts_ns, base_total[c])
                mc[c].push(ts_ns, base_crit[c])
        if self.history is not None:
            self.history.append((ts_ns, base_total, base_crit))
            lo = ts_ns - self.rules.cap_window_s * 1e9
            while self.history and self.history[0][0] < lo:
                self.history.popleft()
        self.last_ctx_ts_ns = ts_ns
        self.n_pushed += 1

    def window_max(self, seconds, now_ns):
        mt, mc = self.maxes[seconds]
        return ({c: mt[c].get(now_ns) for c in CHANNELS}, {c: mc[c].get(now_ns) for c in CHANNELS})

    def cap_context(self, now_ns):
        """Rule e input: the p-percentile context of the last T seconds."""
        p = self.rules.cap_percentile
        if p >= 100 or self.history is None:
            return self.window_max(self.rules.cap_window_s, now_ns)
        lo = now_ns - self.rules.cap_window_s * 1e9
        items = [h for h in self.history if h[0] >= lo]
        if not items:
            return None, None
        idx = min(len(items) - 1, max(0, int(round(p / 100.0 * (len(items) - 1)))))
        pt = {}
        pc = {}
        for c in CHANNELS:
            pt[c] = sorted(h[1][c] for h in items)[idx]
            pc[c] = sorted(h[2][c] for h in items)[idx]
        return pt, pc


def base_times(ctx, rules, cells):
    """Rules a + g. Returns (base_total{ch}, base_crit{ch}) in microseconds."""
    total = {}
    crit = {}
    w = rules.w
    fixed = rules.fixed_us
    for c in CHANNELS:
        agg = ctx.ch[c]
        if agg.count == 0:
            total[c] = 0.0
            crit[c] = 0.0
            continue
        wc = w[c]
        t = 0.0
        for cell, cw in ctx.cell_work.items():
            wk = cw.get(c)
            if not wk:
                continue
            s = cells.scale_of(cell)
            for k, v in wk.items():
                t += wc[k] * v * s
        cr = 0.0
        for k, v in agg.critical.items():
            cr += wc[k] * v
        f = fixed.get(c, 0.0)
        total[c] = f + t
        crit[c] = f + cr
    return total, crit


def latency_from_base(base_total, base_crit, cap_pct, sm_fill):
    """Rule b."""
    sm_avail = 1.0 - (cap_pct / 100.0) * sm_fill
    if sm_avail < 0.05:
        sm_avail = 0.05
    out = {}
    for c in CHANNELS:
        bt = base_total[c]
        if bt > 0:
            v = bt / sm_avail
            bc = base_crit[c]
            out[c] = v if v > bc else bc
        else:
            out[c] = 0.0
    return out


def predict_latency_us(ctx, cap_pct, profile, rules, cells, batch=None):
    """Upper-bound GPU latency per channel for this context if the tenant runs at cap_pct."""
    total, crit = base_times(ctx, rules, cells)
    b = batch or profile.batches[0]
    return latency_from_base(total, crit, cap_pct, profile.fill(b, cap_pct))


def is_safe(lat, margin, rules):
    """True if every active channel meets its deadline: latency + margin <= D_c."""
    D = rules.deadline_us
    for c in CHANNELS:
        v = lat[c]
        if v > 0 and v + margin > D[c]:
            return False
    return True


def decide(ctx_now, ctx_future, profile, state, cells, now_ns=None, batch=None, bases=None):
    """
    ctx_now:    SlotContext of the slot the tenant would overlap first (None if unknown)
    ctx_future: the following confirmed SlotContexts, in slot order
    bases:      optional {ctx.key(): (base_total, base_crit)} cache
    returns (gate_slots, cap_pct, info); info = {reason, lat_now, lat_win, need_slots, safe_k}
    """
    rules = state.rules
    state.n_decisions += 1
    info = {"reason": "", "lat_now": None, "lat_win": None, "need_slots": 0.0, "safe_k": 0}
    if profile is None:                                    # rule f
        state.cap_pct = 0
        info["reason"] = "no profile"
        return 0, 0, info
    b = batch or profile.batches[0]
    margin = profile.kernel_max_us + rules.jitter_us      # rule c

    if ctx_now is None or not ctx_now.complete:            # rule f
        state.n_gate_closed += 1
        state.n_stale += 1
        info["reason"] = "missing context"
        return 0, state.cap_pct, info
    t_now = now_ns if now_ns is not None else ctx_now.ts_end_ns
    if state.last_ctx_ts_ns and (t_now - state.last_ctx_ts_ns) > rules.stale_after_ms * 1e6:
        state.n_gate_closed += 1
        state.n_stale += 1
        info["reason"] = "ring stale"
        return 0, state.cap_pct, info

    # ---- rule e: cap from the percentile context of the last T seconds ----
    pt, pc = state.cap_context(t_now)
    cand = 0
    if pt is not None:
        for cap in rules.caps:
            if cap == 0:
                continue
            if is_safe(latency_from_base(pt, pc, cap, profile.fill(b, cap)), margin, rules):
                cand = cap
        info["lat_win"] = latency_from_base(pt, pc, cand, profile.fill(b, cand if cand else rules.caps[-1]))
    if cand < state.cap_pct:
        state.cap_pct = cand                               # lower immediately
        state.raise_votes = 0
    elif cand > state.cap_pct:
        state.raise_votes += 1
        if state.raise_votes >= rules.raise_after:
            state.cap_pct = cand
            state.raise_votes = 0
    else:
        state.raise_votes = 0
    cap = state.cap_pct
    if cap == 0:
        state.n_gate_closed += 1
        if cand > 0:
            info["reason"] = "cap pending hysteresis (%d/%d)" % (state.raise_votes, rules.raise_after)
        else:
            info["reason"] = "no cap satisfies deadlines"
        return 0, 0, info
    fill = profile.fill(b, cap)

    # ---- rule d: gate over the confirmed slots, then the window max ----
    k = 0
    confirmed = [ctx_now]
    for c in ctx_future:
        if len(confirmed) >= rules.known_slots:
            break
        if c is not None:
            confirmed.append(c)
    all_safe = True
    for c in confirmed:
        if not c.complete:
            all_safe = False
            break
        cached = bases.get(c.key()) if bases is not None else None
        bt, bc = cached if cached is not None else base_times(c, rules, cells)
        lat = latency_from_base(bt, bc, cap, fill)
        if c is ctx_now:
            info["lat_now"] = lat
        if is_safe(lat, margin, rules):
            k += 1
        else:
            all_safe = False
            break
    if all_safe:
        mt, mc = state.window_max(rules.gate_window_s, t_now)
        if is_safe(latency_from_base(mt, mc, cap, fill), margin, rules):
            k = rules.max_gate_slots
    need_slots = profile.duration_ms(b, cap) * 1000.0 / rules.slot_us
    info["need_slots"] = need_slots
    info["safe_k"] = k
    if k < need_slots:
        state.n_gate_closed += 1
        info["reason"] = "not enough safe slots (%d < %.1f)" % (k, need_slots)
        return 0, cap, info
    info["reason"] = "ok"
    return min(k, rules.max_gate_slots), cap, info
