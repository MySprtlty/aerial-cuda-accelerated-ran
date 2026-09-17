"""
The deterministic core shared by replay.py and live.py: ring records in,
decisions out. Both modes feed the same records in the same order, so the
decision logs are identical (tests/test_pipeline.py checks this).

Timeline: L2 sends FAPI for slot s at about T0(s) - slot_advance slots, and
the L1 writes SLOT_END(s) when it has enqueued the slot. So when SLOT_END(s)
arrives, slots s-2, s-1 and s are confirmed but not yet on the GPU (DL of s
starts right away, UL of s starts ~2 ms later). A decision is made at every
SLOT_END; "now" is the oldest of the last known_slots confirmed slots and
the newer ones are its future. The tenant reads the newest control block.
"""
import time

from . import decision_log as L
from . import estimator as E
from . import ringio as R
from . import slot_context as S


class Runner:
    def __init__(self, rules, cells, profile, batch=None, mu=1, ctrl=None, log=None):
        self.rules = rules
        self.cells = cells
        self.profile = profile
        self.batch = batch or (profile.batches[0] if profile else 0)
        self.slots_per_frame = 10 << mu
        self.ctrl = ctrl
        self.log = log
        self.reset()

    def reset(self):
        self.builder = S.ContextBuilder()
        self.state = E.State(self.rules)
        self.recent = []
        self.bases = {}
        self.n_decisions = 0
        self.n_records = 0
        self.last_log_us = 0.0     # time spent building/writing the decision record (debug path)

    def feed(self, rec):
        """One ring record. Returns the list of decision records it produced (0 or 1 normally)."""
        self.n_records += 1
        out = []
        for ctx in self.builder.feed(rec):
            d = self.on_context(ctx)
            if d is not None:
                out.append(d)
        return out

    def flush(self):
        out = []
        for ctx in self.builder.flush():
            d = self.on_context(ctx)
            if d is not None:
                out.append(d)
        return out

    def on_context(self, ctx):
        t0 = time.perf_counter_ns()
        bt, bc = E.base_times(ctx, self.rules, self.cells)
        self.bases[ctx.key()] = (bt, bc)
        if ctx.complete:
            self.state.push(ctx.ts_end_ns, bt, bc)
        self.recent.append(ctx)
        if len(self.recent) < self.rules.known_slots:
            return None
        while len(self.recent) > self.rules.known_slots:
            old = self.recent.pop(0)
            self.bases.pop(old.key(), None)
        ctx_now = self.recent[0]
        future = self.recent[1:]
        now_ns = ctx.ts_end_ns or ctx.ts_first_ns
        gate, cap, info = E.decide(ctx_now, future, self.profile, self.state, self.cells,
                                   now_ns=now_ns, batch=self.batch, bases=self.bases)
        self.n_decisions += 1
        slot_id = ctx_now.sfn * self.slots_per_frame + ctx_now.slot
        if self.ctrl is not None:
            self.ctrl.write(gate, cap, slot_id, ctx_now.sfn, ctx_now.slot, now_ns, self.n_decisions, self.batch)
        t1 = time.perf_counter_ns()
        proc_us = (t1 - t0) / 1000.0
        base_now = self.bases.get(ctx_now.key(), (bt, bc))[0]
        rec = L.make_record(ctx_now, now_ns, slot_id, base_now, info.get("lat_now"), gate, cap, info, proc_us)
        if self.log is not None:
            self.log.write(rec)
        self.last_log_us = (time.perf_counter_ns() - t1) / 1000.0
        return rec


def replay_file(path, runner, hdr_path=None, max_slots=None):
    """Feed a recorded ring file through the runner. Returns the decision records."""
    out = []
    for rec in R.RecordFile(path, hdr_path):
        out.extend(runner.feed(rec))
        if max_slots and len(out) >= max_slots:
            return out
    out.extend(runner.flush())
    return out
