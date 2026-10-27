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

Several tenant profiles can be decided side by side (one State each, the same
slot context): the control block (ABI 2) then carries one {name, cap, gate}
entry per profile and the decision log one row per (slot, profile). The
estimator only answers what is safe per model; which model runs is the
dispatcher's choice.
"""
import time

from . import decision_log as L
from . import estimator as E
from . import ringio as R
from . import slot_context as S


class Runner:
    def __init__(self, rules, cells, profile, batch=None, mu=1, ctrl=None, log=None, ctrl_wall_clock=False):
        """profile: one Profile or a list of Profiles (decided side by side).
        ctrl_wall_clock: stamp the control block with time.time_ns() instead of the ring clock
        (replay --realtime, so a reader's staleness check works on a recorded ring)."""
        self.rules = rules
        self.cells = cells
        self.profiles = list(profile) if isinstance(profile, (list, tuple)) else [profile]
        self.batches = [batch or (p.batches[0] if p else 0) for p in self.profiles]
        self.profile = self.profiles[0]          # backwards compatible single-profile view
        self.batch = self.batches[0]
        self.slots_per_frame = 10 << mu
        self.ctrl = ctrl
        self.ctrl_wall_clock = ctrl_wall_clock
        self.log = log
        self.reset()

    def reset(self):
        self.builder = S.ContextBuilder()
        self.states = [E.State(self.rules) for _ in self.profiles]
        self.state = self.states[0]
        self.recent = []
        self.bases = {}
        self.n_decisions = 0
        self.n_records = 0
        self.last_log_us = 0.0     # time spent building/writing the decision records (debug path)

    def feed(self, rec):
        """One ring record. Returns the decision records it produced (0 or one per profile normally)."""
        self.n_records += 1
        out = []
        for ctx in self.builder.feed(rec):
            out.extend(self.on_context(ctx))
        return out

    def flush(self):
        out = []
        for ctx in self.builder.flush():
            out.extend(self.on_context(ctx))
        return out

    def on_context(self, ctx):
        t0 = time.perf_counter_ns()
        bt, bc = E.base_times(ctx, self.rules, self.cells)
        self.bases[ctx.key()] = (bt, bc)
        if ctx.complete:
            for st in self.states:
                st.push(ctx.ts_end_ns, bt, bc)
        self.recent.append(ctx)
        if len(self.recent) < self.rules.known_slots:
            return []
        while len(self.recent) > self.rules.known_slots:
            old = self.recent.pop(0)
            self.bases.pop(old.key(), None)
        ctx_now = self.recent[0]
        future = self.recent[1:]
        now_ns = ctx.ts_end_ns or ctx.ts_first_ns
        results = []
        for prof, st, b in zip(self.profiles, self.states, self.batches):
            gate, cap, info = E.decide(ctx_now, future, prof, st, self.cells, now_ns=now_ns, batch=b, bases=self.bases)
            results.append((prof, gate, cap, info))
        self.n_decisions += 1
        slot_id = ctx_now.sfn * self.slots_per_frame + ctx_now.slot
        if self.ctrl is not None:
            ts = time.time_ns() if self.ctrl_wall_clock else now_ns
            self.ctrl.write(slot_id, ctx_now.sfn, ctx_now.slot, ts, self.n_decisions,
                            [(prof.ctrl_name, cap, gate) for prof, gate, cap, _ in results])
        t1 = time.perf_counter_ns()
        proc_us = (t1 - t0) / 1000.0            # all profiles together
        base_now = self.bases.get(ctx_now.key(), (bt, bc))[0]
        recs = []
        for prof, gate, cap, info in results:
            rec = L.make_record(ctx_now, now_ns, slot_id, base_now, info.get("lat_now"), gate, cap, info, proc_us,
                                profile=prof.name)
            if self.log is not None:
                self.log.write(rec)
            recs.append(rec)
        self.last_log_us = (time.perf_counter_ns() - t1) / 1000.0
        return recs


def replay_file(path, runner, hdr_path=None, max_slots=None):
    """Feed a recorded ring file through the runner. Returns the decision records."""
    out = []
    for rec in R.RecordFile(path, hdr_path):
        out.extend(runner.feed(rec))
        if max_slots and len(out) >= max_slots:
            return out
    out.extend(runner.flush())
    return out
