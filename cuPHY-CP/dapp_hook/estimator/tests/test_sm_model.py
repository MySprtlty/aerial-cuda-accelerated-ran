"""sm_model: duration monotone in s, inversion picks the smallest feasible multiple of g, infeasible flagged, burst queueing."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from dapp_est import sm_model as M   # noqa: E402


class Ctx:
    def __init__(self, sfn, slot, t0_ns, totals):
        self.sfn, self.slot, self.t0_ns, self._t = sfn, slot, t0_ns, totals

    def channel_totals(self):
        base = {c: {k: 0 for k in M.W.KERNELS[c]} for c in M.W.CHANNELS}
        for c, d in self._t.items():
            base[c].update(d)
        return base


def test_duration_monotone_and_min_sm():
    m = M.SmModel()
    ch = m.channels["PUSCH"]
    work = {"chest": 2e5, "eq": 1e6, "llr": 8e6, "ldpc": 4e5, "uci": 0}
    d = [ch.duration_us(work, s) for s in (8, 16, 32, 64, 100)]
    assert all(d[i] >= d[i + 1] for i in range(len(d) - 1)), d
    s, t, ok = m.min_sm("PUSCH", work)
    assert s % m.g == 0 and s >= ch.s_min
    if ok:
        assert t + ch.margin_us <= ch.budget_us
        assert s == ch.s_min or ch.duration_us(work, s - m.g) + ch.margin_us > ch.budget_us   # smallest feasible
    big = {k: v * 50 for k, v in work.items()}
    s2, t2, ok2 = m.min_sm("PUSCH", big)
    assert not ok2 and s2 == ch.s_max


def test_plan_and_burst():
    m = M.SmModel()
    work = {"chest": 2e5, "eq": 1e6, "llr": 8e6, "ldpc": 4e5, "uci": 0}
    c7 = Ctx(1, 7, 1_000_000_000, {"PUSCH": work, "PUCCH": {"f01": 8}})
    r7 = m.plan(c7)
    assert r7["PUSCH_s"] > 0 and r7["PUCCH_s"] == m.channels["PUCCH"].s_fixed and "tenant_sm" in r7
    # a second U slot 500 us later that would start before the first one's predicted end queues behind it
    c8 = Ctx(1, 8, 1_000_500_000, {"PUSCH": work})
    prev = (r7["pusch_end_rel_us"], c7.t0_ns)
    r8 = m.plan(c8, prev)
    expected_q = max(0.0, (c7.t0_ns + r7["pusch_end_rel_us"] * 1e3 - (c8.t0_ns + m.ul_start.get(8, 300.0) * 1e3)) / 1e3)
    assert abs(r8["pusch_queue_us"] - expected_q) < 1e-6
    assert r8["tenant_sm"] == max(0, (m.sm_total - r8["footprint_sm"]) // m.g * m.g)


if __name__ == "__main__":
    test_duration_monotone_and_min_sm(); test_plan_and_burst(); print("ok")
