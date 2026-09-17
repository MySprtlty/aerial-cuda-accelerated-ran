import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from dapp_est import config as C          # noqa: E402
from dapp_est import estimator as E       # noqa: E402
from dapp_est import ringio as R          # noqa: E402
from dapp_est import slot_context as S    # noqa: E402
import synth                              # noqa: E402

RULES = C.Rules()
CELLS = C.Cells()
PROF_N = C.Profile(os.path.join(os.path.dirname(HERE), "profiles", "yolov8n_b1.yaml"))
PROF_M = C.Profile(os.path.join(os.path.dirname(HERE), "profiles", "yolov8m_b8.yaml"))


def ctx_ul(sfn, slot, cells=2, n_ue=3, tb=50000, ts=1000):
    c = S.SlotContext(sfn, slot)
    for cell in range(cells):
        for u in range(n_ue):
            c.add_pdu(synth.make_rec(R.REC_UL_PDU, sfn, slot, cell, **synth.pusch(sfn, slot, cell, 100 + u, tb=tb)))
    c.add_slot_end(synth.make_rec(R.REC_SLOT_END, sfn, slot, 0, ts_ns=ts, t0_ns=ts, enqueued=1, is_ul=1, num_cells=cells))
    return c


def ctx_dl(sfn, slot, cells=2, n_ue=3, tb=100000, ts=1000):
    c = S.SlotContext(sfn, slot)
    for cell in range(cells):
        for u in range(n_ue):
            c.add_pdu(synth.make_rec(R.REC_DL_PDU, sfn, slot, cell, **synth.pdsch(100 + u, tb=tb)))
    c.add_slot_end(synth.make_rec(R.REC_SLOT_END, sfn, slot, 0, ts_ns=ts, t0_ns=ts, enqueued=1, is_dl=1, num_cells=cells))
    return c


class TestRules(unittest.TestCase):
    def test_base_and_latency_monotone_in_cap(self):
        c = ctx_ul(1, 4)
        bt, bc = E.base_times(c, RULES, CELLS)
        self.assertGreater(bt["PUSCH"], bc["PUSCH"])
        self.assertEqual(bt["PDSCH"], 0.0)
        prev = 0
        for cap in RULES.caps:
            lat = E.predict_latency_us(c, cap, PROF_N, RULES, CELLS)
            self.assertGreaterEqual(lat["PUSCH"], prev)
            prev = lat["PUSCH"]
        # critical floor: with a tiny total the latency is the critical path
        lat0 = E.latency_from_base({k: 1.0 for k in E.CHANNELS}, {k: 50.0 for k in E.CHANNELS}, 0, 0.5)
        self.assertEqual(lat0["PUSCH"], 50.0)

    def test_cell_scale(self):
        c = ctx_ul(1, 4, cells=2)
        bt, _ = E.base_times(c, RULES, CELLS)
        scaled = C.Cells()
        scaled.scale[1] = 2.0
        bt2, _ = E.base_times(c, RULES, scaled)
        f = RULES.fixed_us["PUSCH"]
        self.assertAlmostEqual((bt2["PUSCH"] - f), (bt["PUSCH"] - f) * 1.5, places=6)

    def test_no_profile_and_missing_context(self):
        st = E.State(RULES)
        g, cap, info = E.decide(ctx_ul(1, 4), [], None, st, CELLS)
        self.assertEqual((g, cap), (0, 0))
        st.cap_pct = 40
        g, cap, info = E.decide(None, [], PROF_N, st, CELLS)
        self.assertEqual((g, cap), (0, 40))
        self.assertEqual(info["reason"], "missing context")
        incomplete = S.SlotContext(1, 5)
        g, cap, _ = E.decide(incomplete, [], PROF_N, st, CELLS)
        self.assertEqual((g, cap), (0, 40))

    def test_stale_ring(self):
        st = E.State(RULES)
        c = ctx_ul(1, 4, ts=10 ** 9)
        st.push(c.ts_end_ns, *E.base_times(c, RULES, CELLS))
        late = int(10 ** 9 + (RULES.stale_after_ms + 1) * 1e6)
        g, cap, info = E.decide(c, [], PROF_N, st, CELLS, now_ns=late)
        self.assertEqual(g, 0)
        self.assertEqual(info["reason"], "ring stale")

    def _run(self, st, ctxs, prof):
        out = []
        for c in ctxs:
            st.push(c.ts_end_ns, *E.base_times(c, RULES, CELLS))
            out.append(E.decide(c, [], prof, st, CELLS, now_ns=c.ts_end_ns))
        return out

    def test_hysteresis_raise_slowly_lower_at_once(self):
        st = E.State(RULES)
        light = [ctx_ul(0, i, cells=1, n_ue=1, tb=1000, ts=1000 + i * 500000) for i in range(6)]
        res = self._run(st, light, PROF_N)
        caps = [r[1] for r in res]
        self.assertEqual(caps[:RULES.raise_after - 1], [0] * (RULES.raise_after - 1))
        self.assertEqual(caps[RULES.raise_after - 1], RULES.caps[-1])
        self.assertTrue(all(c == RULES.caps[-1] for c in caps[RULES.raise_after - 1:]))
        # a heavy slot with a batch-8 tenant (fill 1.0) drops the cap immediately
        heavy = ctx_ul(0, 10, cells=8, n_ue=6, tb=200000, ts=1000 + 10 * 500000)
        st2 = E.State(RULES)
        self._run(st2, light, PROF_M)
        before = st2.cap_pct
        self.assertGreater(before, 0)
        g, cap, info = self._run(st2, [heavy], PROF_M)[0]
        self.assertLess(cap, before)

    def test_gate_needs_duration_and_window(self):
        st = E.State(RULES)
        light = [ctx_ul(0, i, cells=1, n_ue=1, tb=1000, ts=1000 + i * 500000) for i in range(5)]
        self._run(st, light, PROF_N)
        c = light[-1]
        g, cap, info = E.decide(c, light[-3:-1], PROF_N, st, CELLS, now_ns=c.ts_end_ns)
        self.assertEqual(info["reason"], "ok")
        self.assertEqual(g, RULES.max_gate_slots)            # whole window safe
        # a heavy slot in the window: only the confirmed slots count -> k = 3 < needed for yolov8m (dur >> 1.5 ms)
        heavy = ctx_ul(0, 6, cells=8, n_ue=6, tb=150000, ts=1000 + 6 * 500000)
        st.push(heavy.ts_end_ns, *E.base_times(heavy, RULES, CELLS))
        cap_m = st.cap_pct
        g, cap, info = E.decide(light[-1], light[-3:-1], PROF_M, st, CELLS, now_ns=heavy.ts_end_ns)
        self.assertEqual(g, 0)
        self.assertLessEqual(info["safe_k"], RULES.known_slots)

    def test_sliding_max_matches_bruteforce(self):
        rnd = random.Random(7)
        win = 1000
        sm = E.SlidingMax(win)
        hist = []
        for i in range(2000):
            ts = i * 37
            v = rnd.random() * 100
            sm.push(ts, v)
            hist.append((ts, v))
            brute = max(x for t, x in hist if t >= ts - win)
            self.assertEqual(sm.get(ts), brute)

    def test_percentile_path(self):
        rules = C.Rules()
        rules.cap_percentile = 50
        st = E.State(rules)
        ctxs = [ctx_ul(0, i, cells=1 + (i % 3), ts=1000 + i * 500000) for i in range(9)]
        for c in ctxs:
            st.push(c.ts_end_ns, *E.base_times(c, rules, CELLS))
        pt, pc = st.cap_context(ctxs[-1].ts_end_ns)
        vals = sorted(E.base_times(c, rules, CELLS)[0]["PUSCH"] for c in ctxs)
        self.assertEqual(pt["PUSCH"], vals[4])


if __name__ == "__main__":
    unittest.main()
