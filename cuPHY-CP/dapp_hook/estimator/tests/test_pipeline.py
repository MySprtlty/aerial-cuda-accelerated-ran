import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from dapp_est import config as C            # noqa: E402
from dapp_est import control_block as CB    # noqa: E402
from dapp_est import decision_log as L      # noqa: E402
from dapp_est import pipeline as P          # noqa: E402
from dapp_est import ringio as R            # noqa: E402
import synth                                # noqa: E402

RULES = C.Rules()
CELLS = C.Cells()
PROF = C.Profile(os.path.join(os.path.dirname(HERE), "profiles", "yolov8n_b1.yaml"))
TEST_RING = "/dapp_est_test_ring"          # never the production /aerial_dapp_ring


def strip(recs):
    return [{k: r[k] for k in L.COMPARE_FIELDS} for r in recs]


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream, cls.n = synth.make_stream(n_slots=80, cells=2, heavy_slots=(44, 45))
        cls.image = synth.make_image(cls.stream, cls.n)
        cls.tmp = tempfile.mkdtemp()
        cls.raw = os.path.join(cls.tmp, "s.bin")
        with open(cls.raw, "wb") as f:
            f.write(cls.stream)
        cls.img = os.path.join(cls.tmp, "img.bin")
        with open(cls.img, "wb") as f:
            f.write(cls.image)

    def test_replay_deterministic_and_image_equals_stream(self):
        a = strip(P.replay_file(self.raw, P.Runner(RULES, CELLS, PROF)))
        b = strip(P.replay_file(self.raw, P.Runner(RULES, CELLS, PROF)))
        c = strip(P.replay_file(self.img, P.Runner(RULES, CELLS, PROF)))
        self.assertEqual(len(a), 80 - RULES.known_slots + 1)
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        # heavy slots exist and the cap reacts (drops) at or after them
        caps = [d["cap_pct"] for d in a]
        self.assertGreater(max(caps), 0)
        self.assertTrue(all(d["gate_slots"] == 0 for d in a[:RULES.raise_after - 1]))

    def test_live_ring_equals_replay(self):
        path = "/dev/shm" + TEST_RING
        with open(path, "wb") as f:
            f.write(self.image)
        try:
            ring = R.LiveRing(TEST_RING)
            runner = P.Runner(RULES, CELLS, PROF)
            live = []
            idle = 0
            while idle < 3:
                rec = ring.next()
                if rec is None:
                    idle += 1
                    continue
                live.extend(runner.feed(rec))
            live.extend(runner.flush())
            self.assertEqual(ring.lost, 0)
        finally:
            os.unlink(path)
        rep = P.replay_file(self.raw, P.Runner(RULES, CELLS, PROF))
        self.assertEqual(strip(live), strip(rep))

    def test_control_block_carries_last_decision(self):
        cb = CB.ControlBlock("/dapp_est_test_ctrl")
        try:
            out = P.replay_file(self.raw, P.Runner(RULES, CELLS, PROF, ctrl=cb))
            last = out[-1]
            got = cb.read()
            self.assertEqual(got["gate_slots"], last["gate_slots"])
            self.assertEqual(got["cap_pct"], last["cap_pct"])
            self.assertEqual(got["slot_id"], last["slot_id"])
            self.assertEqual((got["sfn"], got["slot"]), (last["sfn"], last["slot"]))
            self.assertEqual(got["n_decisions"], len(out))
            self.assertEqual(got["seq"] % 2, 0)
            rd = CB.ControlBlock("/dapp_est_test_ctrl", create=False)
            self.assertEqual(rd.read()["cap_pct"], last["cap_pct"])
            rd.close()
        finally:
            cb.close()
            cb.unlink()

    def test_csv_log_roundtrip(self):
        path = os.path.join(self.tmp, "dec.csv")
        log = L.CsvLog(path)
        out = P.replay_file(self.raw, P.Runner(RULES, CELLS, PROF, log=log))
        log.close()
        rows = list(L.read_log(path))
        self.assertEqual(len(rows), len(out))
        self.assertEqual(int(rows[-1]["cap_pct"]), out[-1]["cap_pct"])
        self.assertEqual(rows[0].keys().__len__(), len(L.FIELDS))


if __name__ == "__main__":
    unittest.main()
