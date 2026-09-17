import csv
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import evaluate as EV                          # noqa: E402
from dapp_est import config as C               # noqa: E402
from dapp_est import decision_log as L         # noqa: E402
from dapp_est import labels as LB              # noqa: E402
from dapp_est import pipeline as P             # noqa: E402
import synth                                   # noqa: E402

RULES = C.Rules()
CELLS = C.Cells()
PROF = C.Profile(os.path.join(ROOT, "profiles", "yolov8n_b1.yaml"))

NVLOG = """11:46:31.640513 CON 24593 0 [MAC.FAPI] Finished running 40000 slots test.
11:46:32.000001 ERR 24593 12 [DRV.UL] SFN 12.4 Slot Map 7 Order kernel timeout error (exit condition 2) for cell index 1 Dyn index 0! x
11:46:32.000002 WRN 24593 12 [DRV.UL] task_work_function_ul_aggr_3 timeout waiting for ULC Tasks, Slot Map 7
11:46:32.000003 ERR 24593 12 [DRV.UL] SFN 12.5 Slot Map 8 PUSCH Post Early Harq Wait kernel timeout!
11:46:32.000004 ERR 24593 3 [SCF.PHY] Send Err.ind for SFN 13.6 cell_id=0 msg_id=0x82 err_code=0x05
11:46:32.000005 INF 24593 3 [SCF.PHY] Send Err.ind for SFN 13.7 cell_id=0 msg_id=0x82 err_code=0x05
11:46:32.000006 ERR 24593 3 [NVIPC] something broke without a slot
11:46:32.000007 INF 24593 3 [SCF.PHY] all good
"""


class TestLabels(unittest.TestCase):
    def test_parse_nvlog(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "phy.log")
            with open(p, "w") as f:
                f.write(NVLOG)
            v = LB.parse_nvlog(p)
        finally:
            shutil.rmtree(tmp)
        reasons = [(x.reason, x.sfn, x.slot, x.cell, x.side) for x in v]
        self.assertIn(("ul_order_timeout", 12, 4, 1, "UL"), reasons)
        self.assertIn(("ulc_task_timeout", -1, -1, -1, "UL"), reasons)
        self.assertIn(("pusch_wait_timeout", 12, 5, -1, "UL"), reasons)
        self.assertIn(("error_ind_0x05", 13, 6, 0, "ANY"), reasons)
        self.assertIn(("error_ind_0x05", 13, 7, 0, "ANY"), reasons)       # INF level but a real Err.ind
        self.assertIn(("nvlog_error", -1, -1, -1, "ANY"), reasons)
        self.assertEqual(len(v), 6)
        ls = LB.LabelSet()
        ls.add_violations(v)
        self.assertEqual(len(ls.unattributed), 2)
        self.assertEqual(len(ls.violations_for(12, 4, "PUSCH")), 1)
        self.assertEqual(len(ls.violations_for(12, 4, "PDSCH")), 0)      # UL-side violation
        self.assertEqual(len(ls.violations_for(12, 5, "PUCCH")), 0)      # PUSCH hint
        self.assertEqual(len(ls.violations_for(13, 6, "PDSCH")), 1)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        stream, n = synth.make_stream(n_slots=60, cells=2)
        raw = os.path.join(self.tmp, "s.bin")
        with open(raw, "wb") as f:
            f.write(stream)
        self.dec = os.path.join(self.tmp, "dec.csv")
        log = L.CsvLog(self.dec)
        self.out = P.replay_file(raw, P.Runner(RULES, CELLS, PROF, log=log))
        log.close()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_partial_labels_ul_only(self):
        ul_rows = [r for r in self.out if r["PUSCH_n"] > 0]
        ind = []
        D = RULES.deadline_us["PUSCH"]
        for i, r in enumerate(ul_rows):
            start = r["t0_ns"] + int(RULES.raw["label"]["ul_start_offset_us"] * 1000)
            lat_us = D + 100 if i == 0 else D - 300                     # first one misses
            for cell in range(2):
                ind.append(LB.IndSent(r["sfn"], r["slot"], cell, "PUSCH", start + int(lat_us * 1000)))
        ls = LB.LabelSet()
        ls.add_ind(ind)
        rep = EV.evaluate(L.read_log(self.dec), ls, RULES, min_slots=10 ** 6)
        pu = rep["channels"]["PUSCH"]
        self.assertEqual(pu["observed_slots"], len(ul_rows))
        self.assertAlmostEqual(pu["miss_rate"], 1.0 / len(ul_rows))
        self.assertAlmostEqual(pu["reliability"], 1 - 1.0 / len(ul_rows))
        self.assertEqual(pu["status"], EV.UNCONFIRMED)
        pd = rep["channels"]["PDSCH"]
        self.assertEqual(pd["status"], EV.UNMEASURED)
        self.assertIsNone(pd["miss_rate"])
        self.assertIsNone(pd["reliability"])
        self.assertIsNotNone(rep["gate_closed_ratio"])
        self.assertTrue(rep["cap_distribution"])
        if pu["gate_open_slots"] > 0 and pu["tightness_us"] is not None:
            self.assertLess(pu["tightness_us"], D)

    def test_violations_and_chan_done(self):
        ul_rows = [r for r in self.out if r["PUSCH_n"] > 0]
        r0 = ul_rows[0]
        ls = LB.LabelSet()
        ls.add_violations([LB.SlotViolation(r0["sfn"], r0["slot"], 0, "ul_order_timeout", 0, "UL")])
        r1 = ul_rows[1]
        ls.add_done([LB.ChanDone(r1["sfn"], r1["slot"], 0, "PUSCH", 0, int(RULES.deadline_us["PUSCH"] * 1000) - 1000, "test")])
        rep = EV.evaluate(L.read_log(self.dec), ls, RULES, min_slots=1)
        pu = rep["channels"]["PUSCH"]
        self.assertEqual(pu["observed_slots"], 2)
        self.assertAlmostEqual(pu["miss_rate"], 0.5)
        self.assertEqual(pu["status"], "ok")
        self.assertEqual(rep["channels"]["PDSCH"]["status"], EV.UNMEASURED)


if __name__ == "__main__":
    unittest.main()
