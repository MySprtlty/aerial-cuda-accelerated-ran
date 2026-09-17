import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import calibrate as K                    # noqa: E402
from dapp_est import config as C         # noqa: E402


class TestCalibrate(unittest.TestCase):
    def test_fit_origin(self):
        rnd = random.Random(3)
        xs = [rnd.uniform(1e3, 1e5) for _ in range(50)]
        ys = [2.5e-3 * x + rnd.gauss(0, 5) for x in xs]
        w, r2, n = K.fit_origin(xs, ys)
        self.assertAlmostEqual(w, 2.5e-3, delta=1e-4)
        self.assertGreater(r2, 0.95)
        noise = [rnd.uniform(0, 100) for _ in xs]
        _, r2n, _ = K.fit_origin(xs, noise)
        self.assertLess(r2n, 0.8)

    def test_measure_csv_direct_and_write(self):
        tmp = tempfile.mkdtemp()
        try:
            rules_copy = os.path.join(tmp, "rules.yaml")
            shutil.copy(os.path.join(ROOT, "rules.yaml"), rules_copy)
            meas = os.path.join(tmp, "m.csv")
            rnd = random.Random(5)
            with open(meas, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["tv", "channel", "kernel", "work", "duration_us"])
                for i in range(30):
                    x = rnd.uniform(1e4, 4e5)
                    w.writerow(["tvA", "PUSCH", "ldpc", x, 2.0e-3 * x + rnd.gauss(0, 10)])
                    w.writerow(["tvA", "PDSCH", "enc", x, rnd.uniform(0, 500)])       # noise: must not be written
                w.writerow(["tvA", "PUCCH", "f01", 1, 5])                              # n < 3: skipped
            samples, missing = K.read_measurements(meas, "")
            rules = C.Rules(rules_copy)
            rows = K.fit_all(samples, rules, rules.calibrate_min_r2)
            by = {(r["channel"], r["kernel"]): r for r in rows}
            self.assertEqual(by[("PUSCH", "ldpc")]["action"], "update")
            self.assertTrue(by[("PDSCH", "enc")]["action"].startswith("WARN"))
            self.assertTrue(by[("PUCCH", "f01")]["action"].startswith("skip"))
            out = subprocess.run([sys.executable, os.path.join(ROOT, "calibrate.py"), "--measure", meas,
                                  "--rules", rules_copy, "--write"], stdout=subprocess.PIPE, text=True, check=True).stdout
            self.assertIn("wrote 1 weights", out)
            new = C.Rules(rules_copy)
            self.assertAlmostEqual(new.w["PUSCH"]["ldpc"], 2.0e-3, delta=1e-4)
            self.assertEqual(new.w["PDSCH"]["enc"], rules.w["PDSCH"]["enc"])
            with open(rules_copy) as f:
                text = f.read()
            self.assertIn("# per (num_cb*Zc)", text)          # comments survive
            self.assertIn("provisional: true", text)
        finally:
            shutil.rmtree(tmp)

    def test_missing_column(self):
        tmp = tempfile.mkdtemp()
        try:
            meas = os.path.join(tmp, "bad.csv")
            with open(meas, "w") as f:
                f.write("tv,channel,duration_us\nx,PUSCH,1\n")
            with self.assertRaises(ValueError):
                K.read_measurements(meas, "")
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
