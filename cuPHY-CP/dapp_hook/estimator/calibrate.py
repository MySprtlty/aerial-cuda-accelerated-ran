#!/usr/bin/env python3
"""
calibrate: fit the per-kernel weights w_ck of rules.yaml from measured kernel
durations (duration_us ~ w_ck * work_ck, least squares through the origin).

Input CSV (--measure), one row per (slot, channel, kernel):
    tv,sfn,slot,channel,kernel,duration_us          work is looked up in <dump-dir>/<tv>.bin
 or tv,channel,kernel,work,duration_us               work given directly (no dump needed)

Output: a table channel/kernel/n/w_old/w_new/R2/action. With --write the
weights with R2 >= calibrate.min_r2 (rules.yaml) replace the numbers in
rules.yaml in place (comments kept); worse fits are reported and left alone.

The duration source is nsys (CUPTI kernel trace of a cuPHY run on a test
vector) mapped to (channel, kernel) by kernel_map.md; that mapping is not
part of this file. Only the input format and the fit are implemented here.
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dapp_est import config as C            # noqa: E402
from dapp_est import slot_context as S      # noqa: E402
from dapp_est import work_formulas as W     # noqa: E402

MEASURE_FIELDS_SLOT = ("tv", "sfn", "slot", "channel", "kernel", "duration_us")
MEASURE_FIELDS_WORK = ("tv", "channel", "kernel", "work", "duration_us")
MIN_SAMPLES = 3


def fit_origin(xs, ys):
    """w = sum(xy)/sum(x^2); R^2 against the mean of y. Returns (w, r2, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0
    sxx = sum(x * x for x in xs)
    if sxx == 0:
        return 0.0, 0.0, n
    w = sum(x * y for x, y in zip(xs, ys)) / sxx
    ym = sum(ys) / n
    ss_tot = sum((y - ym) ** 2 for y in ys)
    ss_res = sum((y - w * x) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    return w, r2, n


class WorkIndex:
    """(tv, sfn, slot) -> SlotContext work totals, built lazily from dumps."""

    def __init__(self, dump_dir):
        self.dump_dir = dump_dir
        self.cache = {}

    def get(self, tv, sfn, slot, channel, kernel):
        if tv not in self.cache:
            path = os.path.join(self.dump_dir, tv + ".bin")
            idx = {}
            for ctx in S.contexts_from_file(path):
                idx[(ctx.sfn, ctx.slot)] = ctx
            self.cache[tv] = idx
        ctx = self.cache[tv].get((int(sfn), int(slot)))
        if ctx is None:
            return None
        return ctx.ch[channel].total[kernel]


def read_measurements(path, dump_dir):
    """Returns {(channel, kernel): ([work], [duration_us])}, and the number of rows without work."""
    samples = {}
    missing = 0
    widx = WorkIndex(dump_dir) if dump_dir else None
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        cols = tuple(rd.fieldnames or ())
        direct = "work" in cols
        need = MEASURE_FIELDS_WORK if direct else MEASURE_FIELDS_SLOT
        for c in need:
            if c not in cols:
                raise ValueError("measure csv: missing column %s (have %s)" % (c, cols))
        for row in rd:
            ch = row["channel"].strip().upper()
            k = row["kernel"].strip()
            if ch not in W.KERNELS or k not in W.KERNELS[ch]:
                raise ValueError("unknown channel/kernel %s.%s" % (ch, k))
            if direct:
                work = float(row["work"])
            else:
                if widx is None:
                    raise ValueError("rows carry sfn/slot: --dump-dir is required")
                w = widx.get(row["tv"], row["sfn"], row["slot"], ch, k)
                if w is None:
                    missing += 1
                    continue
                work = float(w)
            xs, ys = samples.setdefault((ch, k), ([], []))
            xs.append(work)
            ys.append(float(row["duration_us"]))
    return samples, missing


def fit_all(samples, rules, min_r2):
    rows = []
    for (ch, k), (xs, ys) in sorted(samples.items()):
        w, r2, n = fit_origin(xs, ys)
        old = rules.w[ch][k]
        if n < MIN_SAMPLES:
            action = "skip (n<%d)" % MIN_SAMPLES
        elif r2 < min_r2:
            action = "WARN R2<%.2f, kept" % min_r2
        else:
            action = "update"
        rows.append({"channel": ch, "kernel": k, "n": n, "w_old": old, "w_new": w, "r2": r2, "action": action})
    return rows


_WEIGHT_LINE = re.compile(r"^(\s+)(%s):(\s+)([-+0-9.eE]+)(.*)$")


def update_rules_text(text, channel, kernel, value):
    """Replace the number of `kernel:` inside the `channel:` block of weights_us_per_work; keeps comments."""
    lines = text.split("\n")
    in_weights = False
    in_channel = False
    pat = re.compile(r"^(\s+)(%s):(\s+)([-+0-9.eE]+)(.*)$" % re.escape(kernel))
    for i, ln in enumerate(lines):
        if ln.startswith("weights_us_per_work:"):
            in_weights = True
            continue
        if in_weights and ln and not ln[0].isspace() and not ln.startswith("#"):
            in_weights = False
        if not in_weights:
            continue
        m = re.match(r"^  (\w+):\s*$", ln)
        if m:
            in_channel = (m.group(1) == channel)
            continue
        if in_channel:
            m = pat.match(ln)
            if m:
                lines[i] = "%s%s:%s%.4g%s" % (m.group(1), m.group(2), m.group(3), value, m.group(5))
                return "\n".join(lines), True
    return text, False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", required=True, help="csv of measured kernel durations")
    ap.add_argument("--dump-dir", default="", help="directory with <tv>.bin ring dumps (for sfn/slot rows)")
    ap.add_argument("--rules", default=C.DEFAULT_RULES)
    ap.add_argument("--min-r2", type=float, default=None, help="override rules.yaml calibrate.min_r2")
    ap.add_argument("--write", action="store_true", help="update rules.yaml in place")
    a = ap.parse_args()
    rules = C.Rules(a.rules)
    min_r2 = a.min_r2 if a.min_r2 is not None else rules.calibrate_min_r2
    samples, missing = read_measurements(a.measure, a.dump_dir)
    rows = fit_all(samples, rules, min_r2)
    print("| channel | kernel | n | w_old | w_new | R2 | action |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %d | %.4g | %.4g | %.3f | %s |" % (r["channel"], r["kernel"], r["n"], r["w_old"], r["w_new"], r["r2"], r["action"]))
    if missing:
        print("rows without a matching slot in the dumps: %d" % missing)
    if a.write:
        text = open(a.rules).read()
        n_upd = 0
        for r in rows:
            if r["action"] != "update":
                continue
            text, ok = update_rules_text(text, r["channel"], r["kernel"], r["w_new"])
            if ok:
                n_upd += 1
            else:
                print("WARN: could not find %s.%s in %s" % (r["channel"], r["kernel"], a.rules))
        open(a.rules, "w").write(text)
        print("wrote %d weights to %s" % (n_upd, a.rules))
    return 0


if __name__ == "__main__":
    sys.exit(main())
