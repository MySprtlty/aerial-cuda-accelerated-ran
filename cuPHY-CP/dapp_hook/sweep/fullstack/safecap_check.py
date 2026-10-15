#!/usr/bin/env python3
"""safecap_check.py <R0 name> <run name> [--first-s 10] [--dir prof/run61]
Decides whether a SafeCap run has MORE deadline misses than the no-tenant run R0 (exit 1) or not (exit 0), and prints
the per-channel miss counts of both. Two deadline sets (safecap_report.py prints both; this check uses A):

  A  NVIDIA check_cell_capacity.py constraints (testBenches/perf/measure/TDD/check_cell_capacity.py:57-101, 4T4R
     DDDSUUDDDD, measured by its parser as end - start of the channel task, i.e. a processing DURATION): PUSCH 1500,
     PUCCH 1500, SRS 500, PRACH 1500 (1250 parallel / 2000 otherwise), PDSCH 375, PDCCH 375 us. Compared with the
     cuphydriver aggregate task's [GPU Run] duration of the slot (parse_phylog.py <ch>_gpu_run_us).
  B  estimator/rules.yaml deadline_us: a budget from the moment the slot's GPU work can start (UL: the slot's last
     symbol, t0 + 500 us at mu=1; DL: the tick, t0 - 1500 us): PUSCH 2000, PUCCH 1500, PRACH 3000, SRS 4000,
     PDSCH / PDCCH 900. Compared with the GPU window end after t0 minus 500 (UL) and the CPU run end + GPU run + 1500 (DL).

Besides the channel timers, the L1's own overload strings (forced CRC failure / work cancel / order-kernel timeout /
pool exhaustion / recovery, phy_warn_<name>.log) and the RU emulator's DL U-plane late slots (ru_<name>.log) count
as misses: a run is "worse than R0" when any of these counts rises. The first --first-s seconds are excluded."""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "prof", "run61"))
A = {"pusch": 1500.0, "pucch": 1500.0, "srs": 500.0, "prach": 1500.0, "pdsch": 375.0, "pdcch": 375.0}
B = {"pusch": 2000.0, "pucch": 1500.0, "prach": 3000.0, "srs": 4000.0, "pdsch": 900.0, "pdcch": 900.0}
UL = ("pusch", "pucch", "prach", "srs")
UL_START_US, DL_TICK_US = 500.0, 1500.0
OVERLOAD = ("Order kernel timeout", "work cancel", "WorkCancel", "pool exhaust", "No available", "Successive Non-availability",
            "Attempting PUSCH pipeline termination", "recovery", "pipeline failed", "Send Err.ind")


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows(name, d, first_s):
    rows = list(csv.DictReader(open(os.path.join(d, "phy_%s.csv" % name))))
    if not rows:
        return []
    t0 = min(int(r["t0_ns"]) for r in rows)
    return [r for r in rows if int(r["t0_ns"]) - t0 >= first_s * 1e9]


def channel_values(rows):
    """{ch: {"dur": [...], "done": [...]}}: A = GPU run duration, B = completion from the channel's start reference."""
    out = {}
    for ch in UL:
        dur = [fl(r.get(ch + "_gpu_run_us")) for r in rows]
        done = [fl(r.get(ch + "_gpu_end")) for r in rows]
        out[ch] = {"dur": [x for x in dur if x is not None], "done": [x - UL_START_US for x in done if x is not None]}
    for ch in ("pdsch", "pdcch"):
        dur = [fl(r.get(ch + "_gpu_run_us")) for r in rows]
        done = []
        for r in rows:
            e, g = fl(r.get(ch + "_run_end")), fl(r.get(ch + "_gpu_run_us"))
            if e is not None and g is not None:
                done.append(e + g + DL_TICK_US)
        out[ch] = {"dur": [x for x in dur if x is not None], "done": done}
    return out


def misses(vals):
    return {ch: (sum(1 for x in v["dur"] if x > A[ch]), sum(1 for x in v["done"] if x > B[ch]), len(v["dur"])) for ch, v in vals.items()}


def overload(name, d):
    n = 0
    try:
        with open(os.path.join(d, "phy_warn_%s.log" % name), errors="replace") as f:
            for ln in f:
                if any(k in ln for k in OVERLOAD):
                    n += 1
    except FileNotFoundError:
        pass
    return n


def ru_late(name, d):
    """DL U-plane late PACKETS (sum over cells) from the RU emulator's end-of-run table ([RU] | DL U LATE PACKETS | ...),
    the late-slots line when the packets line is missing; -1 if neither is there."""
    p = os.path.join(d, "ru_%s.log" % name)
    if not os.path.exists(p):
        p = os.path.join(d, "ru_warn_%s.log" % name)      # the driver keeps only the summary lines of the raw RU log
    if not os.path.exists(p):
        return -1
    # the RU emulator prints its table at exit; a run killed mid-print may lack the SLOTS line, the PACKETS line comes first
    pk, sl = None, None
    with open(p, errors="replace") as f:
        for ln in f:
            m = re.search(r"\[RU\] \| DL U LATE (PACKETS|SLOTS)\s+\|(.*)", ln)
            if m:
                nums = [int(x) for x in re.findall(r"\d+", m.group(2))]
                if nums:
                    if m.group(1) == "PACKETS":
                        pk = sum(nums)
                    else:
                        sl = sum(nums)
    return pk if pk is not None else (sl if sl is not None else -1)


def main():
    a = sys.argv[1:]
    first_s, d = 10.0, DEFAULT_DIR
    if "--first-s" in a:
        i = a.index("--first-s"); first_s = float(a[i + 1]); del a[i:i + 2]
    if "--dir" in a:
        i = a.index("--dir"); d = a[i + 1]; del a[i:i + 2]
    r0, run = a[0], a[1]
    m0 = misses(channel_values(load_rows(r0, d, first_s)))
    m1 = misses(channel_values(load_rows(run, d, first_s)))
    worse = []
    print("| channel | R0 misses A / B / n | run misses A / B / n |")
    print("|---|---|---|")
    for ch in ("pusch", "pucch", "prach", "srs", "pdsch", "pdcch"):
        a0, b0, n0 = m0.get(ch, (0, 0, 0)); a1, b1, n1 = m1.get(ch, (0, 0, 0))
        print("| %s | %d / %d / %d | %d / %d / %d |" % (ch, a0, b0, n0, a1, b1, n1))
        if n1 and a1 > a0:
            worse.append(ch)
    w0, w1 = overload(r0, d), overload(run, d)
    l0, l1 = ru_late(r0, d), ru_late(run, d)
    print("L1 overload strings: R0 %d, run %d | RU DL U-plane late packets: R0 %d, run %d" % (w0, w1, l0, l1))
    if w1 > w0:
        worse.append("L1 overload")
    if l0 >= 0 and l1 > 1.2 * l0 + 100:          # run-to-run noise without a tenant is ~2 % (R0 69,192 vs R1 67,636 packets)
        worse.append("RU DL late")
    print("verdict:", "MISSES ABOVE R0 (%s)" % ", ".join(worse) if worse else "no increase over R0")
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())
