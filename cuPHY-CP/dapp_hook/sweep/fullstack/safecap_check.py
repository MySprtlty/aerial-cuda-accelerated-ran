#!/usr/bin/env python3
"""safecap_check.py <R0 name> <run name> [--dir D] [--first-s 10] [--err-s 1] [--r0-late-slots N] [--legacy]
Pass / fail of one tenant run against the no-tenant reference R0 (fixed criteria of the 2026-10-16 re-sweep):
  1. B misses (rules.yaml completion budgets, all channels, after --first-s)  <= max(1, B misses of R0)
  2. Err.ind lines, slot-drop lines ('Dropping the slot command', 'FAPI msg dropped due to latency', 'SFN mismatch')
     and L1 CRC errors after the first --err-s seconds of traffic  == 0
  3. RU emulator DL U-plane LATE SLOTS (end-of-run table, whole run)  <= 1.25 x the R0 reference
     (--r0-late-slots N overrides the reference, e.g. a mean of several R0 runs; default: R0's own table)
Prints the numbers of both runs and 'verdict: PASS' / 'verdict: FAIL (...)'; exit 0 on pass, 1 on fail.
--legacy keeps the earlier check (A-set misses, overload strings, late packets <= 1.2x + 100).
The module also keeps the helpers used by safecap_report.py / r0_ladder_report.py (load_rows, channel_values, misses,
overload, ru_late; deadline sets A = NVIDIA duration constraints, B = rules.yaml completion budgets)."""
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


def ru_late_slots(name, d):
    """DL U-plane LATE SLOTS from the RU end-of-run table (sum over cells), -1 if missing."""
    for p in (os.path.join(d, "ru_%s.log" % name), os.path.join(d, "ru_warn_%s.log" % name)):
        if os.path.exists(p):
            with open(p, errors="replace") as f:
                for ln in f:
                    m = re.search(r"\[RU\] \| DL U LATE SLOTS\s+\|(.*)", ln)
                    if m:
                        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
                        if nums:
                            return sum(nums)
    return -1


def stamp_s(line):
    m = re.match(r"(\d\d):(\d\d):(\d\d)\.(\d+)", line)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + float("0." + m.group(4)) if m else None


def traffic_start(name, d):
    p = os.path.join(d, "l2a_%s.log" % name)
    if os.path.exists(p):
        with open(p, errors="replace") as f:
            for ln in f:
                t = stamp_s(ln)
                if t is not None:
                    return t
    return None


def after(t, t0, s):
    if t is None or t0 is None:
        return True
    if t < t0 - 43200:
        t += 86400
    return t >= t0 + s


def errors_after(name, d, err_s):
    """(Err.ind lines, slot-drop lines, L1 CRC errors) after the first err_s seconds of traffic."""
    t0 = traffic_start(name, d)
    errind = drops = 0
    p = os.path.join(d, "phy_warn_%s.log" % name)
    if os.path.exists(p):
        with open(p, errors="replace") as f:
            for ln in f:
                if not after(stamp_s(ln), t0, err_s):
                    continue
                if "Send Err.ind" in ln:
                    errind += 1
                if "Dropping the slot command" in ln or "dropped due to latency" in ln or "SFN mismatch" in ln:
                    drops += 1
    crc = 0
    rx = re.compile(r"Cell\s+(\d+) \| DL .* UL .* Slots CRC\s+(\d+) \(\s*(\d+)\)")
    for p in (os.path.join(d, "l1_persec_%s.log" % name), os.path.join(d, "run2_%s.log" % name)):
        if os.path.exists(p):
            with open(p, errors="replace") as f:
                for ln in f:
                    m = rx.search(ln)
                    if m and after(stamp_s(ln), t0, err_s):
                        crc += int(m.group(2))
            break
    return errind, drops, crc


def b_misses(name, d, first_s):
    return sum(v[1] for v in misses(channel_values(load_rows(name, d, first_s))).values())


def check(r0, run, d, first_s=10.0, err_s=1.0, r0_late=None):
    """Returns (passed, reasons, numbers)."""
    b0, b1 = b_misses(r0, d, first_s), b_misses(run, d, first_s)
    e1 = errors_after(run, d, err_s)
    l0 = ru_late_slots(r0, d) if r0_late is None else r0_late
    l1 = ru_late_slots(run, d)
    why = []
    if b1 > max(1, b0):
        why.append("B misses %d > max(1, R0 %d)" % (b1, b0))
    if e1[0]:
        why.append("Err.ind %d" % e1[0])
    if e1[1]:
        why.append("slot drops %d" % e1[1])
    if e1[2]:
        why.append("CRC errors %d" % e1[2])
    if l1 < 0:
        why.append("RU late slots unknown")
    elif l0 is not None and l0 >= 0 and l1 > 1.25 * l0:
        why.append("RU late slots %d > 1.25 x %.0f" % (l1, l0))
    return (not why), why, {"b_miss_r0": b0, "b_miss": b1, "errind": e1[0], "drops": e1[1], "crc": e1[2], "ru_late_slots_r0": l0, "ru_late_slots": l1}


def main():
    a = sys.argv[1:]
    first_s, err_s, d, r0_late, legacy = 10.0, 1.0, DEFAULT_DIR, None, False
    if "--first-s" in a:
        i = a.index("--first-s"); first_s = float(a[i + 1]); del a[i:i + 2]
    if "--err-s" in a:
        i = a.index("--err-s"); err_s = float(a[i + 1]); del a[i:i + 2]
    if "--dir" in a:
        i = a.index("--dir"); d = a[i + 1]; del a[i:i + 2]
    if "--r0-late-slots" in a:
        i = a.index("--r0-late-slots"); r0_late = float(a[i + 1]); del a[i:i + 2]
    if "--legacy" in a:
        a.remove("--legacy"); legacy = True
    r0, run = a[0], a[1]
    if legacy:
        return main_legacy(r0, run, d, first_s)
    ok, why, n = check(r0, run, d, first_s, err_s, r0_late)
    print("| run | B misses (R0) | Err.ind | drops | CRC | RU late slots (R0 ref) |")
    print("|---|---|---|---|---|---|")
    print("| %s | %d (%d) | %d | %d | %d | %d (%.0f) |" % (run, n["b_miss"], n["b_miss_r0"], n["errind"], n["drops"], n["crc"], n["ru_late_slots"], n["ru_late_slots_r0"] if n["ru_late_slots_r0"] is not None else -1))
    print("verdict:", "PASS" if ok else "FAIL (%s)" % "; ".join(why))
    return 0 if ok else 1


def main_legacy(r0, run, d, first_s):
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
