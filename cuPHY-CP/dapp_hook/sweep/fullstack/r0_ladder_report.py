#!/usr/bin/env python3
"""r0_ladder_report.py --dir D [--first-s 10] [--out D/ladder_table.csv] (--append NAME | --table)
One row per no-tenant run of the R0 ladder (r0_ladder.sh), accumulated in --out (a row with the same name is replaced):

  ptp                   ptp4l / phc2sys Cpus_allowed_list and last cpu at run start (ptp_<name>.txt)
  slots                 rows of phy_<name>.csv after the first --first-s seconds
  l1_overload           L1 overload strings (safecap_check.OVERLOAD) in phy_warn_<name>.log
  errind, slot_drop     'Send Err.ind' lines / slot-drop lines ('Dropping the slot command', 'FAPI msg dropped due to
                        latency', 'SFN mismatch') after the first seconds (line stamps vs the first L2A line)
  l1_crc_max_s, l1_crc  L1 per-second line '[SCF.PHY] Cell N | ... Slots CRC a ( b)': max a over cells and seconds, sum of
                        the final b (l1_persec_<name>.log or run2_<name>.log)
  mac_err, mac_inv_med  testMAC per-second line '[MAC.FAPI] Cell N | ... ERR e | INV i': sum of e, median of i
  ru_late_pkts/slots    RU end-of-run table (all cells, whole run); ru_late_pos<k> = late (slot, cell) rows per TDD position
                        k = slot % 10 and ru_late_us p50 / p99 / max = lateness of the slot's latest packet
                        (max_ns / 1e3 + dl_u_plane_timing_delay - dl_u_plane_window_size, from ru_<name>_late.csv)
  per channel           A misses (GPU run duration > NVIDIA constraint), B misses (completion > rules.yaml budget),
                        completion p99.99 / max (us after t0), duration p99.99 / max
  pusch_pos4 / pos5     first / second consecutive UL slot (59c: positions 4 and 5): duration p50 / p99 / p99.99 / max and
                        completion p50 / p99 / p99.99 / max
Deadline sets and parsing come from safecap_check.py. numpy only."""
import argparse
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import safecap_check as SC   # noqa: E402

CH = ("pusch", "pucch", "prach", "srs", "pdsch", "pdcch")


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def stamp_s(line):
    m = re.match(r"(\d\d):(\d\d):(\d\d)\.(\d+)", line)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + float("0." + m.group(4)) if m else None


def traffic_start(d, name):
    for p in (os.path.join(d, "l2a_%s.log" % name), os.path.join(d, "phy_warn_%s.log" % name)):
        if os.path.exists(p):
            with open(p, errors="replace") as f:
                for ln in f:
                    t = stamp_s(ln)
                    if t is not None and ("L2A.PROCESSING_TIMES" in ln or p.endswith("phy_warn_%s.log" % name)):
                        return t
    return None


def after(t, t0, first_s):
    if t is None or t0 is None:
        return True
    if t < t0 - 43200:
        t += 86400
    return t >= t0 + first_s


def count_lines(path, keys, t0, first_s):
    n = {k: 0 for k in keys}
    try:
        with open(path, errors="replace") as f:
            for ln in f:
                if not after(stamp_s(ln), t0, first_s):
                    continue
                for k in keys:
                    if k in ln:
                        n[k] += 1
    except FileNotFoundError:
        pass
    return n


def l1_persec(d, name, t0, first_s):
    p = os.path.join(d, "l1_persec_%s.log" % name)
    if not os.path.exists(p):
        p = os.path.join(d, "run2_%s.log" % name)
    mx, last = 0, {}
    rx = re.compile(r"Cell\s+(\d+) \| DL .* UL .* Slots CRC\s+(\d+) \(\s*(\d+)\)")
    try:
        with open(p, errors="replace") as f:
            for ln in f:
                m = rx.search(ln)
                if not m or not after(stamp_s(ln), t0, first_s):
                    continue
                mx = max(mx, int(m.group(2))); last[int(m.group(1))] = int(m.group(3))
    except FileNotFoundError:
        return None, None
    return mx, sum(last.values())


def mac_persec(d, name, t0, first_s):
    p = os.path.join(d, "mac_persec_%s.log" % name)
    if not os.path.exists(p):
        p = os.path.join(d, "run3_%s.log" % name)
    err, inv = 0, []
    rx = re.compile(r"Cell\s+(\d+) \| DL .* ERR\s+(\d+) \| INV\s+(\d+)")
    try:
        with open(p, errors="replace") as f:
            for ln in f:
                m = rx.search(ln)
                if not m or not after(stamp_s(ln), t0, first_s):
                    continue
                err += int(m.group(2)); inv.append(int(m.group(3)))
    except FileNotFoundError:
        return None, None
    return err, (float(np.median(inv)) if inv else None)


def ru_window(d, name):
    """dl_u_plane_timing_delay - dl_u_plane_window_size (us) from the RU yaml snapshot of the run, default 345 - 51."""
    for p in (os.path.join(d, "before_%s.config.yaml" % name), os.path.join(d, "ru_config_%s.yaml" % name)):
        if os.path.exists(p):
            s = open(p, errors="replace").read()
            m1 = re.search(r"dl_u_plane_timing_delay:\s*(\d+)", s); m2 = re.search(r"dl_u_plane_window_size:\s*(\d+)", s)
            if m1 and m2:
                return int(m1.group(1)) - int(m2.group(1))
    return 345 - 51


def ru_late_detail(d, name, first_s):
    p = os.path.join(d, "ru_%s_late.csv" % name)
    if not os.path.exists(p):
        return {}, []
    rows = list(csv.DictReader(open(p)))
    if not rows or "max_ns" not in rows[0]:
        return {}, []
    t0 = min(int(r["t0_ns"]) for r in rows)
    rows = [r for r in rows if int(r["t0_ns"]) - t0 >= first_s * 1e9]
    pos = {}
    for r in rows:
        k = int(r["slot"]) % 10; pos[k] = pos.get(k, 0) + 1
    late_us = [int(r["max_ns"]) / 1e3 + ru_window(d, name) for r in rows]
    return pos, late_us


def ru_table_slots(d, name):
    """late SLOTS from the RU end-of-run table (safecap_check.ru_late returns packets)."""
    for p in (os.path.join(d, "ru_%s.log" % name), os.path.join(d, "ru_warn_%s.log" % name)):
        if os.path.exists(p):
            with open(p, errors="replace") as f:
                for ln in f:
                    m = re.search(r"\[RU\] \| DL U LATE SLOTS\s+\|(.*)", ln)
                    if m:
                        return sum(int(x) for x in re.findall(r"\d+", m.group(1)))
    return None


def run_row(d, name, first_s):
    rows = SC.load_rows(name, d, first_s)
    vals = SC.channel_values(rows)
    miss = SC.misses(vals)
    t0 = traffic_start(d, name)
    drops = count_lines(os.path.join(d, "phy_warn_%s.log" % name), ("Send Err.ind", "Dropping the slot command", "dropped due to latency", "SFN mismatch"), t0, first_s)
    crc_max, crc_cum = l1_persec(d, name, t0, first_s)
    mac_err, mac_inv = mac_persec(d, name, t0, first_s)
    pos, late_us = ru_late_detail(d, name, first_s)
    ptp = "-"
    pp = os.path.join(d, "ptp_%s.txt" % name)
    if os.path.exists(pp):
        ptp = " ".join(re.sub(r"^\d+ ", "", ln.strip()).replace(" last_cpu=", "@") for ln in open(pp) if ln.strip())
    row = {"name": name, "ptp": ptp, "slots": len(rows), "l1_overload": SC.overload(name, d), "errind": drops["Send Err.ind"],
           "slot_drop": drops["Dropping the slot command"] + drops["dropped due to latency"] + drops["SFN mismatch"],
           "l1_crc_max_s": crc_max, "l1_crc": crc_cum, "mac_err": mac_err, "mac_inv_med": mac_inv,
           "ru_late_pkts": SC.ru_late(name, d), "ru_late_slots": ru_table_slots(d, name),
           "ru_late_us_p50": round(pct(late_us, 50), 1) if late_us else None, "ru_late_us_p99": round(pct(late_us, 99), 1) if late_us else None,
           "ru_late_us_max": round(max(late_us), 1) if late_us else None}
    for k in range(10):
        row["ru_late_pos%d" % k] = pos.get(k, 0) if pos else None
    for ch in CH:
        dur = np.array(vals[ch]["dur"]); done = np.array(vals[ch]["done"])
        row.update({ch + "_n": len(dur), ch + "_missA": miss[ch][0], ch + "_missB": miss[ch][1],
                    ch + "_done_p9999": round(pct(done, 99.99), 0) if len(done) else None, ch + "_done_max": round(float(done.max()), 0) if len(done) else None,
                    ch + "_dur_p9999": round(pct(dur, 99.99), 0) if len(dur) else None, ch + "_dur_max": round(float(dur.max()), 0) if len(dur) else None})
    for p in (4, 5):
        sub = [r for r in rows if int(r["pos"]) == p]
        dur = np.array([SC.fl(r.get("pusch_gpu_run_us")) for r in sub if SC.fl(r.get("pusch_gpu_run_us")) is not None])
        done = np.array([SC.fl(r.get("pusch_gpu_end")) - SC.UL_START_US for r in sub if SC.fl(r.get("pusch_gpu_end")) is not None])
        for lab, v in (("dur", dur), ("done", done)):
            row.update({"pusch_pos%d_%s_p50" % (p, lab): round(pct(v, 50), 0) if len(v) else None, "pusch_pos%d_%s_p99" % (p, lab): round(pct(v, 99), 0) if len(v) else None,
                        "pusch_pos%d_%s_p9999" % (p, lab): round(pct(v, 99.99), 0) if len(v) else None, "pusch_pos%d_%s_max" % (p, lab): round(float(v.max()), 0) if len(v) else None})
    return row


def load_table(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def save_table(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


def fmt(v):
    return "-" if v in (None, "", "None") else str(v)


def print_table(rows):
    print("## R0 ladder: run summary (statistics after the first seconds; RU table counts cover the whole run)")
    print("| run | PTP cores (ptp4l, phc2sys allowed@last) | slots | L1 overload | Err.ind | slot drop | L1 CRC max/s / cum | MAC ERR sum / INV med | RU late pkts / slots | RU late pos 0..9 | RU late us p50 / p99 / max |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %s | %s | %s | %s | %s / %s | %s / %s | %s / %s | %s | %s / %s / %s |" % (
            r["name"], fmt(r.get("ptp")), fmt(r["slots"]), fmt(r["l1_overload"]), fmt(r["errind"]), fmt(r["slot_drop"]), fmt(r["l1_crc_max_s"]), fmt(r["l1_crc"]),
            fmt(r["mac_err"]), fmt(r["mac_inv_med"]), fmt(r["ru_late_pkts"]), fmt(r["ru_late_slots"]),
            " ".join(fmt(r.get("ru_late_pos%d" % k)) for k in range(10)), fmt(r["ru_late_us_p50"]), fmt(r["ru_late_us_p99"]), fmt(r["ru_late_us_max"])))
    print("\n## per channel: A misses (duration > NVIDIA constraint) / B misses (completion > rules.yaml budget) | completion p99.99 / max us | duration p99.99 / max us")
    print("| run | " + " | ".join(CH) + " |")
    print("|---|" + "---|" * len(CH))
    for r in rows:
        cells = []
        for ch in CH:
            if fmt(r.get(ch + "_n")) in ("-", "0"):
                cells.append("-"); continue
            cells.append("%s / %s \\| %s / %s \\| %s / %s" % (fmt(r.get(ch + "_missA")), fmt(r.get(ch + "_missB")), fmt(r.get(ch + "_done_p9999")), fmt(r.get(ch + "_done_max")), fmt(r.get(ch + "_dur_p9999")), fmt(r.get(ch + "_dur_max"))))
        print("| %s | %s |" % (r["name"], " | ".join(cells)))
    print("\n## PUSCH in the two consecutive UL slots (TDD position 4 = first, 5 = second): duration p50 / p99 / p99.99 / max | completion after t0 p50 / p99 / p99.99 / max (us)")
    print("| run | pos 4 duration | pos 4 completion | pos 5 duration | pos 5 completion |")
    print("|---|---|---|---|---|")
    for r in rows:
        c = []
        for p in (4, 5):
            for lab in ("dur", "done"):
                c.append(" / ".join(fmt(r.get("pusch_pos%d_%s_%s" % (p, lab, q))) for q in ("p50", "p99", "p9999", "max")))
        print("| %s | %s |" % (r["name"], " | ".join(c)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=SC.DEFAULT_DIR)
    ap.add_argument("--first-s", type=float, default=10.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--append", default="", help="compute this run and add / replace its row")
    ap.add_argument("--table", action="store_true", help="print the accumulated table only")
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "ladder_table.csv")
    rows = load_table(out)
    if a.append:
        r = run_row(a.dir, a.append, a.first_s)
        rows = [x for x in rows if x["name"] != a.append] + [r]
        save_table(out, rows)
        rows = load_table(out)
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
