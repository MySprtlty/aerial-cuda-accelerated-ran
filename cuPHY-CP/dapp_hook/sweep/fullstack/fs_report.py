#!/usr/bin/env python3
"""fs_report.py <name> [<name> ...] [--csv out.csv]
Full-stack run with an AI tenant (fullstack_run.sh + TENANT_SPEC): the terms of the UE-to-UE latency that our
control knobs can move, measured without a UE.

  E2E(arm) = GIVEN (upload air time, core, UE, DL slot timing - identical in both arms)
           + last-UL-slot term : PUSCH GPU completion after the slot's air time t0 (L1 log, per TDD position)
           + GH200 term        : server time = receive -> NVJPG decode -> inference -> NMS -> reply (server log)
           + step terms        : events that would cost a HARQ round trip or a TDD period with a real UE:
                                 forced CRC failures / pool exhaustion / work cancel / order timeout / recovery (L1 log),
                                 UL C-plane late at the RU (the grant's slot is lost), DL C/U-plane late at the RU

RAN statistics are restricted to the window in which the client was sending (client csv unix times) when a tenant ran,
otherwise to the whole run minus its first second. UL slots are also split by whether their PUSCH GPU window
overlapped an inference of the tenant. The RU packet table and the L1 warning counts cover the whole run. numpy only.
"""
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ladder_report import O, console, pct, warn_counts  # noqa: E402


def load_client(name):
    p = os.path.join(O, "client_%s.csv" % name)
    if not os.path.exists(p):
        return None
    rows = [r for r in csv.DictReader(open(p)) if r.get("t_send_unix_ns")]
    if not rows:
        return None
    f = lambda k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    d = {"n": len(rows), "send": f("t_send_unix_ns"), "reply": f("t_reply_unix_ns"), "rtt": f("rtt_ms"), "late": f("late_ms"), "server": f("srv_server_ms"),
         "decode": f("srv_decode_ms"), "infer": f("srv_infer_ms"), "post": f("srv_post_ms"), "queue": f("srv_queue_ms"),
         "ndet": f("n_det"), "status": f("status"), "engine": rows[0]["engine"]}
    return d


def ran(name, cl):
    p = os.path.join(O, "phy_%s.csv" % name)
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    t0 = np.array([float(x["t0_ns"]) for x in rows])
    if cl is not None:
        keep = (t0 >= cl["send"].min()) & (t0 <= cl["reply"].max())
        win = "client window %.0f s" % ((cl["reply"].max() - cl["send"].min()) / 1e9)
        # inference windows (unix ns): the reply leaves right after the NMS, the network ran just before it
        inf_end = cl["reply"] - np.nan_to_num(cl["post"]) * 1e6
        inf_start = inf_end - np.nan_to_num(cl["infer"]) * 1e6
        order = np.argsort(inf_start)
        inf_start, inf_end = inf_start[order], inf_end[order]
    else:
        keep = t0 - t0.min() >= 1e9
        win = "whole run minus the first second"
        inf_start = inf_end = np.array([])
    out = {"window": win, "slots": int(keep.sum()), "ul": {}, "dl": {}}
    pos = np.array([int(x["pos"]) for x in rows])

    def arr(key):
        return np.array([float(x[key]) if x.get(key) not in ("", None) else np.nan for x in rows])
    gend, grun = arr("pusch_gpu_end"), arr("pusch_gpu_run_us")
    for p_ in sorted(set(pos[~np.isnan(gend)])):
        m = keep & (pos == p_) & ~np.isnan(gend)
        v = gend[m]
        if not len(v):
            continue
        d = {"n": len(v), "p50": pct(v, 50), "p99": pct(v, 99), "max": float(v.max()), "gt2500": float((v > 2500).mean()), "gt3000": float((v > 3000).mean())}
        if len(inf_start):
            w_end = t0[m] + gend[m] * 1e3
            w_start = w_end - np.nan_to_num(grun[m]) * 1e3
            i = np.searchsorted(inf_start, w_end, side="left") - 1          # last inference starting before the PUSCH window ends
            ov = (i >= 0) & (inf_end[np.clip(i, 0, len(inf_end) - 1)] > w_start)
            d["ov_share"] = float(ov.mean())
            d["ov_p50"], d["ov_p99"] = (pct(v[ov], 50), pct(v[ov], 99)) if ov.any() else (float("nan"), float("nan"))
            d["no_p50"], d["no_p99"] = (pct(v[~ov], 50), pct(v[~ov], 99)) if (~ov).any() else (float("nan"), float("nan"))
        out["ul"][int(p_)] = d
    pr, cw = arr("pdsch_run_end"), arr("comp_wait_end")
    for p_ in (0, 6, 7):
        m = keep & (pos == p_) & ~np.isnan(pr)
        if m.any():
            out["dl"][p_] = (int(m.sum()), pct(pr[m], 50), pct(pr[m], 99), float(pr[m].max()), pct(cw[m & ~np.isnan(cw)], 50), pct(cw[m & ~np.isnan(cw)], 99))
    return out


def l1_events(name, cl):
    """order-kernel timeouts / PUSCH work cancels / pool exhaustion with their time relative to the client window:
    (inside the window, outside it). Log times are HH:MM:SS.us UTC; the date comes from the client's first request."""
    p = os.path.join(O, "phy_warn_%s.log" % name)
    if cl is None or not os.path.exists(p):
        return None
    import datetime
    w0, w1 = cl["send"].min() / 1e9, cl["reply"].max() / 1e9
    day = datetime.datetime.utcfromtimestamp(w0).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=datetime.timezone.utc).timestamp()
    inside, outside, slots_in = 0, [], set()
    for line in open(p, errors="replace"):
        if not ("Order kernel timeout" in line or "Attempting PUSCH pipeline termination" in line or "No available" in line or "recovery" in line):
            continue
        m = re.match(r"(\d+):(\d+):(\d+\.\d+)", line)
        if not m:
            continue
        t = day + int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        if t < w0 - 43200:
            t += 86400
        if w0 <= t <= w1:
            inside += 1
            sm = re.search(r"SFN (\d+\.\d+)", line)
            slots_in.add(sm.group(1) if sm else "%.3f" % t)
        else:
            outside.append(t - w1 if t > w1 else t - w0)
    return inside, len(slots_in), outside


def ru_slots(name):
    """share (%) of slots per cell with at least one late packet, from the RU emulator's end-of-run slot table"""
    p = os.path.join(O, "run1_%s.log" % name)
    if not os.path.exists(p):
        return None
    tab = {}
    for line in open(p, errors="replace"):
        m = re.search(r"\[RU\] \| ((?:DL U|DL C|UL C) (?:LATE|ON TIME) SLOTS)\s+\|(.*)", line)
        if m:
            v = [int(x) for x in re.findall(r"(\d+)\s*\|", m.group(2))]
            if v:
                tab[m.group(1)] = v
    out = {}
    for k in ("DL U", "DL C", "UL C"):
        late, on = tab.get(k + " LATE SLOTS"), tab.get(k + " ON TIME SLOTS")
        if late and on:
            out[k] = (100.0 * sum(late) / max(1, sum(late) + sum(on)), min(late), max(late), on[0] + late[0])
    return out


def ru_packets(name):
    """late share (%) per TDD position from the RU emulator's FH.PACKET_SUMMARY count table (whole run, all cells)"""
    p = os.path.join(O, "run1_%s.log" % name)
    if not os.path.exists(p):
        return None
    acc = {}
    for line in open(p, errors="replace"):
        m = re.search(r"\[FH\.PACKET_SUMMARY\] (DLU|DLC|ULC) Slot (\d+) \| (.*)", line)
        if not m:
            continue
        kind, slot = m.group(1), int(m.group(2))
        cells = [c.strip() for c in m.group(3).split("|") if c.strip()]
        if not cells or "." in cells[0]:          # the second table repeats the numbers as percentages
            continue
        on = late = 0
        for c in cells:
            e, o, l = (int(x) for x in c.split(",")[:3])
            on += o + e
            late += l
        a = acc.setdefault(kind, {}).setdefault(slot % 10, [0, 0])
        a[0] += on
        a[1] += late
    return acc


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else ""
    if out_csv in names:
        names.remove(out_csv)
    table = []
    for name in names:
        print("=== %s" % name)
        cl = load_client(name)
        row = {"run": name}
        r = ran(name, cl)
        if r is None:
            print("  no per-slot L1 csv")
        else:
            print("  RAN (%s, %d slots)" % (r["window"], r["slots"]))
            for p_, d in r["ul"].items():
                s = "  UL pos %d: PUSCH GPU end after t0  p50 %5.0f  p99 %5.0f  max %5.0f us  (>2.5 ms %.3f %%, >3.0 ms %.3f %%; n=%d)" % (
                    p_, d["p50"], d["p99"], d["max"], 100 * d["gt2500"], 100 * d["gt3000"], d["n"])
                if "ov_share" in d:
                    s += "\n            overlapping an inference %.0f %% of slots: p50 %5.0f p99 %5.0f | not overlapping: p50 %5.0f p99 %5.0f" % (
                        100 * d["ov_share"], d["ov_p50"], d["ov_p99"], d["no_p50"], d["no_p99"])
                print(s)
                row["ul%d_p50" % p_], row["ul%d_p99" % p_], row["ul%d_max" % p_] = round(d["p50"]), round(d["p99"]), round(d["max"])
            for p_, (n, a, b, mx, c50, c99) in r["dl"].items():
                print("  DL pos %d: PDSCH run end p50 %5.0f p99 %5.0f max %5.0f us, compression end p50 %5.0f p99 %5.0f (relative to t0; n=%d)" % (p_, a, b, mx, c50, c99, n))
                row["dl%d_pdsch_p99" % p_], row["dl%d_comp_p99" % p_] = round(b), round(c99)
        c, err, n = warn_counts(name)
        if c is not None:
            print("  L1 (whole run): WRN/ERR lines %d | pool 'No available' %d | recovery %d | not responsive %d | FATAL %d | PUSCH work cancel %d | order timeout %d | exit %d | Err.ind %s" % (
                n, c["No available"], c["recovery"], c["not responsive"], c["FATAL"], c["Attempting PUSCH pipeline termination"], c["Order kernel timeout"], c["exit handler"],
                " ".join("%s x%d" % kv for kv in sorted(err.items())) or "-"))
            row.update(l1_pool=c["No available"], l1_recovery=c["recovery"], l1_cancel=c["Attempting PUSCH pipeline termination"], l1_order_to=c["Order kernel timeout"],
                       l1_errind=" ".join("%s x%d" % kv for kv in sorted(err.items())))
        ev = l1_events(name, cl)
        if ev is not None:
            ins, nslot, outs = ev
            print("  L1 overload events inside the client window: %d (%d UL slots) | outside: %d%s" % (
                ins, nslot, len(outs), (" at %s s relative to the window" % ", ".join(sorted({"%+.2f" % o for o in outs})[:4])) if outs else ""))
            row.update(l1_events_in_window=ins, l1_event_slots_in_window=nslot, l1_events_outside=len(outs))
        co = console(name)
        if "l1" in co:
            print("  L1 counters (2nd half, per cell per s): UL %.2f Mbps (min %.2f)  DL %.2f Mbps  CRC errors/s max %d, cumulative %d" % co["l1"])
            row.update(l1_ul_mbps=co["l1"][0], l1_crc_cum=co["l1"][4])
        if "mac" in co:
            print("  testMAC (2nd half): UL %.2f Mbps (min %.2f)  DL %.2f Mbps  ERR max %d  INV median %.0f max %d  finished=%s" % (co["mac"] + (co.get("mac_finished"),)))
            row.update(mac_ul_mbps=co["mac"][0], mac_inv=co["mac"][4])
        ru = ru_packets(name)
        if ru:
            for kind in ("ULC", "DLC", "DLU"):
                if kind in ru:
                    tot_on = sum(v[0] for v in ru[kind].values()); tot_late = sum(v[1] for v in ru[kind].values())
                    per = "  ".join("%d: %.3f" % (p_, 100.0 * v[1] / max(1, v[0] + v[1])) for p_, v in sorted(ru[kind].items()) if v[1])
                    print("  RU %s late packets %.4f %% (%d of %d)%s" % (kind, 100.0 * tot_late / max(1, tot_on + tot_late), tot_late, tot_on + tot_late, ("  by TDD position (%): " + per) if per else ""))
                    row["ru_%s_late_pct" % kind.lower()] = round(100.0 * tot_late / max(1, tot_on + tot_late), 4)
        rs = ru_slots(name)
        if rs:
            print("  RU slots with a late packet (whole run, per cell): " + " | ".join("%s %.3f %% (%d-%d of %d)" % (k, v[0], v[1], v[2], v[3]) for k, v in rs.items()))
            for k, v in rs.items():
                row["ru_%s_late_slots_pct" % k.lower().replace(" ", "")] = round(v[0], 4)
        if cl is not None:
            ok = cl["status"] == 0
            print("  tenant %s: %d requests (%d ok), client late p99 %.2f ms" % (cl["engine"], cl["n"], int(ok.sum()), pct(cl["late"], 99)))
            for k, lab in (("decode", "NVJPG decode"), ("infer", "inference   "), ("post", "NMS (CPU)   "), ("queue", "queue       "), ("server", "server total"), ("rtt", "client RTT  ")):
                v = cl[k][~np.isnan(cl[k])]
                if len(v):
                    print("    %s p50 %7.2f  p99 %7.2f  max %7.2f ms" % (lab, pct(v, 50), pct(v, 99), float(v.max())))
                    row["%s_p50" % k], row["%s_p99" % k] = round(pct(v, 50), 2), round(pct(v, 99), 2)
            print("    detections per image: median %.0f" % np.nanmedian(cl["ndet"]))
            row.update(engine=cl["engine"], requests=cl["n"], ndet=float(np.nanmedian(cl["ndet"])))
        if r is not None and r["ul"]:
            worst = max(r["ul"].values(), key=lambda d: d["p50"])
            srv50 = row.get("server_p50", 0.0); srv99 = row.get("server_p99", 0.0)
            print("  -> knob-dependent part of the E2E: last UL slot %.2f ms (p50, worst position; p99 %.2f) + GH200 %.2f ms (p50; p99 %.2f) = %.2f ms (p50 + p50), %.2f ms (p99 + p99)" % (
                worst["p50"] / 1e3, worst["p99"] / 1e3, srv50, srv99, worst["p50"] / 1e3 + srv50, worst["p99"] / 1e3 + srv99))
            row.update(last_ul_p50_ms=round(worst["p50"] / 1e3, 3), last_ul_p99_ms=round(worst["p99"] / 1e3, 3), var_p50_ms=round(worst["p50"] / 1e3 + srv50, 2),
                       var_p99_ms=round(worst["p99"] / 1e3 + srv99, 2))
        table.append(row)
    if out_csv and table:
        keys = []
        for r_ in table:
            for k in r_:
                if k not in keys:
                    keys.append(k)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(table)


if __name__ == "__main__":
    main()
