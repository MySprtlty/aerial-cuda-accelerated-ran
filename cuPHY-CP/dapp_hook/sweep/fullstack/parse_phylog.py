#!/usr/bin/env python3
"""parse_phylog.py <phy.log> <out.csv>
One row per slot. Times in microseconds relative to the slot's air time t0 = L1 slot-indication tick + 1500 us
(all CLOCK_REALTIME): L2 adapter enqueue, PDSCH/PDCCH aggregate pipeline stages, compression completion (cell 0),
per-cell DL communication tasks (min/max over cells), PUSCH / PUCCH / PRACH / SRS aggregate GPU window end (Max GPU Time END) and GPU run, order kernel run.
Lines of one slot arrive within a few ms; the same (sfn, slot) recurs every 10.24 s, so a record is closed when the
next L2A line for that key arrives. Streaming, string search only."""
import sys, csv
src, dst = sys.argv[1], sys.argv[2]
COLS = ["sfn", "slot", "pos", "t0_ns", "enq_us", "l2_est_us", "fapi_us", "ul", "dl", "cells",
        "pdsch_setup_start", "pdsch_setup_end", "pdsch_run_end", "pdsch_gpu_setup_us", "pdsch_gpu_run_us", "pdsch_compl_end", "pdsch_wait_start",
        "pdcch_setup_start", "pdcch_gpu_run_us", "comp_cpu_start", "comp_wait_start", "comp_wait_end", "comp_gpu_us", "chan_to_comp_gap_us",
        "cplane_start_min", "cplane_end_max", "uprep_end_max", "utx_start_min", "utx_end_max",
        "pusch_setup_start", "pusch_gpu_end", "pusch_gpu_run_us", "order_run_us",
        "pucch_gpu_end", "pucch_gpu_run_us", "prach_gpu_end", "prach_gpu_run_us", "srs_gpu_end", "srs_gpu_run_us"]
def num(s, key, end=" "):
    i = s.find(key)
    if i < 0: return None
    i += len(key); j = s.find(end, i)
    try: return int(s[i:j] if j > 0 else s[i:].strip())
    except ValueError: return None
def fnum(s, key):
    i = s.find(key)
    if i < 0: return None
    i += len(key); j = s.find(" ", i)
    try: return float(s[i:j])
    except ValueError: return None
rows = {}
n = 0
with open(src, errors="replace") as f, open(dst, "w", newline="") as out:
    w = csv.DictWriter(out, fieldnames=COLS, extrasaction="ignore"); w.writeheader()
    def close(key):
        r = rows.pop(key, None)
        if r: w.writerow({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
    for ln in f:
        n += 1
        if "[L2A.PROCESSING_TIMES]" in ln:
            i = ln.find("SFN "); j = ln.find(" ", i + 4)
            try: sfn, slot = ln[i + 4:j].split(".")
            except ValueError: continue
            key = (int(sfn), int(slot)); close(key)
            tick = num(ln, "l1_slot_ind_tick="); enq = num(ln, "l1_enqueue_complete_time=")
            if tick is None or enq is None: continue
            t0 = tick + 1_500_000
            rows[key] = {"sfn": key[0], "slot": key[1], "pos": key[1] % 10, "t0_ns": t0, "enq_us": (enq - t0) / 1e3,
                         "l2_est_us": (num(ln, "L2 Estimate=", "ns") or 0) / 1e3, "fapi_us": (num(ln, "FAPI Proc Dur=", "ns") or 0) / 1e3,
                         "ul": num(ln, "UL=", " "), "dl": num(ln, "DL=", " "), "cells": 0}
            continue
        if "times ===>" not in ln: continue
        i = ln.find("SFN "); j = ln.find(" ", i + 4)
        try: sfn, slot = ln[i + 4:j].split(".")
        except ValueError: continue
        r = rows.get((int(sfn), int(slot)))
        if r is None: continue
        t0 = r["t0_ns"]
        def rel(v): return None if v is None else (v - t0) / 1e3
        if " PDSCH Aggr " in ln:
            s0 = ln.find("[CPU CUDA Setup]"); s1 = ln.find("[CPU CUDA Run]"); s2 = ln.find("[Max GPU Time]"); s3 = ln.find("[CPU wait GPU Time]")
            r["pdsch_setup_start"] = rel(num(ln[s0:], "START: ")); r["pdsch_setup_end"] = rel(num(ln[s0:], "END: ")); r["pdsch_run_end"] = rel(num(ln[s1:], "END: "))
            r["pdsch_gpu_setup_us"] = fnum(ln, "[GPU Setup] { DURATION: "); r["pdsch_gpu_run_us"] = fnum(ln, "[GPU Run] { DURATION: ")
            r["pdsch_compl_end"] = rel(num(ln[s2:], "END: ")); r["pdsch_wait_start"] = rel(num(ln[s3:], "START: "))
        elif " Aggr PDCCH DL " in ln:
            s0 = ln.find("[CPU CUDA Setup]"); r["pdcch_setup_start"] = rel(num(ln[s0:], "START: ")); r["pdcch_gpu_run_us"] = fnum(ln, "[GPU Run] { DURATION: ")
        elif " COMPRESSION DL " in ln:
            s0 = ln.find("[CPU CUDA time]"); s1 = ln.find("[CPU wait GPU Time]")
            r["comp_cpu_start"] = rel(num(ln[s0:], "START: ")); r["comp_wait_start"] = rel(num(ln[s1:], "START: ")); r["comp_wait_end"] = rel(num(ln[s1:], "END: "))
            r["comp_gpu_us"] = fnum(ln, "[GPU Execution Duration] { DURATION: "); r["chan_to_comp_gap_us"] = fnum(ln, "[Channel To Compression Gap] { DURATION: ")
        elif " DL Communication Tasks Cell " in ln:
            if "[DRV.MAP_DL_VERBOSE]" not in ln: continue     # cell 0 is also logged under the base tag; count each cell once
            c0 = ln.find("[C-plane]"); u0 = ln.find("[U-plane prepare]"); x0 = ln.find("[U-plane tx]")
            vals = {"cplane_start_min": (rel(num(ln[c0:], "START: ")), min), "cplane_end_max": (rel(num(ln[c0:], "END: ")), max),
                    "uprep_end_max": (rel(num(ln[u0:], "END: ")), max), "utx_start_min": (rel(num(ln[x0:], "START: ")), min), "utx_end_max": (rel(num(ln[x0:], "END: ")), max)}
            r["cells"] += 1
            for k, (v, fn) in vals.items():
                if v is not None: r[k] = v if r.get(k) is None else fn(r[k], v)
        elif " PUSCH Aggr " in ln:
            s2 = ln.find("[Max GPU Time]"); s0 = ln.find("[CPU CUDA Setup]")
            r["pusch_setup_start"] = rel(num(ln[s0:], "START: ")); r["pusch_gpu_end"] = rel(num(ln[s2:], "END: ")); r["pusch_gpu_run_us"] = fnum(ln, "[GPU Run] { DURATION: ")
        elif " PUCCH Aggr " in ln or " PRACH Aggr " in ln or " SRS Aggr " in ln:
            ch = "pucch" if " PUCCH Aggr " in ln else "prach" if " PRACH Aggr " in ln else "srs"
            s2 = ln.find("[Max GPU Time]")
            r[ch + "_gpu_end"] = rel(num(ln[s2:], "END: ")); r[ch + "_gpu_run_us"] = fnum(ln, "[GPU Run] { DURATION: ")
        elif " ORDER " in ln and " Cell 41 " in ln:
            r["order_run_us"] = fnum(ln, "[GPU Order Run] { DURATION: ")
    for key in list(rows): close(key)
print("parsed %d lines -> %s" % (n, dst))
