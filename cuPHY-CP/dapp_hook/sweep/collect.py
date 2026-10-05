#!/usr/bin/env python3
"""
collect: turn the raw files of one run directory into tables.

    python3 collect.py --run runs/<mid>/<run_id> [--drop-first-s 10] [--gap-us 150]

Inputs (written by run_one.py)            Outputs
  bench.out      bench stdout              bench_avg.csv         per channel/slot average run time (END-START, us) + pattern total
  nsys_trace.*   CUDA kernel trace         chan_samples.parquet  one row per (channel, slot execution): t_start_ns, dur_us, n_kernels, slot_rel
                                           chan_samples.csv      same (CSV for plotting without pyarrow)
  worker.csv     tenant fps/latency        tenant summary in summary.json
  dcgm.csv       dcgmi dmon                dcgm_parsed.csv       1 s counters (SMACT/SMOCC/DRAMA are fractions 0..1, GPUTL %, POWER W)
  smi.csv        nvidia-smi -l 1           (kept)
                                           summary.json          per channel: avg (stdout), n/mean/p50/p99/p99.99/max (nsys), tenant, dcgm

Channel attribution: cubb_gpu_test_bench gives every enabled channel its own
CUDA context, created in the --M order [PRACH][PDCCH][PUCCH] PDSCH PUSCH [SSB][SRS];
nsys numbers contexts in creation order, the lowest id being the main thread's
context that only runs the pattern delay kernel. The mapping is therefore
positional, and the kernel names of each context are checked against per-channel
regexes as a sanity check (summary.json "nsys.contexts" / "nsys.name_votes").
Scheduling kernels (delay_kernel_us, delay_ns_wait_until, get_ptimer_kernel,
get_sm_id_kernel, convert_kernel) are not channel work and are dropped before
bursts are formed. A "slot execution" is a burst of work kernels in one context
separated from the next by more than --gap-us; slots are 500 us apart so bursts
never merge. Runs inside the Aerial container (numpy, pandas/pyarrow optional).
"""
import argparse
import csv
import json
import os
import re
import sqlite3
import subprocess
import sys

import numpy as np
import yaml

SM_ORDER = ("PRACH", "PDCCH", "PUCCH", "PDSCH", "PUSCH", "SSB", "SRS")
CH_RE = {   # most specific first (srsChEst* would otherwise match the PUSCH chEst pattern)
    "SRS": re.compile(r"srs", re.I),
    "PRACH": re.compile(r"prach|block_fft|memcpyRssi", re.I),
    "PUCCH": re.compile(r"pucch", re.I),
    "PDCCH": re.compile(r"pdcch|MultipleDCIs|polar", re.I),
    "PDSCH": re.compile(r"pdsch|ldpc_encode|fused_dl|fused_dmrs|prepare_crc|crcDownlink|dl_rm|modulation", re.I),
    "PUSCH": re.compile(r"pusch|chEst|eqMmse|ldpc2|de_rate|rsrp|rssi|cfoTa|noiseIntf|crcUplink|softDemap|uci", re.I),
    "SSB": re.compile(r"ssb|pbch", re.I),
    "CSIRS": re.compile(r"csirs|csi_rs", re.I),
}
SKIP_RE = re.compile(r"^(delay_kernel|delay_ns_wait|get_ptimer|get_sm_id|convert_kernel|spin)", re.I)
BENCH_RE = re.compile(r"^(?:Slot # (\d+): )?[Aa]verage (\S+) run time: ([\d.]+) us from ([\d.]+) \(averaged over (\d+) iterations\)")
PATTERN_RE = re.compile(r"^average slot pattern run time: ([\d.]+) us \(averaged over (\d+) iterations\)")


def parse_bench(path):
    rows = []
    pattern = -1
    for line in open(path, errors="replace"):
        s = line.strip()
        if s.startswith("Slot pattern #"):
            pattern = int(s.split("#")[1])
            continue
        m = PATTERN_RE.match(s)
        if m:
            rows.append({"pattern": pattern, "channel": "PATTERN", "slot": -1, "end_us": float(m.group(1)), "start_us": 0.0,
                         "dur_us": float(m.group(1)), "iterations": int(m.group(2))})
            continue
        m = BENCH_RE.match(s)
        if m:
            name = m.group(2)
            base = re.sub(r"[12]$", "", name) if re.match(r"^(PUSCH|PUCCH|SRS|ULBFW)[12]$", name) else name
            slot = int(m.group(1)) if m.group(1) else (1 if name.endswith("2") and base != "SRS" else 0)
            rows.append({"pattern": pattern, "channel": base, "slot": slot, "end_us": float(m.group(3)), "start_us": float(m.group(4)),
                         "dur_us": round(float(m.group(3)) - float(m.group(4)), 2), "iterations": int(m.group(5))})
    return rows


def channels_in_run(run_dir):
    """Enabled channels in --M order, from bench.yaml."""
    try:
        y = yaml.safe_load(open(os.path.join(run_dir, "bench.yaml")))
        present = set()
        for s in y.get("slots", []):
            present.update(s.keys())
        return [ch for ch in SM_ORDER if ch in present]
    except Exception:
        return []


def parse_nsys(run_dir, gap_us, drop_first_s):
    rep = os.path.join(run_dir, "nsys_trace.nsys-rep")
    sq = os.path.join(run_dir, "nsys_trace.sqlite")
    if not os.path.exists(rep):
        return [], {"error": "no nsys report"}
    if not os.path.exists(sq) or os.path.getmtime(sq) < os.path.getmtime(rep):
        subprocess.run(["nsys", "export", "--type", "sqlite", "-o", sq, "--force-overwrite", "true", rep], capture_output=True)
    db = sqlite3.connect(sq)
    tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tabs:
        return [], {"error": "no kernel table"}
    names = {i: n for i, n in db.execute("select id, value from StringIds")}
    work = {}       # ctx -> kernel count of work kernels
    votes = {}      # ctx -> {channel: count}
    for ctx, sid, n in db.execute("select contextId, shortName, count(*) from CUPTI_ACTIVITY_KIND_KERNEL group by contextId, shortName"):
        nm = names.get(sid, "")
        if SKIP_RE.match(nm):
            continue
        work[ctx] = work.get(ctx, 0) + n
        for ch, rx in CH_RE.items():
            if rx.search(nm):
                votes.setdefault(ctx, {}).setdefault(ch, 0)
                votes[ctx][ch] += n
                break
    order = channels_in_run(run_dir)
    work_ctx = sorted(c for c in work if work[c] > 0)
    name_votes = {int(c): max(v.items(), key=lambda kv: kv[1])[0] for c, v in votes.items()}
    by_order = {c: ch for c, ch in zip(work_ctx, order)} if order and len(work_ctx) == len(order) else {}
    by_name = {c: name_votes[int(c)] for c in work_ctx if int(c) in name_votes}
    ctx_ch = {}
    # The bench creates the channel contexts from concurrent threads, so the creation order (= nsys context id
    # order) is usually the --M order but can race (seen 2026-09-21: PUSCH before PDSCH with all six channels on).
    # Kernel names identify a channel unambiguously, so use them whenever they cover the expected channel set
    # one-to-one; fall back to the creation order, then to bare votes.
    if order and len(by_name) == len(order) and sorted(by_name.values()) == sorted(order):
        ctx_ch, mapping_source = by_name, "names"
    elif by_order:
        ctx_ch, mapping_source = by_order, "order"
    else:
        ctx_ch, mapping_source = {c: name_votes[int(c)] for c in votes}, "votes"
    mismatch = [int(c) for c in ctx_ch if name_votes.get(int(c)) not in (None, ctx_ch[c])]
    order_mismatch = [int(c) for c in by_order if ctx_ch.get(c) != by_order[c]]   # informational: creation order != --M order

    skip_ids = [i for i, n in names.items() if SKIP_RE.match(n)]
    skip_sql = ",".join(str(i) for i in skip_ids) or "-1"
    t_first = db.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    t_cut = t_first + int(drop_first_s * 1e9)
    slots_of = channel_slots(run_dir)

    # pattern starts: the main thread's context (lowest id, no work kernels) runs one delay kernel per pattern;
    # its end is the pattern's t0 and slot k of the pattern is scheduled at t0 + k * 500 us.
    main_ctx = min(c for (c,) in db.execute("select distinct contextId from CUPTI_ACTIVITY_KIND_KERNEL"))
    t0s = np.array([r[0] for r in db.execute("select end from CUPTI_ACTIVITY_KIND_KERNEL where contextId=? order by start", (main_ctx,))], dtype=np.int64)
    samples, seg_info = [], {}
    for ctx, ch in ctx_ch.items():
        E = len(slots_of.get(ch, [])) or 1
        rows = db.execute("select streamId, start, end from CUPTI_ACTIVITY_KIND_KERNEL where contextId=? and shortName not in (%s) order by start" % skip_sql, (ctx,)).fetchall()
        by_stream = {}
        for sid, st, en in rows:
            by_stream.setdefault(sid, []).append((st, en))
        ok = len(t0s) > 2 and bool(by_stream)
        plan = {}     # stream -> ("own", exec_idx, k) | ("shared", None, k)
        if ok:
            npat = len(t0s)
            ks = {sid: int(round(len(v) / float(npat))) for sid, v in by_stream.items()}
            if any(k == 0 for k in ks.values()) or any(abs(len(by_stream[sid]) / float(npat) - k) > 0.05 * k + 0.05 for sid, k in ks.items()):
                ok = False
            else:
                by_k = {}
                for sid, k in ks.items():
                    by_k.setdefault(k, []).append(sid)
                for k, sids in by_k.items():
                    sids.sort()
                    if len(sids) % E == 0:                # one set of streams per execution (PUSCH1/PUSCH2, PUCCH1/PUCCH2, per-slot PDSCH streams)
                        per = len(sids) // E
                        for i, sid in enumerate(sids):
                            plan[sid] = ("own", i // per, k)
                    elif k % E == 0:                      # one stream shared by the E executions, run in order
                        for sid in sids:
                            plan[sid] = ("shared", None, k)
                    else:
                        ok = False
        if not ok:
            # fallback: gap-based bursts (fine for light load, merges executions when a slot overruns)
            st = np.array([r[1] for r in rows], dtype=np.int64)
            en = np.array([r[2] for r in rows], dtype=np.int64)
            if len(st) == 0:
                continue
            cum_end = np.maximum.accumulate(en)
            cuts = np.where((st[1:] - cum_end[:-1]) > gap_us * 1000)[0] + 1
            idx = np.concatenate([[0], cuts, [len(st)]])
            for a, b in zip(idx[:-1], idx[1:]):
                b0, b1 = int(st[a]), int(en[a:b].max())
                if b0 >= t_cut:
                    samples.append({"channel": ch, "ctx": int(ctx), "pattern": -1, "exec": -1, "slot": -1, "t_start_ns": b0, "dur_us": (b1 - b0) / 1000.0,
                                    "start_delay_us": "", "lat_us": (b1 - b0) / 1000.0, "n_kernels": int(b - a)})
            seg_info[ch] = "gap"
            continue
        n_complete = min(len(by_stream[sid]) // plan[sid][2] for sid in plan)
        n_complete = min(n_complete, len(t0s))
        agg = {}      # (pattern, exec) -> [start_min, end_max, n]
        for sid, (kind, ex, k) in plan.items():
            arr = by_stream[sid]
            per_exec = k // E if kind == "shared" else k
            for i in range(n_complete * k):
                pidx = i // k
                e = ((i % k) // per_exec) if kind == "shared" else ex
                st, en = arr[i]
                cur = agg.get((pidx, e))
                if cur is None:
                    agg[(pidx, e)] = [st, en, 1]
                else:
                    if st < cur[0]:
                        cur[0] = st
                    if en > cur[1]:
                        cur[1] = en
                    cur[2] += 1
        for (pidx, e), (st, en, n) in sorted(agg.items()):
            t0 = int(t0s[pidx])
            if t0 < t_cut:
                continue
            slot = slots_of[ch][e] if ch in slots_of and e < len(slots_of[ch]) else -1
            slot_t = t0 + slot * 500_000 if slot >= 0 else st
            samples.append({"channel": ch, "ctx": int(ctx), "pattern": int(pidx), "exec": int(e), "slot": int(slot), "t_start_ns": int(st),
                            "dur_us": (en - st) / 1000.0, "start_delay_us": (st - slot_t) / 1000.0, "lat_us": (en - slot_t) / 1000.0, "n_kernels": int(n)})
        seg_info[ch] = "pattern"
    samples.sort(key=lambda x: x["t_start_ns"])
    info = {"contexts": {int(k): v for k, v in ctx_ch.items()}, "name_votes": name_votes, "mapping_mismatch": mismatch,
            "mapping_source": mapping_source, "order_mismatch": order_mismatch, "segmentation": seg_info,
            "t_first_ns": int(t_first), "t_cut_ns": int(t_cut), "n_patterns": int(len(t0s)),
            "pattern_period_us": float(np.median(np.diff(t0s)) / 1000.0) if len(t0s) > 2 else None,
            "window_s": round((db.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] - t_first) / 1e9, 1)}
    return samples, info


def channel_slots(run_dir):
    """{channel: [slot indices of the 8-slot pattern in which it runs]} from bench.yaml."""
    out = {}
    try:
        y = yaml.safe_load(open(os.path.join(run_dir, "bench.yaml")))
        for i, sl in enumerate(y.get("slots", [])):
            for ch in sl:
                out.setdefault(ch, []).append(i)
    except Exception:
        pass
    return out


def parse_dcgm(path):
    rows = []
    if not os.path.exists(path):
        return rows
    hdr = None
    for line in open(path, errors="replace"):
        if line.startswith("#Entity"):
            hdr = line.replace("#", "").split()[1:]   # metric names after "Entity"
            continue
        if line.startswith("GPU") and hdr:
            parts = line.split()
            vals = parts[2:]   # "GPU", "<id>", metrics...
            if len(vals) >= len(hdr):
                rows.append({"entity": parts[0] + parts[1], **{hdr[i]: vals[i] for i in range(len(hdr))}})
    return rows


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--drop-first-s", type=float, default=10.0)
    ap.add_argument("--gap-us", type=float, default=150.0)
    a = ap.parse_args()
    rd = a.run
    summary = {"run_dir": rd, "channels": {}}

    bench = parse_bench(os.path.join(rd, "bench.out")) if os.path.exists(os.path.join(rd, "bench.out")) else []
    with open(os.path.join(rd, "bench_avg.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pattern", "channel", "slot", "start_us", "end_us", "dur_us", "iterations"])
        w.writeheader()
        w.writerows(bench)
    for r in bench:
        if r["channel"] == "PATTERN":
            summary["pattern_avg_us"] = r["dur_us"]
            summary["iterations"] = r["iterations"]
            continue
        c = summary["channels"].setdefault(r["channel"], {})
        c.setdefault("stdout_avg_us", {})[str(r["slot"])] = r["dur_us"]
    for ch, c in summary["channels"].items():
        if "stdout_avg_us" in c:
            c["stdout_mean_us"] = float(np.mean(list(c["stdout_avg_us"].values())))

    samples, info = parse_nsys(rd, a.gap_us, a.drop_first_s)
    summary["nsys"] = info
    if samples:
        keys = ["channel", "ctx", "pattern", "exec", "slot", "t_start_ns", "dur_us", "start_delay_us", "lat_us", "n_kernels"]
        with open(os.path.join(rd, "chan_samples.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(samples)
        try:
            import pandas as pd
            pd.DataFrame(samples)[keys].to_parquet(os.path.join(rd, "chan_samples.parquet"), index=False)
        except Exception as e:
            summary["parquet_error"] = repr(e)
        for ch in sorted(set(s["channel"] for s in samples)):
            # primary statistic = dur_us: first kernel start -> last kernel end of ONE slot execution of the pipeline. This is what the bench
            # prints (END-START) and what NVIDIA's budgets refer to. lat_us (scheduled slot start -> end) is kept as a secondary column:
            # it adds the launch delay and, for PRACH/SRS, the bench's deliberate ~500 us scheduling offset.
            v = np.array([s["dur_us"] for s in samples if s["channel"] == ch])
            l = np.array([float(s["lat_us"]) for s in samples if s["channel"] == ch and s["lat_us"] != ""])
            c = summary["channels"].setdefault(ch, {})
            c["nsys"] = {"n": int(len(v)), "mean": float(v.mean()), "p50": pct(v, 50), "p99": pct(v, 99), "p99_99": pct(v, 99.99), "max": float(v.max()),
                         "lat_mean": float(l.mean()) if len(l) else None, "lat_p99": pct(l, 99),
                         "kernels_per_exec": float(np.mean([s["n_kernels"] for s in samples if s["channel"] == ch]))}

    if os.path.exists(os.path.join(rd, "worker.csv")):
        rows = list(csv.DictReader(open(os.path.join(rd, "worker.csv"))))
        if rows and "total_ms" in rows[0]:      # openloop_worker.py: one row per request (rate, queue_ms, service_ms, total_ms)
            tot = np.array([float(r["total_ms"]) for r in rows])
            svc = np.array([float(r["service_ms"]) for r in rows])
            span_s = (int(rows[-1]["t_arrival_ns"]) - int(rows[0]["t_arrival_ns"])) / 1e9 if len(rows) > 1 else 0.0
            summary["tenant"] = {"open_loop": True, "requests": len(rows), "rate_offered": rows[0]["rate"], "rate_served": len(rows) / span_s if span_s > 0 else None,
                                 "fps_mean": len(rows) / span_s if span_s > 0 else 0.0, "lat_ms_p50": pct(tot, 50), "lat_ms_p99": pct(tot, 99), "lat_ms_max": float(tot.max()),
                                 "service_ms_p50": pct(svc, 50), "service_ms_p99": pct(svc, 99)}
        elif rows:
            fps = np.array([float(r["fps"]) for r in rows])
            p50 = np.array([float(r["lat_ms_p50"]) for r in rows])
            p99 = np.array([float(r["lat_ms_p99"]) for r in rows])
            summary["tenant"] = {"seconds": len(rows), "fps_mean": float(fps.mean()), "fps_p5": pct(fps, 5), "lat_ms_p50": float(np.median(p50)),
                                 "lat_ms_p99": float(np.median(p99)), "lat_ms_max": max(float(r["lat_ms_max"]) for r in rows),
                                 "cap_pct": rows[0]["cap_pct"], "prio": rows[0]["prio"], "batch": rows[0]["batch"]}
    dc = parse_dcgm(os.path.join(rd, "dcgm.csv"))
    if dc:
        with open(os.path.join(rd, "dcgm_parsed.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dc[0].keys()))
            w.writeheader()
            w.writerows(dc)

        def col(name):
            v = []
            for r in dc:
                try:
                    v.append(float(r[name]))
                except (KeyError, ValueError):
                    pass
            return np.array(v)
        summary["dcgm"] = {k: {"mean": float(v.mean()), "p95": pct(v, 95), "max": float(v.max())} for k, v in
                           (("SMACT", col("SMACT")), ("SMOCC", col("SMOCC")), ("TENSO", col("TENSO")), ("DRAMA", col("DRAMA")),
                            ("GPUTL", col("GPUTL")), ("POWER", col("POWER"))) if len(v)}
    json.dump(summary, open(os.path.join(rd, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "run_dir"}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
