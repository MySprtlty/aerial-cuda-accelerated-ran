#!/usr/bin/env python3
"""
Static design points for the 16-cell bench attribution experiment (CONTEXT.md step 3): 60 points made of
existing test vectors, per-cell feature columns named after the hook record fields (dapp_est.ringio
UL_PDU_FIELDS / DL_PDU_FIELDS), and the run schedule of step 4. Nothing is executed.

    python3 design_points.py            -> points/design_points.csv, points/design_points.yaml, points/design_schedule.csv

Blocks
  A  the 16-cell load ladder already measured (c16_scan / c16_capgrid): L1..L5 + 4 variants          9 points
  B  one-axis-at-a-time around the stage-1 base, re-run at 16 cells (30 of the 8-cell scan points
     + PUSCH nUE 1/2/8, PDSCH nUE 4, PDCCH nDCI 8, which separate nUE from PRB and give 3-4 levels)   35 points
  C  2^(6-2) resolution-IV fractional factorial across the six channels (low/high TV per channel)     16 points
Feature values come from the FAPI PDUs recorded in testVectors/sweep_tv_manifest.yaml (actual.fapi.pdus),
i.e. what the hook would read, not the generator's request; the code-block count comes from the manifest's
38.212 segmentation (derived.ue_alloc.nCb). Per-cell values; the bench replicates the cell 16 times.
"""
import csv
import os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MANIFEST = os.path.join(ROOT, "testVectors", "sweep_tv_manifest.yaml")
OUT = os.path.join(HERE, "points")
CELLS = 16
MAX_UE_PER_CELL = 8            # non-64C build: 128 TBs / 128 PDSCH UEs per cell group
CHANNELS = ("PUSCH", "PUCCH", "PRACH", "SRS", "PDSCH", "PDCCH")
BASE = {"PUSCH": 60000, "PUCCH": 63000, "PDSCH": 61200, "PDCCH": 62000, "PRACH": 64000, "SRS": 65000}   # stage-1 base + light PRACH/SRS
LADDER_CTL = {"PUCCH": 63000, "PRACH": 64000, "SRS": 65000, "PDCCH": 62000}

# ---- feature extraction: FAPI PDU field -> hook field name ----
UL_MAP = {"rbSize": "rb_size", "nrOfLayers": "num_layers", "qamModOrder": "qam_mod_order", "targetCodeRate": "target_code_rate",
          "TBSize": "tb_size", "NrOfSymbols": "num_sym", "DmrsSymbPos": "ul_dmrs_sym_pos", "harqAckBitLength": "harq_ack_bit_len",
          "csiPart1BitLength": "csi_part1_bit_len", "FormatType": "format_type", "BitLenHarq": "bit_len_harq", "BitLenCsiPart1": "bit_len_csi1",
          "prbSize": "rb_size", "numAntPorts": "num_layers", "numSymbols": "srs_num_symbols", "numRa": "num_ra"}
DL_MAP = {"rbSize": "rb_size", "nrOfLayers": "num_layers", "qamModOrder": "qam_mod_order", "targetCodeRate": "target_code_rate",
          "TBSize": "tb_size", "NrOfSymbols": "num_sym", "DmrsSymbPos": "dl_dmrs_sym_pos", "numPRGs": "num_prgs", "numDlDci": "num_dl_dci"}
SUM_FIELDS = {"rb_size", "tb_size", "harq_ack_bit_len", "csi_part1_bit_len", "bit_len_harq", "bit_len_csi1", "num_prgs", "num_dl_dci"}
MAX_FIELDS = {"rb_size", "tb_size", "num_layers", "qam_mod_order", "target_code_rate", "num_sym", "format_type", "srs_num_symbols", "num_ra"}


def popcount(v):
    return bin(int(v)).count("1")


def load_manifest():
    m = yaml.safe_load(open(MANIFEST))
    tvs = m["tvs"] if isinstance(m["tvs"], list) else [dict(v, id=k) for k, v in m["tvs"].items()]
    return {int(t["id"]): t for t in tvs}


def tv_features(tv, ch):
    """{hook_field_agg: value} for one TV of channel ch, per cell."""
    f = {}
    pdus = ((tv.get("actual") or {}).get("fapi") or {}).get("pdus") or []
    der = tv.get("derived") or {}
    f["n_pdu"] = len(pdus)
    mp = DL_MAP if ch in ("PDSCH", "PDCCH") else UL_MAP
    for fapi_key, hook in mp.items():
        vals = [p[fapi_key] for p in pdus if fapi_key in p]
        if not vals:
            continue
        if hook in ("ul_dmrs_sym_pos", "dl_dmrs_sym_pos"):
            f["dmrs_sym_count_max"] = max(popcount(v) for v in vals)
            continue
        if hook in SUM_FIELDS:
            f[hook + "_sum"] = sum(vals)
        if hook in MAX_FIELDS:
            f[hook + "_max"] = max(vals)
    if ch in ("PUSCH", "PDSCH"):
        ua = der.get("ue_alloc") or []
        if ua:
            f["num_cb_sum"] = sum(u.get("nCb", 0) for u in ua)
            f["num_cb_max"] = max(u.get("nCb", 0) for u in ua)
    if ch == "PDSCH":
        f["prg_bf_sum"] = sum(p.get("numPRGs", 0) for p in pdus if p.get("enablePrcdBf"))
    if ch == "PDCCH":
        f["agg_level_sum"] = der.get("aggrL_sum")
        dcis = ((tv.get("actual") or {}).get("cuphy") or {}).get("dci") or []
        f["dci_payload_bits_sum"] = sum(d.get("Npayload", 0) for d in dcis)
        f["agg_level_max"] = max((d.get("aggrL", 0) for d in dcis), default=None)
    if ch == "PRACH":
        f["num_prach_ocas_sum"] = len(der.get("occasions") or [])
        f["prach_format"] = 0 if str(der.get("format")) == "0" else 1
    if ch == "SRS":
        f["ports_x_symbols_sum"] = sum(p.get("numAntPorts", 0) * p.get("numSymbols", 0) for p in pdus)
    return f


def level_label(tv):
    r = tv.get("requested") or {}
    keys = ("nUE", "total_prb", "layers", "mcs", "nCb", "spread", "nDCI", "AL", "fmt_bits", "format", "bits", "count", "fdm", "preambles", "ports", "symbols", "precoding", "uci")
    return " ".join("%s=%s" % (k, r[k] if not isinstance(r[k], dict) else "on") for k in keys if k in r)


def main():
    tvs = load_manifest()
    m = yaml.safe_load(open(os.path.join(HERE, "matrix.yaml")))
    mpts = m["points"]
    c16 = yaml.safe_load(open(os.path.join(OUT, "c16_points.yaml")))["points"]
    points = []          # (id, block, name, source, measured_cells, {ch: tv}, note)

    # A: ladder (existing, 16 cells)
    for nm in ("c16_L1", "c16_L2", "c16_L3", "c16_L4", "c16_L5", "c16_L5_2L", "c16_L5_uci", "c16_L5_crcfail", "c16_L5_ctlmax"):
        chans = {k: v for k, v in c16[nm].items() if not k.startswith("_")}
        points.append(("A%02d" % (len(points) + 1), "A", nm, "existing c16_scan/c16_capgrid (old bench slot layout)", 16, chans, "ladder: nUE and PRB move together (34 PRB/UE), 4 layers, MCS 27"))

    # B: one axis at a time around the base, at 16 cells
    b_names = m["profiles"]["c8_scan"]["points"] + ["pusch_ue_1", "pusch_ue_2", "pusch_ue_8", "pdsch_ue_4", "pdcch_dci_8"]
    extra = {"pdsch_ue_4": {"PDSCH": 61202}, "pdcch_dci_8": {"PDCCH": 62003}}
    for nm in b_names:
        ov = mpts.get(nm, extra.get(nm, {}))
        chans = dict(BASE); chans.update({k: v for k, v in ov.items() if not k.startswith("_")})
        src = "existing c8_scan (8 cells, PRACH/SRS absent there)" if nm in m["profiles"]["c8_scan"]["points"] else "new"
        note = ""
        if nm == "pdsch_ue_16":
            note = "INFEASIBLE at 16 cells: 16 UE x 16 cells > 128 PDSCH UEs per cell group; keep the 8-cell data only"
        points.append(("B%02d" % (len([p for p in points if p[1] == "B"]) + 1), "B", nm, src, 8 if src.startswith("existing") else 0, chans, note))

    # C: 2^(6-2) resolution IV fractional factorial, generators E = ABC, F = BCD
    low = {"PUSCH": 68000, "PDSCH": 68100, "PDCCH": 62001, "PUCCH": 63010, "PRACH": 64000, "SRS": 65000}
    high = {"PUSCH": 68004, "PDSCH": 68104, "PDCCH": 62004, "PUCCH": 63032, "PRACH": 64007, "SRS": 65014}
    order = ("PUSCH", "PDSCH", "PDCCH", "PUCCH", "PRACH", "SRS")
    for i in range(16):
        a, b, c, d = ((i >> 3) & 1) * 2 - 1, ((i >> 2) & 1) * 2 - 1, ((i >> 1) & 1) * 2 - 1, (i & 1) * 2 - 1
        e, fq = a * b * c, b * c * d
        signs = dict(zip(order, (a, b, c, d, e, fq)))
        chans = {ch: (high[ch] if signs[ch] > 0 else low[ch]) for ch in order}
        lab = "".join("+" if signs[ch] > 0 else "-" for ch in order)
        points.append(("C%02d" % (i + 1), "C", "ff_" + lab, "new", 0, chans, "2^(6-2) res IV; E=ABC, F=BCD; order PUSCH PDSCH PDCCH PUCCH PRACH SRS"))

    # ---- feature columns ----
    feat_cols = []
    rows = []
    for pid, block, nm, src, meas, chans, note in points:
        row = {"point_id": pid, "block": block, "name": nm, "source": src, "measured_cells": meas, "cells": CELLS,
               "feasible_16c": 1, "note": note}
        for ch in CHANNELS:
            tv = tvs.get(chans.get(ch))
            row["tv_" + ch] = chans.get(ch, "")
            row["levels_" + ch] = level_label(tv) if tv else ""
            feats = tv_features(tv, ch) if tv else {"n_pdu": 0}
            for k, v in feats.items():
                col = "%s_%s" % (ch, k)
                if col not in feat_cols:
                    feat_cols.append(col)
                row[col] = v
        ue = max(row.get("PUSCH_n_pdu", 0), row.get("PDSCH_n_pdu", 0))
        if ue * CELLS > 128:
            row["feasible_16c"] = 0
        rows.append(row)
    cols = ["point_id", "block", "name", "source", "measured_cells", "cells", "feasible_16c"] + ["tv_" + c for c in CHANNELS] + ["levels_" + c for c in CHANNELS] + sorted(feat_cols) + ["note"]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "design_points.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    # points yaml for sweep.py (feasible points only)
    ypts = {}
    for pid, block, nm, src, meas, chans, note in points:
        r = next(x for x in rows if x["point_id"] == pid)
        if r["feasible_16c"]:
            ypts["dp_" + pid] = dict({"_cells": CELLS}, **{ch: chans[ch] for ch in CHANNELS if ch in chans})
    with open(os.path.join(OUT, "design_points.yaml"), "w") as f:
        f.write("# generated by design_points.py; use with: sweep.py --points-file points/design_points.yaml\n")
        yaml.safe_dump({"note": "60 static design points for the 16-cell attribution experiment (blocks A ladder, B one-axis, C 2^(6-2))", "points": ypts}, f, sort_keys=False)

    # ---- schedule (step 4): none + 26x b8 cap30 + cap100 for every point, +{26x b8 cap60, 26n b1 cap100} for 10 points ----
    RUN_MIN = 4.0         # 120 s traffic + 30 s warm-up + ~90 s setup/collect
    ten_extra = ["A01", "A04", "A05", "B01", "B35", "B14", "C01", "C16", "C06", "C11"]      # ladder ends, base, PUSCH ue8, PDSCH prb273, factorial corners/mixes
    sched = []
    feas = [r for r in rows if r["feasible_16c"]]
    for r in feas:
        for tenant, cap, batch in (("none", 100, 1), ("yolo26x", 30, 8), ("yolo26x", 100, 8)):
            sched.append({"point_id": r["point_id"], "tenant": tenant, "cap_pct": cap, "batch": batch, "minutes": RUN_MIN, "phase": "main"})
        if r["point_id"] in ten_extra:
            for tenant, cap, batch in (("yolo26x", 60, 8), ("yolo26n", 100, 1)):
                sched.append({"point_id": r["point_id"], "tenant": tenant, "cap_pct": cap, "batch": batch, "minutes": RUN_MIN, "phase": "extra"})
    for tenant, cap, batch in (("yolo26x", 30, 8), ("yolo26x", 100, 8), ("yolo26x", 60, 8), ("yolo26n", 100, 1)):
        sched.append({"point_id": "tenant_alone", "tenant": tenant, "cap_pct": cap, "batch": batch, "minutes": 1.5, "phase": "baseline"})
    pilot = [s for s in sched if s["point_id"] in ("A01", "A04", "C01")]
    for s in sched:
        s["order"] = 0 if s in pilot else 1
    sched.sort(key=lambda s: (s["order"], s["point_id"], s["phase"], s["tenant"], s["cap_pct"]))
    with open(os.path.join(OUT, "design_schedule.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order", "phase", "point_id", "tenant", "cap_pct", "batch", "minutes"])
        w.writeheader(); w.writerows(sched)
    tot = sum(s["minutes"] for s in sched)
    print("points: %d (feasible at 16 cells: %d), feature columns: %d" % (len(rows), len(feas), len(feat_cols)))
    print("blocks:", {b: sum(1 for r in rows if r["block"] == b) for b in "ABC"})
    print("runs: %d (pilot %d), total %.1f h at %.0f min/run" % (len(sched), len(pilot), tot / 60, RUN_MIN))
    print("wrote", os.path.join(OUT, "design_points.csv"), os.path.join(OUT, "design_points.yaml"), os.path.join(OUT, "design_schedule.csv"))


if __name__ == "__main__":
    main()
