#!/usr/bin/env python3
"""
tv_index: build a parameter index of every per-channel cuPHY test vector
(testVectors/TVnr_*_<CH>_gNB_CUPHY_s<slot>p<pdu>.h5) so a sweep can pick the
TV closest to a requested (UEs, PRB, layers, MCS, ...) point without MATLAB.

    python3 tv_index.py --tv-dir /opt/nvidia/cuBB/testVectors --out tv_index.csv [--jobs 8]

One row per file. Curated columns per channel (see CURATED) plus `params`:
a JSON dict of every small compound dataset in the file (first element),
so anything not curated is still searchable. Run inside the Aerial container
(h5py 3.13, numpy 2.2). ~7k files, a few minutes.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np

NAME_RE = re.compile(r"^TVnr_(?:(DLMIX|ULMIX)_)?(\d+)_([A-Z0-9_]+?)_gNB_CUPHY_s(\d+)p(\d+)\.h5$")

CURATED = ["n_ue", "n_grp", "n_tb", "n_cw", "n_layers_max", "n_layers_sum", "n_prb_tot", "n_prb_max", "n_prb_bwp",
           "mcs_min", "mcs_max", "mcs_table", "qam_max", "tb_bytes_tot", "tb_bytes_max", "n_cb_tot", "n_cb_max",
           "rv_max", "n_sym", "n_dmrs_sym", "n_rx", "n_tx", "mu", "sinr_db", "harq_bits", "csi1_bits",
           "n_dci", "agg_sum", "agg_max", "dci_bits_sum", "n_coreset",
           "pucch_format", "n_uci", "uci_bits", "n_prach", "prach_lra", "prach_nrep", "prach_cfg_idx", "prach_nroot",
           "srs_n_ue", "srs_n_ports", "srs_n_sym", "n_ssb", "n_csirs", "precoded"]


def _val(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return _val(x.reshape(-1)[0])
        return [_val(v) for v in x.reshape(-1)[:16]]
    if isinstance(x, bytes):
        return x.decode(errors="replace")
    return x


def compound_rows(ds, max_rows=64):
    """All rows of a small compound dataset as list of dicts."""
    if ds.size == 0 or ds.size > max_rows or ds.dtype.names is None:
        return []
    arr = ds[()].reshape(-1)
    return [{n: _val(r[n]) for n in ds.dtype.names} for r in arr]


def popcount(v):
    return bin(int(v)).count("1")


def index_file(path):
    name = os.path.basename(path)
    m = NAME_RE.match(name)
    if not m:
        return None
    fam, tvid, ch, slot, pdu = m.group(1) or "SINGLE", int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
    row = {"file": name, "fam": fam, "tvid": tvid, "channel": ch, "slot": slot, "pdu": pdu, "size_mb": round(os.path.getsize(path) / 1e6, 1)}
    for k in CURATED:
        row[k] = ""
    params = {}
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for k in keys:
                d = f[k]
                if isinstance(d, h5py.Dataset) and d.dtype.names is not None and 0 < d.size <= 64 and not k.startswith("reference") \
                        and "PM_W" not in k and k not in ("DataRx", "X_tf", "Xtf", "Noise_pwr", "ShiftSeq", "UnShiftSeq"):
                    rows = compound_rows(d)
                    if rows:
                        params[k] = rows if len(rows) > 1 else rows[0]
            # scalar-ish SNR/noise references
            for k in ["reference_sinrdB", "reference_postEqSinrdB"] + [k for k in keys if "sinr" in k.lower() or "snr" in k.lower()]:
                if k in f and isinstance(f[k], h5py.Dataset) and f[k].dtype.names is None and 0 < f[k].size <= 64:
                    try:
                        row["sinr_db"] = round(float(np.mean(f[k][()])), 2)
                        break
                    except Exception:
                        pass

            if ch == "PUSCH":
                g = params.get("gnb_pars", {})
                tbs = params.get("tb_pars", [])
                tbs = tbs if isinstance(tbs, list) else [tbs]
                grps = params.get("ueGrp_pars", [])
                grps = grps if isinstance(grps, list) else [grps]
                row.update(n_ue=len(tbs), n_grp=g.get("nUserGroups", len(grps)), n_tb=g.get("numTb", len(tbs)),
                           n_prb_bwp=g.get("nPrb"), n_rx=g.get("nRx"), mu=g.get("mu"))
                if tbs:
                    row.update(n_layers_max=max(t["numLayers"] for t in tbs), n_layers_sum=sum(t["numLayers"] for t in tbs),
                               mcs_min=min(t["mcsIndex"] for t in tbs), mcs_max=max(t["mcsIndex"] for t in tbs),
                               mcs_table=tbs[0].get("mcsTableIndex"), qam_max=max(t["qamModOrder"] for t in tbs),
                               tb_bytes_tot=sum(t["nTbByte"] for t in tbs), tb_bytes_max=max(t["nTbByte"] for t in tbs),
                               n_cb_tot=sum(t["nCb"] for t in tbs), n_cb_max=max(t["nCb"] for t in tbs),
                               rv_max=max(t["rv"] for t in tbs), n_sym=tbs[0].get("numSym"),
                               harq_bits=sum(t.get("nBitsHarq", 0) for t in tbs), csi1_bits=sum(t.get("nBitsCsi1", 0) for t in tbs))
                if grps:
                    row.update(n_prb_tot=sum(gp["nPrb"] for gp in grps), n_prb_max=max(gp["nPrb"] for gp in grps),
                               n_dmrs_sym=popcount(grps[0].get("dmrsSymLocBmsk", 0)))
            elif ch == "PDSCH":
                cg = params.get("cellGrpDyn_pars", {})
                cws = params.get("cw_pars", [])
                cws = cws if isinstance(cws, list) else [cws]
                ues = params.get("ue_pars", [])
                ues = ues if isinstance(ues, list) else [ues]
                grps = params.get("ueGrp_pars", [])
                grps = grps if isinstance(grps, list) else [grps]
                cs = params.get("cellStat_pars", {})
                cd = params.get("cellDyn_pars", {})
                row.update(n_ue=cg.get("nUes", len(ues)), n_grp=cg.get("nUeGrps", len(grps)), n_cw=cg.get("nCws", len(cws)),
                           n_tx=cs.get("nTxAnt"), n_prb_bwp=cs.get("nPrbDlBwp"), mu=cs.get("mu"), n_sym=cd.get("nPdschSym"),
                           n_dmrs_sym=popcount(cd.get("dmrsSymLocBmsk", 0)))
                if cws:
                    row.update(mcs_min=min(c["mcsIndex"] for c in cws), mcs_max=max(c["mcsIndex"] for c in cws),
                               mcs_table=cws[0].get("mcsTableIndex"), qam_max=max(c["qamModOrder"] for c in cws),
                               tb_bytes_tot=sum(c["tbSize"] for c in cws), tb_bytes_max=max(c["tbSize"] for c in cws),
                               rv_max=max(c["rv"] for c in cws))
                if ues:
                    row.update(n_layers_max=max(u["nUeLayers"] for u in ues), n_layers_sum=sum(u["nUeLayers"] for u in ues),
                               precoded=int(any(u.get("enablePrcdBf", 0) for u in ues)))
                if grps:
                    row.update(n_prb_tot=sum(gp["nPrb"] for gp in grps), n_prb_max=max(gp["nPrb"] for gp in grps))
                row["n_cb_tot"] = sum(f[k].shape[0] for k in keys if re.match(r"tb\d+_cbs$", k))
            elif ch == "PDCCH":
                dcis = [v for k, v in params.items() if k.startswith("DciParams_")]
                pp = params.get("PdcchParams", [])
                pp = pp if isinstance(pp, list) else [pp]
                row.update(n_dci=len(dcis), n_coreset=len(pp), agg_sum=sum(d["aggrL"] for d in dcis) if dcis else 0,
                           agg_max=max((d["aggrL"] for d in dcis), default=0), dci_bits_sum=sum(d["Npayload"] for d in dcis) if dcis else 0,
                           n_sym=pp[0].get("n_sym") if pp else "")
            elif ch.startswith("PUCCH"):
                cg = params.get("cellGrpDynPrm", {})
                fmt = ch.split("_")[-1]
                row["pucch_format"] = fmt
                nu = {"F0": cg.get("nF0Ucis"), "F1": cg.get("nF1Ucis"), "F2": cg.get("nF2Ucis"), "F3": cg.get("nF3Ucis")}.get(fmt)
                row["n_uci"] = nu
                pr = params.get(fmt + "UciPrms", [])
                pr = pr if isinstance(pr, list) else [pr]
                if pr:
                    row["uci_bits"] = sum(p.get("bitLenHarq", 0) + p.get("bitLenSr", 0) + p.get("bitLenCsiPart1", 0) for p in pr)
                    row["n_prb_tot"] = sum(p.get("prbSize", 1) for p in pr)
                    row["n_sym"] = pr[0].get("nSym")
                cs = params.get("cellStatPrm", {})
                row.update(n_rx=cs.get("nRxAnt"), mu=cs.get("mu"))
            elif ch == "PRACH":
                row["n_prach"] = int(f["nPrach"][()].reshape(-1)[0]) if "nPrach" in f else ""
                p0 = params.get("prachParams_0", {})
                cs = params.get("PrachCellStatPrms", {})
                row.update(prach_lra=p0.get("L_RA"), prach_nrep=p0.get("N_rep"), prach_nroot=p0.get("uCount"),
                           prach_cfg_idx=cs.get("configurationIndex"), n_rx=p0.get("N_ant"), mu=cs.get("mu"))
            elif ch == "SRS":
                nue = len([k for k in keys if re.match(r"^HestUe\d+$", k)])
                row["srs_n_ue"] = nue
                if "DataRx" in f:
                    sh = f["DataRx"].shape
                    row["n_rx"] = sh[0]
                    row["srs_n_sym"] = sh[1] if len(sh) == 3 else ""
                sc = params.get("srsCellParams", {})
                up = params.get("srsUePrms", [])
                up = up if isinstance(up, list) else [up]
                row.update(srs_n_ue=sc.get("nSrsUes", nue), srs_n_sym=sc.get("nSrsSym", row.get("srs_n_sym")),
                           n_prb_bwp=sc.get("nPrbs"), mu=sc.get("mu"), n_rx=sc.get("nRxAntSrs", row.get("n_rx")))
                if up:
                    row["srs_n_ports"] = max(u.get("nAntPorts", 0) for u in up)
                    row["n_sym"] = max(u.get("nSyms", 0) for u in up)
            elif ch == "SSB":
                row["n_ssb"] = int(f["nSsb"][()].reshape(-1)[0]) if "nSsb" in f else ""
            elif ch == "CSIRS":
                row["n_csirs"] = int(f["nCsirs"][()].reshape(-1)[0]) if "nCsirs" in f else ""
    except Exception as e:  # keep going; record the error
        row["error"] = repr(e)[:120]
    row["params"] = json.dumps(params, separators=(",", ":"))[:20000]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tv-dir", default="/opt/nvidia/cuBB/testVectors")
    ap.add_argument("--out", default="tv_index.csv")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.tv_dir, "TVnr_*_gNB_CUPHY_*.h5")))
    if a.limit:
        files = files[: a.limit]
    print("indexing %d files with %d workers" % (len(files), a.jobs), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, r in enumerate(ex.map(index_file, files, chunksize=16)):
            if r:
                rows.append(r)
            if i % 1000 == 0:
                print("  %d/%d" % (i, len(files)), flush=True)
    cols = ["file", "fam", "tvid", "channel", "slot", "pdu", "size_mb"] + CURATED + ["error", "params"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print("done:", len(rows), "rows ->", a.out)
    print("per channel:", dict(Counter(r["channel"] for r in rows)))
    print("errors:", sum(1 for r in rows if r.get("error")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
