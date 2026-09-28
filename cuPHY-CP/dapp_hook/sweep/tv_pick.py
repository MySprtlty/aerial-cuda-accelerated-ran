#!/usr/bin/env python3
"""
tv_pick: choose, from tv_index.csv, the per-channel cuPHY test vector closest
to a requested parameter point, and report how far it is.

    python3 tv_pick.py --index tv_index.csv coverage            # default point + every sweep list
    python3 tv_pick.py --index tv_index.csv pick PUSCH n_ue=4 n_prb_tot=136 n_layers_max=2 mcs_max=16

Distance: per requested key, |actual - wanted| / scale(key); keys not present
in the row count as a miss. The best row has the smallest sum; "exact" means
every requested key matched, "approx" otherwise. Rows are restricted to
mu=1 / 4 Rx (the lab's 4T4R 100 MHz cells) unless --any-cell is given.
"""
import argparse
import csv
import json
import math
import sys
from collections import OrderedDict

SCALE = {"n_ue": 1.0, "n_prb_tot": 20.0, "n_layers_max": 1.0, "mcs_max": 2.0, "mcs_min": 2.0, "sinr_db": 3.0,
         "rv_max": 1.0, "n_cb_tot": 4.0, "n_cb_max": 4.0, "tb_bytes_tot": 5000.0, "n_dci": 1.0, "agg_max": 2.0,
         "agg_sum": 4.0, "n_uci": 2.0, "uci_bits": 4.0, "n_prach": 1.0, "prach_lra": 100.0, "prach_nrep": 1.0,
         "srs_n_ue": 1.0, "srs_n_ports": 1.0, "srs_n_sym": 1.0, "precoded": 1.0, "pucch_format": 1.0, "n_layers_sum": 2.0}

# The one-at-a-time sweep of the experiment brief, expressed in index columns.
DEFAULT = {
    "PUSCH": dict(n_ue=4, n_prb_tot=136, n_layers_max=2, mcs_max=16, mcs_min=16, rv_max=0),
    "PDSCH": dict(n_ue=8, n_prb_tot=136, n_layers_max=2, mcs_max=16, mcs_min=16, precoded=0),
    "PDCCH": dict(n_dci=4, agg_max=4),
    "PUCCH": dict(pucch_format="F1", n_uci=4),
}
SWEEPS = {
    "PUSCH": OrderedDict([("n_ue", [1, 2, 4, 8]), ("n_prb_tot", [34, 68, 136, 273]), ("n_layers_max", [1, 2, 4]),
                          ("mcs_max", [4, 10, 16, 22, 27]), ("rv_max", [0, 2])]),
    "PDSCH": OrderedDict([("n_ue", [1, 4, 8, 16]), ("n_prb_tot", [34, 68, 136, 273]), ("n_layers_max", [1, 2, 4]),
                          ("mcs_max", [4, 16, 27]), ("precoded", [0, 1])]),
    "PDCCH": OrderedDict([("n_dci", [1, 4, 8, 16]), ("agg_max", [1, 4, 16])]),
    "PUCCH": OrderedDict([("pucch_format", ["F0", "F1", "F2", "F3"]), ("n_uci", [2, 8, 16]), ("uci_bits", [2, 11, 22])]),
    "PRACH": OrderedDict([("prach_lra", [139, 839]), ("n_prach", [1, 4])]),
    "SRS": OrderedDict([("srs_n_ue", [4, 8]), ("srs_n_ports", [1, 2, 4]), ("srs_n_sym", [1, 2, 4])]),
}
PUCCH_CHANNELS = ("PUCCH_F0", "PUCCH_F1", "PUCCH_F2", "PUCCH_F3")


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_index(path, any_cell=False):
    rows = list(csv.DictReader(open(path, newline="")))
    if not any_cell:
        rows = [r for r in rows if (r.get("mu") in ("", "1")) and (r.get("n_rx") in ("", "4") or r["channel"] in ("PDSCH", "PDCCH", "SSB", "CSIRS"))]
    return rows


def rows_of(rows, channel):
    if channel == "PUCCH":
        return [r for r in rows if r["channel"] in PUCCH_CHANNELS]
    return [r for r in rows if r["channel"] == channel]


def distance(row, want):
    d = 0.0
    misses = []
    for k, v in want.items():
        a = row.get(k, "")
        if isinstance(v, str):
            if a != v:
                d += 10.0
                misses.append(k)
            continue
        av = num(a)
        if av is None:
            d += 10.0
            misses.append(k)
            continue
        dd = abs(av - v) / SCALE.get(k, 1.0)
        if dd > 1e-9:
            misses.append(k)
        d += dd
    return d, misses


def pick(rows, channel, want, n=1, prefer_single=True):
    cands = []
    for r in rows_of(rows, channel):
        d, misses = distance(r, want)
        # prefer TVs whose other load-relevant fields are moderate (avoid huge TB when asking for small PRB, etc.)
        tie = 0.001 * (num(r.get("size_mb")) or 0)
        if prefer_single and r.get("fam") != "SINGLE":
            tie += 0.01
        cands.append((d + tie, d, misses, r))
    cands.sort(key=lambda t: t[0])
    return cands[:n]


def describe(r):
    keys = {"PUSCH": ["n_ue", "n_prb_tot", "n_layers_max", "mcs_min", "mcs_max", "tb_bytes_tot", "n_cb_tot", "rv_max", "sinr_db"],
            "PDSCH": ["n_ue", "n_prb_tot", "n_layers_max", "mcs_min", "mcs_max", "tb_bytes_tot", "n_cb_tot", "precoded"],
            "PDCCH": ["n_dci", "agg_max", "agg_sum", "dci_bits_sum"], "PRACH": ["n_prach", "prach_lra", "prach_nrep", "prach_nroot", "prach_cfg_idx"],
            "SRS": ["srs_n_ue", "srs_n_ports", "srs_n_sym"], "SSB": ["n_ssb"], "CSIRS": ["n_csirs"]}
    ch = "PDCCH" if r["channel"] == "PDCCH" else r["channel"]
    ks = keys.get(ch, ["pucch_format", "n_uci", "uci_bits", "n_prb_tot", "n_sym"] if ch.startswith("PUCCH") else [])
    return " ".join("%s=%s" % (k, r.get(k, "")) for k in ks if r.get(k, "") != "")


def coverage(rows):
    out = []
    for ch, base in DEFAULT.items():
        best = pick(rows, ch, base)
        if best:
            _, d, misses, r = best[0]
            out.append((ch, "default", str(base), r["file"], "exact" if not misses else "approx(%s)" % ",".join(misses), describe(r)))
    for ch, sweeps in SWEEPS.items():
        base = dict(DEFAULT.get(ch, {}))
        for key, values in sweeps.items():
            for v in values:
                want = dict(base)
                if key == "mcs_max":
                    want["mcs_min"] = v
                want[key] = v
                best = pick(rows, ch, want)
                if not best:
                    out.append((ch, key, str(v), "-", "missing", ""))
                    continue
                _, d, misses, r = best[0]
                out.append((ch, key, str(v), r["file"], "exact" if not misses else "approx(%s)" % ",".join(misses), describe(r)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="tv_index.csv")
    ap.add_argument("--any-cell", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("coverage")
    pp = sub.add_parser("pick")
    pp.add_argument("channel")
    pp.add_argument("kv", nargs="*")
    pp.add_argument("-n", type=int, default=5)
    a = ap.parse_args()
    rows = load_index(a.index, a.any_cell)
    if a.cmd == "pick":
        want = {}
        for kv in a.kv:
            k, v = kv.split("=")
            want[k] = v if k == "pucch_format" else float(v)
        for _, d, misses, r in pick(rows, a.channel, want, n=a.n):
            print("%-52s d=%.2f %s | %s" % (r["file"], d, "exact" if not misses else "approx:" + ",".join(misses), describe(r)))
        return 0
    res = coverage(rows)
    w = csv.writer(sys.stdout)
    w.writerow(["channel", "sweep_key", "value", "tv", "match", "actual"])
    for row in res:
        w.writerow(row)
    n_exact = sum(1 for r in res if r[4] == "exact")
    print("# %d points: %d exact, %d approx, %d missing" % (len(res), n_exact, sum(1 for r in res if r[4].startswith("approx")), sum(1 for r in res if r[4] == "missing")), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
