#!/usr/bin/env python3
"""tv_inventory.py <out.csv>: one row per DLMIX/ULMIX FAPI test vector in /opt/nvidia/cuBB/testVectors.
Per TV: cell config (grid, antennas, mu) and per channel the PDU count, PRB sum, layers, MCS, symbol span."""
import csv, os, re, sys
import h5py
D = "/opt/nvidia/cuBB/testVectors"
out = sys.argv[1]
names = sorted(f for f in os.listdir(D) if re.match(r"TVnr_(DLMIX|ULMIX)_\d+_gNB_FAPI_s0\.h5$", f))
TYPES = {1: "ssb", 2: "pdcch", 3: "pdsch", 4: "csirs", 6: "prach", 7: "pucch", 8: "pusch", 9: "srs"}
cols = ["tv", "kind", "id", "grid", "ntx", "nrx", "mu", "n_pdu", "types",
        "pdsch_n", "pdsch_rb", "pdsch_layers", "pdsch_mcs", "pdsch_sym0", "pdsch_sym1", "pdsch_tb",
        "pusch_n", "pusch_rb", "pusch_layers", "pusch_mcs", "pusch_sym0", "pusch_sym1", "pusch_tb",
        "pucch_n", "pucch_fmt", "pucch_sym0", "pucch_sym1", "prach_n", "prach_fmt", "srs_n", "srs_sym",
        "pdcch_n", "pdcch_dci", "pdcch_sym", "csirs_n", "csirs_sym", "ssb_n", "ssb_sym", "err"]
def g(v, k):
    try:
        x = v[k]
        return int(x) if getattr(x, "shape", ()) == () else None
    except (ValueError, KeyError, IndexError):
        return None
w = csv.DictWriter(open(out, "w", newline=""), fieldnames=cols); w.writeheader()
for i, f in enumerate(names):
    m = re.match(r"TVnr_(DLMIX|ULMIX)_(\d+)_", f)
    row = {c: "" for c in cols}; row.update(tv=f, kind=m.group(1), id=int(m.group(2)))
    try:
        h = h5py.File(os.path.join(D, f), "r")
        cc = h["Cell_Config"][()][0]
        row.update(grid=g(cc, "dlGridSize"), ntx=g(cc, "numTxAnt"), nrx=g(cc, "numRxAnt"), mu=g(cc, "mu"))
        pdus = [k for k in h.keys() if re.match(r"PDU\d+$", k)]
        row["n_pdu"] = len(pdus)
        acc = {}
        for k in pdus:
            d = h[k]
            v = d[()][0] if d.shape else d[()]
            t = g(v, "type"); name = TYPES.get(t, "t%s" % t)
            a = acc.setdefault(name, {"n": 0, "rb": 0, "layers": set(), "mcs": set(), "s0": [], "s1": [], "tb": 0, "fmt": set(), "dci": 0, "sym": set()})
            a["n"] += 1
            rb = g(v, "rbSize"); rb = g(v, "prbSize") if rb is None else rb; rb = g(v, "NrOfRBs") if rb is None else rb
            if rb: a["rb"] += rb
            l = g(v, "nrOfLayers")
            if l is not None: a["layers"].add(l)
            mc = g(v, "mcsIndex")
            if mc is not None: a["mcs"].add(mc)
            s0 = g(v, "StartSymbolIndex"); ns = g(v, "NrOfSymbols"); ns = g(v, "DurationSymbols") if ns is None else ns
            if s0 is not None:
                a["s0"].append(s0)
                if ns is not None: a["s1"].append(s0 + ns)
            tb = g(v, "TBSize")
            if tb: a["tb"] += tb
            for fk in ("FormatType", "prachFormat"):
                fv = g(v, fk)
                if fv is not None: a["fmt"].add(fv)
            nd = g(v, "numDlDci")
            if nd: a["dci"] += nd
            for sk in ("SymbL0", "SymbL1", "SymbolIndex"):
                sv = g(v, sk)
                if sv is not None and sv < 14: a["sym"].add(sv)
        row["types"] = " ".join("%s:%d" % (k, a["n"]) for k, a in sorted(acc.items()))
        for ch in ("pdsch", "pusch"):
            a = acc.get(ch)
            if a:
                row.update({ch + "_n": a["n"], ch + "_rb": a["rb"], ch + "_layers": "/".join(map(str, sorted(a["layers"]))),
                            ch + "_mcs": "/".join(map(str, sorted(a["mcs"]))), ch + "_sym0": min(a["s0"]) if a["s0"] else "",
                            ch + "_sym1": max(a["s1"]) if a["s1"] else "", ch + "_tb": a["tb"]})
        a = acc.get("pucch")
        if a: row.update(pucch_n=a["n"], pucch_fmt="/".join(map(str, sorted(a["fmt"]))), pucch_sym0=min(a["s0"]) if a["s0"] else "", pucch_sym1=max(a["s1"]) if a["s1"] else "")
        a = acc.get("prach")
        if a: row.update(prach_n=a["n"], prach_fmt="/".join(map(str, sorted(a["fmt"]))))
        a = acc.get("srs")
        if a: row.update(srs_n=a["n"], srs_sym="/".join(map(str, sorted(a["sym"]))))
        a = acc.get("pdcch")
        if a: row.update(pdcch_n=a["n"], pdcch_dci=a["dci"], pdcch_sym=max(a["s1"]) if a["s1"] else "")
        a = acc.get("csirs")
        if a: row.update(csirs_n=a["n"], csirs_sym="/".join(map(str, sorted(a["sym"]))))
        a = acc.get("ssb")
        if a: row.update(ssb_n=a["n"], ssb_sym="/".join(map(str, sorted(a["sym"]))))
        h.close()
    except Exception as e:
        row["err"] = str(e)[:80]
    w.writerow(row)
    if i % 2000 == 0:
        print(i, f, flush=True)
print("done", len(names))
