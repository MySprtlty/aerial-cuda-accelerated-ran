#!/usr/bin/env python3
"""ring_load_profile.py --dump ring.bin [--window-s 1.0] [--max-records N] [--out prefix] [--sfn0-offset 3.64] [--period 10.24]
Load profile seen by the hook: per slot the sum over cells of the UL_TTI summaries' tot_pusch_prb and tot_pusch_tb_bytes
(and the DL_TTI tot_pdsch_prb), then a sliding --window-s mean; written as <out>.csv (t_s, phase_s = (ts - 3.64) mod 10.24,
pusch_prb_per_s, pusch_tb_bytes_per_s, pdsch_prb_per_s) and, when matplotlib is available, <out>.png. Prints the window
means of the Average and Peak phases of the alternating pattern (Peak frames 583-1015 of the 1024-frame period) and the
separation (ratio) - the basis of the load classifier."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "estimator"))
from dapp_est import ringio as R   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--window-s", type=float, default=1.0); ap.add_argument("--max-records", type=int, default=40000000)
    ap.add_argument("--out", default=""); ap.add_argument("--sfn0-offset", type=float, default=3.64); ap.add_argument("--period", type=float, default=10.24)
    ap.add_argument("--peak-frames", default="583,1015", help="first,last Peak frame of the alternating pattern's 1024-frame period")
    a = ap.parse_args()
    with open(a.dump, "rb") as f:
        buf = f.read(a.max_records * R.REC_SIZE + 4096)
    off = R.HDR_SIZE if buf[:8] == R.MAGIC else 0
    ul, dl = {}, {}
    n = 0
    for i in range(a.max_records):
        o = off + i * R.REC_SIZE
        if o + R.REC_SIZE > len(buf):
            break
        rec = R.decode(buf, o)
        if rec is None:
            continue
        n += 1
        if rec.type == R.REC_UL_TTI:
            k = (rec.sfn, rec.slot, rec.ts_ns // 5_000_000_000)
            e = ul.setdefault(k, [rec.ts_ns, 0, 0]); e[1] += rec.tot_pusch_prb; e[2] += rec.tot_pusch_tb_bytes
        elif rec.type == R.REC_DL_TTI:
            k = (rec.sfn, rec.slot, rec.ts_ns // 5_000_000_000)
            e = dl.setdefault(k, [rec.ts_ns, 0]); e[1] += rec.tot_pdsch_prb
    if not ul:
        print("no UL_TTI records"); return 1
    t0 = min(v[0] for v in ul.values())
    tmax = max(max(v[0] for v in ul.values()), max(v[0] for v in dl.values()) if dl else 0)
    edges = np.arange(0, (tmax - t0) / 1e9 + a.window_s, a.window_s)
    def binned(d, idx):
        t = np.array([(v[0] - t0) / 1e9 for v in d.values()]); x = np.array([v[idx] for v in d.values()], dtype=float)
        h, _ = np.histogram(t, bins=edges, weights=x); return h / a.window_s
    prb, tb, dprb = binned(ul, 1), binned(ul, 2), (binned(dl, 1) if dl else np.zeros(len(edges) - 1))
    tc = edges[:-1] + a.window_s / 2
    phase = ((t0 / 1e9 + tc) - a.sfn0_offset) % a.period
    f0, f1 = [int(x) for x in a.peak_frames.split(",")]
    is_peak = (phase >= f0 * 0.01) & (phase <= (f1 + 1) * 0.01)
    out = a.out or os.path.splitext(a.dump)[0] + "_load"
    with open(out + ".csv", "w") as f:
        f.write("t_s,phase_s,pusch_prb_per_s,pusch_tb_bytes_per_s,pdsch_prb_per_s,peak_phase\n")
        for i in range(len(tc)):
            f.write("%.2f,%.3f,%.0f,%.0f,%.0f,%d\n" % (tc[i], phase[i], prb[i], tb[i], dprb[i], is_peak[i]))
    traffic = prb > 0
    print("records %d, UL slots %d, DL slots %d, %.1f s with traffic; window %.1f s" % (n, len(ul), len(dl), traffic.sum() * a.window_s, a.window_s))
    for lab, m in (("Average phase", traffic & ~is_peak), ("Peak phase", traffic & is_peak)):
        if m.any():
            print("  %s: PUSCH PRB/s mean %.0f (min %.0f max %.0f), TB bytes/s mean %.2e, PDSCH PRB/s mean %.0f, windows %d" % (lab, prb[m].mean(), prb[m].min(), prb[m].max(), tb[m].mean(), dprb[m].mean(), m.sum()))
    if (traffic & is_peak).any() and (traffic & ~is_peak).any():
        print("  separation PUSCH PRB/s: Peak min / Average max = %.2f; TB: %.2f" % (prb[traffic & is_peak].min() / max(1, prb[traffic & ~is_peak].max()), tb[traffic & is_peak].min() / max(1, tb[traffic & ~is_peak].max())))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].plot(tc, prb, lw=1); ax[0].set_ylabel("PUSCH PRB / s (16 cells)"); ax[0].grid(alpha=.3)
        ax[0].fill_between(tc, 0, prb.max(), where=is_peak, color="tab:red", alpha=.12, label="Peak phase (pattern)"); ax[0].legend()
        ax[1].plot(tc, tb / 1e6, lw=1, color="tab:green"); ax[1].set_ylabel("PUSCH TB MB / s"); ax[1].set_xlabel("s since first UL_TTI"); ax[1].grid(alpha=.3)
        fig.suptitle(os.path.basename(a.dump) + ": hook-ring UL load, %.1f s windows" % a.window_s); fig.tight_layout(); fig.savefig(out + ".png", dpi=110)
        print("plot:", out + ".png")
    except Exception as e:
        print("(no plot: %s)" % e)
    print("csv:", out + ".csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
