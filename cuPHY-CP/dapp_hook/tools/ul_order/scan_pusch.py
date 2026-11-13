#!/usr/bin/env python3
# scan_pusch.py < "PUSCH Aggr ... times" lines of phy.log  ->  one TSV row per UL slot (stdout)
# columns: abs_slot sfn slot map thread setup_status run_status setup_start_ns gpu_run_us maxgpu_us maxgpu_end_ns cb_start_ns cb_end_ns log_time
import sys, re
pat = re.compile(rb'^(\S+) INF (\S+) \d+ \[DRV\.MAP_UL\] \[PHYDRV\] SFN (\d+)\.(\d+) PUSCH Aggr \S+ Map (\d+) times ===> \[CPU CUDA Setup\] \{ START: (\d+) END: \d+ DURATION: \d+ us Setup STATUS: (\d+) \} \[CPU CUDA Run\] \{ START: \d+ END: \d+ DURATION: \d+ us Run STATUS: (\d+) \}.*?\[GPU Run\] \{ DURATION: (-?[\d.]+) us \}.*?\[Max GPU Time\] \{ START: \d+ END: (\d+) DURATION: (-?\d+) us \}.*?\[Callback\] \{ START: (\d+) END: (\d+)')
out = sys.stdout; n = bad = 0
for l in sys.stdin.buffer:
    if b'[GPU Run EH]' not in l: continue          # every slot is logged twice (EH / front-loaded DMRS wording): keep one
    m = pat.match(l)
    if not m: bad += 1; continue
    t, th, sfn, slot, mp, st, sst, rst, grun, mge, mgu, cbs, cbe = m.groups()
    s = int(sfn) * 20 + int(slot); st = int(st)
    k = round((st / 500000 - s) / 20480); a = k * 20480 + s
    out.write(f"{a}\t{int(sfn)}\t{int(slot)}\t{int(mp)}\t{th.decode()}\t{int(sst)}\t{int(rst)}\t{st}\t{grun.decode()}\t{int(mgu)}\t{int(mge)}\t{int(cbs)}\t{int(cbe)}\t{t.decode()}\n"); n += 1
sys.stderr.write(f"rows {n} unparsed {bad}\n")
