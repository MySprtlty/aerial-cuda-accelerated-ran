#!/bin/bash
# verify.sh <run name> [from HH:MM:SS UTC] [to HH:MM:SS UTC]   - order / storm check of one _dgist generator run (run as troy; uses sudo for reads)
# Niced and kept on cores 0-3,65-71 (the L1 may be live). Writes rows_<run>.tsv into $OUT (default: the current directory).
R=$1; A=${2:-00:00:00}; B=${3:-99:99:99}
T=$(cd "$(dirname "$0")" && pwd); H=${OUT:-$PWD}
LOG=/data/dgist_tgen/log/$R/phy.log; CON=/home/nvidia/_dgist/runs/$R/l1_console.log
NI="nice -n 19 taskset -c 0-3,65-71"
echo "== $R  window $A .. $B (L1 clock, UTC)  $(date '+%T KST')"
echo "-- gate lines (console)"; sudo -n grep -a 'PUSCH in-order gate' $CON | sed -n '1p;$p' | cut -c1-200
win() { awk -v a="$A" -v b="$B" '{t=substr($1,1,8)} t>=a && t<=b'; }
echo "-- counts inside the window (phy.log)"
for p in 'PUSCH setup returned error' 'MAC reusing hqbuf' 'Send Err.ind' 'FATAL' 'AGGR 3 task waiting'; do
  n=$(sudo -n ionice -c3 $NI /usr/bin/grep -a "$p" $LOG | win | wc -l); printf '   %-28s %s\n' "$p" "$n"; done
echo "-- PUSCH result order between neighbouring UL slots (inv.py on the PUSCH callback start times)"
sudo -n ionice -c3 $NI /usr/bin/grep -a 'PUSCH Aggr' $LOG | win | $NI python3 $T/scan_pusch.py > $H/rows_$R.tsv
$NI python3 $T/inv.py $H/rows_$R.tsv 2>&1 | tail -${TAILN:-14}
