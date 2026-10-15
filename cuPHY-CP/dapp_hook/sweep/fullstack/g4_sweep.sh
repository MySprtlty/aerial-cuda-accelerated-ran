#!/bin/bash
# g4_sweep.sh <phase> [prefix]   — SafeCap re-sweep (2026-10-16): 16-cell Peak (59c) and Average (60c), tenant through the
# dispatcher (dapp_sched, 3 models x SM classes 56/80/132), closed-loop client, 60 s (120,000 slots) per run.
#   phase grid   : R0 per load (no tenant) + models 26n/26s/26x x caps 43/61/100 x loads + 4 Poisson spots (10 req/s)
#   phase eval   : safecap_check.py (fixed criteria) per tenant run -> g4_results.csv; R0 reference late slots:
#                  Peak = mean(A1a/2, A1b/2, R0_peak) of prof/run62 (120 s runs halved), Average = R0_avg
#   phase rerun  : one re-run of every failed grid/spot run (suffix _r2); a cap fails only when both runs fail;
#                  26x Peak cap 24 (SM 32 engine) when 26x fails at cap 43 in Peak
#   phase table  : final table (g4_table.py)
# env: FS_OUT (default prof/run63), SLOTS (120000), TDELAY (28 s from testMAC start = ~10 s after the first slot with the
#      patched parser), TSECS (45)
set -u
S=$(cd "$(dirname "$0")" && pwd); R=$(cd "$S/../../../.." && pwd); O=${FS_OUT:-$R/prof/run63}
PHASE=$1; PREFIX=${2:-G4}; SLOTS=${SLOTS:-120000}; TDELAY=${TDELAY:-28}; TSECS=${TSECS:-45}
LOADS="59c 60c"; MODELS="26n 26s 26x"; CAPS="43 61 100"
cd $R; mkdir -p $O
line() { echo "$1 pattern=$2 slots=$SLOTS tags=311,223 ${3:-}"; }
tline() { local name=$1 pat=$2 m=$3 cap=$4 arr=${5:-closed} extra=${6:-}; echo "$name pattern=$pat slots=$SLOTS tags=311,223 tenant=dispatcher model=$m cap=$cap arrivals=$arr rate=10 tdelay=$TDELAY tsecs=$TSECS $extra"; }
case $PHASE in
  grid)
    L=$O/ladder_${PREFIX}_grid.txt; : > $L
    for pat in $LOADS; do line R0_$pat $pat >> $L; done
    for pat in $LOADS; do for m in $MODELS; do for cap in $CAPS; do tline S_${m}_c${cap}_$pat $pat $m $cap >> $L; done; done; done
    tline P_26x_c100_59c 59c 26x 100 poisson >> $L; tline P_26x_c61_59c 59c 26x 61 poisson >> $L; tline P_26s_c43_59c 59c 26s 43 poisson >> $L; tline P_26x_c100_60c 60c 26x 100 poisson >> $L
    echo "== grid list $L ($(grep -c . $L) runs)"; FS_OUT=$O bash $S/r0_ladder.sh $L $PREFIX ;;
  eval|rerun|table)
    python3 $S/g4_table.py --dir $O --prefix $PREFIX --phase $PHASE > $O/g4_${PHASE}.out 2>&1; cat $O/g4_${PHASE}.out
    if [ "$PHASE" = rerun ] && [ -s $O/ladder_${PREFIX}_rerun.txt ]; then
      echo "== rerun list ($(grep -c . $O/ladder_${PREFIX}_rerun.txt) runs)"; FS_OUT=$O bash $S/r0_ladder.sh $O/ladder_${PREFIX}_rerun.txt $PREFIX
      python3 $S/g4_table.py --dir $O --prefix $PREFIX --phase eval > $O/g4_eval2.out 2>&1; cat $O/g4_eval2.out
    fi ;;
  *) echo "usage: $0 grid|eval|rerun|table [prefix]"; exit 1 ;;
esac
