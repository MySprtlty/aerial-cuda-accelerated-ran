#!/bin/bash
# g3_sweep.sh grid|eval|rerun|table [prefix]  — SafeCap sweep for the DDDDDDSUUU patterns (2026-10-16 morning), 16 cells, 60 s,
# closed-loop tenant through the dispatcher. Loads: ddddddsuuu_peak / ddddddsuuu_average (R0 runs come from G1/G2: prefix G1_P1/G1_A1,
# G2_P2/G2_A2 in the same directory). Grid: 26n, 26s x caps 43/61/100; 26x x 43/61/80/100 (80 = SM-104 engine, class 104).
# Spots (10 req/s Poisson): 26x@61 avg, 26x@80 avg, 26s@43 peak, 26n@43 peak. Criteria (g3_table.py): B misses <= 5 (absolute) with a
# relative column (<= R0 + 5), Err.ind / drops / CRC 0 after 1 s, RU late slots <= 1.25 x mean Peak R0 (absolute for both loads).
set -u
S=$(cd "$(dirname "$0")" && pwd); R=$(cd "$S/../../../.." && pwd); O=${FS_OUT:-$R/prof/run64}
PHASE=$1; PREFIX=${2:-G3}; SLOTS=${SLOTS:-120000}; TDELAY=${TDELAY:-28}; TSECS=${TSECS:-45}
cd $R; mkdir -p $O
tline() { local name=$1 pat=$2 m=$3 cap=$4 arr=${5:-closed} extra=""; [ "$cap" = 80 ] && extra="xclass=104 xmodels=26x"; [ "$cap" = 24 ] && extra="xclass=32 xmodels=26x"
  echo "$name pattern=$pat slots=$SLOTS tags=${TAGS:-311,223} keep_raw=2 tenant=dispatcher model=$m cap=$cap arrivals=$arr rate=10 tdelay=$TDELAY tsecs=$TSECS $extra"; }
case $PHASE in
  grid)
    L=$O/ladder_${PREFIX}_grid.txt; : > $L
    # a third Peak R0 first (P1 1,694 / P2 3,082 late slots: the two-run mean is too uncertain for an absolute 1.25 x line)
    [ "${P3:-1}" = 1 ] && echo "P3 pattern=ddddddsuuu_peak slots=$SLOTS tags=${TAGS:-311,223}" >> $L
    for pat in ddddddsuuu_peak ddddddsuuu_average; do sfx=${pat#ddddddsuuu_}
      for m in 26n 26s; do for cap in 43 61 100; do tline S_${m}_c${cap}_$sfx $pat $m $cap >> $L; done; done
      for cap in 43 61 80 100; do tline S_26x_c${cap}_$sfx $pat 26x $cap >> $L; done
    done
    tline P_26x_c61_average ddddddsuuu_average 26x 61 poisson >> $L; tline P_26x_c80_average ddddddsuuu_average 26x 80 poisson >> $L
    tline P_26s_c43_peak ddddddsuuu_peak 26s 43 poisson >> $L; tline P_26n_c43_peak ddddddsuuu_peak 26n 43 poisson >> $L
    echo "== grid list $L ($(grep -c . $L) runs)"; cat $L; [ "${NORUN:-0}" = 1 ] && exit 0
    export TENANT_GLOB=${TENANT_GLOB:-uhd_*_q60.jpg}      # demo image set (q60, ~1 MB) for every tenant run; inherited by fs_tenant.sh
    FS_OUT=$O bash $S/r0_ladder.sh $L $PREFIX ;;
  eval|rerun|table)
    python3 $S/g3_table.py --dir $O --prefix $PREFIX --phase $PHASE > $O/g3_${PHASE}.out 2>&1; cat $O/g3_${PHASE}.out
    if [ "$PHASE" = rerun ] && [ -s $O/ladder_${PREFIX}_rerun.txt ]; then
      echo "== rerun list ($(grep -c . $O/ladder_${PREFIX}_rerun.txt) runs)"; FS_OUT=$O bash $S/r0_ladder.sh $O/ladder_${PREFIX}_rerun.txt $PREFIX
      python3 $S/g3_table.py --dir $O --prefix $PREFIX --phase table > $O/g3_table2.out 2>&1; cat $O/g3_table2.out
    fi ;;
  *) echo "usage: $0 grid|eval|rerun|table [prefix]"; exit 1 ;;
esac
