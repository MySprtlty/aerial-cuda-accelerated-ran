#!/bin/bash
# tn_alt.sh [name:mode ...] (2026-11-01): slow-alternation runs (Peak 2 frames | L3 (light PUSCH load, same PRACH) 2 frames, each block replayed
# BLOCK_REPEAT x 20 ms, default 1500 = 30 s per phase) on 8 RU-emulator cells with the one-process tenant. Pattern
# testVectors/multi-cell/launch_pattern_F08_8C_tmp_alt_l3.yaml (make_launch_pattern_tdd.py from 59c, frames 2-3 UL vectors swapped for their L3 copies as alt10_run.sh does; a 59c|60c mix made the RU emulator flood its log with PRACH startPrb mismatch warnings) is
# registered in valid_perf_patterns.sh for the duration of the runs (restored on exit). Modes as in tn_runs.sh
# (base | ours | static32 | profiles). Needs the block-repeat RU emulator / testMAC (build.aarch64_f1002 has them).
set -u
R=/home/troy/build/aerial-cuda-accelerated-ran; T=$R/cuPHY-CP/dapp_hook/tenant; V=$R/testBenches/phase4_test_scripts/valid_perf_patterns.sh
O=${FS_OUT:-$R/prof/tenant}; mkdir -p $O; cd $R
BLOCK_REPEAT=${BLOCK_REPEAT:-1500}; SLOTS=${SLOTS:-360000}
[ -f testVectors/multi-cell/launch_pattern_F08_8C_tmp_alt_l3.yaml ] || { echo "!! pattern missing (make_launch_pattern_tdd.py)"; exit 1; }
cp -p $V $O/valid_perf_patterns.orig.sh
restore() { cp -p $O/valid_perf_patterns.orig.sh $V && echo "   valid_perf_patterns.sh restored"; }
trap restore EXIT
grep -q tmp_alt_l3 $V || sed -i 's/valid_perf_patterns=(/valid_perf_patterns=("tmp_alt_l3" /; s/peak_patterns=(/peak_patterns=("tmp_alt_l3" /' $V
echo "   registered tmp_alt_l3: $(grep -c tmp_alt_l3 $V)"
export RUN_ENV="RU_LP_BLOCK_FRAMES=2 RU_LP_BLOCK_REPEAT=$BLOCK_REPEAT TESTMAC_LP_BLOCK_FRAMES=2 TESTMAC_LP_BLOCK_REPEAT=$BLOCK_REPEAT"
specs=""; for s in "$@"; do IFS=: read -r n mode <<< "$s"; specs="$specs $n:tmp_alt_l3:$mode"; done
SLOTS=$SLOTS bash $T/tn_runs.sh $specs
