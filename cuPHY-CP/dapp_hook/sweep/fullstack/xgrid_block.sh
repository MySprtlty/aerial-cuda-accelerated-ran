#!/bin/bash
# xgrid_block.sh <X> <peak|average> [prefix]   (2026-10-16 night: demo cell-count X grid, see xgrid_table.py)
# One block = L1/RU/testMAC restarted per run by fullstack_run.sh with pattern tmp_ddddddsuuu_<load> and X cells, 100 s traffic
# (200,000 slots), measurement window [T0+35 s, T0+95 s]: R0 x2 (first with a 60 s hook-ring dump -> PRB window profile) ->
# per-TDD-position PUSCH budgets (p99.99 x 1.05) and RU late reference -> cells in the priority order of the night prompt
# (model-wise ascending caps 43 -> 61 -> 80, first failure stops the model: higher caps 'monotone-fail'; rate cells and the naive
# 26x@100 always run) -> closing R0 (block 'unstable' when its B_rel overruns or late rows reach 2x the start R0 mean) -> table.
# Every run goes through r0_ladder.sh (precheck: a foreign RAN process aborts; disk rule; PTP record; lab files restored and diffed).
# Tenant: dapp_sched + dispatcher (manual policy, model forced), 10 req/s Poisson, in-flight 1, from ~T0+5 s for 90 s.
# env: FS_OUT (default prof/run65), MAXMIN (block time limit, default 60), FILL=1 (run the fill cells too), SLOTS (200000),
#      CELLS_ONLY=1 (no start R0 runs: reuse budgets_<prefix>.json, run only the cells missing from the csv), NO_R0Z=1 (skip the closing R0),
#      SPOTS="26m:43 26s:43" (only these cells, kind spot), RATE=8 (request rate; names get _r8), FILL_ONLY=1 (average: fill cells only),
#      XMODELS ("26n 26s 26m 26l 26x"), XCLASSES ("56 80 104 132"); MIG_UUID/MIG_SM/TENANT_MIG_UUID pass through to fullstack_run/fs_tenant.
set -u
S=$(cd "$(dirname "$0")" && pwd); R=$(cd "$S/../../../.." && pwd); O=${FS_OUT:-$R/prof/run65}
X=$1; LOAD=$2; PREFIX=${3:-X${X}$( [ "$LOAD" = peak ] && echo P || echo A )}; MAXMIN=${MAXMIN:-60}; SLOTS=${SLOTS:-200000}
PAT=tmp_ddddddsuuu_$LOAD; CSV=$O/xgrid_$PREFIX.csv; BUD=$O/budgets_$PREFIX.json; SKIP=$O/skipped_$PREFIX.txt; LOG=$O/block_$PREFIX.log
export TENANT_MODELS=${XMODELS:-"26n 26s 26m 26l 26x"} TENANT_CLASSES=${XCLASSES:-"56 80 104 132"} TENANT_GLOB=${TENANT_GLOB:-uhd_*_q60.jpg}
COMMON="pattern=$PAT ncells=$X slots=$SLOTS tags=311,223,503 keep_raw=2"
mkdir -p $O; cd $R; [ "${CELLS_ONLY:-0}" = 1 ] || : > $SKIP; T_BLOCK=$(date +%s); CRASH=0; SUSPECT=0
cls() { case $1 in 43) echo 56;; 61) echo 80;; 80) echo 104;; 100) echo 132;; 24) echo 32;; *) echo $1;; esac; }
say() { echo "$(date +%T) $*" | tee -a $LOG; }
elapsed_min() { echo $(( ($(date +%s) - T_BLOCK) / 60 )); }
ladder() {  # <cname> <keys...> -> runs one r0_ladder line; returns r0_ladder's exit code (1 foreign run, 3 disk)
  local cname=$1; shift; local list=$O/list_${PREFIX}_$cname.txt
  echo "$cname $COMMON $*" > $list
  FS_OUT=$O bash $S/r0_ladder.sh $list $PREFIX >> $O/driver_${PREFIX}.log 2>&1; local rc=$?
  grep -E '^== |!!|precheck:|disk:|L1 did not|did not finish|report row' $O/driver_${PREFIX}.log | tail -4 | sed 's/^/   /' | tee -a $LOG
  return $rc
}
abort_check() { local rc=$1; [ $rc -eq 1 ] && { say "!! r0_ladder aborted (foreign run or precheck) -> block stops"; exit 2; }; [ $rc -eq 3 ] && { say "!! disk below 20 GB -> block stops"; exit 3; }; }
evaluate() {  # <name> <kind> <model> <cap> -> prints VERDICT
  local v; v=$(python3 $S/xgrid_table.py eval --dir $O --name $1 --budgets $BUD --csv $CSV --kind $2 --model "$3" --cap "$4" 2>&1 | tail -1)
  say "   $1: $v"; echo "$v" | grep -q 'no data' && CRASH=$((CRASH + 1)); echo "$v"
}
run_r0() {  # <cname> [record]
  local cname=$1 rec=""; [ "${2:-}" = record ] && rec="record=1 record_secs=60"
  ladder $cname $rec; abort_check $?
  local name=${PREFIX}_${cname}_${X}c
  if [ -n "$rec" ] && [ -s $R/prof/ring_16C_$name.bin ]; then
    python3 $S/ring_load_profile.py --dump $R/prof/ring_16C_$name.bin --window-s 0.1 --out $O/ringprof_$name > $O/ringprof_$name.txt 2>&1 && say "   PRB profile -> ringprof_$name.csv" || say "   PRB profile failed (see ringprof_$name.txt)"
    rm -f $R/prof/ring_16C_$name.bin $R/prof/ring_16C_$name.bin.hdr
  fi
}
run_cell() {  # <kind> <model> <cap>  -> sets V (PASS/FAIL/...)
  local kind=$1 m=$2 cap=$3; local cname=${m}_c${cap}; local c; c=$(cls $cap); local eng=$R/cuPHY-CP/dapp_hook/models/yolo${m}_1280_b8_e2e_fp16_sm${c}.engine
  [ "$kind" = rate ] && cname=rate_${m}_c${cap}; [ "$kind" = naive ] && cname=naive_${m}_c${cap}; [ "$kind" = spot ] && cname=spot_${m}_c${cap}
  [ "${RATE:-10}" != 10 ] && cname=${cname}_r${RATE}
  local name=${PREFIX}_${cname}_${X}c
  # the same (model, cap) measured under another kind (rate / naive / cell) counts: reuse its verdict instead of re-running
  local alts="$name"; [ "${RATE:-10}" = 10 ] && alts="$name ${PREFIX}_${m}_c${cap}_${X}c ${PREFIX}_rate_${m}_c${cap}_${X}c ${PREFIX}_naive_${m}_c${cap}_${X}c"
  local alt; for alt in $alts; do
    if [ -s $CSV ] && grep -q "^$alt," $CSV; then V=$(python3 -c "import csv,sys; print(next(r['verdict'] for r in csv.DictReader(open('$CSV')) if r['name']=='$alt'))"); say "   $name already measured as $alt: $V"; return; fi
  done
  [ -s $eng ] || { say "   $name: no engine $(basename $eng) -> skipped"; echo "$name no-engine" >> $SKIP; V=SKIP; return; }
  [ $(elapsed_min) -ge $MAXMIN ] && { say "   $name: block time limit ($MAXMIN min) -> unmeasured"; echo "$name time-limit" >> $SKIP; V=SKIP; return; }
  [ $CRASH -ge 2 ] && { say "   $name: two L1 crashes in this block -> unmeasured"; echo "$name crash-limit" >> $SKIP; V=SKIP; return; }
  ladder $cname tenant=dispatcher model=$m cap=$cap arrivals=poisson rate=${RATE:-10} tdelay=25 tsecs=90; abort_check $?
  V=$(evaluate $name $kind $m $cap | tail -1 | sed 's/VERDICT //')      # evaluate() also echoes its log line: keep the verdict only
}
suspect() {  # safety rule (01:05): >= 80 % of the block's measured cells (at least 3) FAIL for the same reason -> criterion suspect
  [ -s $CSV ] || return 1
  python3 - $CSV <<'PY'
import csv, sys, collections
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["kind"] in ("cell", "rate", "naive") and r["verdict"] not in ("", "no data") and not r["verdict"].startswith("INVALID")]
c = collections.Counter(r["verdict"] for r in rows if r["verdict"].startswith("FAIL"))
top = c.most_common(1)
ok = len(rows) >= 3 and top and top[0][1] >= 0.8 * len(rows)
print("%d measured, same-reason failures %s -> %s" % (len(rows), top[0] if top else "-", "SUSPECT" if ok else "normal"))
sys.exit(0 if ok else 1)
PY
}
ladder_model() {  # <model> <caps ascending...>: stop at the first failure, higher caps monotone-fail - unless the criterion is suspect
  local m=$1; shift; local cap stop=0
  for cap in "$@"; do
    if [ $stop = 1 ]; then
      if suspect > /dev/null; then say "   ${m}@$cap: criterion suspect (>= 80 % same-reason FAIL) -> measured anyway"; SUSPECT=1
      else echo "${PREFIX}_${m}_c${cap}_${X}c monotone-fail" >> $SKIP; say "   ${m}@$cap monotone-fail (unmeasured)"; continue; fi
    fi
    run_cell cell $m $cap; case "$V" in PASS) ;; SKIP) ;; *) stop=1 ;; esac
  done
}
say "== block $PREFIX: X=$X load=$LOAD pattern=$PAT slots=$SLOTS models '$TENANT_MODELS' classes '$TENANT_CLASSES' limit $MAXMIN min cells_only ${CELLS_ONLY:-0}"
if [ "${CELLS_ONLY:-0}" != 1 ]; then      # CELLS_ONLY=1: reuse the block's budgets, run only the cells not yet in the csv, no R0 runs
  run_r0 R0a record; run_r0 R0b
  python3 $S/xgrid_table.py budgets --dir $O --r0 ${PREFIX}_R0a_${X}c,${PREFIX}_R0b_${X}c --out $BUD | tee -a $LOG
  for r in R0a R0b; do evaluate ${PREFIX}_${r}_${X}c R0 "" "" > /dev/null; done
fi
if [ -n "${SPOTS:-}" ]; then          # SPOTS="26m:43 26s:43" [RATE=8]: only these cells (kind spot), no R0s unless asked
  for mc in $SPOTS; do run_cell spot ${mc%%:*} ${mc##*:}; done
elif [ "$LOAD" = peak ]; then
  for m in 26s 26m 26l; do ladder_model $m 43 61 80; done
  run_cell naive 26x 100
  for mc in 26x:61 26x:80 26l:61 26l:80; do run_cell rate ${mc%%:*} ${mc##*:}; done
  ladder_model 26n 43 61 80
else
  if [ "${FILL_ONLY:-0}" != 1 ]; then   # FILL_ONLY=1: only the fill cells (26m/26l/26x@43), no ladders
    for m in 26x 26l 26m; do ladder_model $m 61 80; done
    run_cell naive 26x 100
    ladder_model 26s 43 61 80; ladder_model 26n 43 61 80
  fi
  if [ "${FILL:-0}" = 1 ]; then for m in 26m 26l 26x; do run_cell cell $m 43; done; else for m in 26m 26l 26x; do echo "${PREFIX}_${m}_c43_${X}c fill-later" >> $SKIP; done; fi
fi
[ "${NO_R0Z:-0}" = 1 ] || { run_r0 R0z; evaluate ${PREFIX}_R0z_${X}c R0 "" "" > /dev/null; }
python3 $S/xgrid_stability.py --csv $CSV --prefix $PREFIX --x $X | tee -a $LOG
SUSPECT=${SUSPECT:-0}; suspect | sed 's/^/   final check: /' | tee -a $LOG; suspect > /dev/null && SUSPECT=1
[ "$SUSPECT" = 1 ] && { say "== block $PREFIX: 판정 기준 의심 (>= 80 % of the measured cells failed for the same reason; no monotone skipping applied from that point)"; echo "$PREFIX criterion-suspect" >> $SKIP; }
say "== block $PREFIX done in $(elapsed_min) min; skipped: $(tr '\n' ';' < $SKIP)"
python3 $S/xgrid_table.py table --csv $CSV | tee -a $LOG
