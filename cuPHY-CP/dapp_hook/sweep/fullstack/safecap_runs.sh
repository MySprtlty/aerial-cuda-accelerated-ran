#!/bin/bash
# SafeCap sweep: 16-cell 59c full stack + the tiled YOLO tenant through the dispatcher path (fs_tenant.sh TENANT_MODE=dispatcher),
# one model and one SM cap per run, closed-loop client (one request in flight). Per model the caps go 43 -> 61 -> 100 and stop at
# the first cap whose deadline-miss count rises above the no-tenant run R0 (safecap_check.py).
#   safecap_runs.sh <prefix> [models="26x 26s 26n"] [caps="43 61 100"]
# Runs (prof/run61/<prefix>_<run>_16c): R0 = no tenant (hook ring recorded: prof/ring_16C_<prefix>_R0_16c.bin), R1 = dapp_sched +
# dispatcher idle, then <model>_c<cap>. Each 240,000 slots (120 s); statistics skip the first 10 s (safecap_report.py).
# Logs kept per run: phy_<name>.csv (parse_phylog.py: PUSCH/PUCCH/PRACH/SRS GPU end, PDSCH/PDCCH stages), phy_warn_<name>.log,
# l2a_<name>.log, ru_<name>.log (RU.LATE_PACKETS INFO on: nvlog tag 503), disp_<name>.csv / client_<name>.csv (tenant).
S=$(cd "$(dirname "$0")" && pwd); R=$(cd "$S/../../../.." && pwd); O=$R/prof/run61
PREFIX=$1; MODELS=${2:-"26x 26s 26n"}; CAPS=${3:-"43 61 100"}; SLOTS=${SAFECAP_SLOTS:-240000}
run() {  # <name> [env...]
  local name=$1; shift
  env "$@" KEEP_RAW=1 MPSTAT=0 PATTERN=59c NCELLS=16 bash $S/fullstack_run.sh $name $SLOTS "" "311,223,503" async
  if [ -s $O/phy_$name.log ]; then
    python3 $S/parse_phylog.py $O/phy_$name.log $O/phy_$name.csv
    grep -F '[L2A.PROCESSING_TIMES]' $O/phy_$name.log > $O/l2a_$name.log
    grep -E ' (WRN|ERR|FATAL) |Send Err.ind|recovery|not responsive|No available|Successive Non-availability|Attempting PUSCH pipeline termination|Order kernel timeout|exit handler|pipeline failed' $O/phy_$name.log | head -2000000 > $O/phy_warn_$name.log
    rm -f $O/phy_$name.log $O/testmac_$name.log
    if [ -s $O/ru_$name.log ]; then      # RU late packets (tag 503): ru_late.py tables, then keep only the summary lines of the raw log
      python3 $S/ru_late.py $O/ru_$name.log $O/ru_$name > $O/ru_late_$name.txt 2>&1; rm -f $O/ru_${name}_slots.csv
      grep -E '\[RU\] \||PACKET_SUMMARY| (WRN|ERR) ' $O/ru_$name.log | head -400000 > $O/ru_warn_$name.log; rm -f $O/ru_$name.log
    fi
  else
    echo "no phy log for $name"
  fi
  echo "== safecap $name done $(date +%T)"
}
run ${PREFIX}_R0_16c RECORD=1 RECORD_SECS=${RECORD_SECS:-75}
[ -s $R/prof/ring_16C_${PREFIX}_R0_16c.bin ] && truncate -s ${RING_KEEP:-3G} $R/prof/ring_16C_${PREFIX}_R0_16c.bin   # raw stream prefix (~50 s) is enough for the replay
run ${PREFIX}_R1_16c TENANT_MODE=dispatcher TENANT_NOCLIENT=1 TENANT_CAP=100
for m in $MODELS; do
  for cap in $CAPS; do
    name=${PREFIX}_${m}_c${cap}_16c
    run $name TENANT_MODE=dispatcher TENANT_FORCE=$m TENANT_CAP=$cap TENANT_ARRIVALS=closed
    if ! python3 $S/safecap_check.py ${PREFIX}_R0_16c $name; then
      echo "== $m: deadline misses above R0 at cap $cap -> higher caps skipped"; break
    fi
  done
done
echo SAFECAP_DONE_$PREFIX $(date +%T)
