#!/bin/bash
# hybrid_config.sh apply [--ncells N] [--pattern P] [--slots S] [--wait-all 0|1] [--force]   |   restore   |   status
#   Prepares (or undoes) the configuration for ONE cuphycontroller serving the Benetel RU cell (OAI gNB, COTS UE; L1 cell 0, nvIPC prefix
#   'nvipc' = the baseline) plus N RU-emulator cells (testMAC on prefix 'nvipc1'; L1 cells 1..N). The L1/L2A yamls are GENERATED FROM THE
#   BASELINE FILES in ./baseline (the researchers' proven single-cell setup) by hybrid_yaml.py; the lab files only contribute the emulator
#   cell blocks, the capacity/instrumentation/core keys listed in benetel_cell.env and the hook line.
#   apply: backs up the lab files (prof/hybrid_runs/lab_backup), generates/registers the DDDDDDSUUU 'tmp_' pattern for N cells, runs the NVIDIA
#   phase-4 configuration inside c_aerial_troy (setup1/setup2/test_config.sh - they rewrite the lab yamls, so the generation comes AFTER),
#   then hybrid_yaml.py, then the OAI compose .env. restore: puts the backups back, removes generated files/patterns. Nothing here touches the GPU.
#   Values: benetel_cell.env (HYBRID_ENV=... to use another file). Run as troy on the host.
set -u
H=$(cd "$(dirname "$0")" && pwd); R=$(cd "$H/../../../.." && pwd); cd $R
ENVF=${HYBRID_ENV:-$H/benetel_cell.env}; . $ENVF
OUT=$R/prof/hybrid_runs; BK=$OUT/lab_backup; mkdir -p $OUT
P=/opt/nvidia/cuBB/testBenches/phase4_test_scripts
CFG=cuPHY-CP/cuphycontroller/config
LAB="$CFG/l2_adapter_config_F08_CG1.yaml $CFG/cuphycontroller_F08_CG1.yaml cuPHY-CP/ru-emulator/config/config.yaml cuPHY/nvlog/config/nvlog_config.yaml cuPHY-CP/testMAC/testMAC/test_mac_config.yaml testBenches/phase4_test_scripts/valid_perf_patterns.sh"
GEN="$CFG/nvipc_multi_hybrid.yaml"
cmd=${1:-status}; shift || true
N=${NCELLS_DEFAULT:-9}; PATTERN=tmp_ddddddsuuu_peak; SLOTS=7200000; FORCE=0; WAITALL=""
while [ $# -gt 0 ]; do case "$1" in
  --ncells) N=$2; shift 2;; --pattern) PATTERN=$2; shift 2;; --slots) SLOTS=$2; shift 2;; --force) FORCE=1; shift;; --wait-all) WAITALL=$2; shift 2;;
  --mode) echo "note: --mode is gone (one mode: Benetel cell 0 + N emulator cells); ignoring '$2'"; shift 2;;
  *) echo "unknown argument $1"; exit 2;; esac; done
container_up() { [ "$(docker inspect -f '{{.State.Running}}' c_aerial_troy 2>/dev/null)" = "true" ]; }

register_pattern() {  # <cells> : generate testVectors/multi-cell/launch_pattern_F08_<cells>C_$PATTERN.yaml from the official 59c/60c file and register the name
  local c=$1 src=59c f
  f=testVectors/multi-cell/launch_pattern_F08_${c}C_$PATTERN.yaml
  case "$PATTERN" in *average*) src=60c;; esac
  if [ ! -f $f ]; then
    python3 cuPHY-CP/dapp_hook/tools/make_launch_pattern_tdd.py --tdd DDDDDDSUUU --src testVectors/multi-cell/launch_pattern_F08_${c}C_$src.yaml --slots 80 --out $f > /dev/null || { echo "pattern generation failed ($f)"; return 1; }
    echo "$f" >> $BK/generated_patterns.txt; echo "   pattern generated: $f (from ${c}C_$src)"
  fi
  python3 - "$PATTERN" <<'PY'
import re, sys
p = "testBenches/phase4_test_scripts/valid_perf_patterns.sh"; name = sys.argv[1]; s = open(p).read()
if ('"%s"' % name) not in s:
    s = s.replace('valid_perf_patterns=(', 'valid_perf_patterns=("%s" ' % name, 1)
    if "average" not in name: s = s.replace('peak_patterns=(', 'peak_patterns=("%s" ' % name, 1)
    open(p, "w").write(s); print("   pattern registered in valid_perf_patterns.sh:", name)
PY
}

case "$cmd" in
apply)
  if [ -f $BK/applied ] && [ $FORCE = 0 ]; then echo "already applied ($(cat $BK/applied)); run 'restore' first or use --force"; exit 1; fi
  container_up || { echo "c_aerial_troy is not running (setup1/setup2/test_config.sh run inside it): start it with cuPHY-CP/container/run_aerial.sh -d"; exit 1; }
  pgrep -x phy_main > /dev/null && { echo "a cuphycontroller (phy_main) is running - stop it first"; exit 1; }
  for f in $H/$BASE_CUPHY $H/$BASE_L2A; do [ -f $f ] || { echo "baseline file missing: $f"; exit 1; }; done
  mkdir -p $BK; : > $BK/generated_patterns.txt
  if [ $FORCE = 0 ] || [ ! -f $BK/applied ]; then for f in $LAB; do cp -p $f $BK/$(basename $f); done; echo "   lab files backed up to $BK"; fi
  register_pattern $N || exit 1
  echo "   phase-4 configuration: $N emulator cells, pattern $PATTERN, $SLOTS slots (test_config.sh in the container)"
  docker exec -u root c_aerial_troy bash -lc "cd $P && ./setup1_DU.sh --ru-host-type=_LOOPBACK --du-eth0=aerial02 -y F08_CG1 && ./setup2_RU.sh --ru-eth0=aerial03" > $OUT/setup.log 2>&1 || { echo "setup failed: $(tail -2 $OUT/setup.log)"; exit 1; }
  docker exec -u root c_aerial_troy bash -lc "cd $P && ./test_config.sh $PATTERN --num-cells=$N --num-slots=$SLOTS" > $OUT/test_config.log 2>&1
  grep -q "total $N cells" $OUT/test_config.log || { echo "test_config failed: $(tail -3 $OUT/test_config.log)"; exit 1; }
  python3 $H/hybrid_yaml.py --ncells $N --env $ENVF --lab-cuphy $CFG/cuphycontroller_F08_CG1.yaml --lab-l2a $CFG/l2_adapter_config_F08_CG1.yaml \
    --ru cuPHY-CP/ru-emulator/config/config.yaml --testmac cuPHY-CP/testMAC/testMAC/test_mac_config.yaml --nvipc-out $CFG/nvipc_multi_hybrid.yaml \
    ${WAITALL:+--wait-all $WAITALL} > $OUT/hybrid_yaml.log || { echo "hybrid_yaml failed: $(tail -1 $OUT/hybrid_yaml.log)"; exit 1; }
  sed 's/^hybrid_yaml: /   /' $OUT/hybrid_yaml.log | grep -v -E '^   (written|base L1)'
  python3 - $CFG/cuphycontroller_F08_CG1.yaml $CFG/l2_adapter_config_F08_CG1.yaml $CFG/nvipc_multi_hybrid.yaml cuPHY-CP/ru-emulator/config/config.yaml cuPHY-CP/testMAC/testMAC/test_mac_config.yaml <<'PY' || exit 1
import sys, yaml
for f in sys.argv[1:]:
    try: yaml.safe_load(open(f))
    except Exception as e: sys.exit("generated file does not parse: %s: %s" % (f, e))
c = yaml.safe_load(open(sys.argv[1]))["cuphydriver_config"]; nv = yaml.safe_load(open(sys.argv[3]))
assert c["cell_group_num"] == len(c["cells"]) == sum(len(t["phy_cells"]) for t in nv["transport"]), "cell count mismatch"
print("   check: %d cells (%s), transports %s, parse ok" % (c["cell_group_num"], ", ".join(x["nic"] for x in c["cells"][:2]) + ", ...", [(t["shm_config"]["prefix"], t["phy_cells"]) for t in nv["transport"]]))
PY
  printf 'REGISTRY=""\nTAG="%s"\nOAI_CPUSET="%s"\n' "$OAI_TAG" "$OAI_CPUSET" > $OAI_DIR/.env
  grep -q "tr_s_shm_prefix *= *\"$OAI_PREFIX\"" $OAI_DIR/gnb_hybrid.conf || echo "!! $OAI_DIR/gnb_hybrid.conf does not use prefix $OAI_PREFIX"
  grep -q "tr_s_poll_core *= *$OAI_POLL_CORE;" $OAI_DIR/gnb_hybrid.conf || echo "!! gnb_hybrid.conf tr_s_poll_core != $OAI_POLL_CORE"
  diff -q <(sed -e "s/^\(\s*tr_s_poll_core\s*=\s*\)[0-9]*;/\1$OAI_POLL_CORE;/" $H/baseline/gnb_original.txt) $OAI_DIR/gnb_hybrid.conf > /dev/null || echo "!! gnb_hybrid.conf differs from baseline/gnb_original.txt beyond the poll core"
  TS=$(date +%Y%m%d_%H%M%S); mkdir -p $OUT/applied_$TS; for f in $LAB $GEN testBenches/phase4_test_scripts/test_config_summary.sh $OAI_DIR/gnb_hybrid.conf; do [ -f $f ] && cp $f $OUT/applied_$TS/; done; cp $OUT/hybrid_yaml.log $OUT/applied_$TS/
  OAI_CELL=$(sed -n 's/^hybrid_yaml: OAI_CELL=//p' $OUT/hybrid_yaml.log | tail -1); OAI_CELL=${OAI_CELL:-$N}
  echo "N=$N NC=$((N + 1)) OAI_CELL=$OAI_CELL PATTERN=$PATTERN SLOTS=$SLOTS applied $TS" > $BK/applied; cp $BK/applied $OUT/current.env
  echo "== applied (Benetel = L1 cell $OAI_CELL on '$OAI_PREFIX', $N emulator cells on '$TESTMAC_PREFIX'). Snapshot: $OUT/applied_$TS. Next: $H/hybrid_run.sh start"
  ;;
restore)
  [ -d $BK ] || { echo "no backup directory $BK - nothing to restore"; exit 0; }
  pgrep -x phy_main > /dev/null && echo "!! a cuphycontroller is still running; restoring files anyway (they are read at start-up only)"
  for f in $LAB; do [ -f $BK/$(basename $f) ] && cp -p $BK/$(basename $f) $f; done
  for f in $GEN; do rm -f $f; done
  if [ -f $BK/generated_patterns.txt ]; then while read -r f; do [ -n "$f" ] && rm -f "$f"; done < $BK/generated_patterns.txt; fi
  rm -f $BK/applied $OUT/current.env
  echo "== restored: $(for f in $LAB; do cmp -s $BK/$(basename $f) $f && printf '%s ok; ' $(basename $f) || printf '%s DIFFERS; ' $(basename $f); done)"
  echo "   git status of the lab files: $(git status --short -- $LAB | tr '\n' ' ')"
  ;;
status)
  if [ -f $BK/applied ]; then echo "applied: $(cat $BK/applied)"; for f in $LAB; do cmp -s $BK/$(basename $f) $f && echo "   $(basename $f): unchanged" || echo "   $(basename $f): MODIFIED vs backup"; done
  else echo "not applied"; fi
  grep -H '^nvipc_config_file' $CFG/l2_adapter_config_F08_CG1.yaml; grep -HE '^\s*(cell_group_num|fix_beta_dl):' $CFG/cuphycontroller_F08_CG1.yaml | tr '\n' ' '; echo; ls $CFG/nvipc_multi_hybrid.yaml 2>/dev/null
  ;;
*) echo "usage: $0 apply [--ncells N] [--pattern P] [--slots S] [--wait-all 0|1] [--force] | restore | status"; exit 2;;
esac
