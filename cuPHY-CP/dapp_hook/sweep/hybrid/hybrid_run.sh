#!/bin/bash
# hybrid_run.sh start [name] | stop [--now] | status
#   Brings up, in order, RU emulator (N cells) -> cuphycontroller (N+1 cells: Benetel = cell 0 on 'nvipc', emulator = cells 1..N on 'nvipc1')
#   -> OAI gNB container (CONFIG/START of cell 0) -> testMAC, then monitors until 'stop' (or a STOP file / the configured slot count) and
#   tears everything down in reverse order, copies the logs to prof/hybrid_runs/<name>/ and (unless KEEP_CONFIG=1) restores the lab files.
#   Prerequisites: hybrid_config.sh apply (reads prof/hybrid_runs/current.env), c_aerial_troy running, no other L1 / OAI stack (never next to
#   the researchers' nv-cubb stack), Open5GS up (NO_CORE=1 skips that check), PTP locked. Start order: with wait_all_cell_config 1 the L1 sends
#   no SLOT.indication until the OAI cell is CONFIG-ed, so OAI is started before testMAC.
#   L1 binary: L1_BUILD_DIR (default benetel_cell.env L1_BUILD_DIR_DEFAULT = build.aarch64_f1002: SCF_FAPI_10_04=OFF like the baseline build,
#   + the env-gated PRACH threshold change; CUPHY_PRACH_FORCE_THR0 is exported from PRACH_FORCE_THR0). Env: DURATION_S, NO_CORE=1, KEEP_CONFIG=1.
set -u
H=$(cd "$(dirname "$0")" && pwd); R=$(cd "$H/../../../.." && pwd); cd $R
ENVF=${HYBRID_ENV:-$H/benetel_cell.env}; . $ENVF
OUT=$R/prof/hybrid_runs; P=/opt/nvidia/cuBB/testBenches/phase4_test_scripts
cmd=${1:-status}; NAME=${2:-hybrid_$(date +%m%d_%H%M)}
RUN=$OUT/$NAME; CUR=$OUT/current_run
L1_BUILD_DIR=${L1_BUILD_DIR:-${L1_BUILD_DIR_DEFAULT:-build.aarch64}}
c_up() { [ "$(docker inspect -f '{{.State.Running}}' c_aerial_troy 2>/dev/null)" = "true" ]; }
in_c() { docker exec c_aerial_troy bash -c "$1" 2>/dev/null; }
ran_alive() { ps -eo stat,comm --no-headers | awk '$1 !~ /^Z/ && $2 ~ /^(phy_main|cuphycontroller|ru_emulator|test_mac|nr-softmodem)/' | wc -l; }

stop_all() {   # reverse order; idempotent
  local D=${1:-$RUN}; mkdir -p $D
  echo "== stop $(date +%T)"
  in_c 'pkill -INT -x test_mac' ; for i in $(seq 1 20); do in_c 'pgrep -x test_mac' > /dev/null || break; sleep 1; done; in_c 'pkill -KILL -x test_mac'
  if docker ps --format '{{.Names}}' | grep -qx oai-gnb-aerial-hybrid; then docker logs oai-gnb-aerial-hybrid > $D/oai.log 2>&1; docker compose -f $OAI_DIR/docker-compose.yaml down > /dev/null 2>&1 && echo "   OAI gNB container stopped"; fi
  docker exec -u root c_aerial_troy bash -c 'pkill -TERM -x phy_main; sleep 6; pkill -KILL -x phy_main; pkill -TERM -x ru_emulator; for i in $(seq 1 20); do pgrep -x ru_emulator > /dev/null || break; sleep 1; done; pkill -KILL -x ru_emulator' 2>/dev/null
  in_c "grep -E ' (WRN|ERR|FATAL) |Send Err.ind|recovery|not responsive|Order kernel timeout|exit handler|pipeline failed|CONFIG.req received|CONFIG.response|START.req|STOP.req|PHY reset|L1 is ready|RACH|prach' /tmp/phy.log" > $D/phy_warn.log 2>/dev/null
  in_c "grep -E '\] Cell +[0-9]+ \| DL .* Slots CRC' /tmp/phy.log" > $D/l1_persec.log 2>/dev/null
  in_c "tail -n 2000000 /tmp/phy.log" > $D/phy_tail.log 2>/dev/null   # full L1 log tail (all levels, ~1-2 min) for post-mortem
  in_c "tail -n 400000 /tmp/testmac.log" > $D/testmac_tail.log 2>/dev/null; in_c "grep -E '\[RU\] \||PACKET_SUMMARY| (WRN|ERR) ' /tmp/ru.log | tail -n 400000" > $D/ru_warn.log 2>/dev/null
  in_c "grep -E 'DL U (LATE|ON TIME|EARLY) PACKETS|DL C LATE SLOTS|DL U LATE SLOTS' /tmp/ru.log" > $D/ru_table.txt 2>/dev/null
  in_c "grep -c 'Payload Validation ERROR' /tmp/ru.log" > $D/ru_payload_errors.txt 2>/dev/null
  docker exec -u root c_aerial_troy bash -c 'rm -f /tmp/phy.log /tmp/ru.log /tmp/testmac.log' 2>/dev/null
  echo "   Err.ind in L1 log: $(grep -c 'Send Err.ind' $D/phy_warn.log 2>/dev/null); FATAL: $(grep -c FATAL $D/phy_warn.log 2>/dev/null); OAI RRC connections: $(grep -cE 'RRCSetupComplete|RRC_CONNECTED' $D/oai.log 2>/dev/null); logs in $D"
  rm -f $CUR
  if [ "${KEEP_CONFIG:-0}" != "1" ]; then bash $H/hybrid_config.sh restore; else echo "   KEEP_CONFIG=1: lab files left in the hybrid state (hybrid_config.sh restore later)"; fi
}

case "$cmd" in
start)
  [ -f $OUT/current.env ] || { echo "configuration not applied: run $H/hybrid_config.sh apply first"; exit 1; }
  . <(sed 's/ /\n/g' $OUT/current.env | grep -E '^(N|NC|OAI_CELL|PATTERN|SLOTS)=')
  OC=${OAI_CELL:-$N}     # L1 cell index of the Benetel/OAI cell (BENETEL_CELL_INDEX last -> N, first -> 0)
  DUR=${DURATION_S:-$(( SLOTS / 2000 ))}
  mkdir -p $RUN; echo "$NAME" > $CUR; rm -f $RUN/STOP
  echo "== $NAME: Benetel = L1 cell $OC + $N emulator cells, pattern $PATTERN, $SLOTS slots (~$DUR s), L1 $L1_BUILD_DIR $(date +%T)"
  # ---- prechecks
  ok=1
  c_up || { echo "!! c_aerial_troy not running (cuPHY-CP/container/run_aerial.sh -d)"; ok=0; }
  docker ps --format '{{.Names}}' | grep -qE '^(nv-cubb|oai-gnb-aerial_hsw|oai-gnb-aerial-hybrid)$' && { echo "!! an nv-cubb / oai-gnb container is running (the researchers' compose stack?) - stop it first"; ok=0; }
  [ "$(ran_alive)" = 0 ] || { echo "!! RAN processes alive: $(ps -eo comm --no-headers | grep -E '^(phy_main|cuphycontroller|ru_emulator|test_mac|nr-softmodem)' | sort | uniq -c | tr '\n' ' ')"; ok=0; }
  grep -q nvipc_multi_hybrid cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml || { echo "!! l2_adapter yaml is not in the hybrid state"; ok=0; }
  [ -x $R/$L1_BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf ] || { echo "!! $L1_BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf missing"; ok=0; }
  # FAPI wire layout of the build must equal the baseline build (SCF 222.10.02 as OAI sends it): SCF_FAPI_10_04 OFF and
  # ENABLE_CONFORMANCE_TM_PDSCH_PDCCH OFF (ON inserts a testMode byte into DL_TTI.request -> OAI's DL_TTI is misparsed, no SSB/SIB1; ue1-ue5 10-29).
  for kv in SCF_FAPI_10_04:BOOL=OFF ENABLE_CONFORMANCE_TM_PDSCH_PDCCH:BOOL=OFF SCF_FAPI_10_04_SRS:BOOL=ON; do
    grep -q "^$kv\$" $R/$L1_BUILD_DIR/CMakeCache.txt 2>/dev/null || { echo "!! $L1_BUILD_DIR/CMakeCache.txt: expected $kv (baseline build flags; reconfigure with cmake -D<flag> and rebuild cuphycontroller_scf test_mac)"; ok=0; }
  done
  if [ -n "${PRACH_FORCE_THR0:-}" ]; then
    grep -q CUPHY_PRACH_FORCE_THR0 $R/$L1_BUILD_DIR/cuPHY-CP/cuphydriver/src/libcuphydriver.so 2>/dev/null || echo "!! PRACH_FORCE_THR0=$PRACH_FORCE_THR0 set but $L1_BUILD_DIR/cuPHY-CP/cuphydriver/src/libcuphydriver.so does not contain the CUPHY_PRACH_FORCE_THR0 change (rebuild as root in c_aerial_troy: ninja -C $L1_BUILD_DIR cuphycontroller_scf)"
  fi
  hp=$(awk '/HugePages_Free/{print $2}' /proc/meminfo); [ "$hp" -ge 20 ] || echo "!! only $hp hugepages free"
  pgrep -x ptp4l > /dev/null && pgrep -x phc2sys > /dev/null || { echo "!! ptp4l/phc2sys not running"; ok=0; }
  [ "$(cat /sys/class/net/aerial00/operstate 2>/dev/null)" = up ] || { echo "!! aerial00 (Benetel link) is not up"; ok=0; }
  ping -c1 -W1 10.10.0.100 > /dev/null 2>&1 && echo "   Benetel RU 10.10.0.100 answers ping" || echo "   note: Benetel RU 10.10.0.100 does not answer ping"
  if [ "${NO_CORE:-0}" != 1 ]; then
    if pgrep -x open5gs-amfd > /dev/null && ip -br addr show ogstun 2>/dev/null | grep -q '10.45.0.1'; then echo "   Open5GS up (amfd running, ogstun 10.45.0.1); NAT rules: $(sudo -n iptables -t nat -S POSTROUTING 2>/dev/null | grep -c OPEN5GS)"
    else echo "!! Open5GS not up: /home/nvidia/hsw/25_3_hsw/sh/start_cn.sh && sudo /home/nvidia/hsw/25_3_hsw/sh/routing.sh (NO_CORE=1 to skip)"; ok=0; fi
  fi
  docker image inspect oai-gnb-aerial_hsw:$OAI_TAG > /dev/null 2>&1 || { echo "!! image oai-gnb-aerial_hsw:$OAI_TAG missing"; ok=0; }
  [ $(df --output=avail -BG $R | tail -1 | tr -dc 0-9) -ge 15 ] || echo "!! less than 15 GB free on $R"
  [ $ok = 1 ] || { echo "precheck failed"; rm -f $CUR; exit 1; }
  trap 'echo; echo "   interrupted"; stop_all; exit 130' INT TERM
  # ---- RU emulator
  docker exec c_aerial_troy bash -lc "$P/run1_RU.sh --timeout $((DUR + 600))" > $RUN/run1_ru.log 2>&1 & RU=$!; sleep 25
  # ---- cuphycontroller (N+1 cells, two transports); PRACH threshold env for the real cell
  EXP=""; [ -n "${PRACH_FORCE_THR0:-}" ] && EXP="export CUPHY_PRACH_FORCE_THR0=$PRACH_FORCE_THR0;"
  # L1_EXTRA_ARGS: extra run2 options, parsed by the container's shell (quotes allowed), e.g. a backtrace on a crash:
  #   L1_EXTRA_ARGS="--gdb_script 'gdb -q -batch -x /opt/nvidia/cuBB/prof/hybrid_runs/gdbcmds --args'"
  docker exec -u root c_aerial_troy bash -lc "$EXP $P/run2_cuPHYcontroller.sh --build_dir=$L1_BUILD_DIR --timeout $((DUR + 500)) ${L1_EXTRA_ARGS:-}" > $RUN/run2_l1.log 2>&1 & L1=$!
  for i in $(seq 1 150); do grep -q "L1 is ready" $RUN/run2_l1.log 2>/dev/null && break; kill -0 $L1 2>/dev/null || break; sleep 2; done
  grep -q "L1 is ready" $RUN/run2_l1.log || { echo "!! L1 did not come up: $(grep -E 'FATAL|ERR' $RUN/run2_l1.log | tail -3 | cut -c1-160)"; stop_all; exit 1; }
  echo "   L1 ready $(date +%T) (cell_group_num $((N + 1)), OAI on $OAI_PREFIX -> cell $OC, testMAC on $TESTMAC_PREFIX${PRACH_FORCE_THR0:+, CUPHY_PRACH_FORCE_THR0=$PRACH_FORCE_THR0})"
  # ---- OAI gNB : CONFIG/START of the Benetel cell (baseline prefix)
  (cd $OAI_DIR && TAG=$OAI_TAG OAI_CPUSET=$OAI_CPUSET docker compose -f docker-compose.yaml up -d) > $RUN/oai_compose.log 2>&1 || { echo "!! OAI compose up failed: $(tail -2 $RUN/oai_compose.log)"; stop_all; exit 1; }
  echo "   OAI gNB container started (cpuset $OAI_CPUSET, prefix $OAI_PREFIX, image $OAI_TAG) $(date +%T); waiting for CONFIG.request of cell $OC"
  for i in $(seq 1 90); do in_c "grep -q 'CONFIG.req received for cell_id=$OC ' /tmp/phy.log" && break; docker ps --format '{{.Names}}' | grep -qx oai-gnb-aerial-hybrid || break; sleep 2; done
  if in_c "grep -q 'CONFIG.req received for cell_id=$OC ' /tmp/phy.log"; then
    echo "   cell $OC: $(in_c "grep -E 'cell_id=$OC .*(CONFIG|START)|send_cell_config_response.*cell_id=$OC|Cell $OC Received START' /tmp/phy.log" | tail -2 | cut -c1-150 | tr '\n' ';')"
    in_c "grep -q 'Cell $OC Received START.request' /tmp/phy.log" || echo "   note: no START.request for cell $OC yet - watch $RUN/oai_live.log (CONFIG.response error -> OAI stays silent)"
  else echo "!! no CONFIG.request from OAI for cell $OC within 180 s (docker logs oai-gnb-aerial-hybrid); continuing - with wait_all_cell_config 1 testMAC will idle until it arrives"; fi
  docker logs -f oai-gnb-aerial-hybrid > $RUN/oai_live.log 2>&1 & OL=$!
  # ---- testMAC (cells 0..N-1 on the wire = L1 cells 1..N)
  docker exec c_aerial_troy bash -lc "$P/run3_testMAC.sh --build_dir=$L1_BUILD_DIR --timeout $((DUR + 300))" > $RUN/run3_testmac.log 2>&1 & MAC=$!
  echo "   testMAC started $(date +%T) ($N cells on $TESTMAC_PREFIX); monitoring (stop: $0 stop, or touch $RUN/STOP)"
  t0=$(date +%s)
  while :; do
    sleep 10
    [ -f $RUN/STOP ] && { echo "   STOP file seen"; break; }
    kill -0 $MAC 2>/dev/null || { echo "   testMAC finished/exited $(date +%T)"; break; }
    kill -0 $L1 2>/dev/null || { echo "!! cuphycontroller exited $(date +%T)"; break; }
    docker ps --format '{{.Names}}' | grep -qx oai-gnb-aerial-hybrid || { echo "!! OAI gNB container exited $(date +%T)"; break; }
    el=$(( $(date +%s) - t0 )); [ $el -ge $((DUR + 60)) ] && { echo "   duration reached"; break; }
    e=$(in_c "grep -c 'Send Err.ind' /tmp/phy.log"); f=$(in_c "grep -c FATAL /tmp/phy.log"); m=$(in_c "grep 'Cell  0 ' /tmp/testmac.log | tail -1" | sed -E 's/.*(DL +[0-9.]+ Mbps).*(UL +[0-9.]+ Mbps).*(ERR +[0-9]+).*/\1 \2 \3/')
    c0=$(in_c "grep -E '\] Cell +$OC \| DL' /tmp/phy.log | tail -1" | sed -E 's/.*(UL +[0-9.]+ Mbps +[0-9]+ Slots CRC +[0-9]+).*/\1/')
    c1=$(in_c "grep -E '\] Cell +$(( OC == 0 ? 1 : 0 )) \| DL' /tmp/phy.log | tail -1" | sed -E 's/.*(DL +[0-9.]+ Mbps).*(UL +[0-9.]+ Mbps +[0-9]+ Slots CRC +[0-9]+).*/\1 \2/')
    o="OAI: $(grep -cE 'RRCSetupComplete|RRC_CONNECTED' $RUN/oai_live.log 2>/dev/null) RRC, $(grep -c 'PDU SESSION' $RUN/oai_live.log 2>/dev/null) PDU, $(grep -cE 'RA failed' $RUN/oai_live.log 2>/dev/null) RA-fail"
    printf '   %5ds  L1 Err.ind %s FATAL %s | Benetel cell %s: %s | emu cell: %s | testMAC: %s | %s\n' "$el" "${e:-?}" "${f:-?}" "$OC" "${c0:-n/a}" "${c1:-n/a}" "${m:-n/a}" "$o"
  done
  stop_all
  echo "== $NAME done $(date +%T)"
  ;;
stop)
  if [ -f $CUR ]; then touch $OUT/$(cat $CUR)/STOP; echo "STOP requested for $(cat $CUR) (the foreground runner tears down); if it is gone: $0 stop --now"; fi
  [ "${2:-}" = "--now" ] || [ ! -f $CUR ] && { KEEP_CONFIG=${KEEP_CONFIG:-0} stop_all $OUT/stop_$(date +%m%d_%H%M); }
  ;;
status)
  echo "config: $(cat $OUT/current.env 2>/dev/null || echo 'not applied') | run: $(cat $CUR 2>/dev/null || echo none)"
  echo "RAN processes: $(ps -eo comm --no-headers | grep -E '^(phy_main|cuphycontroller|ru_emulator|test_mac|nr-softmodem)' | sort | uniq -c | tr '\n' ' ')"
  echo "containers: $(docker ps --format '{{.Names}}' | grep -E 'c_aerial|oai-gnb|nv-cubb' | tr '\n' ' ')"
  ;;
*) echo "usage: $0 start [name] | stop [--now] | status"; exit 2;;
esac
