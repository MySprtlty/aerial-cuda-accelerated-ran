#!/bin/bash
# cots_run.sh start [name] | stop [--now] | status | ue [name] | test [model] [req/s] [seconds]
#   1-cell COTS-UE demo on OUR cuphycontroller: Benetel RU + OAI gNB (container) + Open5GS (host) with CUDA green contexts, the in-process
#   YOLO dApp tenant and the xApp agent (TCP 9810). No MPS. Settings: cots.env. Everything host-side lives in this directory; the L1 runs
#   inside the Aerial container ($CONTAINER) via docker exec, so any user in the docker group can run this (the nvidia account included).
#   start  : prechecks -> MPS off -> our L1/L2A yamls into the container -> cuphycontroller_scf (green contexts + dApp) -> wait 'L1 is ready'
#            -> wait for the dApp tenant (engines loaded, agent port open) -> OAI gNB container (CONFIG/START of cell 0, NG Setup) -> monitor
#            every 10 s (L1 Err.ind/FATAL, cell 0 DL/UL, UE RA/RRC/PDU counts, dApp stats) until 'stop', Ctrl-C, a STOP file or DURATION_S
#   stop   : tear down in reverse order (OAI container, L1), collect the logs into runs/<name>/ (MPS stays off, the core network stays up)
#   ue     : UE attach report from the logs of a run (phases RA -> RRC -> registration -> PDU session, failure signals)
#   test   : xApp smoke test from inside the container (ra_stub.py sends 4K JPEGs to the agent; default 26x, 2 req/s, 15 s)
#   Env: DURATION_S=<s> (default: until stop), NO_CORE=1 (skip the Open5GS check), L1_EXTRA_ENV="VAR=val ..." (extra L1 environment)
set -u
D=$(cd "$(dirname "$0")" && pwd); . $D/cots.env
OUT=$D/runs; CUR=$OUT/current_run; SDK=/opt/nvidia/cuBB; CFG=$SDK/cuPHY-CP/cuphycontroller/config; PHYLOG=/tmp/phy.log; mkdir -p $OUT
cmd=${1:-status}; NAME=${2:-cots_$(date +%m%d_%H%M)}; RUN=$OUT/$NAME
in_c() { docker exec $CONTAINER bash -c "$1" 2>/dev/null; }
in_r() { docker exec -u root $CONTAINER bash -c "$1" 2>/dev/null; }
c_up() { [ "$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null)" = true ]; }
mps_on() { in_c 'pgrep -f "nvidia-cuda-mps-serve[r]"' > /dev/null; }
mps_off() { in_r 'export CUDA_MPS_PIPE_DIRECTORY=/var CUDA_MPS_LOG_DIRECTORY=/var; echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 1; pkill -f "nvidia-cuda-mps-contro[l]"; pkill -f "nvidia-cuda-mps-serve[r]"; true'; sleep 1; ! mps_on; }
ran_procs() { ps -eo stat,comm --no-headers | awk '$1 !~ /^Z/ && $2 ~ /^(phy_main|cuphycontroller|ru_emulator|test_mac|nr-softmodem)/ {print $2}' | sort | uniq -c | tr '\n' ' '; }
core_up() { pgrep -x open5gs-amfd > /dev/null && ip -br addr show ogstun 2>/dev/null | grep -q '10.45.0.1'; }
port_open() { (echo > /dev/tcp/127.0.0.1/$DAPP_AGENT_PORT) 2>/dev/null; }
# engine set of the input profile (.env DAPP_INPUT, 2026-11-09): 4k = models/full26<m> batch-8 tiles, fhd = models/fhd26<m> batch-3 (+ FHD warm-up frames)
ESET=full26; FHD_ENV=""
if [ "${DAPP_INPUT:-4k}" = fhd ]; then ESET=fhd26; M=$SDK/cuPHY-CP/dapp_hook/models
  FHD_ENV="DAPP_TENANT_MODELS=$(c=""; for m in n s m l x; do printf "%s26%s=%s" "$c" $m $M/fhd26$m; c=,; done) DAPP_AGENT_WARMUP=$M/imgfhd/fhd_0_q85.jpg,$M/imgfhd/fhd_1_q85.jpg DAPP_TENANT_CLASSES=${DAPP_TENANT_CLASSES_FHD:-24,40,48,56,64,72,80}"; fi
# dApp SM classes of the profile (.env DAPP_TENANT_CLASSES / DAPP_TENANT_CLASSES_FHD). Both ladders end at 80 since 2026-11-10 11:00: the tenant now
# deserializes every engine in the primary context (TensorRT 10.3 Myelin caches per-kernel GEMM grids process-wide from the context of the first
# deserialization) and the fhd26s/fhd26l sm80 engines are built with the sm72 timing cache (their native sm80 tactics fail "Cask Gemm execution"
# next to another class of the same model) - see tenant/dapp_tenant.cu load_set and _dgist scripts/build_yolo_engines.sh.
CL=$DAPP_TENANT_CLASSES; [ "${DAPP_INPUT:-4k}" = fhd ] && CL=${DAPP_TENANT_CLASSES_FHD:-24,40,48,56,64,72,80}
oai_up() { docker ps --format '{{.Names}}' | grep -qx $OAI_CONTAINER; }
compose() { (cd $D/oai && OAI_IMAGE=$OAI_IMAGE OAI_CPUSET=$OAI_CPUSET OAI_CONTAINER=$OAI_CONTAINER docker compose "$@"); }

stop_all() {   # reverse order; idempotent; logs into $1 (default $RUN)
  local Dd=${1:-$RUN}; mkdir -p $Dd; echo "== stop $(date +%T)"
  if docker ps -a --format '{{.Names}}' | grep -qx $OAI_CONTAINER; then docker logs $OAI_CONTAINER > $Dd/oai.log 2>&1; compose down > /dev/null 2>&1 && echo "   OAI gNB container stopped"; fi
  in_r 'pkill -TERM -x phy_main; for i in $(seq 1 10); do pgrep -x phy_main > /dev/null || break; sleep 1; done; pkill -KILL -x phy_main; true'
  pgrep -x phy_main > /dev/null || echo "   L1 stopped"
  in_c "grep -E ' (WRN|ERR|FATAL) |Send Err.ind|recovery|not responsive|Order kernel timeout|exit handler|CONFIG.req received|START.req|L1 is ready|RACH|Preambles|dApp sizer' $PHYLOG" > $Dd/phy_warn.log 2>/dev/null
  in_c "grep -E '\] Cell +0 \| DL' $PHYLOG" > $Dd/l1_persec.log 2>/dev/null
  in_c "tail -n 2000000 $PHYLOG" > $Dd/phy_tail.log 2>/dev/null
  in_c "cat /tmp/dapp_tenant.log" > $Dd/dapp_tenant.log 2>/dev/null; in_c "cat /tmp/dapp_tenant_slots.csv" > $Dd/dapp_tenant_slots.csv 2>/dev/null
  in_r "rm -f $PHYLOG /tmp/dapp_tenant.log /tmp/dapp_tenant_slots.csv"
  echo "   L1 Err.ind $(grep -c 'Send Err.ind' $Dd/phy_warn.log 2>/dev/null), FATAL $(grep -c FATAL $Dd/phy_warn.log 2>/dev/null) | UE RRC connections $(grep -c RRCSetupComplete $Dd/oai.log 2>/dev/null), PDU sessions $(grep -c 'PDU SESSION' $Dd/oai.log 2>/dev/null) | logs: $Dd"
  rm -f $CUR
}

case "$cmd" in
start)
  mkdir -p $RUN; rm -f $RUN/STOP
  echo "== $NAME: 1-cell COTS demo on our L1 ($BUILD_DIR, config $L1_CONFIG, green contexts, dApp classes $CL, input ${DAPP_INPUT:-4k}) $(date +%T)"
  ok=1
  c_up || { echo "!! container $CONTAINER is not running (in the SDK tree: cuPHY-CP/container/run_aerial.sh -d)"; ok=0; }
  [ -f $CUR ] && { echo "!! a run is registered as current ($(cat $CUR)); '$0 stop --now' first"; ok=0; }
  other=$(docker ps --format '{{.Names}}' | grep -E "^(nv-cubb|oai-gnb.*|$OAI_CONTAINER)$" | tr '\n' ' '); [ -z "$other" ] || { echo "!! L1/gNB container(s) of another stack running: $other - stop that stack first (never two L1s on this GPU/NIC)"; ok=0; }
  rp=$(ran_procs); [ -z "$rp" ] || { echo "!! RAN processes alive: $rp"; ok=0; }
  if c_up; then
    in_c "test -x $SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf" || { echo "!! $BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf missing in $CONTAINER"; ok=0; }
    for kv in SCF_FAPI_10_04:BOOL=OFF ENABLE_CONFORMANCE_TM_PDSCH_PDCCH:BOOL=OFF SCF_FAPI_10_04_SRS:BOOL=ON ENABLE_DAPP_HOOK:BOOL=ON; do in_c "grep -q '^$kv\$' $SDK/$BUILD_DIR/CMakeCache.txt" || { echo "!! $BUILD_DIR/CMakeCache.txt: expected $kv (baseline FAPI wire layout + dApp hook)"; ok=0; }; done
    for m in n s m l x; do in_c "test -e $SDK/cuPHY-CP/dapp_hook/models/${ESET}$m/seg00_sm${CL##*,}.engine" || { echo "!! YOLO engine models/${ESET}$m/seg00_sm${CL##*,}.engine missing (models/build_engines.sh)"; ok=0; }; done
    [ -n "${PRACH_FORCE_THR0:-}" ] && { in_c "grep -q CUPHY_PRACH_FORCE_THR0 $SDK/$BUILD_DIR/cuPHY-CP/cuphydriver/src/libcuphydriver.so" || echo "!! PRACH_FORCE_THR0 set but libcuphydriver.so lacks the CUPHY_PRACH_FORCE_THR0 change"; }
  fi
  [ "$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null)" = Disabled ] || { echo "!! GPU MIG mode is not Disabled"; ok=0; }
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v nvidia-cuda-mps-server | grep -c .); [ "$apps" = 0 ] || echo "!! $apps other process(es) on the GPU: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"
  hp=$(awk '/HugePages_Free/{print $2}' /proc/meminfo); [ "$hp" -ge 20 ] || echo "!! only $hp hugepages free"
  pgrep -x ptp4l > /dev/null && pgrep -x phc2sys > /dev/null || { echo "!! ptp4l/phc2sys not running (PTP)"; ok=0; }
  [ "$(cat /sys/class/net/$FH_IF/operstate 2>/dev/null)" = up ] || { echo "!! $FH_IF (Benetel fronthaul link) is not up"; ok=0; }
  ping -c1 -W1 $RU_IP > /dev/null 2>&1 && echo "   Benetel RU $RU_IP answers ping" || echo "   note: Benetel RU $RU_IP does not answer ping"
  if [ "${NO_CORE:-0}" != 1 ]; then core_up && echo "   Open5GS up (amfd running, ogstun 10.45.0.1)" || { echo "!! Open5GS not up: $D/cots_core.sh up   (NO_CORE=1 skips this check)"; ok=0; }; fi
  docker image inspect $OAI_IMAGE > /dev/null 2>&1 || { echo "!! OAI image $OAI_IMAGE missing"; ok=0; }
  [ $(df --output=avail -BG $D | tail -1 | tr -dc 0-9) -ge 10 ] || echo "!! less than 10 GB free on $D"
  [ $ok = 1 ] || { echo "precheck failed"; exit 1; }
  echo "$NAME" > $CUR
  trap 'echo; echo "   interrupted"; stop_all; exit 130' INT TERM
  # ---- MPS off (green contexts partition the GPU; the researchers' entrypoint starts MPS, the dApp needs it off)
  mps_off && echo "   MPS off" || { echo "!! MPS server still running in $CONTAINER"; rm -f $CUR; exit 1; }
  # ---- our yamls into the container's config directory (new file names; the lab files are not touched)
  docker exec -i -u root $CONTAINER bash -c "cat > $CFG/cuphycontroller_$L1_CONFIG.yaml" < $D/config/cuphycontroller_$L1_CONFIG.yaml
  docker exec -i -u root $CONTAINER bash -c "cat > $CFG/l2_adapter_$L1_CONFIG.yaml" < $D/config/l2_adapter_$L1_CONFIG.yaml
  in_r "rm -f $PHYLOG /tmp/dapp_tenant.log /tmp/dapp_tenant_slots.csv"
  # ---- L1 = cuphycontroller_scf with the dApp tenant + xApp agent
  ENVS="export cuBB_SDK=$SDK DAPP_TENANT=1 DAPP_AGENT=1 DAPP_AGENT_CELLS=1 DAPP_PUSCH_PROFILES=$DAPP_PUSCH_PROFILES DAPP_PDSCH_PROFILES=$DAPP_PDSCH_PROFILES DAPP_SIZER_MODEL=$SDK/cuPHY-CP/dapp_hook/estimator/sm_model.txt DAPP_TENANT_CLASSES=$DAPP_TENANT_CLASSES DAPP_TENANT_FB_GUARD=0 DAPP_TENANT_LAYOUT=union DAPP_TENANT_CPU=$DAPP_TENANT_CPU DAPP_AGENT_CPU=$DAPP_AGENT_CPU DAPP_AGENT_PORT=$DAPP_AGENT_PORT DAPP_AGENT_GRAPH_CHUNK=$DAPP_AGENT_GRAPH_CHUNK DAPP_AGENT_CHUNK_AHEAD=$DAPP_AGENT_CHUNK_AHEAD DAPP_AGENT_QOS=$DAPP_AGENT_QOS DAPP_TENANT_LOG=/tmp/dapp_tenant.log DAPP_TENANT_CSV=/tmp/dapp_tenant_slots.csv ${PRACH_FORCE_THR0:+CUPHY_PRACH_FORCE_THR0=$PRACH_FORCE_THR0} DAPP_INPUT=${DAPP_INPUT:-4k} ${FHD_ENV:-} ${L1_EXTRA_ENV:-}"
  echo "$ENVS" > $RUN/l1_env.txt
  docker exec -u root $CONTAINER bash -c "$ENVS; cd $SDK && exec $SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf $L1_CONFIG" > $RUN/l1_console.log 2>&1 & L1=$!
  echo "   L1 starting: cuphycontroller_scf $L1_CONFIG${PRACH_FORCE_THR0:+ (CUPHY_PRACH_FORCE_THR0=$PRACH_FORCE_THR0)} $(date +%T)"
  l1_ready() { grep -q 'L1 is ready' $RUN/l1_console.log 2>/dev/null || in_c "grep -q 'L1 is ready' $PHYLOG"; }
  for i in $(seq 1 150); do l1_ready && break; kill -0 $L1 2>/dev/null || break; sleep 2; done
  l1_ready || { echo "!! L1 did not come up: $(grep -E 'FATAL|ERR' $RUN/l1_console.log | tail -3 | cut -c1-160 | tr '\n' ';')"; stop_all; exit 1; }
  echo "   L1 is ready $(date +%T); waiting for the dApp tenant (engine loading, up to 8 min)"
  for i in $(seq 1 240); do port_open && break; in_c "grep -qE 'tenant NOT started|FATAL' /tmp/dapp_tenant.log $PHYLOG" && break; kill -0 $L1 2>/dev/null || break; sleep 2; done
  port_open || { echo "!! dApp tenant did not start: $(in_c "grep -E 'tenant NOT started|FATAL' $PHYLOG /tmp/dapp_tenant.log" | tail -1 | cut -c1-160)"; stop_all; exit 1; }
  echo "   dApp tenant ready: xApp agent on TCP $DAPP_AGENT_PORT $(date +%T) | $(in_c 'grep -E "CUDA graphs captured" /tmp/dapp_tenant.log' | tail -1 | cut -c1-90) | GPU mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  # ---- OAI gNB: CONFIG/START of cell 0 on nvIPC 'nvipc', NG Setup with the AMF
  compose up -d > $RUN/oai_compose.log 2>&1 || { echo "!! OAI compose up failed: $(tail -2 $RUN/oai_compose.log)"; stop_all; exit 1; }
  echo "   OAI gNB container $OAI_CONTAINER started (image $OAI_IMAGE, cpuset $OAI_CPUSET) $(date +%T); waiting for CONFIG/START of cell 0"
  for i in $(seq 1 90); do in_c "grep -qE 'START.req received|Received START.request' $PHYLOG" && break; oai_up || break; sleep 2; done
  in_c "grep -qE 'START.req received|Received START.request' $PHYLOG" && echo "   cell 0 CONFIG/START done $(date +%T)" || echo "!! no START.request for cell 0 within 180 s (docker logs $OAI_CONTAINER)"
  docker logs -f $OAI_CONTAINER > $RUN/oai.log 2>&1 & OL=$!
  sleep 5; grep -q 'NGSetupResponse' $RUN/oai.log && echo "   NG Setup with the AMF ok" || echo "   note: no NGSetupResponse yet (core network? watch $RUN/oai.log)"
  echo "   running - a COTS UE can attach now. stop: $0 stop   (or Ctrl-C, or touch $RUN/STOP)"
  t0=$(date +%s); DUR=${DURATION_S:-0}
  while :; do
    sleep 10
    [ -f $RUN/STOP ] && { echo "   STOP file seen"; break; }
    kill -0 $L1 2>/dev/null || { echo "!! L1 exited $(date +%T): $(grep -E 'FATAL|Segmentation|exit' $RUN/l1_console.log | tail -2 | cut -c1-140 | tr '\n' ';')"; break; }
    oai_up || { echo "!! OAI gNB container exited $(date +%T)"; break; }
    el=$(( $(date +%s) - t0 )); [ $DUR -gt 0 ] && [ $el -ge $DUR ] && { echo "   duration reached"; break; }
    e=$(in_c "grep -c 'Send Err.ind' $PHYLOG"); f=$(in_c "grep -c FATAL $PHYLOG")
    c0=$(in_c "grep -E '\] Cell +0 \| DL' $PHYLOG | tail -1" | sed -E 's/.*(DL +[0-9.]+ Mbps).*(UL +[0-9.]+ Mbps +[0-9]+ Slots CRC +[0-9]+).*/\1 \2/')
    tn=$(in_c "grep -E 'agent ok' /tmp/dapp_tenant.log | tail -1" | sed -E 's/.*(agent ok [0-9]+ fail [0-9]+ queue [0-9]+).*/\1/')
    o="RA $(grep -c 'initiating RA procedure' $RUN/oai.log) RRC $(grep -c 'RRCSetupComplete' $RUN/oai.log) PDU $(grep -c 'PDU SESSION' $RUN/oai.log) RA-fail $(grep -cE 'RA failed|Contention Resolution timer expired' $RUN/oai.log)"
    printf '   %5ds  L1 Err.ind %s FATAL %s | cell 0: %s | UE: %s | dApp: %s\n' "$el" "${e:-?}" "${f:-?}" "${c0:-no traffic line yet}" "$o" "${tn:-idle}"
  done
  kill $OL 2>/dev/null; stop_all; echo "== $NAME done $(date +%T)"
  ;;
stop)
  if [ -f $CUR ] && [ "${2:-}" != "--now" ]; then touch $OUT/$(cat $CUR)/STOP; echo "STOP requested for $(cat $CUR) (the foreground runner tears down within 10 s); if that runner is gone: $0 stop --now"
  else stop_all $OUT/stop_$(date +%m%d_%H%M); fi
  ;;
status)
  echo "container $CONTAINER: $(c_up && echo up || echo DOWN) | MPS: $(mps_on && echo ON || echo off) | RAN processes: $(rp=$(ran_procs); echo ${rp:-none}) | gNB/L1 containers: $(docker ps --format '{{.Names}}' | grep -E 'oai-gnb|nv-cubb' | tr '\n' ' ')"
  echo "Open5GS: $(core_up && echo up || echo down) | xApp agent port $DAPP_AGENT_PORT: $(port_open && echo open || echo closed) | current run: $(cat $CUR 2>/dev/null || echo none) | GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  ;;
ue)
  R2=$OUT/${2:-$(cat $CUR 2>/dev/null || ls -t $OUT | grep -vE 'current_run|^stop_' | head -1)}; L=$R2/oai.log
  if oai_up; then docker logs $OAI_CONTAINER > $OUT/oai_live.log 2>&1; L=$OUT/oai_live.log; fi
  [ -f "$L" ] || { echo "no OAI log ($L)"; exit 1; }
  first() { grep -m1 -E "$1" $L | cut -c1-24; }; cnt() { grep -cE "$1" $L; }
  echo "== UE attach phases ($L)"; printf '   %-44s %-26s %s\n' phase first count
  printf '   %-44s %-26s %s\n' "NG Setup with the AMF" "$(first NGSetupResponse)" "$(cnt NGSetupResponse)"
  printf '   %-44s %-26s %s\n' "RA started (Msg1)" "$(first 'initiating RA procedure')" "$(cnt 'initiating RA procedure')"
  printf '   %-44s %-26s %s\n' "Msg3 / CBRA succeeded" "$(first 'CBRA procedure succeeded|Received Ack of')" "$(cnt 'CBRA procedure succeeded|Received Ack of')"
  printf '   %-44s %-26s %s\n' "RRCSetupComplete (RRC_CONNECTED)" "$(first RRCSetupComplete)" "$(cnt RRCSetupComplete)"
  printf '   %-44s %-26s %s\n' "RRCReconfigurationComplete" "$(first RRCReconfigurationComplete)" "$(cnt RRCReconfigurationComplete)"
  printf '   %-44s %-26s %s\n' "PDU session" "$(first 'PDU SESSION')" "$(cnt 'PDU SESSION')"
  for k in 'RA failed' 'Contention Resolution timer expired' 'out-of-sync' 'Assertion|Exiting' 'Release'; do n=$(cnt "$k"); [ "$n" -gt 0 ] && printf '   !! %-40s %4d  %s\n' "$k" "$n" "$(grep -m1 -E "$k" $L | cut -c1-120)"; done
  echo "-- last UE stats lines:"; grep -E 'UE RNTI|dlsch_rounds|ulsch_rounds|MAC:' $L | tail -4 | cut -c1-180 | sed 's/^/   /'
  [ -f $R2/phy_warn.log ] && echo "-- L1: Err.ind $(grep -c 'Send Err.ind' $R2/phy_warn.log), FATAL $(grep -c FATAL $R2/phy_warn.log), preamble detections $(grep -c 'Preambles detected' $R2/phy_warn.log)"
  ;;
test)
  m=${2:-26x}; rate=${3:-2}; sec=${4:-15}
  port_open || { echo "xApp agent port $DAPP_AGENT_PORT closed - is the L1 running ($0 status)?"; exit 1; }
  echo "== xApp smoke test: model $m, $rate req/s, $sec s, 4K JPEGs from models/img4k (closed loop)"
  docker exec $CONTAINER bash -c "cd $SDK/cuPHY-CP/dapp_hook/python && TS=\$(command -v taskset > /dev/null && echo 'taskset -c 65-71'); \$TS python3 ra_stub.py --rate $rate --duration $sec --policy fixed:$m --deadline-ms 1000 --max-inflight 1 --csv /tmp/ra_test.csv" 2>&1 | tail -12
  ;;
*) echo "usage: $0 start [name] | stop [--now] | status | ue [name] | test [model] [req/s] [seconds]"; exit 2 ;;
esac
