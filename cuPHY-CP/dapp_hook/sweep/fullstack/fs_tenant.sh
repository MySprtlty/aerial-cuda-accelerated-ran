#!/bin/bash
# The AI tenant of a full-stack run: the e2e inference server (container) and the open-loop client (host), started by
# fullstack_run.sh once the L1 is up and testMAC is running, so the tenant's requests overlap the 16-cell traffic.
#   fs_tenant.sh start <name> <traffic_s>     server up -> wait TENANT_DELAY s -> client for TENANT_SECS s -> copy server log
#   fs_tenant.sh stop  <name>                 kill the server
# Environment (set by the caller of fullstack_run.sh)
#   TENANT_SPEC     "name=engine_file [name=engine_file ...]" under dapp_hook/models; the first is the one the client uses
#                   (a batch-8 engine = tiled request: 6 native 1280 crops + global views)
#   TENANT_CAP      MPS active-thread percentage of the server (default 100); the engines must be built for it
#   TENANT_MIG_UUID tenant MIG instance (static-MIG baseline): direct CUDA on that device, no MPS, no cap
#   TENANT_RATE     requests per second (default 10), TENANT_ARRIVALS (default jitter: periodic +-5 ms), TENANT_SEL (engine index, default 0)
#   TENANT_POLICY   manual (default) | timetable (TENANT_TIMETABLE csv, TENANT_TT_PERIOD s, TENANT_TT_ORIGIN start|sfn0) |
#                   table (TENANT_TABLE SafeCap table yaml, estimator/table_policy.py live classifier); TENANT_GLOB image glob (uhd_*_q60.jpg)
#   TENANT_BACKEND  nvjpeg backend of the server: hw (default, NVJPG engines) | gpu | hybrid
#   TENANT_DELAY    seconds between the start of this script and the first request (default 30: server load + traffic ramp-up)
#   TENANT_SECS     client duration (default traffic_s - 45)
#   TENANT_MODELS   dispatcher mode: models to load (default "26n 26s 26x"), TENANT_CLASSES SM classes (default "56 80 132"); missing engines skipped;
#                   TENANT_MIG_UUID in dispatcher mode runs dapp_sched directly on that MIG instance (no MPS, uncapped)
#   TENANT_MODE     server (default: infer_server.py as above) | dispatcher: dapp_sched (9 engines, 3 models x SM 56/80/132,
#                   --ctrl) in the container + e2e/dispatcher.py (host, manual policy: cap TENANT_CAP, gate TENANT_GATE=40,
#                   --force-model TENANT_FORCE) + e2e_client.py (TENANT_ARRIVALS default closed = closed loop, one in flight).
#                   Engines: models/yolo26{n,s,x}_1280_b8${TENANT_ENGINE_SFX:-_e2e}_fp16_sm{56,80,132}.engine.
#                   TENANT_NOCLIENT=1 starts dapp_sched + dispatcher only (idle tenant). Cores: dapp_sched 44-47,
#                   dispatcher 48-55, client 56-63 (L1 4-14, testMAC 15-18, RU 19-42, export 43 stay untouched).
# Outputs in prof/run61: tenant_<name>.log (this script), client_<name>.csv (per request, with unix times), server_<name>.csv
#   (dispatcher mode: disp_<name>.csv per request with every hand-over stamp, dapp_sched_<name>.log)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$HERE/../../../.." && pwd)
O=${FS_OUT:-$R/prof/run61}
C=c_aerial_troy
CROOT=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook
ACT=$1; NAME=$2
if [ "$ACT" = "stop" ]; then
  docker exec -u root $C bash -lc "pkill -f infer_serve[r].py; true"
  for p in $(pgrep -f "dispatcher.py --sched-sock"); do kill -INT $p 2>/dev/null; done
  sleep 1
  docker exec -u root $C bash -lc 'for p in $(pgrep -x dapp_sched); do kill -INT $p 2>/dev/null; done; sleep 2; for p in $(pgrep -x dapp_sched); do kill -KILL $p 2>/dev/null; done; rm -f /dev/shm/dapp_sched.sock; true'
  rm -f /dev/shm/dapq_*.jpg
  exit 0
fi
TRAFFIC_S=$3
CAP=${TENANT_CAP:-100}; RATE=${TENANT_RATE:-10}; ARR=${TENANT_ARRIVALS:-jitter}; SEL=${TENANT_SEL:-0}
DELAY=${TENANT_DELAY:-30}; SECS=${TENANT_SECS:-$((TRAFFIC_S - 45))}
T0=$(date +%s)
if [ "${TENANT_MODE:-server}" = "dispatcher" ]; then
  GATE=${TENANT_GATE:-40}; SFX=${TENANT_ENGINE_SFX:-_e2e}; ARR=${TENANT_ARRIVALS:-closed}; SOCK=/dev/shm/dapp_sched.sock; CTRL=/aerial_dapp_ctrl
  echo "$(date +%T) tenant $NAME: dispatcher mode, force '${TENANT_FORCE:-}' cap $CAP gate $GATE arrivals $ARR engines *$SFX*, delay $DELAY secs $SECS noclient ${TENANT_NOCLIENT:-0}"
  docker exec -u root $C bash -lc 'for p in $(pgrep -x dapp_sched); do kill -KILL $p 2>/dev/null; done; rm -f /dev/shm/dapp_sched.sock; true'
  for p in $(pgrep -f "dispatcher.py --sched-sock"); do kill -KILL $p 2>/dev/null; done
  # TENANT_MODELS / TENANT_CLASSES (2026-10-16 night): the engine grid dapp_sched loads; engines that do not exist are skipped
  args=""; CLASSES=""; HROOT=$R/cuPHY-CP/dapp_hook
  for m in ${TENANT_MODELS:-26n 26s 26x}; do for sm in ${TENANT_CLASSES:-56 80 132}; do
    f=yolo${m}_1280_b8${SFX}_fp16_sm${sm}.engine
    if [ -s $HROOT/models/$f ]; then args="$args --engine $m:$sm=$CROOT/models/$f"; CLASSES="$CLASSES,$sm"; else echo "$(date +%T) tenant $NAME: no engine $f (skipped)"; fi
  done; done
  CLASSES=$(echo ${CLASSES#,} | tr ',' '\n' | sort -n | uniq | tr '\n' ',' | sed 's/,$//')
  if [ -n "${TENANT_EXTRA_CLASS:-}" ]; then     # e.g. "32" or "32,104" with TENANT_EXTRA_MODELS="26x": extra SM-class engines (cap 24 / cap 80)
    for xc in ${TENANT_EXTRA_CLASS//,/ }; do
      CLASSES=$xc,$CLASSES
      for m in ${TENANT_EXTRA_MODELS:-26x}; do args="$args --engine $m:$xc=$CROOT/models/yolo${m}_1280_b8${SFX}_fp16_sm${xc}.engine"; done
    done
    CLASSES=$(echo $CLASSES | tr ',' '\n' | sort -n | uniq | tr '\n' ',' | sed 's/,$//')
  fi
  SCHED_ENV="-e CUDA_MPS_PIPE_DIRECTORY=/var"; [ -n "${TENANT_MIG_UUID:-}" ] && SCHED_ENV="-e CUDA_VISIBLE_DEVICES=$TENANT_MIG_UUID -e CUDA_MPS_PIPE_DIRECTORY=/var/mig_tenant_no_mps"   # MIG slice: direct CUDA, no MPS (dapp_sched falls back to an uncapped context)
  docker exec -d -u root $SCHED_ENV $C bash -lc "taskset -c 44-47 /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/dapp_hook/dapp_sched --classes $CLASSES $args --image $CROOT/models/img4k/uhd_0_q85.jpg -s $SOCK -n /aerial_dapp_ring_unused --ctrl $CTRL --ctrl-max-age-ms 50 --cells 16 > /tmp/dapp_sched_$NAME.out 2>&1"
  for i in $(seq 1 180); do [ -S $SOCK ] && break; sleep 1; done
  [ -S $SOCK ] || { echo "$(date +%T) dapp_sched did not come up"; docker exec $C tail -5 /tmp/dapp_sched_$NAME.out; exit 1; }
  echo "$(date +%T) dapp_sched up after $(( $(date +%s) - T0 )) s ($(docker exec $C grep -c 'yolo: model' /tmp/dapp_sched_$NAME.out) engine lines)"
  force=""; [ -n "${TENANT_FORCE:-}" ] && force="--force-model $TENANT_FORCE"
  # TENANT_POLICY=timetable: TENANT_TIMETABLE csv (t_s,cap_26x,cap_26s,cap_26n,gate[,model]), TENANT_TT_PERIOD (s, e.g. 10.24),
  # TENANT_TT_ORIGIN (start | sfn0); default: manual caps TENANT_CAP for every model
  if [ "${TENANT_POLICY:-manual}" = "timetable" ]; then
    policy="--policy-source timetable --timetable ${TENANT_TIMETABLE} --timetable-period ${TENANT_TT_PERIOD:-0} --timetable-origin ${TENANT_TT_ORIGIN:-start}"
  elif [ "${TENANT_POLICY:-manual}" = "table" ]; then
    # TENANT_POLICY=table (H2): estimator/table_policy.py classifies the live hook ring and writes the SafeCap table row
    # (TENANT_TABLE yaml) into the control block; the dispatcher reads it (live) and picks inside the allowed set (rule table)
    for p in $(pgrep -f "table_policy.py.*--ring"); do kill -KILL $p 2>/dev/null; done     # live instances only (an offline --replay has no --ring)
    taskset -c 64-65 python3 $R/cuPHY-CP/dapp_hook/estimator/table_policy.py --table ${TENANT_TABLE} --ring ${TENANT_RING:-/aerial_dapp_ring} --ctrl $CTRL \
      --log $O/table_policy_$NAME.csv > $O/table_policy_$NAME.log 2>&1 &
    sleep 1; grep -q 'table_policy live' $O/table_policy_$NAME.log || { echo "$(date +%T) table_policy failed"; tail -3 $O/table_policy_$NAME.log; exit 1; }
    policy="--policy-source live --rule table"
  else
    policy="--policy-source manual --manual 26x=$CAP/$GATE,26s=$CAP/$GATE,26n=$CAP/$GATE"
    case "${TENANT_FORCE:-}" in ""|26x|26s|26n) ;; *) policy="--policy-source manual --manual ${TENANT_FORCE}=$CAP/$GATE,26x=$CAP/$GATE,26s=$CAP/$GATE,26n=$CAP/$GATE" ;; esac   # forced 26m/26l: its own control-block entry (ABI 2 holds 4)
  fi
  taskset -c 48-55 python3 $R/cuPHY-CP/dapp_hook/e2e/dispatcher.py --sched-sock $SOCK --port 9007 --log $O/disp_$NAME.csv --ctrl $CTRL \
    $policy $force > $O/dispatcher_$NAME.log 2>&1 &
  for i in $(seq 1 60); do grep -q listening $O/dispatcher_$NAME.log 2>/dev/null && break; sleep 0.5; done
  grep -q listening $O/dispatcher_$NAME.log || { echo "$(date +%T) dispatcher failed"; tail -3 $O/dispatcher_$NAME.log; exit 1; }
  echo "$(date +%T) dispatcher ready after $(( $(date +%s) - T0 )) s"
  if [ "${TENANT_NOCLIENT:-0}" = "1" ]; then echo "$(date +%T) no client (idle tenant)"; exit 0; fi
  while [ $(( $(date +%s) - T0 )) -lt $DELAY ]; do sleep 1; done
  echo "$(date +%T) client start"
  taskset -c 56-63 python3 $R/cuPHY-CP/dapp_hook/e2e/e2e_client.py --server 127.0.0.1:9007 --images $R/cuPHY-CP/dapp_hook/models/img4k --glob "${TENANT_GLOB:-uhd_*_q85.jpg}" --rate $RATE --seconds $SECS \
    --arrivals $ARR --warm 3 --log $O/client_$NAME.csv 2>&1 | tail -2
  echo "$(date +%T) client done"
  docker cp $C:/tmp/dapp_sched_$NAME.out $O/dapp_sched_$NAME.log >/dev/null 2>&1
  exit 0
fi
eng=""; for e in $TENANT_SPEC; do eng="$eng --engine ${e%%=*}=$CROOT/models/${e#*=}"; done
if [ -n "${TENANT_MIG_UUID:-}" ]; then
  envs="-e CUDA_VISIBLE_DEVICES=$TENANT_MIG_UUID -e CUDA_MPS_PIPE_DIRECTORY=/var/mig_tenant_no_mps"
else
  envs="-e CUDA_MPS_PIPE_DIRECTORY=/var"; [ "$CAP" -lt 100 ] && envs="$envs -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CAP"
fi
echo "$(date +%T) tenant $NAME: spec '$TENANT_SPEC' cap $CAP mig '${TENANT_MIG_UUID:-}' rate $RATE ($ARR) sel $SEL delay $DELAY secs $SECS"
docker exec -u root $C bash -lc "pkill -f infer_serve[r].py; rm -f /tmp/server_$NAME.csv /tmp/server_$NAME.out; true"
docker exec -d -u root $envs $C bash -lc "cd $CROOT/e2e && taskset -c 52-55 python3 infer_server.py $eng --port 9000 --backend ${TENANT_BACKEND:-hw} --log /tmp/server_$NAME.csv > /tmp/server_$NAME.out 2>&1"
until docker exec $C grep -q "listening\|Traceback\|unavailable\|Error" /tmp/server_$NAME.out 2>/dev/null; do sleep 1; [ $(( $(date +%s) - T0 )) -gt 120 ] && break; done
docker exec $C grep -v "TRT\|^$" /tmp/server_$NAME.out | tail -8
if ! docker exec $C grep -q listening /tmp/server_$NAME.out; then echo "$(date +%T) server failed to start"; exit 1; fi
echo "$(date +%T) server ready after $(( $(date +%s) - T0 )) s"
while [ $(( $(date +%s) - T0 )) -lt $DELAY ]; do sleep 1; done
echo "$(date +%T) client start"
taskset -c 56-59 python3 $R/cuPHY-CP/dapp_hook/e2e/e2e_client.py --server 127.0.0.1:9000 --images $R/cuPHY-CP/dapp_hook/models/img4k --rate $RATE --seconds $SECS \
  --engine-sel $SEL --arrivals $ARR --log $O/client_$NAME.csv 2>&1 | tail -2
echo "$(date +%T) client done"
docker cp $C:/tmp/server_$NAME.csv $O/server_$NAME.csv >/dev/null 2>&1
# The server is NOT killed here: tearing an MPS client down while the traffic is still running stalled the GPU long enough
# for an order-kernel timeout on all 16 cells in one UL slot (sh_n30_65a_16c, 0.12 s after the last reply). fullstack_run.sh
# calls "stop" once testMAC has finished.
