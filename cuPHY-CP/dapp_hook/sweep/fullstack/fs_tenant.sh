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
#   TENANT_BACKEND  nvjpeg backend of the server: hw (default, NVJPG engines) | gpu | hybrid
#   TENANT_DELAY    seconds between the start of this script and the first request (default 30: server load + traffic ramp-up)
#   TENANT_SECS     client duration (default traffic_s - 45)
# Outputs in prof/run61: tenant_<name>.log (this script), client_<name>.csv (per request, with unix times), server_<name>.csv
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$HERE/../../../.." && pwd)
O=${FS_OUT:-$R/prof/run61}
C=c_aerial_troy
CROOT=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook
ACT=$1; NAME=$2
if [ "$ACT" = "stop" ]; then
  docker exec -u root $C bash -lc "pkill -f infer_serve[r].py; true"
  exit 0
fi
TRAFFIC_S=$3
CAP=${TENANT_CAP:-100}; RATE=${TENANT_RATE:-10}; ARR=${TENANT_ARRIVALS:-jitter}; SEL=${TENANT_SEL:-0}
DELAY=${TENANT_DELAY:-30}; SECS=${TENANT_SECS:-$((TRAFFIC_S - 45))}
T0=$(date +%s)
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
