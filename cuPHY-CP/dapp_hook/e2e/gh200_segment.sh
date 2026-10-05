#!/bin/bash
# GH200 segment of the E2E path with the real pipeline (TCP receive -> NVJPG decode -> letterbox -> TensorRT -> reply),
# measured over loopback (client on the host, server in the container), with and without the 16-cell cuPHY bench load,
# at tenant caps 100 and 30, as an open-loop stream at several request rates. No UE needed; the air part is added later.
#
#   bash gh200_segment.sh [out_dir]      (default: ../sweep/results/e2e_prep/gh200_segment)
#
# Loads: none | c16_L1 (lowest) | c16_L4 (highest), bench.yaml taken from an earlier sweep run of the same point.
# Engines: 26x1280 (default, sel 0), 26s1280 (1), 26n1280 (2), 26n640 (3), all batch 1, built for the server's cap.
# MIG variant (tenant in the 3g instance, bench in the 4g instance behind its own MPS daemon):
#   MIG_TENANT_UUID=MIG-... MIG_RAN_PIPE=/var/mig_ran SM_LIST=2,5,2,49,48,8 CAPS=100 ENGINE_TAG=mig3g bash gh200_segment.sh <out_dir>
set -u
CAPS=${CAPS:-"100 30"}
SM_LIST=${SM_LIST:-2,10,2,102,100,16}
MIG_TENANT_UUID=${MIG_TENANT_UUID:-}
MIG_RAN_PIPE=${MIG_RAN_PIPE:-/var}
ENGINE_TAG=${ENGINE_TAG:-}          # empty = atp<cap>; mig3g = the engines built inside the 3g instance
# Arrivals default to Poisson: fixed intervals at 5/10/20/40 req/s are multiples of the bench's 5 ms pattern and would phase-lock.
BATCH=${BATCH:-1}                   # 8 = tiled requests (one batch-8 inference: 6 native 1280 tiles + global views); then set RATES_N="" (no 640 pass)
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/../sweep/results/e2e_prep/gh200_segment}
mkdir -p "$OUT"
C=c_aerial_troy
CROOT=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook
BENCH=/opt/nvidia/cuBB/build.aarch64/testBenches/cubb_gpu_test_bench/cubb_gpu_test_bench
RATES_X=${RATES_X:-"5 10 20 40"}
RATES_N=${RATES_N-"10 40"}          # 26n-640 pass; RATES_N="" skips it
RATES_S=${RATES_S-""}               # optional 26s-1280 pass (engine index 1)
SECS=${SECS:-30}
LOG=$OUT/gh200_segment.log
log() { echo "$(date +%T) $*" | tee -a "$LOG"; }

bench_yaml() {   # host path of a bench.yaml for the point
  ls -d $HERE/../sweep/runs/c16_capgrid_1280/$1__none0/bench.yaml $HERE/../sweep/runs/c16_capgrid_26s/$1__none0/bench.yaml 2>/dev/null | head -1
}
to_c() { echo "$1" | sed "s#.*/cuPHY-CP/dapp_hook#$CROOT#"; }

docker exec -u root $C bash -lc "pkill -f infer_server.py; pkill -9 -x cubb_gpu_test_b; true"
for load in none c16_L1 c16_L4; do
  for cap in $CAPS; do
    tag=${load}_cap${cap}; [ -n "$ENGINE_TAG" ] && tag=${load}_${ENGINE_TAG}
    etag=${ENGINE_TAG:-atp${cap}}
    log "== $tag"
    if [ -n "$MIG_TENANT_UUID" ]; then
      srvenv="-e CUDA_VISIBLE_DEVICES=$MIG_TENANT_UUID -e CUDA_MPS_PIPE_DIRECTORY=/var/mig_tenant_no_mps"      # the tenant owns its instance: no MPS, no cap
    else
      srvenv="-e CUDA_MPS_PIPE_DIRECTORY=/var"; [ $cap -lt 100 ] && srvenv="$srvenv -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$cap"
    fi
    # server
    docker exec -d -u root $srvenv $C bash -lc "cd $CROOT/e2e && taskset -c 52-55 python3 infer_server.py \
      --engine 26x1280=$CROOT/models/yolo26x_1280_b${BATCH}_fp16_${etag}.engine --engine 26s1280=$CROOT/models/yolo26s_1280_b${BATCH}_fp16_${etag}.engine \
      --engine 26n1280=$CROOT/models/yolo26n_1280_b${BATCH}_fp16_${etag}.engine --engine 26n640=$CROOT/models/yolo26n_b${BATCH}_fp16_${etag}.engine \
      --port 9000 --log /tmp/server_${tag}.csv > /tmp/server_${tag}.out 2>&1"
    t0=$(date +%s)
    until docker exec $C grep -q "listening\|Traceback\|unavailable" /tmp/server_${tag}.out 2>/dev/null; do sleep 2; [ $(( $(date +%s) - t0 )) -gt 300 ] && break; done
    if ! docker exec $C grep -q listening /tmp/server_${tag}.out; then log "server failed:"; docker exec $C tail -5 /tmp/server_${tag}.out | tee -a "$LOG"; continue; fi
    # bench (background, long enough for all client stages)
    if [ $load != none ]; then
      by=$(bench_yaml $load); [ -z "$by" ] && { log "no bench.yaml for $load"; continue; }
      nst=$(( $(echo $RATES_X | wc -w) + $(echo $RATES_N | wc -w) + $(echo $RATES_S | wc -w) ))
      dur=$(( nst * (SECS + 8) + 60 )); iters=$(( dur * 200 ))
      log "bench $load for ${dur}s ($by)"
      docker exec -d -u root -e CUDA_MPS_PIPE_DIRECTORY=$MIG_RAN_PIPE -e CUDA_DEVICE_MAX_CONNECTIONS=8 $C bash -lc "cd /tmp && $BENCH -i $(to_c $by) -r $iters -w 850 -u 5 -d 0 -m 0 -c 44 --U --D --P --Q --X --Z --G --g --M $SM_LIST > /tmp/bench_${tag}.out 2>&1"
      sleep 25     # bench start-up (TV load) before the stream begins
    fi
    for r in $RATES_X; do
      log "client 26x1280 rate $r"
      taskset -c 56-59 python3 $HERE/e2e_client.py --server 127.0.0.1:9000 --images $HERE/../models/img4k --rate $r --seconds $SECS --engine-sel 0 --arrivals ${ARRIVALS:-poisson} --log $OUT/client_${tag}_26x1280_r${r}.csv 2>&1 | tail -1 | tee -a "$LOG"
      sleep 3
    done
    for r in $RATES_S; do
      log "client 26s1280 rate $r"
      taskset -c 56-59 python3 $HERE/e2e_client.py --server 127.0.0.1:9000 --images $HERE/../models/img4k --rate $r --seconds $SECS --engine-sel 1 --arrivals ${ARRIVALS:-poisson} --log $OUT/client_${tag}_26s1280_r${r}.csv 2>&1 | tail -1 | tee -a "$LOG"
      sleep 3
    done
    for r in $RATES_N; do
      log "client 26n640 rate $r"
      taskset -c 56-59 python3 $HERE/e2e_client.py --server 127.0.0.1:9000 --images $HERE/../models/img4k --rate $r --seconds $SECS --engine-sel 3 --arrivals ${ARRIVALS:-poisson} --log $OUT/client_${tag}_26n640_r${r}.csv 2>&1 | tail -1 | tee -a "$LOG"
      sleep 3
    done
    docker exec -u root $C bash -lc "pkill -f infer_server.py; true"
    docker cp $C:/tmp/server_${tag}.csv $OUT/server_${tag}.csv 2>/dev/null
    if [ $load != none ]; then
      log "waiting for the bench to end"
      docker exec -u root $C bash -lc "while pgrep -x cubb_gpu_test_b >/dev/null; do sleep 2; done"
      docker cp $C:/tmp/bench_${tag}.out $OUT/bench_${tag}.out 2>/dev/null
      grep -c "Slot pattern #" $OUT/bench_${tag}.out | sed 's/^/bench patterns: /' | tee -a "$LOG"
    fi
  done
done
echo "GH200_SEGMENT_DONE $(date +%T)" | tee -a "$LOG"
