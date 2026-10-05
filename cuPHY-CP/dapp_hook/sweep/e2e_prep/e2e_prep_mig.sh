#!/bin/bash
# E2E scenario preparation, MIG side (static-MIG baseline). Run AFTER a person has switched the GPU to MIG (container, root):
#   echo quit | nvidia-cuda-mps-control; nvidia-smi -i 0 -mig 1; nvidia-smi mig -cgi 4g.48gb,3g.48gb -C; nvidia-smi -L
# Steps
#   1. UUID check against matrix.yaml
#   2. nvJPEG probe inside the 3g (tenant) and 4g instances
#   3. 1280 engines built inside the 3g instance (background, ~50 min) WHILE
#   4. the full-stack load ladder runs inside the 4g instance (cuPHY with NVIDIA's MIG caps; run2 starts its own MPS daemon on pipe /var)
#   5. tenant in 3g: closed loop alone for every model/batch + L4 spot checks next to the bench in 4g (MPS daemon on pipe /var/mig_ran)
#   6. open loop in 3g (batch 1, tiled batch 8), GH200 segment with the tenant in 3g (batch 1 and tiled)
#   7. analysis
# The tenant in 3g is isolated from the RAN load (c16_mig, 2026-09-28: x0.999-1.011), so it is measured alone and only spot-checked under load.
# Log: runs/e2e_prep_mig.log; prints E2E_PREP_MIG_DONE.
set -u
cd "$(dirname "$0")/.."
LOG=runs/e2e_prep_mig.log
PTS=runs/c16_points.yaml
C=c_aerial_troy
CROOT=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook
log() { echo "$(date +%F_%T) $*" | tee -a $LOG; }
only() { log "-- $1 --only $2"; python3 sweep.py --profile $1 --points-file $PTS --out runs/$1 --only "$2" 2>&1 | grep -E "^===|-> |^done|FAILED" | tee -a $LOG; }

RAN_UUID=$(nvidia-smi -L | grep -o "MIG 4g.48gb.*MIG-[0-9a-f-]*" | grep -o "MIG-[0-9a-f-]*$" | head -1)
TEN_UUID=$(nvidia-smi -L | grep -o "MIG 3g.48gb.*MIG-[0-9a-f-]*" | grep -o "MIG-[0-9a-f-]*$" | head -1)
log "MIG instances: ran(4g)=$RAN_UUID tenant(3g)=$TEN_UUID"
if [ -z "$RAN_UUID" ] || [ -z "$TEN_UUID" ]; then log "MIG instances not found (is MIG on?)"; exit 1; fi
for u in $RAN_UUID $TEN_UUID; do grep -q "$u" matrix.yaml || { log "UUID $u is not in matrix.yaml: update the mig: blocks of c16_mig_1280 / c16_openloop_mig_1280(_b8) and restart"; exit 1; }; done

log "== nvJPEG probe inside the 3g instance, then the 4g instance"
docker exec -u root -e CUDA_VISIBLE_DEVICES=$TEN_UUID -e CUDA_MPS_PIPE_DIRECTORY=/var/mig_tenant_no_mps $C bash -lc "cd $CROOT/models/img4k && rm -f $CROOT/sweep/results/e2e_prep/nvjpeg_probe_mig.csv; for be in hw gpu; do ../../tools/nvjpeg_probe uhd_0_q85.jpg --backend \$be --iters 100 --resize 1280 --tag mig3g --csv $CROOT/sweep/results/e2e_prep/nvjpeg_probe_mig.csv; done" 2>&1 | grep -E "device:|hardware decoder|Create|Supported|total" | tee -a $LOG
docker exec -u root -e CUDA_VISIBLE_DEVICES=$RAN_UUID -e CUDA_MPS_PIPE_DIRECTORY=/var/mig_tenant_no_mps $C bash -lc "cd $CROOT/models/img4k && ../../tools/nvjpeg_probe uhd_0_q85.jpg --backend hw --iters 50 --resize 1280 --tag mig4g --csv $CROOT/sweep/results/e2e_prep/nvjpeg_probe_mig.csv" 2>&1 | grep -E "device:|hardware decoder|Create|total" | tee -a $LOG

log "== engines in the 3g instance (background)"
docker exec -u root -e MODELS="yolo26n_1280 yolo26s_1280 yolo26x_1280" $C bash $CROOT/models/build_engines_mig.sh $TEN_UUID mig3g > runs/e2e_prep_mig_engines.log 2>&1 &
ENG=$!

log "== full-stack load ladder inside the 4g instance (engine builds running in 3g at the same time)"
MIG_SM=1 MIG_UUID=$RAN_UUID bash fullstack/ladder_runs.sh mig "${MIG_LADDER:-59c 59e 65c 65a}" > runs/ladder_mig.out 2>&1
grep -E "^== |L1 did not|testMAC did not|LADDER_DONE|Finished" runs/ladder_mig.out | tee -a $LOG
names=""; for it in ${MIG_LADDER:-59c 59e 65c 65a}; do p=${it%%:*}; n=16; [[ $it == *:* ]] && n=${it##*:}; names="$names mig_${p}_${n}c"; done
python3 fullstack/ladder_report.py $names > results/e2e_prep/ladder_mig.txt 2>&1
cut -c1-200 results/e2e_prep/ladder_mig.txt | tee -a $LOG

log "waiting for the engine builds"
wait $ENG; cat runs/e2e_prep_mig_engines.log | tee -a $LOG

# the bench-based runs use an MPS daemon bound to the 4g instance on pipe /var/mig_ran; run2 left one on pipe /var
docker exec -u root $C bash -lc "echo quit | CUDA_MPS_PIPE_DIRECTORY=/var timeout 10 nvidia-cuda-mps-control; mkdir -p /var/mig_ran /var/mig_tenant_no_mps; sleep 2;
  CUDA_VISIBLE_DEVICES=$RAN_UUID CUDA_MPS_PIPE_DIRECTORY=/var/mig_ran CUDA_MPS_LOG_DIRECTORY=/var/mig_ran nvidia-cuda-mps-control -d; sleep 2;
  echo get_default_active_thread_percentage | CUDA_MPS_PIPE_DIRECTORY=/var/mig_ran timeout 5 nvidia-cuda-mps-control; echo mps_mig_ran_ok" 2>&1 | tail -2 | tee -a $LOG

log "== c16_mig_1280: tenant alone (6) + L4 spot checks"
only c16_mig_1280 tenant_alone__
only c16_mig_1280 c16_L4__none0
only c16_mig_1280 c16_L4__yolo26x1280
log "== open loop in 3g: alone (b1 and tiled b8) + L4 spot check"
only c16_openloop_mig_1280 tenant_alone__
only c16_openloop_mig_1280_b8 tenant_alone__
only c16_openloop_mig_1280_b8 c16_L4__olt26x1280
log "== GH200 segment, tenant in 3g (batch 1, then tiled)"
MIG_TENANT_UUID=$TEN_UUID MIG_RAN_PIPE=/var/mig_ran SM_LIST=2,5,2,49,48,8 CAPS=100 ENGINE_TAG=mig3g bash ../e2e/gh200_segment.sh results/e2e_prep/gh200_segment_mig 2>&1 | grep -E "==|offered|GH200_SEGMENT_DONE|failed|patterns" | tee -a $LOG
MIG_TENANT_UUID=$TEN_UUID MIG_RAN_PIPE=/var/mig_ran SM_LIST=2,5,2,49,48,8 CAPS=100 ENGINE_TAG=mig3g BATCH=8 RATES_X="2 5 10 15" RATES_S="10 20" RATES_N="" bash ../e2e/gh200_segment.sh results/e2e_prep/gh200_segment_mig_tiled 2>&1 | grep -E "==|offered|GH200_SEGMENT_DONE|failed|patterns" | tee -a $LOG

log "== analysis"
docker exec -u root $C bash -lc "cd $CROOT/sweep && python3 tenant_latency.py --runs runs/c16_mig_1280 --out results/c16_mig_1280 --slot-ms 0.5 2>&1 | tail -1;
  python3 openloop_report.py --runs runs/c16_openloop_1280 --out results/c16_openloop_1280 --extra 'MIG 3g=runs/c16_openloop_mig_1280' 2>&1 | tail -1;
  python3 openloop_report.py --runs runs/c16_openloop_mig_1280_b8 --out results/c16_openloop_mig_1280_b8 2>&1 | tail -1" | tee -a $LOG
echo "E2E_PREP_MIG_DONE $(date +%T)" | tee -a $LOG
