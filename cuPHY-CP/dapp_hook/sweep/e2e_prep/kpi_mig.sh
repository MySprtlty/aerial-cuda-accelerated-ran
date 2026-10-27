#!/bin/bash
# KPI experiment, baseline arm (static MIG): cuphycontroller in the 4g.48gb instance with NVIDIA's MIG caps, the e2e
# inference server in the 3g.48gb instance (direct CUDA, no MPS, no cap). Run AFTER a person has switched the GPU to MIG:
#   echo quit | nvidia-cuda-mps-control; nvidia-smi -i 0 -mig 1; nvidia-smi mig -cgi 4g.48gb,3g.48gb -C
# Steps: nvJPEG probe in 3g -> batch-8 1280 engines built in 3g (background) while the 4g runs without tenant ->
#        tiled requests with 26x in 3g (the baseline), then 26n and 26s in 3g (like-for-like arms)
# Results: prof/run61/*_mig_<tenant>_<pattern>_16c.*; log runs/kpi_mig.log; prints KPI_MIG_DONE.
set -u
cd "$(dirname "$0")/.."
LOG=runs/kpi_mig.log
C=c_aerial_troy
CROOT=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook
export LADDER_SLOTS=${LADDER_SLOTS:-360000}
log() { echo "$(date +%F_%T) $*" | tee -a $LOG; }
RAN_UUID=$(nvidia-smi -L | grep -o "MIG 4g.48gb.*MIG-[0-9a-f-]*" | grep -o "MIG-[0-9a-f-]*$" | head -1)
TEN_UUID=$(nvidia-smi -L | grep -o "MIG 3g.48gb.*MIG-[0-9a-f-]*" | grep -o "MIG-[0-9a-f-]*$" | head -1)
log "MIG instances: ran(4g)=$RAN_UUID tenant(3g)=$TEN_UUID"
if [ -z "$RAN_UUID" ] || [ -z "$TEN_UUID" ]; then log "MIG instances not found (is MIG on?)"; exit 1; fi

log "== nvJPEG probe inside the 3g instance"
docker exec -u root -e CUDA_VISIBLE_DEVICES=$TEN_UUID -e CUDA_MPS_PIPE_DIRECTORY=/var/mig_tenant_no_mps $C bash -lc "cd $CROOT/models/img4k && for be in hw gpu; do ../../tools/nvjpeg_probe uhd_0_q85.jpg --backend \$be --iters 100 --resize 1280 --tag mig3g --csv $CROOT/sweep/results/e2e_prep/nvjpeg_probe_mig.csv; done" 2>&1 | grep -E "device:|hardware decoder|Create|Supported|total" | tee -a $LOG
BACKEND=hw
grep -q "nvjpegDecoderCreate(hw): SUCCESS" $LOG || { BACKEND=gpu; log "hardware JPEG decoder NOT available in the 3g instance -> the baseline server uses the gpu backend"; }
export TENANT_BACKEND=$BACKEND

log "== batch-8 engines in the 3g instance (background)"
docker exec -u root -e MODELS="yolo26x_1280 yolo26n_1280 yolo26s_1280" -e BATCHES="8" $C bash $CROOT/models/build_engines_mig.sh $TEN_UUID mig3g > runs/kpi_mig_engines.log 2>&1 &
ENG=$!

log "== 4g without tenant (engine builds running in 3g)"
env -u TENANT_SPEC MIG_SM=1 MIG_UUID=$RAN_UUID bash fullstack/ladder_runs.sh mig_none "65a 59c" > runs/kpi_mig_none.out 2>&1
grep -E "^== ladder|did not|failed" runs/kpi_mig_none.out | tee -a $LOG
log "waiting for the engine builds"; wait $ENG; cat runs/kpi_mig_engines.log | tee -a $LOG

run() {   # run <tag> <model> "<patterns>"
  log "-- $1 ($2 tiled in 3g) $3"
  TENANT_SPEC="$2=yolo$2_1280_b8_fp16_mig3g.engine" TENANT_MIG_UUID=$TEN_UUID MIG_SM=1 MIG_UUID=$RAN_UUID bash fullstack/ladder_runs.sh "$1" "$3" > runs/kpi_$1.out 2>&1
  grep -E "^== ladder|did not|failed|offered" runs/kpi_$1.out | cut -c1-200 | tee -a $LOG
}
run mig_x 26x "65a 59c"
run mig_n 26n "65a"
run mig_s 26s "65a"
python3 fullstack/fs_report.py mig_none_65a_16c mig_x_65a_16c mig_n_65a_16c mig_s_65a_16c mig_none_59c_16c mig_x_59c_16c --csv results/e2e_prep/kpi_mig.csv > results/e2e_prep/kpi_mig.txt 2>&1
echo "KPI_MIG_DONE $(date +%T)" | tee -a $LOG
