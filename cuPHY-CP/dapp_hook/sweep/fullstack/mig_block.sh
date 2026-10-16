#!/bin/bash
# mig_block.sh <MIG_4G_UUID> <MIG_3G_UUID> [steps]   (2026-10-16 night §3.3-3.6; MIG already on via mig_night.sh on)
# steps (default "engine alone cuphy iso"):
#   engine : build yolo26x_1280_b8_e2e_fp16_sm<N>.engine inside the 3g slice (N = the slice's SM count), unless present
#   alone  : 3g YOLO26x alone through dapp_sched (direct CUDA, no MPS) + dispatcher + client: 10 req/s then 8 req/s, 100 s each
#   cuphy  : 4g cuPHY alone, 6 configs X=10 P/A -> 11 P/A -> 9 P/A (fullstack_run MIG_UUID + MIG_SM=1: the lab yaml's mps_sm_* take
#            NVIDIA's MIG example values for the run and are restored + diffed by r0_ladder.sh); an L1 that does not come up is retried once
#   iso    : 4g cuPHY + 3g YOLO26x 10 req/s at the same time, X=10 Peak and X=11 Peak
# Every run is scored by xgrid_table.py into $FS_OUT/xgrid_MIG.csv (no budgets yet: B_rel / late ratio are filled in after §5).
set -u
S=$(cd "$(dirname "$0")" && pwd); R=$(cd "$S/../../../.." && pwd); O=${FS_OUT:-$R/prof/run65}; C=c_aerial_troy
MIG4G=$1; MIG3G=$2; STEPS=${3:-"engine alone cuphy iso"}; LOG=$O/block_MIG.log; CSV=$O/xgrid_MIG.csv
export TENANT_GLOB=${TENANT_GLOB:-uhd_*_q60.jpg}
say() { echo "$(date +%T) $*" | tee -a $LOG; }
cd $R; mkdir -p $O
SM3G=$(docker exec -e CUDA_VISIBLE_DEVICES=$MIG3G $C python3 -c "import ctypes;cu=ctypes.CDLL('libcuda.so.1');cu.cuInit(0);d=ctypes.c_int();cu.cuDeviceGet(ctypes.byref(d),0);v=ctypes.c_int();cu.cuDeviceGetAttribute(ctypes.byref(v),16,d);print(v.value)" 2>/dev/null)
SM4G=$(docker exec -e CUDA_VISIBLE_DEVICES=$MIG4G $C python3 -c "import ctypes;cu=ctypes.CDLL('libcuda.so.1');cu.cuInit(0);d=ctypes.c_int();cu.cuDeviceGet(ctypes.byref(d),0);v=ctypes.c_int();cu.cuDeviceGetAttribute(ctypes.byref(v),16,d);print(v.value)" 2>/dev/null)
say "== MIG block: 4g $MIG4G ($SM4G SM), 3g $MIG3G ($SM3G SM); steps: $STEPS"
[ -n "$SM3G" ] || { say "!! cannot query the 3g slice"; exit 1; }
ENG=yolo26x_1280_b8_e2e_fp16_sm${SM3G}.engine
ladder() {  # <prefix> <cname> <keys...>
  local prefix=$1 cname=$2; shift 2; local list=$O/list_${prefix}_$cname.txt; echo "$cname $*" > $list
  FS_OUT=$O bash $S/r0_ladder.sh $list $prefix >> $O/driver_$prefix.log 2>&1; local rc=$?
  grep -E '^== |!!|precheck:|disk:|L1 did not|did not finish' $O/driver_$prefix.log | tail -3 | sed 's/^/   /' | tee -a $LOG; return $rc
}
score() { python3 $S/xgrid_table.py eval --dir $O --name $1 --kind $2 --model "${3:-}" --cap "${4:-}" --csv $CSV 2>&1 | tail -1 | sed "s/^/   $1: /" | tee -a $LOG; }
for st in $STEPS; do case $st in
  engine)
    if docker exec $C test -s /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/$ENG; then say "engine $ENG present"; else
      say "building $ENG in the 3g slice"; docker exec -u root -e CUDA_VISIBLE_DEVICES=$MIG3G -e CUDA_MPS_PIPE_DIRECTORY=/var/mig_no_mps $C bash -c "cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models && /usr/src/tensorrt/bin/trtexec --onnx=yolo26x_1280_b8_e2e.onnx --fp16 --saveEngine=$ENG --skipInference 2>&1 | grep -E 'Engine built|error|Error' | tail -3" | tee -a $LOG
      docker exec $C test -s /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/$ENG && say "   built $ENG" || { say "!! engine build failed"; exit 2; }
    fi ;;
  alone)
    for rate in 10 8; do n=mig3g_26x_r$rate
      say "-- 3g alone: 26x @ $rate req/s (dapp_sched direct on the slice, no MPS)"
      FS_OUT=$O TENANT_MODE=dispatcher TENANT_MIG_UUID=$MIG3G TENANT_MODELS=26x TENANT_CLASSES=$SM3G TENANT_FORCE=26x TENANT_CAP=100 TENANT_ARRIVALS=poisson TENANT_RATE=$rate TENANT_DELAY=5 TENANT_SECS=95 bash $S/fs_tenant.sh start $n 100 > $O/tenant_$n.log 2>&1
      FS_OUT=$O bash $S/fs_tenant.sh stop $n >> $O/tenant_$n.log 2>&1
      grep -E 'offered|failed|did not|engine lines|served' $O/tenant_$n.log | tail -3 | sed 's/^/   /' | tee -a $LOG
      score $n tenant-only 26x 100
    done ;;
  cuphy)
    for cfg in 10:peak 10:average 11:peak 11:average 9:peak 9:average; do x=${cfg%%:*}; l=${cfg##*:}; cname=mig4g_${x}$( [ $l = peak ] && echo P || echo A )
      for attempt in 1 2; do
        MIG_UUID=$MIG4G MIG_SM=1 ladder MIG $cname pattern=tmp_ddddddsuuu_$l ncells=$x slots=200000 tags=311,223,503 keep_raw=2; rc=$?
        [ $rc -eq 1 ] && { say "!! precheck abort"; exit 3; }; [ $rc -eq 3 ] && { say "!! disk"; exit 3; }
        if grep -q "L1 did not come up" $O/driver_MIG.log && [ ! -s $O/phy_MIG_${cname}_${x}c.csv ]; then say "   $cname: L1 did not come up (attempt $attempt)"; [ $attempt = 2 ] && echo "MIG_${cname}_${x}c L1-crash-x2" >> $O/skipped_MIG.txt; continue; fi
        break
      done
      score MIG_${cname}_${x}c mig-cuphy
    done ;;
  iso)
    for x in 10 11; do cname=migiso_${x}P
      MIG_UUID=$MIG4G MIG_SM=1 TENANT_MIG_UUID=$MIG3G TENANT_MODELS=26x TENANT_CLASSES=$SM3G ladder MIG $cname pattern=tmp_ddddddsuuu_peak ncells=$x slots=200000 tags=311,223,503 keep_raw=2 tenant=dispatcher model=26x cap=100 arrivals=poisson rate=10 tdelay=25 tsecs=90; rc=$?
      [ $rc -eq 1 ] && { say "!! precheck abort"; exit 3; }; [ $rc -eq 3 ] && { say "!! disk"; exit 3; }
      score MIG_${cname}_${x}c mig-iso 26x 100
    done ;;
esac; done
say "== MIG block steps done"; python3 $S/xgrid_table.py table --csv $CSV | tee -a $LOG
