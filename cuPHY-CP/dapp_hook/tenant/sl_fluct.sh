#!/bin/bash
# sl_fluct.sh <name> <ours|req|prof|green|base> [slots=480000] [rate=20]   (2026-11-13, branch feature/slot_level)
# Slot-level SM switching vs MIG under a fluctuating multi-cell load: NCELLS (default 10) RU-emulator cells, DDDDDDSUUU, pattern tmp_fluct =
# 4 frames with 1 | 2 | 4 | 3 PUSCH UEs per cell and U slot (Peak = 6; tools/slot_level_fluct_pattern.py), every frame replayed
# BLOCK_REPEAT x 10 ms (default 300 = 3 s per load level, 12-s cycle) by the block-repeat option of the f1002 RU emulator / testMAC.
# The same FHD batch-3 26x requests (Poisson, <rate> req/s, same seed) go to TCP 9810 in every arm, from WARMUP s after the traffic starts:
#   ours : our L1 (green contexts, PUSCH/PDSCH profiles, in-process sizer) + in-process tenant with ONE 132-SM engine whose SM class is chosen
#          per graph piece at the slot boundaries (DAPP_AGENT_PIECE=1). No model selection, no xApp, no QoS refusal.
#   req  : the same L1 and engine with today's request-level rule (one class per request, ladder 24..80)   [comparison arm]
#   prof : the same L1 (profiles + sizer) without the tenant thread and without requests   [the L1's own reference]
#   green: the same L1 binary with green contexts ON and the yaml's own SM partitions, nothing of ours on top (no profiles, no sizer, no tenant
#          thread, no requests)   [2026-11-13: is a behaviour caused by green contexts themselves or by our profiles?]
#   base : MIG. L1 = the same f1002 binary with every dApp feature off (no hook, no green contexts, no profiles, no tenant; MPS inside the 4g
#          instance, NVIDIA's MIG example mps_sm_* values) + the naive YOLO server (stand-alone tenant binary, 60-SM engine, FREERUN / FIXED)
#          on the 3g instance. Needs `sweep/fullstack/mig_night.sh on` first (MIG_4G / MIG_3G are read from nvidia-smi -L).
# Outputs in FS_OUT (default prof/tenant/slot_level/fluct): console_/summary_/ra_<name>.*, dapp_tenant_<name>.{log,csv}, pieces_<name>.csv.
# DRY=1: apply + snapshot the configuration only (fullstack_run.sh DRY), no run. valid_perf_patterns.sh is restored on exit.
set -u
R=/home/troy/build/aerial-cuda-accelerated-ran; D=$R/cuPHY-CP/dapp_hook; T=$D/tenant; F=$D/sweep/fullstack; V=$R/testBenches/phase4_test_scripts/valid_perf_patterns.sh
C=c_aerial_troy; CM=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models
name=${1:?name}; arm=${2:?ours|req|prof|green|base}; slots=${3:-480000}; rate=${4:-20}
O=${FS_OUT:-$R/prof/tenant/slot_level/fluct}; mkdir -p $O; cd $R
MODEL=${SL_MODEL:-26x}; mods=""; for m in ${SL_MODELS:-$MODEL}; do mods="$mods${mods:+,}$m=$CM/fhd132_$m"; done   # 2026-11-13: SL_MODELS="26n 26s 26m 26l 26x" loads several one-engine sets
NC=${NCELLS:-10}; PAT=${PAT:-tmp_fluct}; BLOCK_REPEAT=${BLOCK_REPEAT:-300}; BLOCK_FRAMES=${BLOCK_FRAMES:-1}; WARMUP=${WARMUP:-6}; PORT=9810
[ -f testVectors/multi-cell/launch_pattern_F08_${NC}C_$PAT.yaml ] || { echo "!! launch_pattern_F08_${NC}C_$PAT.yaml missing (tools/slot_level_fluct_pattern.py)"; exit 1; }
apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v nvidia-cuda-mps-server)
[ -z "$apps" ] || { echo "!! the GPU is in use: $apps"; exit 1; }
(echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { echo "!! TCP $PORT is already open"; exit 1; }
[ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 25 ] || { echo "!! less than 25 GB free on /"; exit 1; }
grep -q "\"$PAT\"" $V && { echo "!! $PAT is already registered in valid_perf_patterns.sh (an earlier run did not restore it?)"; exit 1; }
cp -p $V $O/valid_perf_patterns.orig.sh
restore() { cp -p $O/valid_perf_patterns.orig.sh $V && echo "   valid_perf_patterns.sh restored"; }
trap restore EXIT
sed -i "s/^valid_perf_patterns=(/valid_perf_patterns=(\"$PAT\" /; s/^peak_patterns=(/peak_patterns=(\"$PAT\" /" $V
echo "== $name arm $arm: $NC cells, $slots slots, $rate req/s, pattern $PAT ($(grep -c "\"$PAT\"" $V) list entries), block $BLOCK_FRAMES frame(s) x $BLOCK_REPEAT, $(date '+%F %T')"
export RUN_ENV="RU_LP_BLOCK_FRAMES=$BLOCK_FRAMES RU_LP_BLOCK_REPEAT=$BLOCK_REPEAT TESTMAC_LP_BLOCK_FRAMES=$BLOCK_FRAMES TESTMAC_LP_BLOCK_REPEAT=$BLOCK_REPEAT"
common="NCELLS=$NC PATTERN=$PAT KEEP_RAW=2 MPSTAT=0 RECORD=0 FS_OUT=$O BUILD_DIR=build.aarch64_f1002"
case $arm in
  ours|req)
    export PUSCH_PROFILES=${PUSCH_PROFILES:-16,24,32,40,48,56,64,72,88,104} PDSCH_PROFILES=${PDSCH_PROFILES:-24,40}   # PUSCH: requested r -> r + 4 SMs
    export SM_OVERRIDE="${SM_OVERRIDE-}"       # default: the yaml's UL order partition (a 16-SM green context at start-up)
    # 2026-11-13 pilots at 10 cells: 10 PUSCH profiles + 3 PDSCH profiles kill the L1 in its first second (order-kernel launch blocks ~1.4 ms at
    # SFN 0.17 -> UL C-plane aborts -> FATAL); 10 + 2 and 4 + 3 pass. PDSCH 16 SMs misses the 85 % rule at 10-cell Peak DL (p99 344-367 us of 375).
    tenv="-e DAPP_AGENT=1 -e DAPP_AGENT_CELLS=$NC -e DAPP_INPUT=fhd -e DAPP_TENANT_MODELS=$mods -e DAPP_TENANT_ENGINE_SM=132 -e DAPP_AGENT_QOS=0"
    tenv="$tenv -e DAPP_AGENT_WARMUP=$CM/imgfhd/fhd_0_q85.jpg,$CM/imgfhd/fhd_1_q85.jpg -e DAPP_TENANT_EMPTY_US=-1 -e DAPP_TENANT_LOG_S=5"
    if [ $arm = ours ]; then export TENANT_CLASSES=${TENANT_CLASSES:-8,16,24,32,40,48,56,64,72,80,88,96,104}; tenv="$tenv -e DAPP_AGENT_PIECE=1 -e DAPP_AGENT_PIECE_TRACE=/tmp/sl_pieces_$name.csv"
    else export TENANT_CLASSES=${TENANT_CLASSES:-24,32,40,48,56,64,72,80}; tenv="$tenv -e DAPP_AGENT_PIECE=0"; fi
    export TENANT_ENV="$tenv ${SL_TENANT_ENV:-}"
    docker exec -u root $C rm -f /tmp/sl_pieces_$name.csv
    env GREEN=1 TENANT=1 $common bash $F/fullstack_run.sh $name $slots "" "${NVTAGS-311,223}" async > $O/console_$name.log 2>&1 & RUN=$! ;;
  prof)   # the same L1 (profiles + sizer) without the tenant thread and without requests: the L1's own reference
    export PUSCH_PROFILES=${PUSCH_PROFILES:-16,24,32,40,48,56,64,72,88,104} PDSCH_PROFILES=${PDSCH_PROFILES:-24,40} SM_OVERRIDE="${SM_OVERRIDE-}"
    env GREEN=1 PROFILES_ONLY=1 $common bash $F/fullstack_run.sh $name $slots "" "${NVTAGS-311,223}" async > $O/console_$name.log 2>&1 & RUN=$! ;;
  green)  # green contexts with the stock SM values, no dApp env at all (fullstack_run.sh passes L1_ENV only with TENANT / PROFILES_ONLY)
    env GREEN=1 $common bash $F/fullstack_run.sh $name $slots "" "${NVTAGS-311,223}" ${GREEN_HOOK:-async} > $O/console_$name.log 2>&1 & RUN=$! ;;
  base)
    M4=${MIG_4G:-$(nvidia-smi -L | grep 'MIG 4g' | grep -o 'MIG-[0-9a-f-]*')}; M3=${MIG_3G:-$(nvidia-smi -L | grep 'MIG 3g' | grep -o 'MIG-[0-9a-f-]*')}
    [ -n "$M4" ] && [ -n "$M3" ] || { echo "!! MIG instances missing (sweep/fullstack/mig_night.sh on)"; exit 1; }
    [ -s $D/models/migfhd$MODEL/seg00_sm60.engine ] || { echo "!! models/migfhd$MODEL/seg00_sm60.engine missing (Baseline engine of $MODEL)"; exit 1; }
    Y=/tmp/sl_yolo_$name; echo "   MIG 4g $M4 (L1) | 3g $M3 (YOLO)"
    if [ "${DRY:-0}" != 1 ]; then
      docker exec -u root $C rm -f $Y.log $Y.out
      docker exec -d -u root -e CUDA_VISIBLE_DEVICES=$M3 -e CUDA_MPS_PIPE_DIRECTORY=/var/mig_no_mps -e CUDA_MPS_LOG_DIRECTORY=/var/mig_no_mps \
        -e DAPP_TENANT=1 -e DAPP_AGENT=1 -e DAPP_TENANT_FREERUN=1 -e DAPP_TENANT_FIXED=60 -e DAPP_TENANT_CLASSES=60 -e DAPP_PUSCH_PROFILES=16 -e DAPP_PDSCH_PROFILES=0 \
        -e DAPP_AGENT_QOS=0 -e DAPP_AGENT_GRAPH_CHUNK=0 -e DAPP_AGENT_CHUNK_AHEAD=0 -e DAPP_INPUT=fhd -e DAPP_TENANT_MODELS=$MODEL=$CM/migfhd$MODEL \
        -e DAPP_AGENT_WARMUP=$CM/imgfhd/fhd_0_q85.jpg,$CM/imgfhd/fhd_1_q85.jpg -e DAPP_AGENT_PORT=$PORT -e DAPP_TENANT_CPU=45 -e DAPP_AGENT_CPU=46 \
        -e DAPP_TENANT_LOG=$Y.log -e DAPP_TENANT_CSV= -e DAPP_TENANT_LOG_S=5 $C bash -lc "cd /opt/nvidia/cuBB && exec build.aarch64_f1002/cuPHY-CP/dapp_hook/dapp_tenant_main $((slots / 2000 + 1200)) > $Y.out 2>&1"
      for i in $(seq 1 120); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 1; done
      (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo "!! the naive YOLO server did not start"; docker exec $C tail -5 $Y.out $Y.log; exit 1; }
      echo "   naive YOLO server up after $i s: $(docker exec $C grep -E 'CUDA graphs|tenant ready' $Y.log | tail -1 | cut -c1-120)"
    fi
    env MIG_UUID=$M4 MIG_SM=1 $common bash $F/fullstack_run.sh $name $slots "" "${NVTAGS-311,223}" ${BASE_HOOK:-off} > $O/console_$name.log 2>&1 & RUN=$! ;;
  *) echo "unknown arm $arm"; exit 1 ;;
esac
if [ "${DRY:-0}" = 1 ]; then wait $RUN; grep -E "^== |dry-run|total .* cells|!!" $O/console_$name.log | cut -c1-200; exit 0; fi
# requests: from WARMUP s after the first per-second testMAC line (traffic running) until ~6 s before the traffic ends
for i in $(seq 1 700); do grep -q "Cell  0 | DL" $O/run3_$name.log 2>/dev/null && break; kill -0 $RUN 2>/dev/null || break; sleep 1; done
if [ "${NOREQ:-0}" = 1 ]; then echo "   NOREQ=1: no requests (tenant thread idle)"
elif grep -q "Cell  0 | DL" $O/run3_$name.log 2>/dev/null && (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
  dur=$((slots / 2000 - WARMUP - 8)); sleep $WARMUP
  echo "   traffic running after $i s; requests: $rate req/s for $dur s from $(date +%T)"
  taskset -c ${RA_CORES:-65-71} python3 $D/python/ra_stub.py --images $D/models/imgfhd --rate $rate --duration $dur --policy ${RA_POLICY:-fixed:$MODEL} --csv $O/ra_$name.csv --want-dets --seed ${SEED:-1} > $O/ra_$name.txt 2>&1
  tail -4 $O/ra_$name.txt | cut -c1-200
elif [ $arm = prof ] || [ $arm = green ]; then echo "   arm $arm: no requests"; else echo "!! no traffic / no agent port (run ended early?)"; fi
wait $RUN
if [ $arm = base ]; then
  docker exec -u root $C bash -c "pkill -TERM -f 'dapp_tenant_mai[n]'; sleep 3; cp $Y.log /opt/nvidia/cuBB/${O#$R/}/yolo_$name.log; cp $Y.out /opt/nvidia/cuBB/${O#$R/}/yolo_$name.out; chmod a+r /opt/nvidia/cuBB/${O#$R/}/yolo_$name.*; true"
elif [ $arm = ours ]; then docker cp $C:/tmp/sl_pieces_$name.csv $O/pieces_$name.csv > /dev/null 2>&1; docker exec -u root $C rm -f /tmp/sl_pieces_$name.csv; fi
grep -E "^== |L1 did not|tenant ready|profile fallbacks|Dropping|!!" $O/console_$name.log | cut -c1-200
python3 $T/tn_summary.py $name --dir $O > $O/summary_$name.txt 2>&1
grep -E "PUSCH completion|PDSCH compression|PDCCH end|RU: .*LATE SLOTS|L1 error|fallbacks|tenant last stats|^    pos" $O/summary_$name.txt | cut -c1-260
grep -q "FATAL exit\|l1_exit_handler" $O/run2_$name.log 2>/dev/null && echo "!! L1 FATAL exit - phy_$name.log kept" || rm -f $O/phy_$name.log
rm -f $O/ru_$name.log $O/testmac_$name.log
df -h --output=avail / | tail -1 | sed 's/^/   disk free/'
echo "SL_FLUCT_DONE $name $(date +%T)"
