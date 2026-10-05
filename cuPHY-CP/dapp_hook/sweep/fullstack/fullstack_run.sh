#!/bin/bash
# fullstack_run.sh <name> <num_slots> [dl_worker_cores] [nvlog_tag_ids] [hook: async|full|summary|off|legacy|keep]
#   env RECORD=1  record the hook ring on the host (prof/ring_16C_<name>.bin)
#   env RU_HIST=1 RU emulator timing_histogram: 1
#   env FS_OUT    directory for console logs, config snapshots and DCGM (default prof/run61, gitignored)
#   env NCELLS    number of cells (default 16)
#   env PATTERN   official launch pattern id (default 59c = peak, 6 UE x 2 layers UL / 6 UE x 4 layers DL per cell). Heavier 4T4R
#                 16-cell patterns that need nothing but the name: 59e (UL x1.08), 65c (UL x1.07, PUCCH x4), 65a (UL x1.19)
#   env KEEP_RAW=0 do not copy the 8-16 GB container logs; keep only the L2A.PROCESSING_TIMES lines (l2a_<name>.log),
#                 the WRN/ERR lines (phy_warn_<name>.log) and the RU/testMAC console logs (disk)
#   env MPSTAT=1  mpstat -P ALL 1 on the host during traffic (mpstat_<name>.txt) + a thread placement snapshot
#   env RU_UL_CORES / RU_DL_CORES / RU_LOW_CORE / RU_DPDK_CORE  override the RU emulator core lists ("20, 21, ...")
#   env MIG_UUID  run cuphycontroller (and the MPS daemon run2 restarts) on that MIG instance: CUDA_VISIBLE_DEVICES=<uuid>
#   env MIG_SM=1  set the cuphycontroller mps_sm_* caps to NVIDIA's MIG 4g.48gb values (docs: E2E gNodeB on MIG:
#                 pusch 60, pucch 2, prach 2, ul_order 20, pdsch 62, pdcch 10, pbch 2, gpu_comms 16, srs 16; sum 190);
#                 the file is restored after the run like the other lab files
# 16-cell 59c (DDDSUUDDDD) full-stack loopback run without nsys: RU emulator + cuphycontroller + testMAC,
# DCGM 1 s on the host. Edits the lab config in place (hook line, optional DL worker cores, optional nvlog
# INFO tags) and copies the container's /tmp/{phy,ru,testmac}.log to prof/run61/<f>_<name>.log afterwards.
# Statistics scripts (e_analysis.py, analyze_run.py, parse_phylog.py) exclude the first second themselves.
set -u
NAME=$1; SLOTS=$2; DLCORES=${3:-}; NVTAGS=${4:-}; HOOK=${5:-async}; RECORD=${RECORD:-0}; RU_HIST=${RU_HIST:-0}
KEEP_RAW=${KEEP_RAW:-1}; MPSTAT=${MPSTAT:-0}; MIG_UUID=${MIG_UUID:-}; MIG_SM=${MIG_SM:-0}; RU_UL_CORES=${RU_UL_CORES:-}; RU_DL_CORES=${RU_DL_CORES:-}; RU_LOW_CORE=${RU_LOW_CORE:-}; RU_DPDK_CORE=${RU_DPDK_CORE:-}
R=$(cd "$(dirname "$0")/../../../.." && pwd)
S=${FS_OUT:-$R/prof/run61}
P=/opt/nvidia/cuBB/testBenches/phase4_test_scripts
N=${NCELLS:-16}
cd $R; mkdir -p $S
gen() { od -An -tu4 -j24 -N4 /dev/shm/aerial_dapp_ring 2>/dev/null | tr -d ' '; }
strip_summary() {
python3 - <<'PY'
p="testBenches/phase4_test_scripts/test_config_summary.sh"
tv=set("PATTERN PATTERN_MODE CHANNELS NUM_CELLS NUM_PORTS TEST_SLOTS WORK_CANCEL_MODE WC_MODE BFP EARLY_HARQ_ENABLED EHQ_STATUS DEVICE_GRAPH_LAUNCH_ENABLED DGL_STATUS USE_GREEN_CONTEXT GC_STATUS PMU_METRICS STT DLC_TB_ENABLED ML2_CELL_MASK0 ML2_CELL_MASK1 ML2_CELL_LIST0 ML2_CELL_LIST1 TESTMAC1_YAML TEST_CONFIG_DONE".split())
out=[]
for l in open(p).read().splitlines():
    if l.startswith("VARS="): l=l.split(" PATTERN")[0].rstrip('"')+'"'
    elif l.split("=")[0] in tv: continue
    out.append(l)
open(p,"w").write("\n".join(out)+"\n")
PY
}
kill_ran() { docker exec -u root c_aerial_troy bash -c 'pkill -TERM -x phy_main; sleep 6; pkill -KILL -x phy_main; pkill -TERM -x ru_emulator; sleep 2; pkill -KILL -x ru_emulator; pkill -KILL -x test_mac; sleep 1; ps -eo pid,comm | grep -E "phy_main|ru_emulator|test_mac" | grep -v defunct || echo "RAN down"'; }
TRAFFIC_S=$(( SLOTS / 2000 ))
echo "== $NAME: $N cells, $SLOTS slots (${TRAFFIC_S} s), dl cores '${DLCORES}', nvlog tags '${NVTAGS}', hook $HOOK, record $RECORD, ru_hist $RU_HIST, keep_raw $KEEP_RAW, mpstat $MPSTAT, ru cores ul '${RU_UL_CORES}' dl '${RU_DL_CORES}', mig '${MIG_UUID}' mig_sm $MIG_SM $(date +%T)"
docker exec -u root c_aerial_troy bash -lc "cd $P && ./setup1_DU.sh --ru-host-type=_LOOPBACK --du-eth0=aerial02 -y F08_CG1 && ./setup2_RU.sh --ru-eth0=aerial03" > $S/setup_$NAME.log 2>&1 || { echo "setup failed"; tail -3 $S/setup_$NAME.log; exit 1; }
strip_summary
docker exec -u root c_aerial_troy bash -lc "cd $P && ./test_config.sh ${PATTERN:-59c} --num-cells=$N --num-slots=$SLOTS" > $S/test_config_$NAME.log 2>&1
grep -q "total $N cells" $S/test_config_$NAME.log || { echo "test_config failed"; tail -3 $S/test_config_$NAME.log; exit 1; }
# back up the lab files this script edits; restored at the end of the run
for f in cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml cuPHY-CP/ru-emulator/config/config.yaml cuPHY/nvlog/config/nvlog_config.yaml; do cp $f $S/before_$NAME.$(basename $f); done
restore_lab() { for f in cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml cuPHY-CP/ru-emulator/config/config.yaml cuPHY/nvlog/config/nvlog_config.yaml; do cp $S/before_$NAME.$(basename $f) $f; done; }
# hook: async (default) = UL+DL+PDUs, exported by the dApp export thread on core EXPORT_CPU (default 43),
#       full = the same inline on the msg_processing thread, summary = inline without PDUs, off = disabled,
#       legacy = a block without async keys (pre-async yaml: falls back to inline), keep = leave the yaml line as it is
# every mode rewrites the whole line: the file keeps the previous run's values (the 10/01 e_sum and d runs ran with enable: 0)
case "$HOOK" in
  full)    HOOKLINE='dapp_hook: {enable: 1, shm_name: \/aerial_dapp_ring, ring_len: 65536, mlock: 1, export_ul: 1, export_dl: 1, export_pdus: 1, async_export: 0}' ;;
  summary) HOOKLINE='dapp_hook: {enable: 1, shm_name: \/aerial_dapp_ring, ring_len: 65536, mlock: 1, export_ul: 1, export_dl: 1, export_pdus: 0, async_export: 0}' ;;
  off)     HOOKLINE='dapp_hook: {enable: 0, shm_name: \/aerial_dapp_ring, ring_len: 65536, mlock: 1, export_ul: 1, export_dl: 1, export_pdus: 1}' ;;
  async)   HOOKLINE="dapp_hook: {enable: 1, shm_name: \/aerial_dapp_ring, ring_len: 65536, mlock: 1, export_ul: 1, export_dl: 1, export_pdus: 1, async_export: 1, export_cpu: ${EXPORT_CPU:-43}}" ;;
  legacy)  HOOKLINE='dapp_hook: {enable: 1, shm_name: \/aerial_dapp_ring, ring_len: 65536, mlock: 1, export_ul: 1, export_dl: 1, export_pdus: 1}' ;;
  keep)    HOOKLINE="" ;;
esac
[ -n "$HOOKLINE" ] && sed -i "s/^dapp_hook: {.*}/$HOOKLINE/" cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml
if [ "$RU_HIST" = "1" ]; then sed -i 's/^  timing_histogram: 0/  timing_histogram: 1/' cuPHY-CP/ru-emulator/config/config.yaml; grep -n "timing_histogram" cuPHY-CP/ru-emulator/config/config.yaml; fi
# optional: RU emulator core lists (test tool CPU assignment only)
[ -n "$RU_UL_CORES" ] && sed -i "s/^  ul_core_list: .*/  ul_core_list: [$RU_UL_CORES]/" cuPHY-CP/ru-emulator/config/config.yaml
[ -n "$RU_DL_CORES" ] && sed -i "s/^  dl_core_list: .*/  dl_core_list: [$RU_DL_CORES]/" cuPHY-CP/ru-emulator/config/config.yaml
[ -n "$RU_LOW_CORE" ] && sed -i "s/^  low_priority_core: .*/  low_priority_core: $RU_LOW_CORE/" cuPHY-CP/ru-emulator/config/config.yaml
[ -n "$RU_DPDK_CORE" ] && sed -i "s/^  aerial_fh_dpdk_thread: .*/  aerial_fh_dpdk_thread: $RU_DPDK_CORE/" cuPHY-CP/ru-emulator/config/config.yaml
grep -n "^  ul_core_list\|^  dl_core_list\|^  low_priority_core\|^  aerial_fh_dpdk_thread" cuPHY-CP/ru-emulator/config/config.yaml
grep -n "dapp_hook:" cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml
# optional: DL worker cores (CPU assignment only)
if [ -n "$DLCORES" ]; then
  sed -i "s/^  workers_dl: .*/  workers_dl: [$DLCORES]/" cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml
fi
if [ "$MIG_SM" = "1" ]; then
  for kv in pusch:60 pucch:2 prach:2 ul_order:20 pdsch:62 pdcch:10 pbch:2 gpu_comms:16 srs:16; do
    sed -i "s/^  mps_sm_${kv%%:*}: .*/  mps_sm_${kv%%:*}: ${kv##*:}/" cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml
  done
fi
grep -n "^  mps_sm_" cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml | tr -d ' ' | tr '\n' ' '; echo
grep -n "workers_ul\|workers_dl\|debug_worker\|enable_cpu_task_tracing" cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml
# optional: INFO-level shm logging for selected nvlog tag ids (comma separated)
if [ -n "$NVTAGS" ]; then
python3 - "$NVTAGS" <<'PY'
import sys, re
p = "cuPHY/nvlog/config/nvlog_config.yaml"
tags = set(sys.argv[1].split(","))
out = []
lines = open(p).read().splitlines()
for i, ln in enumerate(lines):
    out.append(ln)
    m = re.match(r"(\s*)- (\d+): \"", ln)
    if m and m.group(2) in tags and not (i + 1 < len(lines) and re.match(r"\s+shm_level: \d", lines[i + 1])):
        out.append(m.group(1) + "  shm_level: 5")
open(p, "w").write("\n".join(out) + "\n")
print("nvlog INFO for tags", sorted(tags))
PY
fi
cp cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml $S/cuphycontroller_$NAME.yaml
cp cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml $S/l2_adapter_$NAME.yaml
cp cuPHY/nvlog/config/nvlog_config.yaml $S/nvlog_$NAME.yaml
docker exec c_aerial_troy bash -lc "$P/run1_RU.sh --timeout $((TRAFFIC_S + 330))" > $S/run1_$NAME.log 2>&1 & RU=$!
sleep 25
docker exec -u root ${MIG_UUID:+-e CUDA_VISIBLE_DEVICES=$MIG_UUID} c_aerial_troy bash -lc "$P/run2_cuPHYcontroller.sh --timeout $((TRAFFIC_S + 280))" > $S/run2_$NAME.log 2>&1 & L1=$!
for i in $(seq 1 120); do grep -q "L1 is ready" $S/run2_$NAME.log 2>/dev/null && break; kill -0 $L1 2>/dev/null || break; sleep 2; done
grep -q "L1 is ready" $S/run2_$NAME.log || { echo "L1 did not come up"; kill_ran; wait $RU 2>/dev/null; restore_lab; exit 1; }
echo "   L1 ready $(date +%T); starting DCGM + testMAC (recorder: $RECORD)"
dcgmi dmon -e 1002,1003,1004,1005,203,204,155 -d 1000 > $S/dcgm_$NAME.csv 2>&1 & DC=$!
MP=""
if [ "$MPSTAT" = "1" ]; then mpstat -P ALL 1 > $S/mpstat_$NAME.txt 2>&1 & MP=$!; fi
REC=""
if [ "$RECORD" = "1" ] && [ "$HOOK" != "off" ]; then python3 cuPHY-CP/dapp_hook/tools/dapp_ring_record.py --out prof/ring_16C_$NAME.bin --seconds $((TRAFFIC_S + 90)) > $S/record_$NAME.log 2>&1 & REC=$!; fi
docker exec c_aerial_troy bash -lc "$P/run3_testMAC.sh --timeout $((TRAFFIC_S + 120))" > $S/run3_$NAME.log 2>&1 & MAC=$!
for i in $(seq 1 $((TRAFFIC_S + 150))); do
  grep -q "Finished running" $S/run3_$NAME.log && break
  # thread placement snapshot once the traffic is on (host view: all containers share the host pid namespace view via ps -eL)
  if [ "$MPSTAT" = "1" ] && [ $i -eq 90 ]; then
    docker exec c_aerial_troy bash -c 'ps -eLo pid,tid,psr,pcpu,pri,comm,args | grep -E "phy_main|ru_emulator|test_mac" | grep -v grep' > $S/threads_$NAME.txt 2>&1
    docker exec c_aerial_troy bash -c 'for p in $(pgrep -x phy_main) $(pgrep -x ru_emulator) $(pgrep -x test_mac); do pn=$(cat /proc/$p/comm); for t in /proc/$p/task/*; do echo "$pn $(basename $t) $(cat $t/comm) allowed=$(grep Cpus_allowed_list $t/status | cut -f2) last_cpu=$(awk "{print \$39}" $t/stat)"; done; done' > $S/threads_affinity_$NAME.txt 2>&1
  fi
  sleep 1
done
grep -m1 "Finished running" $S/run3_$NAME.log | cut -c1-80 || echo "testMAC did not finish in time"
sleep 8
kill -INT $DC 2>/dev/null; [ -n "$MP" ] && kill -INT $MP 2>/dev/null; [ -n "$REC" ] && { wait $REC 2>/dev/null; cat $S/record_$NAME.log; }
kill_ran; wait $RU 2>/dev/null; wait $MAC 2>/dev/null
if [ "$KEEP_RAW" = "1" ]; then
  docker exec c_aerial_troy bash -lc "mkdir -p /opt/nvidia/cuBB/prof/run61 && for f in phy ru testmac; do cp /tmp/\$f.log /opt/nvidia/cuBB/prof/run61/\${f}_$NAME.log; done; ls -la /opt/nvidia/cuBB/prof/run61/ | grep _$NAME"
else
  docker exec c_aerial_troy bash -lc "mkdir -p /opt/nvidia/cuBB/prof/run61 && grep -F '[L2A.PROCESSING_TIMES]' /tmp/phy.log > /opt/nvidia/cuBB/prof/run61/l2a_$NAME.log; grep -E ' (WRN|ERR|FATAL) |Send Err.ind|recovery|not responsive|No available|Successive Non-availability|Attempting PUSCH pipeline termination|Order kernel timeout|exit handler|pipeline failed' /tmp/phy.log | head -2000000 > /opt/nvidia/cuBB/prof/run61/phy_warn_$NAME.log; cp /tmp/testmac.log /opt/nvidia/cuBB/prof/run61/testmac_$NAME.log; grep -E ' (WRN|ERR) |PACKET_SUMMARY|LATE' /tmp/ru.log | head -400000 > /opt/nvidia/cuBB/prof/run61/ru_warn_$NAME.log; ls -la /opt/nvidia/cuBB/prof/run61/ | grep _$NAME"
fi
restore_lab
echo "== $NAME done $(date +%T)"
