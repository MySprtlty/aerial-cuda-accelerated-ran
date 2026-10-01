#!/bin/bash
# fullstack_run.sh <name> <num_slots> [dl_worker_cores] [nvlog_tag_ids] [hook: full|summary|off]
#   env RECORD=1  record the hook ring on the host (prof/ring_16C_<name>.bin)
#   env RU_HIST=1 RU emulator timing_histogram: 1
#   env FS_OUT    directory for console logs, config snapshots and DCGM (default prof/run61, gitignored)
# 16-cell 59c (DDDSUUDDDD) full-stack loopback run without nsys: RU emulator + cuphycontroller + testMAC,
# DCGM 1 s on the host. Edits the lab config in place (hook line, optional DL worker cores, optional nvlog
# INFO tags) and copies the container's /tmp/{phy,ru,testmac}.log to prof/run61/<f>_<name>.log afterwards.
# Statistics scripts (e_analysis.py, analyze_run.py, parse_phylog.py) exclude the first second themselves.
set -u
NAME=$1; SLOTS=$2; DLCORES=${3:-}; NVTAGS=${4:-}; HOOK=${5:-full}; RECORD=${RECORD:-0}; RU_HIST=${RU_HIST:-0}
R=$(cd "$(dirname "$0")/../../../.." && pwd)
S=${FS_OUT:-$R/prof/run61}
P=/opt/nvidia/cuBB/testBenches/phase4_test_scripts
N=16
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
echo "== $NAME: $N cells, $SLOTS slots (${TRAFFIC_S} s), dl cores '${DLCORES}', nvlog tags '${NVTAGS}', hook $HOOK, record $RECORD, ru_hist $RU_HIST $(date +%T)"
docker exec -u root c_aerial_troy bash -lc "cd $P && ./setup1_DU.sh --ru-host-type=_LOOPBACK --du-eth0=aerial02 -y F08_CG1 && ./setup2_RU.sh --ru-eth0=aerial03" > $S/setup_$NAME.log 2>&1 || { echo "setup failed"; tail -3 $S/setup_$NAME.log; exit 1; }
strip_summary
docker exec -u root c_aerial_troy bash -lc "cd $P && ./test_config.sh 59c --num-cells=$N --num-slots=$SLOTS" > $S/test_config_$NAME.log 2>&1
grep -q "total $N cells" $S/test_config_$NAME.log || { echo "test_config failed"; tail -3 $S/test_config_$NAME.log; exit 1; }
# back up the lab files this script edits; restored at the end of the run
for f in cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml cuPHY-CP/ru-emulator/config/config.yaml cuPHY/nvlog/config/nvlog_config.yaml; do cp $f $S/before_$NAME.$(basename $f); done
restore_lab() { for f in cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml cuPHY-CP/ru-emulator/config/config.yaml cuPHY/nvlog/config/nvlog_config.yaml; do cp $S/before_$NAME.$(basename $f) $f; done; }
# hook: full = UL+DL+PDUs, summary = UL+DL without PDUs, off = disabled
# every mode rewrites the whole line: the file keeps the previous run's values (the 10/01 e_sum and d runs ran with enable: 0)
case "$HOOK" in
  full)    HOOKLINE='dapp_hook: {enable: 1, shm_name: \/aerial_dapp_ring, ring_len: 65536, mlock: 1, export_ul: 1, export_dl: 1, export_pdus: 1}' ;;
  summary) HOOKLINE='dapp_hook: {enable: 1, shm_name: \/aerial_dapp_ring, ring_len: 65536, mlock: 1, export_ul: 1, export_dl: 1, export_pdus: 0}' ;;
  off)     HOOKLINE='dapp_hook: {enable: 0, shm_name: \/aerial_dapp_ring, ring_len: 65536, mlock: 1, export_ul: 1, export_dl: 1, export_pdus: 1}' ;;
esac
sed -i "s/^dapp_hook: {.*}/$HOOKLINE/" cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml
if [ "$RU_HIST" = "1" ]; then sed -i 's/^  timing_histogram: 0/  timing_histogram: 1/' cuPHY-CP/ru-emulator/config/config.yaml; grep -n "timing_histogram" cuPHY-CP/ru-emulator/config/config.yaml; fi
grep -n "dapp_hook:" cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml
# optional: DL worker cores (CPU assignment only)
if [ -n "$DLCORES" ]; then
  sed -i "s/^  workers_dl: .*/  workers_dl: [$DLCORES]/" cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml
fi
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
docker exec -u root c_aerial_troy bash -lc "$P/run2_cuPHYcontroller.sh --timeout $((TRAFFIC_S + 280))" > $S/run2_$NAME.log 2>&1 & L1=$!
for i in $(seq 1 120); do grep -q "L1 is ready" $S/run2_$NAME.log 2>/dev/null && break; kill -0 $L1 2>/dev/null || break; sleep 2; done
grep -q "L1 is ready" $S/run2_$NAME.log || { echo "L1 did not come up"; kill_ran; wait $RU 2>/dev/null; restore_lab; exit 1; }
echo "   L1 ready $(date +%T); starting DCGM + testMAC (recorder: $RECORD)"
dcgmi dmon -e 1002,1003,1004,1005,203,204,155 -d 1000 > $S/dcgm_$NAME.csv 2>&1 & DC=$!
REC=""
if [ "$RECORD" = "1" ] && [ "$HOOK" != "off" ]; then python3 cuPHY-CP/dapp_hook/tools/dapp_ring_record.py --out prof/ring_16C_$NAME.bin --seconds $((TRAFFIC_S + 90)) > $S/record_$NAME.log 2>&1 & REC=$!; fi
docker exec c_aerial_troy bash -lc "$P/run3_testMAC.sh --timeout $((TRAFFIC_S + 120))" > $S/run3_$NAME.log 2>&1 & MAC=$!
for i in $(seq 1 $((TRAFFIC_S + 150))); do grep -q "Finished running" $S/run3_$NAME.log && break; sleep 1; done
grep -m1 "Finished running" $S/run3_$NAME.log | cut -c1-80 || echo "testMAC did not finish in time"
sleep 8
kill -INT $DC 2>/dev/null; [ -n "$REC" ] && { wait $REC 2>/dev/null; cat $S/record_$NAME.log; }
kill_ran; wait $RU 2>/dev/null; wait $MAC 2>/dev/null
docker exec c_aerial_troy bash -lc "mkdir -p /opt/nvidia/cuBB/prof/run61 && for f in phy ru testmac; do cp /tmp/\$f.log /opt/nvidia/cuBB/prof/run61/\${f}_$NAME.log; done; ls -la /opt/nvidia/cuBB/prof/run61/ | grep _$NAME"
restore_lab
echo "== $NAME done $(date +%T)"
