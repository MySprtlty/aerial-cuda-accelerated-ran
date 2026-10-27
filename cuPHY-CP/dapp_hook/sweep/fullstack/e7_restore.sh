#!/bin/bash
# e7_restore.sh — end of the R0 ladder: restore every host / tool setting the ladder touched and verify the state.
#   1. PTP daemons back to core 20 (ptp_move.sh restore) + 30 s of phc2sys offset afterwards
#   2. GPU clocks: nvidia-smi -rgc (-rmc unsupported on GH200), state printed
#   3. run2_cuPHYcontroller.sh identical to the saved original, B2' flag absent
#   4. testMAC binary / launch_pattern.cpp state (e6_parse_cache.sh status)
#   5. RAN processes down, MPS control daemon alive (started if missing, pipe /var, as root in the container)
#   6. lab files vs git (expected: the usual ' M' lab edits only), temp files under prof/tmp and testVectors/multi-cell
R=$(cd "$(dirname "$0")/../../../.." && pwd); S=$(cd "$(dirname "$0")" && pwd); cd $R
echo "== E7 restore $(date '+%F %T')"
bash $S/ptp_move.sh restore; echo "-- 30 s of phc2sys offset after the restore:"; sudo -n journalctl -u phc2sys -f -n 0 --no-pager 2>/dev/null | head -30 | sed 's/.*phc2sys\[[0-9]*\]: //' | awk '{for(i=1;i<=NF;i++){if($i=="rms")r+=$(i+1); if($i=="max"){m+=$(i+1); if($(i+1)>mx)mx=$(i+1)}}; n++} END{printf "   n=%d mean rms %.1f ns, mean max %.1f ns, max max %d ns\n", n, r/n, m/n, mx}'
bash $S/gpu_clock_lock.sh unlock 2>&1 | grep -v 'not supported\|Treating' 
if [ -f prof/tmp/tmp_run2_cuPHYcontroller.orig.sh ]; then cmp -s prof/tmp/tmp_run2_cuPHYcontroller.orig.sh testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh && echo "-- run2: identical to the saved original" || { echo "!! run2 differs from the original -> restoring"; cp prof/tmp/tmp_run2_cuPHYcontroller.orig.sh testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh; }; fi
[ -f prof/tmp/tmp_gc_keep_mps ] && { rm -f prof/tmp/tmp_gc_keep_mps; echo "-- B2' flag removed"; } || echo "-- B2' flag absent"
echo "-- git state of run2 / launch_pattern.cpp: $(git status --short -- testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh cuPHY-CP/testMAC/testMAC/launch_pattern.cpp | tr '\n' ' ')(empty = clean)"
bash $S/e6_parse_cache.sh status 2>&1 | sed 's/^/   /'
echo "-- RAN processes alive: $(ps -eo stat,comm --no-headers | awk '$1 !~ /^Z/ && $2 ~ /^(phy_main|cuphycontroller|ru_emulator|test_mac|mac_main)/' | wc -l) (expected 0)"
if pgrep -f 'nvidia-cuda-mps-control -d' > /dev/null; then echo "-- MPS control daemon alive"; else echo "-- MPS control daemon missing -> starting"; docker exec -u root c_aerial_troy bash -c 'export CUDA_MPS_PIPE_DIRECTORY=/var CUDA_MPS_LOG_DIRECTORY=/var; nvidia-cuda-mps-control -d && echo start_server -uid 0 | nvidia-cuda-mps-control'; pgrep -f 'nvidia-cuda-mps-control -d' > /dev/null && echo "   started"; fi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/   gpu app: /'
echo "-- lab files: $(git status --short -- cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml cuPHY-CP/ru-emulator/config/config.yaml cuPHY/nvlog/config/nvlog_config.yaml cuPHY-CP/testMAC/testMAC/test_mac_config.yaml | tr '\n' ' ')"
echo "-- temp files: prof/tmp: $(ls prof/tmp 2>/dev/null | tr '\n' ' ' | cut -c1-200)"; echo "   testVectors/multi-cell tmp: $(ls testVectors/multi-cell/ | grep -c 'tmp')"; [ -f timing_histogram.txt ] && echo "!! timing_histogram.txt at the repo root"
echo "-- container /tmp raw logs of the last run (deleted now): $(docker exec c_aerial_troy bash -c 'du -ch /tmp/phy.log /tmp/ru.log /tmp/testmac.log 2>/dev/null | tail -1 | cut -f1')"; docker exec -u root c_aerial_troy bash -c 'rm -f /tmp/phy.log /tmp/ru.log /tmp/testmac.log'
df -h / | tail -1 | awk '{print "-- disk: "$4" free ("$5" used)"}'
