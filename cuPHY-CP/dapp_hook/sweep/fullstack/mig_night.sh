#!/bin/bash
# mig_night.sh on|off|status|stop-mps|start-mps  (2026-10-16 night) — MIG 4g.48gb (cuPHY) + 3g.48gb (YOLO) on GPU 0 with
# `sudo -n nvidia-smi` (passwordless sudoers for nvidia-smi / nvidia-cuda-mps-control only). Our MPS daemon lives INSIDE
# c_aerial_troy (run2 starts it with CUDA_MPS_PIPE_DIRECTORY=/var), so it is stopped / started with docker exec -u root.
#   on   : refuses when a foreign GPU process is present; stops our MPS; -mig 1 (gpu-reset if pending); creates 4g,3g + CIs;
#          prints the two UUIDs (MIG_4G / MIG_3G lines)
#   off  : destroys CIs/GIs, -mig 0 (gpu-reset if pending), restarts our MPS daemon
C=c_aerial_troy; GPU=0
status() { echo "-- $(date +%T) mig mode: $(nvidia-smi -i $GPU --query-gpu=mig.mode.current,mig.mode.pending --format=csv,noheader) | apps: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"; nvidia-smi -L | grep -i mig | sed 's/^/   /'; }
stop_mps() { docker exec -u root $C bash -c 'export CUDA_MPS_PIPE_DIRECTORY=/var CUDA_MPS_LOG_DIRECTORY=/var; echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 1; pgrep -x nvidia-cuda-mps-control > /dev/null && pkill -x nvidia-cuda-mps-control; pgrep -x nvidia-cuda-mps-server > /dev/null && pkill -x nvidia-cuda-mps-server; true'; sleep 2; echo "-- MPS stopped: $(pgrep -c -f 'nvidia-cuda-mps-control -d') control daemon(s) left"; }
start_mps() { docker exec -u root $C bash -c 'export CUDA_MPS_PIPE_DIRECTORY=/var CUDA_MPS_LOG_DIRECTORY=/var; nvidia-cuda-mps-control -d && echo start_server -uid 0 | nvidia-cuda-mps-control'; sleep 2; echo "-- MPS started: $(pgrep -c -f 'nvidia-cuda-mps-control -d') control daemon(s), apps: $(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | tr '\n' ';')"; }
wait_free() {  # no compute app other than our MPS server
  local a; a=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v nvidia-cuda-mps-server | grep -c .)
  [ "$a" -gt 0 ] && { echo "!! foreign GPU process present:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; return 1; }; return 0; }
case ${1:-status} in
  status) status ;;
  stop-mps) stop_mps ;;
  start-mps) start_mps ;;
  on)
    wait_free || exit 5
    stop_mps; nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "!! GPU still has processes"; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; exit 5; }
    sudo -n nvidia-smi -i $GPU -mig 1 | tail -2
    m=$(nvidia-smi -i $GPU --query-gpu=mig.mode.current,mig.mode.pending --format=csv,noheader)
    if [ "$m" != "Enabled, Enabled" ]; then echo "-- mode after -mig 1: $m -> gpu reset"; sudo -n nvidia-smi --gpu-reset -i $GPU | tail -2; m=$(nvidia-smi -i $GPU --query-gpu=mig.mode.current,mig.mode.pending --format=csv,noheader); fi
    [ "$m" = "Enabled, Enabled" ] || { echo "!! MIG mode not enabled ($m) -> reverting"; sudo -n nvidia-smi -i $GPU -mig 0 | tail -1; start_mps; exit 6; }
    sudo -n nvidia-smi mig -i $GPU -cgi 4g.48gb,3g.48gb -C | tail -4
    nvidia-smi -L | grep -q 'MIG 4g' || { echo "!! instance creation failed; profiles:"; sudo -n nvidia-smi mig -i $GPU -lgip | head -20; exit 7; }
    status
    echo "MIG_4G=$(nvidia-smi -L | grep 'MIG 4g' | grep -o 'MIG-[0-9a-f-]*')"; echo "MIG_3G=$(nvidia-smi -L | grep 'MIG 3g' | grep -o 'MIG-[0-9a-f-]*')" ;;
  off)
    stop_mps      # the MPS daemon run2 restarted inside the 4g slice is ours
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v nvidia-cuda-mps-server | grep -q . && { echo "!! foreign processes still on the GPU:"; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; exit 5; }
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -q . && { echo "!! an MPS server is still on the GPU after quit:"; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; exit 5; }
    sudo -n nvidia-smi mig -i $GPU -dci | tail -2; sudo -n nvidia-smi mig -i $GPU -dgi | tail -2
    sudo -n nvidia-smi -i $GPU -mig 0 | tail -1
    m=$(nvidia-smi -i $GPU --query-gpu=mig.mode.current,mig.mode.pending --format=csv,noheader)
    if [ "$m" != "Disabled, Disabled" ]; then echo "-- mode after -mig 0: $m -> gpu reset"; sudo -n nvidia-smi --gpu-reset -i $GPU | tail -2; m=$(nvidia-smi -i $GPU --query-gpu=mig.mode.current,mig.mode.pending --format=csv,noheader); fi
    [ "$m" = "Disabled, Disabled" ] || { echo "!! MIG still $m"; exit 8; }
    start_mps; status ;;
  *) echo "usage: $0 on|off|status|stop-mps|start-mps"; exit 1 ;;
esac
