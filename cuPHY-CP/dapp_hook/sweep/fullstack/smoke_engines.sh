#!/bin/bash
# smoke_engines.sh <model:sm> [...]  — 20-request standalone sweep of dapp_sched for an _e2e_fp16_sm<N> engine under the MPS
# percentage of its class (43/61/80/100 %), printing median / p99 / dets. Not during a measurement.
C=c_aerial_troy
pct() { case $1 in 56) echo 43;; 80) echo 61;; 104) echo 80;; 132) echo 100;; 32) echo 24;; 60) echo 100;; esac; }
for ms in "$@"; do m=${ms%%:*}; sm=${ms##*:}; p=$(pct $sm); f=yolo${m}_1280_b8_e2e_fp16_sm$sm.engine
  if [ "$p" -lt 100 ]; then envs=(-e CUDA_MPS_PIPE_DIRECTORY=/var -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$p); else envs=(-e CUDA_MPS_PIPE_DIRECTORY=/var); fi
  out=$(docker exec -u root "${envs[@]}" $C bash -c "cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook && timeout 300 /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/dapp_hook/dapp_sched --engine $m:$sm=models/$f --classes $sm --image models/img4k/uhd_0_q60.jpg --sweep 20 --sweep-skip 2 2>&1 | grep -E 'median|p99|dets|SM=|error|Error|rejected|fail' | tail -4" | tr '\n' ' ')
  echo "$(date +%T) smoke $m sm$sm (cap $p %): ${out:-no output}"
done
