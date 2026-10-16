#!/bin/bash
# build_night_engines.sh  (2026-10-16 night §4) — full-GPU MPS classes: 26m/26l @ SM 56/80/104, then 26n/26s @ 104 (missing cells),
# each via models/build_engines.sh (MPS percentage that maps to the class, name _e2e_fp16_sm<N>), followed by a 20-request smoke
# sweep of dapp_sched in the same class (single-engine mode, --sweep 20 --sweep-skip 2). Never run during a measurement.
S=$(cd "$(dirname "$0")" && pwd); R=$(cd "$S/../../../.." && pwd); C=c_aerial_troy; O=${FS_OUT:-$R/prof/run65}; LOG=$O/engines_build.log
pct() { case $1 in 56) echo 43;; 80) echo 61;; 104) echo 80;; 132) echo 100;; 32) echo 24;; esac; }
smoke() {  # <model> <sm>
  local f=yolo$1_1280_b8_e2e_fp16_sm$2.engine; local p=$(pct $2)
  local envs=(-e CUDA_MPS_PIPE_DIRECTORY=/var); [ $p -lt 100 ] && envs+=(-e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$p)
  docker exec -u root "${envs[@]}" $C bash -c "cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook && timeout 300 /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/dapp_hook/dapp_sched --engine $1:$2=models/$f --classes $2 --image models/img4k/uhd_0_q60.jpg --sweep 20 --sweep-skip 2 2>&1 | grep -E 'median|p99|net|dets|error|Error|rejected' | tail -3"
}
cd $R; echo "== engine builds $(date +%T)" | tee -a $LOG
for sm in 56 80 104; do
  docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var -e MODELS="yolo26m_1280 yolo26l_1280" -e BATCHES=8 -e ONNX_SUFFIX=_e2e -e SMS="$sm" $C bash /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/build_engines.sh 2>&1 | tee -a $LOG | grep -E '== build|have|skipped|Engine|error' | cut -c1-140
  for m in 26m 26l; do echo "-- smoke $m sm$sm: $(smoke $m $sm | tr '\n' ' ')" | tee -a $LOG; done
done
for m in 26n 26s; do echo "-- smoke $m sm104: $(smoke $m 104 | tr '\n' ' ')" | tee -a $LOG; done
echo "== engines done $(date +%T)" | tee -a $LOG; ls -la $R/cuPHY-CP/dapp_hook/models/yolo26{m,l}_1280_b8_e2e_fp16_sm*.engine $R/cuPHY-CP/dapp_hook/models/yolo26{n,s}_1280_b8_e2e_fp16_sm104.engine 2>&1 | awk '{print $5, $9}' | tee -a $LOG
