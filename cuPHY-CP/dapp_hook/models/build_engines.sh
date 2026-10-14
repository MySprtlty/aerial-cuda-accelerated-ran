#!/bin/bash
# Build FP16 TensorRT engines for the tenant worker / dapp_sched. Idempotent. Run inside the container as root with the
# MPS pipe:  docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var c_aerial_troy bash /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/build_engines.sh [caps...]
# Two naming modes (an engine only runs near the SM count the builder saw: README "Engines and SM classes"):
#   caps as arguments   percentage-cap engines for the bench sweeps:  <model>_b<batch>_fp16_atp<cap>.engine  (default caps 100 80 60 40 20 10)
#   SMS="56 80 132"     SM-class engines for dapp_sched: the cap is the percentage that the driver maps to exactly that SM count
#                       (floor(pct/100 * 66 TPC) * 2 on the GH200), mps_sm_probe.py confirms what the builder saw, and the file is
#                       named by that count:  <model>_b<batch>_fp16_sm<granted>.engine
# MODELS="yolo26s_1280" limits the models (default: yolo26n yolo26s yolo26x); BATCHES="8" the batches (default "1 8");
# ONNX_SUFFIX=_e2e builds from the NMS-free export (<model>_b<batch>_e2e.onnx -> ..._e2e_fp16_sm<N>.engine);
# each needs <model>_b<batch><suffix>.onnx here (export_1280.py [--nms-free] for the 1280 models).
cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models
MODELS=${MODELS:-yolo26n yolo26s yolo26x}; BATCHES=${BATCHES:-1 8}; SFX=${ONNX_SUFFIX:-}   # ONNX_SUFFIX=_e2e: the NMS-free head export
build() {  # <cap> <onnx> <out>
  env $( [ "$1" -lt 100 ] && echo CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$1 ) /usr/src/tensorrt/bin/trtexec --onnx=$2 --fp16 --saveEngine=$3 --skipInference 2>&1 | grep -E "Engine built|PASSED|FAILED|rror" | head -2
}
if [ -n "${SMS:-}" ]; then
  for sm in $SMS; do
    # smallest percentage whose floor(pct * 66 / 100) * 2 reaches sm, then confirmed by the driver
    pct=$(( (sm / 2 * 100 + 65) / 66 )); [ $pct -gt 100 ] && pct=100
    got=$(env $( [ $pct -lt 100 ] && echo CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct ) python3 mps_sm_probe.py)
    if [ "$got" != "$sm" ]; then echo "SM $sm: cap $pct % grants $got SM, not $sm - skipped"; continue; fi
    for m in $MODELS; do for b in $BATCHES; do
      out=${m}_b${b}${SFX}_fp16_sm${sm}.engine
      [ -s $out ] && { echo "have $out"; continue; }
      echo "== build $out (cap $pct % = $got SM) $(date +%T)"
      build $pct ${m}_b${b}${SFX}.onnx $out
    done; done
  done
else
  CAPS=${*:-"100 80 60 40 20 10"}
  for cap in $CAPS; do for m in $MODELS; do for b in $BATCHES; do
    out=${m}_b${b}${SFX}_fp16_atp${cap}.engine
    [ -s $out ] && { echo "have $out"; continue; }
    echo "== build $out $(date +%T)"
    build $cap ${m}_b${b}${SFX}.onnx $out
  done; done; done
fi
echo ENGINES_DONE $(date +%T)
