#!/bin/bash
# Build one FP16 TensorRT engine per (model, batch, MPS cap) for the tenant worker. Idempotent.
# Run inside the container as root with the MPS pipe: docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var c_aerial_troy bash /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/build_engines.sh [caps...]
# MODELS="yolo26s" limits the models (default: yolo26n yolo26s yolo26x); each needs <model>_b1.onnx and <model>_b8.onnx here.
cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models
CAPS=${*:-"100 80 60 40 20 10"}
for cap in $CAPS; do for m in ${MODELS:-yolo26n yolo26s yolo26x}; do for b in 1 8; do
  out=${m}_b${b}_fp16_atp${cap}.engine
  [ -s $out ] && { echo "have $out"; continue; }
  echo "== build $out $(date +%T)"
  env $( [ $cap -lt 100 ] && echo CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$cap ) /usr/src/tensorrt/bin/trtexec --onnx=${m}_b${b}.onnx --fp16 --saveEngine=$out --skipInference 2>&1 | grep -E "Engine built|PASSED|FAILED|rror" | head -2
done; done; done
echo ENGINES_DONE $(date +%T)
