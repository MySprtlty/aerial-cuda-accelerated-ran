#!/bin/bash
# Build the tenant's FP16 TensorRT engines inside a MIG instance. The tenant owns the instance: direct CUDA, no MPS, no cap.
# An engine is tied to the SM count the builder saw, so these are separate from the MPS-cap engines of build_engines.sh.
# Run inside the container as root:
#   docker exec -u root c_aerial_troy bash /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/build_engines_mig.sh MIG-<uuid> [tag]
# <uuid> = the tenant instance from `nvidia-smi -L`; tag defaults to mig3g (matrix.yaml profile c16_mig: engine_template).
# MODELS="yolo26n_1280 yolo26s_1280 yolo26x_1280" selects the ONNX stems (default: yolo26n yolo26x).
cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models
UUID=$1
TAG=${2:-mig3g}
[ -z "$UUID" ] && { echo "usage: $0 MIG-<uuid> [tag]"; exit 1; }
for m in ${MODELS:-yolo26n yolo26x}; do for b in 1 8; do
  out=${m}_b${b}_fp16_${TAG}.engine
  [ -s $out ] && { echo "have $out"; continue; }
  echo "== build $out $(date +%T)"
  env CUDA_VISIBLE_DEVICES=$UUID CUDA_MPS_PIPE_DIRECTORY=/var/mig_tenant_no_mps /usr/src/tensorrt/bin/trtexec --onnx=${m}_b${b}.onnx --fp16 --saveEngine=$out --skipInference 2>&1 | grep -E "Engine built|PASSED|FAILED|rror" | head -2
done; done
echo ENGINES_DONE $(date +%T)
