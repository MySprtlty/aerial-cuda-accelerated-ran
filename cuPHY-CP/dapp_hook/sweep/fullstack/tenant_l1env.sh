#!/bin/bash
# tenant_l1env.sh <tag> <set:fhd26|full26> [extra docker -e args...] (2026-11-10) - start dapp_tenant_main in dgist-fhd-test with the L1 environment, wait for the agent port or exit, print the verdict, stop it
# Container: the one of fhd_b3_test.sh (cstart). Logs /tmp/r<tag>_{console,tenant}.log inside it. Used to reproduce the TensorRT 10.3 engine coexistence failure (prof/tenant/fhd/fhd_b3_2026-11-09.md).
tag=$1; set=$2; shift 2; H=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook; C=dgist-fhd-test
spec=""; for m in n s m l x; do spec="$spec${spec:+,}26$m=$H/models/$set$m"; done
warm=$H/models/imgfhd/fhd_0_q85.jpg,$H/models/imgfhd/fhd_1_q85.jpg; prof=fhd; [ $set = full26 ] && { warm=$H/models/img4k/img4k_0.jpg; prof=4k; }
docker exec -d -u root -e DAPP_TENANT=1 -e DAPP_AGENT=1 -e DAPP_AGENT_CELLS=1 -e DAPP_TENANT_FREERUN=1 -e DAPP_TENANT_CLASSES=24,40,48,56,64,72,80 \
  -e DAPP_PUSCH_PROFILES=16,24,32,40,56 -e DAPP_PDSCH_PROFILES=24,40 -e DAPP_TENANT_LAYOUT=union -e DAPP_TENANT_FB_GUARD=0 \
  -e DAPP_AGENT_QOS=1 -e DAPP_AGENT_GRAPH_CHUNK=3 -e DAPP_AGENT_CHUNK_AHEAD=2 -e DAPP_INPUT=$prof -e DAPP_AGENT_WARMUP=$warm \
  -e DAPP_TENANT_LOG=/tmp/r${tag}_tenant.log -e DAPP_TENANT_CSV=/tmp/r${tag}_tenant.csv -e DAPP_TENANT_MODELS=$spec "$@" \
  $C bash -c "/opt/nvidia/cuBB/build.aarch64_f1002/cuPHY-CP/dapp_hook/dapp_tenant_main 600 > /tmp/r${tag}_console.log 2>&1"
for i in $(seq 1 240); do (echo > /dev/tcp/127.0.0.1/9810) 2>/dev/null && { echo "[$tag] port open after $i s (OK)"; break; }; docker exec -u root $C pgrep -f 'dapp_tenant_mai[n]' >/dev/null || { echo "[$tag] tenant exited after $i s"; break; }; sleep 1; done
echo "[$tag] console:"; docker exec -u root $C bash -c "grep -vE 'different models of devices|lazy loading|aux streams' /tmp/r${tag}_console.log | tail -6" | cut -c1-240
echo "[$tag] tenant log:"; docker exec -u root $C bash -c "grep -E 'part [0-9]+ \(P|MISMATCH|ready|graphs captured|NOT|fail' /tmp/r${tag}_tenant.log | head -14" | cut -c1-160
docker exec -u root $C bash -c 'pkill -TERM -f "dapp_tenant_mai[n]"; sleep 2; pgrep -f "dapp_tenant_mai[n]" >/dev/null && pkill -KILL -f "dapp_tenant_mai[n]"; true'; sleep 1
