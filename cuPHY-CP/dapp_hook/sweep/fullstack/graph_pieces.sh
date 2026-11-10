#!/bin/bash
# graph_pieces.sh (2026-11-10): CUDA-graph pieces per model of the FHD set at DAPP_AGENT_GRAPH_CHUNK=3 (stand-alone tenant, L1 env, classes 24..72, 3 requests per model)
H=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook; C=dgist-fhd-test
spec=""; for m in n s m l x; do spec="$spec${spec:+,}26$m=$H/models/fhd26$m"; done
docker exec -d -u root -e DAPP_TENANT=1 -e DAPP_AGENT=1 -e DAPP_AGENT_CELLS=1 -e DAPP_TENANT_FREERUN=1 -e DAPP_TENANT_CLASSES=24,40,48,56,64,72 \
  -e DAPP_PUSCH_PROFILES=16,24,32,40,56 -e DAPP_PDSCH_PROFILES=24,40 -e DAPP_TENANT_LAYOUT=union -e DAPP_TENANT_FB_GUARD=0 \
  -e DAPP_AGENT_QOS=1 -e DAPP_AGENT_GRAPH_CHUNK=3 -e DAPP_AGENT_CHUNK_AHEAD=2 -e DAPP_INPUT=fhd -e DAPP_AGENT_WARMUP=$H/models/imgfhd/fhd_0_q85.jpg,$H/models/imgfhd/fhd_1_q85.jpg \
  -e DAPP_TENANT_LOG=/tmp/xp_tenant.log -e DAPP_TENANT_CSV=/tmp/xp_tenant.csv -e DAPP_TENANT_MODELS=$spec \
  $C bash -c "/opt/nvidia/cuBB/build.aarch64_f1002/cuPHY-CP/dapp_hook/dapp_tenant_main 300 > /tmp/xp_console.log 2>&1"
for i in $(seq 1 240); do (echo > /dev/tcp/127.0.0.1/9810) 2>/dev/null && { echo "tenant up after $i s"; break; }; docker exec -u root $C pgrep -f 'dapp_tenant_mai[n]' >/dev/null || { echo "tenant exited"; exit 1; }; sleep 1; done
docker exec -u root $C bash -c "grep -E 'graphs captured' /tmp/xp_tenant.log | cut -c1-160"
for m in n s m l x; do docker exec -u root $C bash -c "cd $H/python && python3 ra_stub.py --images $H/models/imgfhd --rate 2 --duration 2 --policy fixed:26$m --deadline-ms 1000 --max-inflight 1 >/dev/null 2>&1"; done
sleep 1; docker exec -u root $C bash -c "grep -E 'graph pieces' /tmp/xp_tenant.log | sed -E 's/, piece ms.*//' | sort -u"
docker exec -u root $C bash -c 'pkill -TERM -f "dapp_tenant_mai[n]"; sleep 2; pgrep -f "dapp_tenant_mai[n]" >/dev/null && pkill -KILL -f "dapp_tenant_mai[n]"; true'
