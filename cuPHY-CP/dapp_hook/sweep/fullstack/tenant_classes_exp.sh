#!/bin/bash
# tenant_classes_exp.sh <tag> <classes e.g. 24,40,48,56,64,72,80> <models e.g. n,s,m,l,x> [extra docker -e args...] (2026-11-10)
# Stand-alone dapp_tenant_main (build.aarch64_f1002 of the mounted _dgist tree) in the container of fhd_b3_test.sh (cstart) with the L1 environment,
# FHD set by default (SET=full26 for 4K), waits for the agent port (OK) or the exit (FAIL), prints the last validation line, stops the tenant.
# Env: TC=1 uses /tmp/tc/fhd26{s,l} symlink dirs with seg00_sm80.tc.engine (the 2026-11-10 timing-cache experiment; 0 = the plain model dirs),
#      PRE="nsys profile ... " or "compute-sanitizer ..." prefixes the tenant command, DUR=<s> tenant run time (600), WAITEXIT=1 waits for the exit.
# Used to isolate the TensorRT 10.3 class-coexistence failures (prof/tenant/fhd/fhd_b3_2026-11-09.md, 2026-11-10 sections). Logs /tmp/x<tag>_{console,tenant}.log in the container.
tag=$1; cls=$2; mods=$3; shift 3; H=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook; C=dgist-fhd-test
docker exec -u root $C bash -c "for m in s l; do d=/tmp/tc/fhd26\$m; mkdir -p \$d; for c in 24 40 48 56 64 72; do ln -sf $H/models/fhd26\$m/seg00_sm\$c.engine \$d/; done; ln -sf $H/models/fhd26\$m/seg00_sm80.tc.engine \$d/seg00_sm80.engine; done"
spec=""; for m in ${mods//,/ }; do d=$H/models/${SET:-fhd26}$m; [ "${TC:-0}" = 1 ] && { [ $m = s ] || [ $m = l ]; } && d=/tmp/tc/fhd26$m; spec="$spec${spec:+,}26$m=$d"; done
warm=$H/models/imgfhd/fhd_0_q85.jpg,$H/models/imgfhd/fhd_1_q85.jpg
docker exec -d -u root -e DAPP_TENANT=1 -e DAPP_AGENT=1 -e DAPP_AGENT_CELLS=1 -e DAPP_TENANT_FREERUN=1 -e DAPP_TENANT_CLASSES=$cls \
  -e DAPP_PUSCH_PROFILES=16,24,32,40,56 -e DAPP_PDSCH_PROFILES=24,40 -e DAPP_TENANT_LAYOUT=union -e DAPP_TENANT_FB_GUARD=0 \
  -e DAPP_AGENT_QOS=1 -e DAPP_AGENT_GRAPH_CHUNK=3 -e DAPP_AGENT_CHUNK_AHEAD=2 -e DAPP_INPUT=fhd -e DAPP_AGENT_WARMUP=$warm \
  -e DAPP_TENANT_LOG=/tmp/x${tag}_tenant.log -e DAPP_TENANT_CSV=/tmp/x${tag}_tenant.csv -e DAPP_TENANT_MODELS=$spec "$@" \
  $C bash -c "${PRE:-}/opt/nvidia/cuBB/build.aarch64_f1002/cuPHY-CP/dapp_hook/dapp_tenant_main ${DUR:-600} > /tmp/x${tag}_console.log 2>&1"
for i in $(seq 1 300); do (echo > /dev/tcp/127.0.0.1/9810) 2>/dev/null && { echo "[$tag] classes $cls models $mods: OK, port open after $i s"; break; }; docker exec -u root $C pgrep -f 'dapp_tenant_mai[n]' >/dev/null || { echo "[$tag] classes $cls models $mods: FAIL, tenant exited after $i s"; break; }; sleep 1; done
docker exec -u root $C bash -c "grep -vE 'different models of devices|lazy loading|aux streams|^$' /tmp/x${tag}_console.log | tail -4" | cut -c1-230
docker exec -u root $C bash -c "grep -cE 'part [0-9]+ \(P' /tmp/x${tag}_tenant.log | sed 's/^/  validations passed: /'; grep -E 'part [0-9]+ \(P' /tmp/x${tag}_tenant.log | tail -1 | cut -c1-100"
[ -n "${WAITEXIT:-}" ] && for i in $(seq 1 180); do docker exec -u root $C pgrep -f "dapp_tenant_mai[n]" >/dev/null || break; sleep 1; done
docker exec -u root $C bash -c 'pkill -TERM -f "dapp_tenant_mai[n]"; sleep 2; pgrep -f "dapp_tenant_mai[n]" >/dev/null && pkill -KILL -f "dapp_tenant_mai[n]"; true'; sleep 1
