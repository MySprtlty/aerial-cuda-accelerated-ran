#!/bin/bash
# fhd_b3_test.sh - validate the FHD batch-3 profile on the stand-alone tenant (dapp_tenant_main of the _dgist tree, no L1, fixed partition)
#   cstart | cstop | tenant-start <sm> [4k|fhd] | tenant-stop | run <tag> <imgdir> [model] [rate] [sec] | welcome | log | summary <csv>
#   env MODELS="n s m l x" (tenant-start): which models the stand-alone tenant loads
set -u
A=/home/nvidia/_dgist/aerial-cuda-accelerated-ran; H=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook
IMG=nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb; C=dgist-fhd-test
OUT=$(cd "$(dirname "$0")" && pwd)/fhd_b3_out; mkdir -p $OUT
inc() { docker exec -u root $C bash -c "$1"; }
case ${1:-} in
cstart) docker rm -f $C >/dev/null 2>&1
  docker run -d --name $C -u root --privileged --network host --ipc host -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e cuBB_SDK=/opt/nvidia/cuBB \
    -v $A:/opt/nvidia/cuBB -w /opt/nvidia/cuBB --entrypoint bash $IMG -c 'sleep infinity' >/dev/null && echo "container $C up" ;;
cstop) docker rm -f $C >/dev/null 2>&1; echo "container $C removed" ;;
tenant-start) sm=$2; prof=${3:-fhd}; set=fhd26; [ $prof = 4k ] && set=full26
  spec=""; for m in ${MODELS:-n s m l x}; do spec="$spec${spec:+,}26$m=$H/models/$set$m"; done   # MODELS="n s" limits the loaded models (2026-11-11)
  warm=$H/models/imgfhd/fhd_0_q85.jpg,$H/models/imgfhd/fhd_1_q85.jpg; [ $prof = 4k ] && warm=$H/models/img4k/img4k_0.jpg
  inc "rm -f /tmp/t_tenant.log /tmp/t_tenant.csv /tmp/t_console.log"
  docker exec -d -u root -e DAPP_TENANT=1 -e DAPP_AGENT=1 -e DAPP_TENANT_FREERUN=1 -e DAPP_TENANT_FIXED=$sm -e DAPP_TENANT_CLASSES=$sm \
    -e DAPP_PUSCH_PROFILES=16 -e DAPP_PDSCH_PROFILES=0 -e DAPP_AGENT_QOS=0 -e DAPP_AGENT_GRAPH_CHUNK=0 -e DAPP_INPUT=$prof -e DAPP_AGENT_WARMUP=$warm \
    -e DAPP_TENANT_LOG=/tmp/t_tenant.log -e DAPP_TENANT_CSV=/tmp/t_tenant.csv -e DAPP_TENANT_MODELS=$spec \
    $C bash -c "/opt/nvidia/cuBB/build.aarch64_f1002/cuPHY-CP/dapp_hook/dapp_tenant_main 7200 > /tmp/t_console.log 2>&1"
  for i in $(seq 1 240); do (echo > /dev/tcp/127.0.0.1/9810) 2>/dev/null && break; inc 'pgrep -f "dapp_tenant_mai[n]"' >/dev/null || break; sleep 1; done
  if (echo > /dev/tcp/127.0.0.1/9810) 2>/dev/null; then echo "tenant up after $i s (fixed $sm SM, profile $prof)"; inc 'grep -E "==== agent|warm-up|b[0-9], |batch" /tmp/t_tenant.log | head -8 | cut -c1-200'
  else echo "!! tenant did not come up"; inc 'tail -5 /tmp/t_console.log; tail -5 /tmp/t_tenant.log' | cut -c1-200; fi ;;
tenant-stop) inc 'pkill -TERM -f "dapp_tenant_mai[n]"; sleep 2; pgrep -f "dapp_tenant_mai[n]" >/dev/null && pkill -KILL -f "dapp_tenant_mai[n]"; true'; echo "tenant stopped" ;;
welcome) inc "cd $H/python && python3 - <<'EOF'
import json, socket
s = socket.create_connection(('127.0.0.1', 9810), timeout=5); f = s.makefile('rwb')
f.write(json.dumps({'type': 'hello', 'role': 'probe', 'name': 'probe'}).encode() + b'\n'); f.flush()
print(f.readline().decode().strip()); s.close()
EOF" ;;
run) tag=$2; imgs=$3; m=${4:-26x}; rate=${5:-10}; sec=${6:-20}
  inc "cd $H/python && taskset -c 65-71 python3 ra_stub.py --images $imgs --rate $rate --duration $sec --policy fixed:$m --deadline-ms 1000 --max-inflight 1 --want-dets --csv /tmp/t_${tag}_$m.csv" 2>&1 | grep -E "welcome|requests|E2E|26[nsmlx]:" | cut -c1-230
  docker cp $C:/tmp/t_${tag}_$m.csv $OUT/ 2>/dev/null && echo "csv: $OUT/t_${tag}_$m.csv" ;;
log) inc 'grep -E "agent:|warm-up|does not fit|tiles" /tmp/t_tenant.log | tail -12 | cut -c1-220' ;;
summary) python3 - "$2" <<'EOF'
import csv, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1])))
ok = [r for r in rows if r.get("status") == "success"]
def col(c): return [float(r[c]) for r in ok if r.get(c) not in (None, "", "None")]
print("rows", len(rows), "success", len(ok), "tiles", sorted({r.get("tiles") for r in ok}), "sm", sorted({r.get("sm") for r in ok}))
for c in ("decode_ms", "infer_ms", "server_ms", "latency_ms", "e2e_ms", "detections", "n_dets"):
    v = col(c) if c in (ok[0] if ok else {}) else []
    if v: print("  %-11s mean %.2f p50 %.2f p99 %.2f max %.2f" % (c, st.mean(v), st.median(v), sorted(v)[int(0.99 * (len(v) - 1))], max(v)))
EOF
  ;;
*) sed -n '2,3p' "$0"; exit 2 ;;
esac
