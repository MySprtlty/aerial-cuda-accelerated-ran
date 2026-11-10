#!/bin/bash
# fhd_bench.sh - FHD (1920x1080) whole-image ("normal": batch-1, 1280 letterbox, NMS-free head) YOLO26 n/s/m/l/x inference latency,
#   MIG 3g.48gb slice (60 SM) vs a green context of 80 SM on the full GPU. Same software in both arms: stand-alone tenant
#   (dapp_tenant_main, no L1, fixed partition, whole-graph launches) + xApp agent + ra_stub (fixed model, 1 in flight).
#   Engines: models/fhd26<m>_b1/ (batch 1; models/fhd26<m>/ is the production batch-3 FHD set of DAPP_INPUT=fhd, 2026-11-09).
# usage: cstart <NVIDIA_VISIBLE_DEVICES> | cstop | mps-on | mps-off | probe [pct] | build <sm> [pct] | tenant-start <sm> | tenant-stop <tag>
#        run <tag> [rate] [sec] | summary | mig-on | mig-off | status
set -u
R=/home/troy/build/aerial-cuda-accelerated-ran; H=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook
IMG=nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb; C=dgist-fhd-bench
SP=$(cd "$(dirname "$0")" && pwd); OUT=$SP/out; mkdir -p $OUT
MODELS="n s m l x"
inc() { docker exec -u root $C bash -c "$1"; }
mode() { nvidia-smi -i 0 --query-gpu=mig.mode.current,mig.mode.pending --format=csv,noheader; }
case ${1:-status} in
status) echo "mig: $(mode) | apps: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')| container: $(docker ps --format '{{.Names}}' | grep -c "^$C\$")"; nvidia-smi -L ;;
cstart) docker rm -f $C >/dev/null 2>&1
  docker run -d --name $C --privileged --network host --ipc host -e NVIDIA_VISIBLE_DEVICES=$2 -e NVIDIA_DRIVER_CAPABILITIES=all -e cuBB_SDK=/opt/nvidia/cuBB \
    -v $R:/opt/nvidia/cuBB -w /opt/nvidia/cuBB --entrypoint bash $IMG -c 'sleep infinity' >/dev/null && echo "container $C up (NVIDIA_VISIBLE_DEVICES=$2)"
  inc 'nvidia-smi -L' ;;
cstop) docker rm -f $C >/dev/null 2>&1; echo "container $C removed" ;;
mps-on) inc 'export CUDA_MPS_PIPE_DIRECTORY=/var CUDA_MPS_LOG_DIRECTORY=/var; nvidia-cuda-mps-control -d && echo start_server -uid 0 | nvidia-cuda-mps-control; sleep 1; pgrep -a nvidia-cuda-mps' ;;
mps-off) inc 'export CUDA_MPS_PIPE_DIRECTORY=/var CUDA_MPS_LOG_DIRECTORY=/var; echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 1; pkill -f "nvidia-cuda-mps-contro[l]"; pkill -f "nvidia-cuda-mps-serve[r]"; true'
  echo "GPU apps after MPS quit: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')" ;;
probe) pct=${2:-}; E=""; [ -n "$pct" ] && E="CUDA_MPS_PIPE_DIRECTORY=/var CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct"
  inc "cd $H/models && env $E python3 mps_sm_probe.py" ;;
build) sm=$2; pct=${3:-}; E=""; [ -n "$pct" ] && E="CUDA_MPS_PIPE_DIRECTORY=/var CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct"
  for m in $MODELS; do d=$H/models/fhd26${m}_b1; out=$d/seg00_sm$sm.engine
    inc "test -s $out" && { echo "have fhd26${m}_b1/seg00_sm$sm.engine"; continue; }
    echo "== build fhd26${m}_b1/seg00_sm$sm.engine $(date +%T)"; t0=$(date +%s)
    inc "mkdir -p $d && cd $H/models && env $E /usr/src/tensorrt/bin/trtexec --onnx=yolo26${m}_1280_b1_e2e.onnx --fp16 --saveEngine=$out --skipInference" > $OUT/build_26${m}_sm$sm.log 2>&1
    grep -E 'Engine built|PASSED|FAILED|rror' $OUT/build_26${m}_sm$sm.log | head -3 | cut -c1-160; echo "   $(( $(date +%s) - t0 )) s"
  done; echo "BUILD_DONE sm$sm $(date +%T)" ;;
tenant-start) sm=$2; spec=""; for m in $MODELS; do spec="$spec${spec:+,}26$m=$H/models/fhd26${m}_b1"; done
  inc "rm -f /tmp/fhd_tenant.log /tmp/fhd_tenant.csv /tmp/fhd_tenant_console.log"
  docker exec -d -u root -e DAPP_TENANT=1 -e DAPP_AGENT=1 -e DAPP_TENANT_FREERUN=1 -e DAPP_TENANT_FIXED=$sm -e DAPP_TENANT_CLASSES=$sm \
    -e DAPP_PUSCH_PROFILES=16 -e DAPP_PDSCH_PROFILES=0 -e DAPP_AGENT_QOS=0 -e DAPP_AGENT_GRAPH_CHUNK=0 \
    -e DAPP_TENANT_LOG=/tmp/fhd_tenant.log -e DAPP_TENANT_CSV=/tmp/fhd_tenant.csv -e DAPP_TENANT_MODELS=$spec \
    $C bash -c "/opt/nvidia/cuBB/build.aarch64_f1002/cuPHY-CP/dapp_hook/dapp_tenant_main 7200 > /tmp/fhd_tenant_console.log 2>&1"
  for i in $(seq 1 240); do (echo > /dev/tcp/127.0.0.1/9810) 2>/dev/null && break; inc 'pgrep -f "dapp_tenant_mai[n]"' >/dev/null || break; sleep 1; done
  if (echo > /dev/tcp/127.0.0.1/9810) 2>/dev/null; then echo "tenant up after $i s (fixed $sm SM)"; inc 'grep -iE "class|engine|model|partition|green|nsm|head" /tmp/fhd_tenant.log | head -14 | cut -c1-200'
  else echo "!! tenant did not come up"; inc 'tail -8 /tmp/fhd_tenant_console.log; tail -8 /tmp/fhd_tenant.log'; exit 1; fi ;;
tenant-stop) tag=${2:-x}; inc 'pkill -INT -f "dapp_tenant_mai[n]"; sleep 2; pkill -KILL -f "dapp_tenant_mai[n]"; true'
  docker cp $C:/tmp/fhd_tenant.log $OUT/tenant_$tag.log 2>/dev/null; docker cp $C:/tmp/fhd_tenant_console.log $OUT/tenant_${tag}_console.log 2>/dev/null; echo "tenant stopped ($tag)" ;;
run) tag=$2; rate=${3:-10}; sec=${4:-40}
  for m in $MODELS; do echo "== $tag 26$m $(date +%T)"
    inc "cd $H/python && taskset -c 60-71 python3 ra_stub.py --images $H/models/imgfhd --rate $rate --duration $sec --policy fixed:26$m --deadline-ms 1000 --max-inflight 1 --csv /tmp/fhd_${tag}_26$m.csv" 2>&1 | grep -E 'server mean|rror|fail|Traceback' | head -4 | cut -c1-200
    docker cp $C:/tmp/fhd_${tag}_26$m.csv $OUT/ 2>/dev/null
  done; echo "RUN_DONE $tag $(date +%T)" ;;
summary) python3 - "$OUT" <<'PY'
import csv, glob, os, statistics as st, sys
out = sys.argv[1]
def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1))))]
def load(tag, m):
    f = os.path.join(out, "fhd_%s_26%s.csv" % (tag, m))
    if not os.path.exists(f): return None
    rows = [r for r in csv.DictReader(open(f)) if r["status"] == "success"]
    t0 = min(float(r["t_arrival"]) for r in rows) if rows else 0
    rows = [r for r in rows if float(r["t_arrival"]) - t0 >= 3.0]      # 3 s warm-up discarded
    g = lambda k: [float(r[k]) for r in rows if r[k] not in ("", "None")]
    inf, dec, srv, wt, sm, det = g("infer_ms"), g("decode_ms"), g("server_ms"), g("wait_ms"), g("sm"), g("detections")
    return dict(n=len(rows), sm=st.mean(sm) if sm else 0, inf_mean=st.mean(inf), inf_p50=pct(inf, 50), inf_p99=pct(inf, 99), inf_max=max(inf),
                dec=st.mean(dec), wait=st.mean(wt) if wt else 0, srv_mean=st.mean(srv), srv_p50=pct(srv, 50), srv_p99=pct(srv, 99), det=st.mean(det) if det else 0)
arms = [("mig60", "MIG 3g (60 SM)"), ("green80", "green ctx 80 SM")]
print("| 모델 | arm | n | SM | 디코드 ms | 추론 mean | 추론 p50 | 추론 p99 | 추론 max | 서버 mean | 서버 p50 | 서버 p99 | 검출 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for m in "nsmlx":
    res = {}
    for tag, name in arms:
        d = load(tag, m); res[tag] = d
        if d: print("| 26%s | %s | %d | %.0f | %.1f | **%.2f** | %.2f | **%.2f** | %.2f | %.1f | %.1f | %.1f | %.0f |" % (m, name, d["n"], d["sm"], d["dec"], d["inf_mean"], d["inf_p50"], d["inf_p99"], d["inf_max"], d["srv_mean"], d["srv_p50"], d["srv_p99"], d["det"]))
    a, b = res.get("mig60"), res.get("green80")
    if a and b:
        r = lambda k: 100.0 * (b[k] - a[k]) / a[k]
        print("| 26%s | 차이 (80 SM vs MIG) | | | | **%+.0f%%** | %+.0f%% | **%+.0f%%** | | %+.0f%% | %+.0f%% | %+.0f%% | |" % (m, r("inf_mean"), r("inf_p50"), r("inf_p99"), r("srv_mean"), r("srv_p50"), r("srv_p99")))
PY
  ;;
mig-on) docker rm -f $C >/dev/null 2>&1
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$a" = 0 ] || { echo "!! GPU busy:"; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; exit 5; }
  sudo -n nvidia-smi -i 0 -mig 1 | tail -2; m=$(mode)
  [ "$m" = "Enabled, Enabled" ] || { echo "-- mode $m -> gpu reset"; sudo -n nvidia-smi --gpu-reset -i 0 | tail -2; m=$(mode); }
  [ "$m" = "Enabled, Enabled" ] || { echo "!! MIG not enabled ($m)"; exit 6; }
  sudo -n nvidia-smi mig -i 0 -cgi 3g.48gb -C | tail -3; nvidia-smi -L
  echo "MIG_3G=$(nvidia-smi -L | grep 'MIG 3g' | grep -o 'MIG-[0-9a-f-]*')" ;;
mig-off) docker rm -f $C >/dev/null 2>&1
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$a" = 0 ] || { echo "!! GPU busy:"; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; exit 5; }
  sudo -n nvidia-smi mig -i 0 -dci | tail -2; sudo -n nvidia-smi mig -i 0 -dgi | tail -2; sudo -n nvidia-smi -i 0 -mig 0 | tail -1; m=$(mode)
  [ "$m" = "Disabled, Disabled" ] || { echo "-- mode $m -> gpu reset"; sudo -n nvidia-smi --gpu-reset -i 0 | tail -2; m=$(mode); }
  [ "$m" = "Disabled, Disabled" ] || { echo "!! MIG still $m"; exit 8; }; echo "MIG off: $(mode)"; nvidia-smi -L ;;
*) sed -n '2,6p' "$0"; exit 2 ;;
esac
