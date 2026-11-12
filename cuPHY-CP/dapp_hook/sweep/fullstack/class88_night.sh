#!/bin/bash
# class88_night.sh [build|standalone|flip|fullstack|all] - FHD class 88 for every model (2026-11-11).
#   88 = the (P132,N24) gap of the union layout (PDSCH 24 + Order 8 reserved, no UL work within ul_back 4 slots -> DDDDDDSUUU slots 4-6), the largest
#   gap the layout produces while cuPHY keeps every channel partition it asks for; until now the tenant ran class 80 engines in it.
#   build      scripts/build_yolo_engines.sh fhd 88 (5 engines, MPS cap 67 % = 88 SMs; s/l with the sm72 timing cache) -> runs/build_engines_fhd_88.log
#   standalone per model: fixed 88-SM partition timing (fhd_b3_test.sh) and the whole ladder 24..88 loaded in one process (tenant_classes_exp.sh);
#              a model whose ladder fails keeps its 88 engine aside as seg00_sm88.failed.engine (the tenant then stays at 80 for that model)
#   flip       .env DAPP_TENANT_CLASSES_FHD, l1_entrypoint.sh / dgist.sh / cots_run.sh / cots.env defaults: 24,40,48,56,64,72,80 -> ...,80,88
#   fullstack  dgist.sh up c88_check, dgist.sh test per model (ra_stub, 10 req/s, 20 s), class-88 use in the tenant stats and the slot CSV,
#              Err.ind / FATAL / Order timeout / PUSCH-PDSCH infeasible + fallback, slot-map render, dgist.sh down
# Run as nvidia from /home/nvidia/_dgist (GPU idle, no L1 of either lab running). Log: runs/class88_night.log
set -u
D=/home/nvidia/_dgist; T=$D/aerial-cuda-accelerated-ran; H=$T/cuPHY-CP/dapp_hook; LOG=$D/runs/class88_night.log; LAD=24,40,48,56,64,72,80; LAD88=$LAD,88
log() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
gpu_idle() { ! docker ps --format '{{.Names}}' | grep -qE '^(dgist-l1|dgist-gnb|nv-cubb|oai-gnb-aerial_hsw|dgist-fhd-test)$' && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
step=${1:-all}; cd $D || exit 1

do_build() {
  gpu_idle || { log "build: GPU not idle ($(docker ps --format '{{.Names}}' | tr '\n' ' ')) - abort"; return 1; }
  log "build: fhd class 88 engines for n s m l x"
  bash scripts/build_yolo_engines.sh fhd 88 > runs/build_engines_fhd_88.log 2>&1; rc=$?
  ok=0; for m in n s m l x; do f=$H/models/fhd26$m/seg00_sm88.engine; [ -s $f ] && { ok=$((ok + 1)); log "  $f $(stat -c %s $f) B"; } || log "  !! missing $f"; done
  grep -E "==|!!|FAILED|rror" runs/build_engines_fhd_88.log | tail -12 | sed 's/^/  /' | tee -a $LOG
  log "build: rc $rc, $ok/5 engines"; [ $ok -ge 1 ]
}

do_standalone() {
  gpu_idle || { log "standalone: GPU not idle - abort"; return 1; }
  bash $H/sweep/fullstack/fhd_b3_test.sh cstart | tee -a $LOG
  for m in n s m l x; do
    [ -s $H/models/fhd26$m/seg00_sm88.engine ] || { log "standalone: 26$m has no 88 engine, skipped"; continue; }
    log "standalone 26$m: fixed 88-SM partition, 10 req/s x 15 s"
    MODELS=$m bash $H/sweep/fullstack/fhd_b3_test.sh tenant-start 88 fhd | tail -2 | sed 's/^/  /' | tee -a $LOG
    bash $H/sweep/fullstack/fhd_b3_test.sh run c88 /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/imgfhd 26$m 10 15   # container path (ra_stub runs inside) | tee -a $LOG
    bash $H/sweep/fullstack/fhd_b3_test.sh tenant-stop > /dev/null
    csv=$(ls -t $H/sweep/fullstack/fhd_b3_out/t_c88_26$m.csv 2>/dev/null | head -1); [ -n "$csv" ] && bash $H/sweep/fullstack/fhd_b3_test.sh summary $csv | sed 's/^/  /' | tee -a $LOG
    log "standalone 26$m: ladder $LAD88 in one process"
    out=$(DUR=120 bash $H/sweep/fullstack/tenant_classes_exp.sh c88$m $LAD88 $m 2>&1); echo "$out" | sed 's/^/  /' | tee -a $LOG
    if ! echo "$out" | grep -q "OK, port open"; then
      mv $H/models/fhd26$m/seg00_sm88.engine $H/models/fhd26$m/seg00_sm88.failed.engine; log "  !! 26$m: ladder with 88 FAILED -> engine set aside (seg00_sm88.failed.engine), 26$m stays at 80"
    fi
  done
  log "standalone all models: ladder $LAD88 in one process"
  DUR=120 bash $H/sweep/fullstack/tenant_classes_exp.sh c88all $LAD88 n,s,m,l,x 2>&1 | sed 's/^/  /' | tee -a $LOG
  bash $H/sweep/fullstack/fhd_b3_test.sh cstop | tee -a $LOG
}

do_flip() {
  n=0; for f in $H/models/fhd26{n,s,m,l,x}/seg00_sm88.engine; do [ -s $f ] && n=$((n + 1)); done
  [ $n -ge 1 ] || { log "flip: no 88 engine present - ladder unchanged"; return 1; }
  sed -i "s/^DAPP_TENANT_CLASSES_FHD=$LAD\$/DAPP_TENANT_CLASSES_FHD=$LAD88/" .env   # anchored at the line end: a second run must not append 88 again
  sed -i "s/\${DAPP_TENANT_CLASSES_FHD:-$LAD}/\${DAPP_TENANT_CLASSES_FHD:-$LAD88}/" scripts/l1_entrypoint.sh dgist.sh $H/cots/cots_run.sh
  sed -i "s/^DAPP_TENANT_CLASSES_FHD=$LAD /DAPP_TENANT_CLASSES_FHD=$LAD88 /" $H/cots/cots.env
  log "flip: FHD ladder -> $LAD88 ($n/5 models have the 88 engine; the tenant keeps a model at 80 when its 88 engine is missing)"
  grep -nE "CLASSES_FHD" .env scripts/l1_entrypoint.sh dgist.sh $H/cots/cots_run.sh $H/cots/cots.env | cut -c1-150 | sed 's/^/  /' | tee -a $LOG
  [ $n -eq 5 ] || log "flip: NOTE dgist.sh check wants the top class (88) of every model - models without it fail the check; put their .failed.engine back or lower the ladder"
}

do_fullstack() {
  gpu_idle || { log "fullstack: GPU not idle - abort"; return 1; }
  grep -q "^DAPP_INPUT=fhd" .env || { log "fullstack: .env DAPP_INPUT is not fhd - abort"; return 1; }
  log "fullstack: dgist.sh up c88_check"; ./dgist.sh up c88_check 2>&1 | tee -a $LOG | tail -3 > /dev/null
  docker ps --format '{{.Names}}' | grep -qx dgist-l1 || { log "fullstack: L1 did not come up"; return 1; }
  sleep 20
  for m in x l m s n; do log "fullstack: dgist.sh test 26$m 10 req/s 20 s"; ./dgist.sh test 26$m 10 20 2>&1 | grep -E "welcome|requests|E2E|26[nsmlx]:|fail" | cut -c1-200 | sed 's/^/  /' | tee -a $LOG; done
  log "fullstack: class use in the tenant stats (last 10-s line; 'c:n' = inferences started at class c)"
  docker logs dgist-l1 2>&1 | grep -E "classes" | grep -E "quiet|busy" | tail -1 | cut -c1-400 | sed 's/^/  /' | tee -a $LOG
  docker exec dgist-l1 bash -c 'grep -E "classes" /tmp/dapp_tenant.log | tail -1' | cut -c1-400 | sed 's/^/  /' | tee -a $LOG
  log "fullstack: dgist.sh down"; ./dgist.sh down 2>&1 | tail -3 | sed 's/^/  /' | tee -a $LOG
  R=runs/c88_check
  log "fullstack: L1 Err.ind $(grep -c 'Send Err.ind' $R/phy_warn.log) FATAL $(grep -c FATAL $R/phy_warn.log) Order-timeout $(grep -c 'Order kernel timeout' $R/phy_warn.log) | $(grep -E 'sizer plans' $R/dapp_tenant.log | tail -1 | grep -oE 'PUSCH slots.*' | cut -c1-200)"
  python3 - $R/dapp_tenant_slots.csv <<'EOF' 2>&1 | sed 's/^/  /' | tee -a $LOG
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1]))); n = len(rows)
if n and "gap_P" in rows[0]:
    gaps = collections.Counter((r["gap_P"], r["gap_N"]) for r in rows); runs = collections.Counter(r["run_cls"] for r in rows if r["run_cls"] != "0")
    free = collections.Counter(int(r["gap_P"]) - int(r["gap_N"]) - 8 for r in rows if int(r["gap_P"]) > 0)
    print("slots", n, "| gap (P,N) top:", gaps.most_common(4), "| free SM top:", free.most_common(4))
    print("slots with an inference running, by class:", sorted(runs.items(), key=lambda x: -x[1]))
    print("slots with free >= 88:", sum(v for k, v in free.items() if k >= 88), "| of them running at 88:", sum(1 for r in rows if r["run_cls"] == "88"))
else: print("slot CSV has no gap_P column (old L1 binary?) rows", n)
EOF
  log "fullstack: slot-map render check"; ./dgist.sh slotmap c88_check --from-sec 30 --sec 3 --fps 30 2>&1 | tail -2 | sed 's/^/  /' | tee -a $LOG
  ls -la $R/slotmap.mp4 2>/dev/null | awk '{print "  slotmap.mp4", $5, "B"}' | tee -a $LOG
}

log "==== class88_night.sh $step start"
case $step in
  build) do_build ;;
  standalone) do_standalone ;;
  flip) do_flip ;;
  fullstack) do_fullstack ;;
  all) do_build && do_standalone && do_flip && do_fullstack ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
rc=$?; log "==== class88_night.sh $step done rc $rc"; exit $rc
