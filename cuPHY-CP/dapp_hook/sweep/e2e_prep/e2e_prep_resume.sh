#!/bin/bash
# E2E scenario preparation (MIG off), resumed after the unclean host reboot of 2026-10-05 00:46 that hit the
# first run of yolo26x-1280 cap 100 together with the 16-cell L4 bench. Cause unknown, so the order goes from the
# lightest combination to the heaviest and the suspect combination comes late; every stage is logged, and
# prof/crashwatch/{sensors.csv,dmesg_follow.log} are synced to disk every line in case it happens again.
#   stages: none L1 -> 26n (cap 30, 100) -> 26s (30, 100) -> open loop 26s -> 26x (30, 100) -> open loop 26x
#           -> GH200 segment (real pipeline) -> cap 50 for all three -> analysis
# sweep.py skips runs whose status.yaml says ok, so the driver can simply be started again after an interruption.
set -u
cd "$(dirname "$0")/.."
LOG=runs/e2e_prep_resume.log
PTS=runs/c16_points.yaml
log() { echo "$(date +%F_%T) $*" | tee -a $LOG; }
grid() { log "-- capgrid --only $1"; python3 sweep.py --profile c16_capgrid_1280 --points-file $PTS --out runs/c16_capgrid_1280 --only "$1" 2>&1 | grep -E "^===|-> |^done|FAILED" | tee -a $LOG; }
olp()  { log "-- openloop --only $1"; python3 sweep.py --profile c16_openloop_1280 --points-file $PTS --out runs/c16_openloop_1280 --only "$1" 2>&1 | grep -E "^===|-> |^done|FAILED" | tee -a $LOG; }

log "== stage 1: baseline without tenant"
grid "c16_L1__none0"; grid "c16_L4__none0"
log "== stage 2: yolo26n1280"
grid "yolo26n1280_cap030"; grid "yolo26n1280_cap100"
log "== stage 3: yolo26s1280"
grid "yolo26s1280_cap030"; grid "yolo26s1280_cap100"
log "== stage 4: open loop yolo26s1280"
olp "ol26s1280"
log "== stage 5: yolo26x1280 (cap 30 first, cap 100 last = the combination that ran when the host went down)"
grid "yolo26x1280_cap030"; grid "yolo26x1280_cap100"
log "== stage 6: open loop yolo26x1280"
olp "ol26x1280"
log "== stage 7: GH200 segment with the real pipeline"
bash ../e2e/gh200_segment.sh results/e2e_prep/gh200_segment 2>&1 | grep -E "==|offered|GH200_SEGMENT_DONE|failed|patterns" | tee -a $LOG
echo "E2E_PREP_ESSENTIAL_DONE $(date +%T)" | tee -a $LOG
log "== stage 8: cap 50"
grid "yolo26n1280_cap050"; grid "yolo26s1280_cap050"; grid "yolo26x1280_cap050"
log "== analysis (inside the container: matplotlib lives there)"
docker exec -u root c_aerial_troy bash -lc 'cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/sweep && python3 plot.py --runs runs/c16_capgrid_1280 --lo c16_L1 --hi c16_L4 2>&1 | grep -v "legend\|UserWarning" | tail -2;
  python3 tenant_latency.py --runs runs/c16_capgrid_1280 --out results/c16_capgrid_1280 --slot-ms 0.5 2>&1 | tail -2;
  python3 openloop_report.py --runs runs/c16_openloop_1280 --out results/c16_openloop_1280 2>&1 | tail -2' | tee -a $LOG
echo "E2E_PREP_DONE $(date +%T)" | tee -a $LOG
