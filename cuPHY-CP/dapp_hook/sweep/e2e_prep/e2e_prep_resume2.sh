#!/bin/bash
# Follow-up to e2e_prep_resume.sh (still MIG off): after E2E_PREP_DONE
#   1. cuPHY channel p99 next to RATE-LIMITED 26x-1280 streams (10/20 req/s batch 1 at cap 100; tiled batch 8 at cap 100/50)
#   2. GH200 segment with tiled requests (batch-8 engines), caps 100 / 50 / 30
#   3. analysis of everything measured with MIG off
# Prints E2E_PREP2_DONE; after that the GPU can be switched to MIG.
set -u
cd "$(dirname "$0")/.."
LOG=runs/e2e_prep_resume.log
PTS=runs/c16_points.yaml
log() { echo "$(date +%F_%T) $*" | tee -a $LOG; }
until grep -q "E2E_PREP_DONE" $LOG 2>/dev/null; do pgrep -f "e2e_prep_resume.sh" >/dev/null || { grep -q "E2E_PREP_DONE" $LOG || { log "stage-1 driver gone without E2E_PREP_DONE"; exit 1; }; }; sleep 30; done
log "== stage 9: rate-limited tenants (channel p99 at a realistic duty cycle)"
for pr in c16_ratelimit_1280 c16_ratelimit_1280_b8; do
  python3 sweep.py --profile $pr --points-file $PTS --out runs/$pr 2>&1 | grep -E "^===|-> |^done|FAILED" | tee -a $LOG
  for pt in c16_L1 c16_L4; do      # the no-tenant reference of the same day, so plot.py has its isolation baseline in the same directory
    [ -e runs/$pr/${pt}__none0 ] || cp -r runs/c16_capgrid_1280/${pt}__none0 runs/$pr/
  done
done
log "== stage 10: GH200 segment, tiled requests (batch 8)"
BATCH=8 CAPS="100 50 30" RATES_X="2 5 10 15" RATES_S="10 20" RATES_N="" bash ../e2e/gh200_segment.sh results/e2e_prep/gh200_segment_tiled 2>&1 | grep -E "==|offered|GH200_SEGMENT_DONE|failed|patterns" | tee -a $LOG
log "== analysis"
docker exec -u root c_aerial_troy bash -lc 'cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/sweep && for r in c16_capgrid_1280 c16_ratelimit_1280 c16_ratelimit_1280_b8; do python3 plot.py --runs runs/$r --lo c16_L1 --hi c16_L4 2>&1 | grep -v "legend\|UserWarning" | tail -1; done;
  python3 tenant_latency.py --runs runs/c16_capgrid_1280 --out results/c16_capgrid_1280 --slot-ms 0.5 2>&1 | tail -1;
  python3 openloop_report.py --runs runs/c16_openloop_1280 --out results/c16_openloop_1280 2>&1 | tail -1' | tee -a $LOG
echo "E2E_PREP2_DONE $(date +%T)" | tee -a $LOG
