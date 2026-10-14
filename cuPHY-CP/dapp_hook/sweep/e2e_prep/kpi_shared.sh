#!/bin/bash
# KPI experiment, "ours" arm (MIG off, MPS sharing): 16-cell full stack + the e2e inference server/client with TILED 4K
# requests (batch-8 engine: 6 native 1280 crops + global views), 10 req/s periodic with +-5 ms jitter, 3-minute runs.
# The tenant's model and cap are fixed per run (what the scheduler would hold at this load); no gate.
#   65a (heaviest official 16-cell 4T4R pattern): no tenant, 26n @ cap 30/50/100, 26s @ 30/50, 26x @ 30/100
#   59c (reference pattern):                      no tenant, 26n @ 30, 26x @ 30
# Results: prof/run61/{phy,client,server,...}_sh_<tenant>_<pattern>_16c.*; report: fullstack/fs_report.py
cd "$(dirname "$0")/.."
export LADDER_SLOTS=${LADDER_SLOTS:-360000}
run() {   # run <tag> <model|none> <cap> "<patterns>"
  if [ "$2" = none ]; then
    env -u TENANT_SPEC bash fullstack/ladder_runs.sh sh_none "$4"
  else
    TENANT_SPEC="$2=yolo$2_1280_b8_fp16_atp$3.engine" TENANT_CAP=$3 bash fullstack/ladder_runs.sh "$1" "$4"
  fi
}
run sh_none none 0   "65a"
run sh_n30  26n  30  "65a"
run sh_x30  26x  30  "65a"
run sh_s30  26s  30  "65a"
run sh_n50  26n  50  "65a"
run sh_x100 26x  100 "65a"
run sh_n100 26n  100 "65a"
run sh_s50  26s  50  "65a"
run sh_n30  26n  30  "59c"
run sh_x30  26x  30  "59c"
run sh_none none 0   "59c"
echo KPI_SHARED_DONE
