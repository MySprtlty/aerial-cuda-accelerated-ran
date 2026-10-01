#!/bin/bash
S=$(cd "$(dirname "$0")" && pwd)
TAGS="216,240,224,311"      # DRV.MAP_DL, DRV.MAP_DL_VERBOSE, DRV.MAP_UL, L2A.PROCESSING_TIMES at INFO
bash $S/fullstack_run.sh a 1200000 "" "$TAGS"
sleep 20
bash $S/fullstack_run.sh b 1200000 "8, 9, 10, 42, 43, 44" "$TAGS"
echo D_RUNS_DONE
