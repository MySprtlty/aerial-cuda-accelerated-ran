#!/bin/bash
# E: hook cost on the L2 adapter path (5 min each, tag 311 = per-slot L2A timing line only), then D confirmation (10 min).
S=$(cd "$(dirname "$0")" && pwd)
bash $S/fullstack_run.sh e_on  600000 "" "311" full
sleep 15
bash $S/fullstack_run.sh e_off 600000 "" "311" off
sleep 15
bash $S/fullstack_run.sh e_sum 600000 "" "311" summary
echo E_RUNS_DONE
sleep 15
RU_HIST=1 bash $S/fullstack_run.sh d 1200000 "" "216,240,311,503" full
echo D_RUN_DONE
