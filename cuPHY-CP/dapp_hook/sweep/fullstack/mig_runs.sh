#!/bin/bash
# Full stack inside MIG 4g.48gb (64 SMs; NVIDIA's MIG mps_sm_* values): 16-cell pilot 2 min, 16 cells 5 min,
# 8 cells 5 min (NVIDIA's supported point), 12 cells 5 min. Hook async (default), no nsys, raw logs not kept.
S=$(cd "$(dirname "$0")" && pwd)
export KEEP_RAW=0 MPSTAT=1 MIG_SM=1 MIG_UUID=MIG-1cd6d0a5-f26d-5e8b-818a-8cd6e0bd3294
NCELLS=16 bash $S/fullstack_run.sh mig16_pilot 240000 "" "311" async
echo MIG_PILOT_DONE
sleep 15
NCELLS=16 bash $S/fullstack_run.sh mig16 600000 "" "311" async
sleep 15
NCELLS=8  bash $S/fullstack_run.sh mig8  600000 "" "311" async
sleep 15
NCELLS=12 bash $S/fullstack_run.sh mig12 600000 "" "311" async
echo MIG_RUNS_DONE
