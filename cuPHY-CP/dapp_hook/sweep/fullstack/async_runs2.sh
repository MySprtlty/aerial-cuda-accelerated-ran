#!/bin/bash
# Replicates for the D comparison (run-to-run spread of the RU late share): off / async alternating, 5 min each.
S=$(cd "$(dirname "$0")" && pwd)
export KEEP_RAW=0 MPSTAT=1 NCELLS=16
for i in 2 3; do
  bash $S/fullstack_run.sh as_off$i   600000 "" "311" off
  sleep 15
  bash $S/fullstack_run.sh as_async$i 600000 "" "311" async
  sleep 15
done
echo ASYNC_REPL_DONE
