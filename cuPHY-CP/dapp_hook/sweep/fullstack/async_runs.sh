#!/bin/bash
# Async-export verification (2026-10-03): same binary, 16 cells 59c, hook off / inline (full) / async (full),
# 5 min each, then a 10 min async soak. Raw logs not kept; mpstat + thread snapshot in every run.
S=$(cd "$(dirname "$0")" && pwd)
export KEEP_RAW=0 MPSTAT=1 NCELLS=16
bash $S/fullstack_run.sh as_off    600000 "" "311" off
sleep 15
bash $S/fullstack_run.sh as_inline 600000 "" "311" full
sleep 15
bash $S/fullstack_run.sh as_async  600000 "" "311" async
echo ASYNC_TIMING_DONE
sleep 15
bash $S/fullstack_run.sh as_soak  1200000 "" "311" async
echo ASYNC_RUNS_DONE
