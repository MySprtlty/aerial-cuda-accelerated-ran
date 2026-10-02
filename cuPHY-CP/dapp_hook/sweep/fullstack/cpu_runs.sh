#!/bin/bash
# CPU-cause check for E and D (2026-10-01): hook off, default logging + tag 311 (needed for the enqueue timing),
# raw container logs not kept. 16 cells with mpstat + thread snapshot, then 8 and 12 cells, then 16 cells with the
# RU emulator core lists doubled (UL 11 -> 22, DL 10 -> 20; all inside isolcpus 4-64, nothing else uses 43-64).
S=$(cd "$(dirname "$0")" && pwd)
export KEEP_RAW=0
NCELLS=16 MPSTAT=1 bash $S/fullstack_run.sh cpu16 600000 "" "311" off
sleep 15
NCELLS=8  MPSTAT=1 bash $S/fullstack_run.sh cpu8  600000 "" "311" off
sleep 15
NCELLS=12 MPSTAT=1 bash $S/fullstack_run.sh cpu12 600000 "" "311" off
sleep 15
NCELLS=16 MPSTAT=1 RU_UL_CORES="20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41" \
  RU_DL_CORES="42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61" RU_LOW_CORE=62 RU_DPDK_CORE=63 \
  bash $S/fullstack_run.sh cpu16ru2 600000 "" "311" off
echo CPU_RUNS_DONE
