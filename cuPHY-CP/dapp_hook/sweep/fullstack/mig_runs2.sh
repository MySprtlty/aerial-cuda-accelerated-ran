#!/bin/bash
# Follow-up inside MIG 4g: 16 cells 2 min with raw logs kept and testMAC validation (408 MAC.VALD) + L1 PUSCH (223)
# at INFO, to explain the testMAC UL-byte deficit (181.95 vs 196.70 Mbps while the L1 reports CRC 0).
S=$(cd "$(dirname "$0")" && pwd)
until grep -q "MIG_RUNS_DONE" $S/../../../../prof/run61/mig_runs_driver.log; do sleep 15; done
sleep 15
export KEEP_RAW=1 MPSTAT=0 MIG_SM=1 MIG_UUID=MIG-1cd6d0a5-f26d-5e8b-818a-8cd6e0bd3294
NCELLS=16 bash $S/fullstack_run.sh mig16_vald 240000 "" "311,408,223" async
echo MIG_RUNS2_DONE
