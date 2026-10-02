#!/bin/bash
# Default-mode check after making async export the default (2026-10-03): 60 s each, 16 cells.
#   keep   = the yaml line as committed (async_export: 1, export_cpu: 43)  -> must start the export thread
#   legacy = a pre-async block without the new keys                        -> must warn and export inline
S=$(cd "$(dirname "$0")" && pwd)
export KEEP_RAW=0 MPSTAT=0 NCELLS=16
bash $S/fullstack_run.sh def_keep   120000 "" "311" keep
sleep 15
bash $S/fullstack_run.sh def_legacy 120000 "" "311" legacy
echo DEFAULT_CHECK_DONE
