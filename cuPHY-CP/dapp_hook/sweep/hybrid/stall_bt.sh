#!/bin/bash
# stall_bt.sh <run-name> [trigger-regex]
#   Waits (in the background) for the first L1 log line matching the trigger (default: the first UL-object exhaustion or
#   early-UCI ULC timeout) and immediately attaches gdb inside c_aerial_troy to the running cuphycontroller_scf to dump
#   every thread's backtrace to prof/hybrid_runs/<run>/stall_bt.txt. Used to catch a UL worker that is blocked without a
#   timeout (2026-10-30: the PRACH task of the first slot x.17 after a COTS UE connects never finishes).
#   Run it right after hybrid_run.sh prints "L1 ready"; the L1 must NOT already be under gdb (no L1_EXTRA_ARGS gdb_script).
set -u
H=$(cd "$(dirname "$0")" && pwd); R=$(cd "$H/../../../.." && pwd)
NAME=${1:?run name}; TRIG=${2:-"No available Aggr PUSCH objects|timeout waiting for ULC Tasks|Successive Non-availability"}
OUT=$R/prof/hybrid_runs/$NAME; mkdir -p $OUT
echo "$(date +%T) stall_bt: waiting for /$TRIG/ in the L1 log" > $OUT/stall_bt.txt
docker exec c_aerial_troy bash -c "tail -F -n0 /tmp/phy.log 2>/dev/null | grep -m1 -E '$TRIG'" >> $OUT/stall_bt.txt 2>&1
echo "$(date +%T) stall_bt: trigger seen, attaching gdb" >> $OUT/stall_bt.txt
docker exec -u root c_aerial_troy bash -c 'P=$(pgrep -f "cuphycontroller_scf F08" | head -1); echo "pid $P"; gdb -q -batch -p $P -ex "set pagination off" -ex "thread apply all bt 25" 2>&1' >> $OUT/stall_bt.txt
echo "$(date +%T) stall_bt: done ($(grep -c '^Thread' $OUT/stall_bt.txt) threads)" >> $OUT/stall_bt.txt
