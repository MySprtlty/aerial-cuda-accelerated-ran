#!/bin/bash
# msg4_trigger.sh <run-name>: when the OAI gNB logs "Received Ack of Msg4" (UE connected), send SIGPWR to the L1 three times
# (0.3 s, 1.0 s, 2.0 s later). With prof/hybrid_runs/gdbcmds_sig the L1 runs under gdb, which stops on SIGPWR, prints every
# thread's backtrace and continues -> snapshots of the UL worker that blocks after the UE connects (2026-10-30 diagnosis).
set -u
H=$(cd "$(dirname "$0")" && pwd); R=$(cd "$H/../../../.." && pwd); OUT=$R/prof/hybrid_runs/${1:?run name}; mkdir -p $OUT
echo "$(date +%T.%N) waiting for Msg4 ack in the OAI log" > $OUT/msg4_trigger.txt
docker logs -f oai-gnb-aerial-hybrid 2>&1 | grep -m1 -E "${TRIG_RE:-Received Ack of Msg4}" >> $OUT/msg4_trigger.txt
P=$(pgrep -f "^/opt/nvidia/cuBB/.*/cuphycontroller_scf" | head -1); [ -n "$P" ] || P=$(pgrep -x cuphycontroller | head -1); echo "$(date +%T.%N) Msg4 ack seen, L1 pid $P" >> $OUT/msg4_trigger.txt; ps -eo pid,comm,args | grep -E "cuphycontroller" | grep -v grep | cut -c1-120 >> $OUT/msg4_trigger.txt
for i in $(seq 1 ${NSNAP:-3}); do sleep ${SNAP_GAP:-0.02}; sudo -n kill -PWR $P 2>/dev/null && echo "$(date +%T.%N) SIGPWR $i sent" >> $OUT/msg4_trigger.txt || { echo "$(date +%T.%N) kill failed (process gone?)" >> $OUT/msg4_trigger.txt; break; }; done
