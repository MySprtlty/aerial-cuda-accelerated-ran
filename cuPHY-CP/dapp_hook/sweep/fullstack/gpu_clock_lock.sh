#!/bin/bash
# gpu_clock_lock.sh lock|unlock|status  — E1 A2: lock the GH200 SM / memory clocks at their maximum (1980 / 2619 MHz) as root
# inside the privileged c_aerial_troy container (host sudo needs a password). unlock = nvidia-smi -rgc / -rmc.
C=c_aerial_troy
case ${1:-status} in
  lock)   docker exec -u root $C bash -c 'nvidia-smi -lgc 1980,1980 && nvidia-smi -lmc 2619,2619' ;;
  unlock) docker exec -u root $C bash -c 'nvidia-smi -rgc; nvidia-smi -rmc' ;;
  status) ;;
  *) echo "usage: $0 lock|unlock|status"; exit 1 ;;
esac
rc=$?
echo "-- $(date '+%F %T') clocks: $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.applications.graphics,clocks.applications.memory,clocks_event_reasons.active --format=csv,noheader)"
nvidia-smi -q -d CLOCK | grep -A3 -i 'Locked Clocks\|Applications Clocks$' | head -8 | sed 's/^/   /'
exit $rc
