#!/bin/bash
# ptp_move.sh <core>|restore|status [log]  — E1 A3: move ptp4l (3420) and phc2sys (6442, CLOCK_REALTIME sync) off RU core 20 with
# `sudo -n /usr/bin/taskset -a -pc <core> <pid>` (passwordless sudoers rule for taskset), restore = back to core 20.
# Prints the two PIDs' Cpus_allowed_list and the last phc2sys offset lines (sudo -n journalctl -u phc2sys) before and after.
P1=$(pgrep -x ptp4l | head -1); P2=$(for p in $(pgrep -x phc2sys); do grep -q aerial00 /proc/$p/cmdline && echo $p; done | head -1); LOG=${2:-/dev/null}
show() { local p; for p in $P1 $P2; do echo "   $p $(cat /proc/$p/comm) allowed=$(awk '/^Cpus_allowed_list:/{print $2}' /proc/$p/status) last_cpu=$(awk '{print $39}' /proc/$p/stat)"; done; echo "   phc2sys: $(sudo -n journalctl -u phc2sys -n 1 --no-pager 2>/dev/null | tail -1 | sed 's/.*phc2sys\[[0-9]*\]: //')"; }
case ${1:-status} in
  status) echo "-- $(date '+%F %T') PTP status" | tee -a $LOG; show | tee -a $LOG ;;
  restore|[0-9]*)
    core=$1; [ "$core" = restore ] && core=20
    { echo "-- $(date '+%F %T') PTP -> core $core: before"; show; } | tee -a $LOG
    sudo -n /usr/bin/taskset -a -pc $core $P1 | tee -a $LOG; sudo -n /usr/bin/taskset -a -pc $core $P2 | tee -a $LOG
    { echo "-- after"; show; } | tee -a $LOG ;;
  *) echo "usage: $0 <core>|restore|status [log]"; exit 1 ;;
esac
