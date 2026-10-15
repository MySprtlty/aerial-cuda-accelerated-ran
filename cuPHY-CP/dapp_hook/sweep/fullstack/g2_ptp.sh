#!/bin/bash
# g2_ptp.sh — G2: PTP-core effect on the no-tenant baseline. Four 60 s runs in the order core 20 -> 64 -> 20 -> 64
# (ptp_move.sh moves ptp4l / phc2sys with the passwordless taskset rule, one minute of phc2sys offset is recorded after
# every move), restored to core 20 at the end. Output rows: prof/run63 G2_PTP20a / G2_PTP64a / G2_PTP20b / G2_PTP64b.
set -u
S=$(cd "$(dirname "$0")" && pwd); R=$(cd "$S/../../../.." && pwd); O=${FS_OUT:-$R/prof/run63}; cd $R
offset() { echo "-- $(date +%T) phc2sys offset, 60 s after the move ($1):"; sudo -n journalctl -u phc2sys -f -n 0 --no-pager 2>/dev/null | head -60 | sed 's/.*phc2sys\[[0-9]*\]: //' | awk '{for(i=1;i<=NF;i++){if($i=="rms")r+=$(i+1); if($i=="max"){m+=$(i+1); if($(i+1)>mx)mx=$(i+1)}}; n++} END{printf "   n=%d mean rms %.1f ns, mean max %.1f ns, max max %d ns\n", n, r/n, m/n, mx}'; }
for step in 20a 64a 20b 64b; do
  core=${step%?}
  bash $S/ptp_move.sh $core $O/ptp_move_G2.log | tail -3; offset $step | tee -a $O/ptp_move_G2.log
  printf 'PTP%s pattern=59c slots=120000 tags=311,223\n' $step > $O/ladder_G2_$step.txt
  FS_OUT=$O bash $S/r0_ladder.sh $O/ladder_G2_$step.txt G2 > $O/driver_G2_$step.log 2>&1
  grep -E '^== .* done|!!|ptp:' $O/driver_G2_$step.log | tail -2
done
bash $S/ptp_move.sh restore $O/ptp_move_G2.log | tail -3; offset "restore" | tee -a $O/ptp_move_G2.log
