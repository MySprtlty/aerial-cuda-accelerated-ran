#!/bin/bash
# xgrid_night.sh [order]  — §5: run the X blocks in order (default "10:peak 10:average 11:peak 11:average 9:peak 9:average"),
# stop on a block exit 2 (foreign run / precheck) or 3 (disk). env FS_OUT, MAXMIN, FILL pass through to xgrid_block.sh.
S=$(cd "$(dirname "$0")" && pwd); O=${FS_OUT:-$(cd "$S/../../../.." && pwd)/prof/run65}
for cfg in ${1:-10:peak 10:average 11:peak 11:average 9:peak 9:average}; do x=${cfg%%:*}; l=${cfg##*:}
  echo "$(date +%T) == xgrid night: block X=$x $l"
  FS_OUT=$O bash $S/xgrid_block.sh $x $l; rc=$?
  [ $rc -eq 2 ] || [ $rc -eq 3 ] && { echo "$(date +%T) !! block X=$x $l stopped (rc $rc) -> night stops"; exit $rc; }
done
echo "$(date +%T) XGRID_NIGHT_DONE"
