#!/bin/bash
# Full-stack load ladder: the official peak patterns of the 16-cell loopback, one 2-minute run each, raw phy log parsed
# into per-slot stage times and then deleted.
#   ladder_runs.sh <prefix> "<pattern[:ncells]> ..."        e.g.  ladder_runs.sh ref "59c 59e 65c 65a"
#   inside MIG:  MIG_SM=1 MIG_UUID=MIG-... ladder_runs.sh mig "59c 59e 65c 65a 65a:20"
# Per run <prefix>_<pattern>_<n>c in prof/run61: phy_<name>.csv (parse_phylog.py), phy_warn_<name>.log (WRN/ERR and the
# overload strings), l2a_<name>.log (tag 311), run{1,2,3}_<name>.log (consoles), ru_<name>.log. ladder_report.py reads them.
S=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$S/../../../.." && pwd)
O=$R/prof/run61
PREFIX=$1
for item in $2; do
  pat=${item%%:*}; n=16; [[ $item == *:* ]] && n=${item##*:}
  name=${PREFIX}_${pat}_${n}c
  KEEP_RAW=1 MPSTAT=0 PATTERN=$pat NCELLS=$n bash $S/fullstack_run.sh $name ${LADDER_SLOTS:-240000} "" "311,223" async
  if [ -s $O/phy_$name.log ]; then
    python3 $S/parse_phylog.py $O/phy_$name.log $O/phy_$name.csv
    grep -F '[L2A.PROCESSING_TIMES]' $O/phy_$name.log > $O/l2a_$name.log
    grep -E ' (WRN|ERR|FATAL) |Send Err.ind|recovery|not responsive|No available|Successive Non-availability|Attempting PUSCH pipeline termination|Order kernel timeout|exit handler|pipeline failed' $O/phy_$name.log | head -2000000 > $O/phy_warn_$name.log
    rm -f $O/phy_$name.log $O/testmac_$name.log
  else
    echo "no phy log for $name"
  fi
  echo "== ladder $name parsed $(date +%T)"
done
echo LADDER_DONE_$PREFIX
