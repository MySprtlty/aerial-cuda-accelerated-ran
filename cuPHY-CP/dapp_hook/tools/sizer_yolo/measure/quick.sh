#!/bin/bash
# quick.sh <run> : cheap checks of one pilot run (console, tenant log, per-slot CSV; no phy.log read). Window = vcam start + 20 s .. vcam stop - 2 s.
R=$1; O=$(cd "$(dirname "$0")" && pwd); W=${YP_OUT:-$O/out}; U=$O/../../ul_order; RD=/home/nvidia/_dgist/runs/$R; CSV=/data/dgist_tgen/log/$R/dapp_tenant_slots.csv
NI="nice -n 19 taskset -c 0-3,65-71"
A=$(date -u -d "$(awk '/vcam_start_utc/{print $2}' $W/$R.times) 20 seconds" +%T); B=$(date -u -d "$(awk '/vcam_stop_utc/{print $2}' $W/$R.times) 2 seconds ago" +%T)
echo "== $R window $A .. $B UTC"; echo "$A $B" > $W/$R.window
/usr/bin/grep -a -m1 'dApp sizer YOLO term' $RD/l1_console.log | cut -c1-200
echo "-- runsum (console)"; $NI python3 $U/runsum.py $RD $A $B 2>&1 | cut -c1-260
echo "-- tenant log last sizer line"; /usr/bin/grep -a 'PUSCH slots' $RD/dapp_tenant.log | tail -1 | sed -E 's/^.*(agent ok)/\1/' | cut -c1-520
echo "-- infeas (10-s deltas, last 8)"; $NI python3 $U/infeas.py $RD/dapp_tenant.log | tail -8
echo "-- vcam"; python3 - $RD/vcam_status.json <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))['total']
print({k: t.get(k) for k in ('offered', 'sent', 'success', 'met', 'fail', 'skipped', 'rt_p50', 'rt_p90', 'rt_p99', 'uptime_s')}, [(c['cam'], c['rt_p50'], c['rt_p90'], c['rt_p99']) for c in json.load(open(sys.argv[1]))['cams']])
PY
echo "-- ymodel (CSV)"; ionice -c3 $NI python3 $O/ymodel.py $CSV $RD/vcam_status.json 2>&1 | cut -c1-330
