#!/bin/bash
# ana.sh <run> : heavy analysis of one pilot run (phy.log greps -> rows -> ovl dump -> st / inv2). Needs quick.sh first (window file).
R=$1; O=$(cd "$(dirname "$0")" && pwd); W=${YP_OUT:-$O/out}; CSV=/data/dgist_tgen/log/$R/dapp_tenant_slots.csv
T=$O/../..
NI="nice -n 19 taskset -c 0-3,65-71"; read A B < $W/$R.window
{
OUT=$W TAILN=14 bash $T/ul_order/verify.sh $R $A $B
echo "== ovl"; OVL_DUMP=$W/ovl_$R.tsv ionice -c3 $NI python3 $O/ovl.py $W/rows_$R.tsv $CSV
echo "== st"; ionice -c3 $NI python3 $O/st.py $W/ovl_$R.tsv $CSV
echo "== inv2"; $NI python3 $O/inv2.py $W/ovl_$R.tsv
} > $W/ana_$R.txt 2>&1
echo "ana done $R $(date +%T) $(wc -l < $W/ana_$R.txt) lines"
