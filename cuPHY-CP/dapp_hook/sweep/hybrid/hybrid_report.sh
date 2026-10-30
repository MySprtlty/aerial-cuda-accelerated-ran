#!/bin/bash
# hybrid_report.sh <run-dir | run-name>   -  compact post-run summary of a hybrid_run.sh run (prof/hybrid_runs/<name>/)
#   L1 per-cell counters at mid-run, Err.ind breakdown, top WRN/ERR kinds, UL packet timing, RU emulator DL lateness / payload errors,
#   testMAC per-cell line, OAI gNB milestones. Read-only.
set -u
D=$1; [ -d "$D" ] || D=$(cd "$(dirname "$0")/../../../.." && pwd)/prof/hybrid_runs/$1
[ -d "$D" ] || { echo "no run dir $1"; exit 1; }
echo "== $D"
[ -f "$D/../$(basename $D).console" ] && grep -E '^== |L1 ready|cell [0-9]+:|!!|duration|exited' "$D/../$(basename $D).console" | cut -c1-200
echo "--- L1 per cell (mid-run tick)"
if [ -s "$D/l1_persec.log" ]; then
  mid=$(grep -o 'Tick [0-9]*' "$D/l1_persec.log" | awk '{print $2}' | sort -n | uniq | awk '{a[NR]=$1} END{print a[int(NR/2)+1]}')
  grep "Tick $mid\$" "$D/l1_persec.log" | sed -E 's/^.*\[SCF.PHY\] //' | cut -c1-110
fi
oc=$(grep -o 'OAI on nvipc[0-9]* -> cell [0-9]*' "$D/../$(basename $D).console" 2>/dev/null | grep -o '[0-9]*$' | head -1)
if [ -n "$oc" ] && [ -s "$D/l1_persec.log" ]; then
  # the OAI cell must show DL bytes (SIB1, ~0.04 Mbps without a UE): DL 0.00 with 50 Slots = DL_TTI not parsed (FAPI layout) -> no SSB/SIB1 on air
  n=$(grep -oE "Cell +$oc \| DL +[0-9.]+" "$D/l1_persec.log" | awk '{ if ($NF+0 > 0) n++ } END {print n+0}'); t=$(grep -cE "Cell +$oc \| DL" "$D/l1_persec.log")
  [ "$n" -gt 0 ] && echo "    OAI cell $oc: DL bytes seen in $n/$t samples (SIB1 scheduled)" || echo "!!  OAI cell $oc: DL 0.00 Mbps in all $t samples -> DL_TTI PDSCH not parsed (check 'PDU count mismatch DL_TTI=0' / build flags)"
fi
echo "--- Err.ind: $(grep -c 'Send Err.ind' "$D/phy_warn.log" 2>/dev/null) total; by msg/err code:"
grep 'Send Err.ind' "$D/phy_warn.log" 2>/dev/null | sed -E 's/.*msg_id=(0x[0-9a-f]+) err_code=(0x[0-9a-f]+).*/\1 \2/' | sort | uniq -c | sort -rn | head -5
echo "    by cell: $(grep 'Send Err.ind' "$D/phy_warn.log" 2>/dev/null | sed -E 's/.*cell_id=([0-9]+).*/\1/' | sort -n | uniq -c | awk '{printf "%s:%s ", $2, $1}')"
echo "--- top WRN/ERR kinds (L1)"
grep -E ' (WRN|ERR) ' "$D/phy_warn.log" 2>/dev/null | grep -v 'Send Err.ind' | sed -E 's/^[0-9:.]+ //; s/SFN [0-9.]+/SFN x/g; s/[0-9]+/N/g' | cut -c1-110 | sort | uniq -c | sort -rn | head -6
echo "--- L1 UL packet timing (early,ontime,late per cell; slots 7/8/9)"
grep 'UL_PACKET_SUMMARY\] Slot [789] ' "$D/run2_l1.log" 2>/dev/null | sed -E 's/^.*\] //' | cut -c1-160
echo "--- RU emulator: DL U late packets / late slots per cell, payload errors $(cat "$D/ru_payload_errors.txt" 2>/dev/null)"
grep -E 'DL U LATE PACKETS|DL U LATE SLOTS|DL U ON TIME PACKETS' "$D/ru_table.txt" 2>/dev/null | sed -E 's/^.*\[RU\] //' | cut -c1-160
echo "--- testMAC last per-cell line (cell 0) / errors"
grep -E '\[MAC.FAPI\] Cell  0 ' "$D/testmac_tail.log" 2>/dev/null | tail -1 | sed -E 's/^.*\[MAC.FAPI\] //' | cut -c1-200
grep -E ' (WRN|ERR) ' "$D/testmac_tail.log" 2>/dev/null | sed -E 's/^[0-9:.]+ //; s/[0-9]+/N/g' | cut -c1-100 | sort | uniq -c | sort -rn | head -3
echo "--- OAI gNB"
printf '    NGSetupResponse %s | RRCSetupComplete %s | PDU SESSION %s | RA failed %s | Assert %s | last: %s\n' \
  "$(grep -c 'NGSetupResponse' "$D/oai.log" 2>/dev/null)" "$(grep -cE 'RRCSetupComplete|RRC_CONNECTED' "$D/oai.log" 2>/dev/null)" "$(grep -c 'PDU SESSION' "$D/oai.log" 2>/dev/null)" \
  "$(grep -c 'RA failed' "$D/oai.log" 2>/dev/null)" "$(grep -c 'Assert' "$D/oai.log" 2>/dev/null)" "$(grep -v -E '^\s*$|E2 AGENT' "$D/oai.log" 2>/dev/null | tail -1 | cut -c1-90)"
grep -E 'Received NGAP|RA-|Msg3|Msg4|rnti|RRCSetup|Registration|PDU SESSION' "$D/oai.log" 2>/dev/null | grep -v -E 'Frame.Slot' | head -8 | cut -c1-160
