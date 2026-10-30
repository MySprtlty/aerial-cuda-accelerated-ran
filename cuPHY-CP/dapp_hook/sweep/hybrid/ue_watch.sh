#!/bin/bash
# ue_watch.sh start | stop | report [name] | tail
#   COTS-UE attach recorder for a running hybrid run (hybrid_run.sh start <name>): merges the attach-relevant lines of
#   three sources into one wall-clock (KST) timeline, prof/hybrid_runs/<name>/ue_timeline.log:
#     OAI  : docker logs -f -t oai-gnb-aerial-hybrid   (RA Msg1-4, RRCSetup*, RRCReconfiguration*, PDU SESSION, RA failed,
#            contention-resolution expiry, out-of-sync, asserts, the periodic per-UE stats block "UE RNTI ..." / "UE xxxx: ...")
#     L1   : /tmp/phy.log in c_aerial_troy  (preamble detection + RACH.indication, Send Err.ind, FATAL, PDU count mismatch,
#            early-UCI timeouts, the per-second "Cell <OC> | DL .. UL .. CRC" counters of the Benetel cell)
#     CORE : /var/log/open5gs/{amf,smf}.log via sudo -n  (gmm Registration, NGAP InitialUEMessage, SUPI/IMSI, PDU session, UE IPv4,
#            Deregistration, WARN/ERROR)
#   plus every 30 s a STATUS line (Err.ind / RRC / PDU counts, ping of every UE IP the SMF assigned).
#   'report' condenses the timeline into the attach phases with first-seen times and the failure signals, and prints the last
#   OAI stats block next to the researchers' Pixel reference (log/debug90/gnb.log 2026-10-12: RSRP -71..-74 dBm, PH 48 dB,
#   CQI 9 RI 2, UL MCS 24 SNR 19 dB BLER 0.06).
#   Timestamps: the L1 and docker lines carry UTC; every timeline line gets a KST prefix from this script.
set -u
H=$(cd "$(dirname "$0")" && pwd); R=$(cd "$H/../../../.." && pwd); OUT=$R/prof/hybrid_runs
cmd=${1:-report}; NAME=${2:-$(cat $OUT/current_run 2>/dev/null)}
[ -n "$NAME" ] || { echo "no run name (no $OUT/current_run): $0 $cmd <name>"; exit 1; }
RUN=$OUT/$NAME; TL=$RUN/ue_timeline.log; PIDF=$RUN/ue_watch.pid
OC=$(sed 's/ /\n/g' $OUT/current.env 2>/dev/null | awk -F= '/^OAI_CELL=/{print $2}'); OC=${OC:-9}

OAI_RE='initiating RA procedure|RA-Msg2|Msg3 reception|Msg3|Msg4|CBRA procedure|CFRA procedure|Received Ack of|RRCSetupRequest|RRCSetupComplete|RRCReconfigurationComplete|RRCReestablishment|SecurityMode|PDU SESSION|PDUSESSION|RA failed|Contention Resolution|out-of-sync|Assertion|Exiting|could not find UE context|release|UE RNTI [0-9a-f]{4} CU-UE-ID|UE [0-9a-f]{4}: (CQI|dlsch_rounds|ulsch_rounds|MAC:|LCID)|NGSetup|SCTP|FAILURE'
L1_RE="Yes Preambles detected|RACH_INDICATION|Send Err.ind|FATAL|PDU count mismatch|Early UCI IND task waiting|not responsive|recovery|Order kernel timeout|Cell +$OC \\| DL"
CORE_RE='gmm|Registration|InitialUE|imsi-|suci-|UE IPv4|PDU session|SUPI|Deregist|WARN|ERROR|Number of'

stamp() { while IFS= read -r line; do printf '%s %-4s %s\n' "$(date +%H:%M:%S.%3N)" "$1" "$line"; done; }

case "$cmd" in
start)
  [ -f $OUT/current_run ] || { echo "no hybrid run in progress ($OUT/current_run missing): start hybrid_run.sh first"; exit 1; }
  [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null && { echo "already watching $NAME (pid $(cat $PIDF))"; exit 0; }
  mkdir -p $RUN
  (
    echo "$(date +%H:%M:%S.%3N) INFO ue_watch start run=$NAME oai_cell=$OC (L1/docker stamps are UTC, this prefix is KST)"
    docker logs -f -t --since "$(date -u +%Y-%m-%dT%H:%M:%S)" oai-gnb-aerial-hybrid 2>&1 | grep --line-buffered -E "$OAI_RE" | stamp OAI &
    docker exec c_aerial_troy tail -F -n0 /tmp/phy.log 2>/dev/null | grep --line-buffered -E "$L1_RE" | stamp L1 &
    sudo -n tail -F -n0 /var/log/open5gs/amf.log /var/log/open5gs/smf.log 2>/dev/null | grep --line-buffered -E "$CORE_RE" | stamp CORE &
    while :; do
      sleep 30
      ips=$(grep -oE 'IPv4\[ *[0-9.]+' $TL 2>/dev/null | grep -oE '[0-9.]+$' | sort -u | tr '\n' ' ')
      p=""; for ip in $ips; do ping -c1 -W1 $ip >/dev/null 2>&1 && p="$p $ip:ok" || p="$p $ip:no-reply"; done
      e=$(docker exec c_aerial_troy grep -c 'Send Err.ind' /tmp/phy.log 2>/dev/null)
      printf '%s STAT L1 Err.ind %s | OAI RA-init %s RRC-setup-complete %s PDU-session %s RA-failed %s out-of-sync %s | preambles(L1) %s | UE ip ping:%s | disk free %s\n' \
        "$(date +%H:%M:%S.%3N)" "${e:-?}" "$(grep -c 'initiating RA procedure' $TL)" "$(grep -c 'RRCSetupComplete' $TL)" "$(grep -c 'PDU SESSION' $TL)" \
        "$(grep -cE 'RA failed|Contention Resolution timer expired' $TL)" "$(grep -v " STAT " $TL | grep -c out-of-sync)" "$(grep -c 'Yes Preambles detected' $TL)" "${p:- none yet}" "$(df -h --output=avail / | tail -1 | tr -d " ")"
    done
  ) >> $TL 2>&1 &
  echo $! > $PIDF
  echo "watching $NAME (pid $!): timeline $TL ; $0 tail to follow, $0 report to summarise, $0 stop to end"
  ;;
stop)
  if [ -f $PIDF ]; then pkill -P "$(cat $PIDF)" 2>/dev/null; kill "$(cat $PIDF)" 2>/dev/null; rm -f $PIDF; echo "stopped (timeline kept: $TL)"; else echo "not running"; fi
  pkill -f "docker logs -f -t .* oai-gnb-aerial-hybrid" 2>/dev/null; pkill -f "tail -F -n0 /tmp/phy.log" 2>/dev/null; sudo -n pkill -f "tail -F -n0 /var/log/open5gs" 2>/dev/null
  ;;
tail) tail -n 30 -f $TL ;;
report)
  [ -s $TL ] || { echo "no timeline for $NAME ($TL)"; exit 1; }
  first() { grep -v " STAT " $TL | grep -m1 -E "$1" | cut -c1-12; }
  cnt() { grep -v " STAT " $TL | grep -cE "$1"; }
  echo "== UE attach report: run $NAME, Benetel = L1 cell $OC, timeline $TL ($(wc -l < $TL) lines, $(head -1 $TL | cut -c1-12) .. $(tail -1 $TL | cut -c1-12) KST)"
  echo "-- phases (first seen / count)"
  printf '   %-44s %-13s %s\n' "1 L1 preamble detected (PRACH, cell $OC)"      "$(first 'Yes Preambles detected')"        "$(cnt 'Yes Preambles detected')"
  printf '   %-44s %-13s %s\n' "  L1 RACH.indication sent"                      "$(first 'RACH_INDICATION')"              "$(cnt 'RACH_INDICATION')"
  printf '   %-44s %-13s %s\n' "2 OAI RA started (Msg1 seen, RA-RNTI/TC-RNTI)" "$(first 'initiating RA procedure')"      "$(cnt 'initiating RA procedure')"
  printf '   %-44s %-13s %s\n' "3 OAI Msg2 (RAR) generated"                     "$(first 'RA-Msg2')"                      "$(cnt 'RA-Msg2')"
  printf '   %-44s %-13s %s\n' "4 OAI Msg3 received / CBRA succeeded"           "$(first 'CBRA procedure succeeded|Received Ack of')" "$(cnt 'CBRA procedure succeeded|Received Ack of')"
  printf '   %-44s %-13s %s\n' "5 RRCSetupRequest"                              "$(first 'RRCSetupRequest')"              "$(cnt 'RRCSetupRequest')"
  printf '   %-44s %-13s %s\n' "6 RRCSetupComplete (RRC_CONNECTED)"             "$(first 'RRCSetupComplete')"             "$(cnt 'RRCSetupComplete')"
  printf '   %-44s %-13s %s\n' "7 CORE registration (gmm)"                      "$(first 'CORE.*(Registration|InitialUE|suci-|imsi-)')" "$(cnt 'CORE.*(Registration|InitialUE|suci-|imsi-)')"
  printf '   %-44s %-13s %s\n' "8 RRCReconfigurationComplete"                   "$(first 'RRCReconfigurationComplete')"   "$(cnt 'RRCReconfigurationComplete')"
  printf '   %-44s %-13s %s\n' "9 PDU session (OAI)"                            "$(first 'PDU SESSION')"                  "$(cnt 'PDU SESSION')"
  printf '   %-44s %-13s %s\n' "  SMF UE IPv4 assigned"                         "$(first 'UE IPv4')"                      "$(grep -oE 'IPv4\[ *[0-9.]+' $TL | grep -oE '[0-9.]+$' | sort -u | tr '\n' ' ')"
  printf '   %-44s %-13s %s\n' "  UE ip ping ok (STAT)"                         "$(first ':ok')"                          "$(cnt ':ok')"
  echo "-- failure signals (count, first)"
  for k in 'RA failed' 'Contention Resolution timer expired' 'out-of-sync' 'Assertion|Exiting' 'Send Err.ind' 'FATAL' 'PDU count mismatch' 'Early UCI IND task waiting' 'CORE.*(WARN|ERROR)' 'could not find UE context' 'Reestablishment'; do
    n=$(cnt "$k"); [ "$n" -gt 0 ] && printf '   %-40s %4d  %s\n' "$k" "$n" "$(grep -m1 -E "$k" $TL | cut -c1-170)"
  done
  echo "-- Benetel cell $OC per-second UL (slots with UL traffic / CRC fails): $(grep -E "Cell +$OC \| DL" $TL | awk '{for(i=1;i<=NF;i++) if($i=="UL"){ if($(i+3)+0>0) s++; c+=$(i+6)+0 }} END{print s+0" samples with UL slots, CRC fails total "c+0}')"
  echo "-- last OAI stats block (reference Pixel 2026-10-12: in-sync, RSRP -71..-74, PH 48 dB, CQI 9 RI 2, UL MCS 24 SNR 19 dB BLER 0.06)"
  grep -E 'UE RNTI|UE [0-9a-f]{4}: (CQI|dlsch_rounds|ulsch_rounds|MAC:)' $TL | tail -5 | cut -c1-200 | sed 's/^/   /'
  echo "-- last STAT line"; grep ' STAT ' $TL | tail -1 | sed 's/^/   /'
  ;;
*) echo "usage: $0 start [name] | stop | report [name] | tail"; exit 2;;
esac
