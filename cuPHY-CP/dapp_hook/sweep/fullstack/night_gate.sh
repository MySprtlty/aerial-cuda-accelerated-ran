#!/bin/bash
# night_gate.sh <start HH:MM> <deadline HH:MM> [log]
# Waits until <start> (host local time), then every 5 minutes checks that the GPU and host are free for the R0 ladder:
#   1. nvidia-smi compute apps: none, or only the nvidia-cuda-mps-server of our MPS daemon (left by run2 as usual)
#   2. no RAN process alive anywhere on the host (phy_main / cuphycontroller / ru_emulator / test_mac / nr-softmodem), zombies ignored
#   3. no container other than c_aerial_troy and icnl-aerial-26-1-kim, and no process of another container above 20 % CPU
#   4. MPS control daemon alive (nvidia-cuda-mps-control -d)
# exit 0 = gate passed (start the ladder), exit 2 = <deadline> reached without passing. Every check is appended to the log.
START=$1; DEADLINE=$2; LOG=${3:-/dev/stdout}
to_s() { local t=$1; echo $(( 10#${t%%:*} * 3600 + 10#${t##*:} * 60 )); }
now_s() { echo $(( 10#$(date +%H) * 3600 + 10#$(date +%M) * 60 + 10#$(date +%S) )); }
start_s=$(to_s $START); dead_s=$(to_s $DEADLINE)
[ $dead_s -le $start_s ] && dead_s=$((dead_s + 86400))          # deadline after midnight
n=$(now_s); [ $n -gt $start_s ] && [ $n -gt $dead_s ] && n=$((n - 86400))
wait_s=$((start_s - n)); [ $wait_s -gt 0 ] && { echo "$(date '+%F %T') gate: sleeping $wait_s s until $START" >> $LOG; sleep $wait_s; }
check() {
  local bad=""
  local apps; apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v '^$')
  local foreign; foreign=$(echo "$apps" | grep -v 'nvidia-cuda-mps-server' | grep -c .)
  [ "$foreign" -gt 0 ] && bad="$bad gpu-apps[$(echo "$apps" | tr '\n' ';')]"
  local ran; ran=$(ps -eo pid,stat,comm --no-headers | awk '$2 !~ /^Z/ && $3 ~ /^(phy_main|cuphycontroller|ru_emulator|test_mac|mac_main|nr-softmodem)/' | wc -l)
  [ "$ran" -gt 0 ] && bad="$bad ran-procs=$ran"
  local others; others=$(docker ps --format '{{.Names}}' | grep -v -x -e c_aerial_troy -e icnl-aerial-26-1-kim | tr '\n' ',')
  [ -n "$others" ] && bad="$bad containers[$others]"
  local busy=0 c
  for c in $(docker ps --format '{{.Names}}' | grep -v -x c_aerial_troy); do
    busy=$(( busy + $(docker top $c -o pid,pcpu,comm 2>/dev/null | awk 'NR>1 && $2+0 > 20' | wc -l) ))
  done
  [ "$busy" -gt 0 ] && bad="$bad other-container-busy=$busy"
  pgrep -f 'nvidia-cuda-mps-control -d' > /dev/null || bad="$bad mps-control-missing"
  echo "$(date '+%F %T') gate: ${bad:-PASS} | gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)" >> $LOG
  [ -z "$bad" ]
}
while :; do
  check && exit 0
  n=$(now_s); [ $n -lt $start_s ] && n=$((n + 86400))
  [ $n -ge $dead_s ] && { echo "$(date '+%F %T') gate: deadline $DEADLINE reached, giving up" >> $LOG; exit 2; }
  sleep 300
done
