#!/bin/bash
# outputs: $YP_OUT (default ./out); run as the account that may sudo to nvidia; the stack must be down and the GPU free
# pilot.sh <run name> <DAPP_SIZER_YOLO_US value | off> [vcam seconds]  - one pilot run of the _dgist stack (Ours, generator load from .env) with 2 x 15 fps virtual cameras
set -u
NAME=$1; VAL=$2; SEC=${3:-330}; [ "$VAL" = off ] && VAL=""
O=$(cd "$(dirname "$0")" && pwd); W=${YP_OUT:-$O/out}; mkdir -p $W; G=/home/nvidia/_dgist
nv() { sudo -n -u nvidia -H bash -c "cd $G && $1" < /dev/null; }
slp() { python3 -c "import time; time.sleep($1)"; }
say() { echo "$(date +%T) $*"; }
sudo -n test -e $G/runs/$NAME && { say "run name exists: $NAME"; exit 1; }
docker ps --format '{{.Names}}' | /usr/bin/grep -qE '^(dgist-l1|dgist-gnb|nv-cubb|oai-gnb-aerial_hsw)$' && { say "a RAN stack is up"; exit 1; }
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { say "GPU busy"; exit 1; }
sudo -n -u nvidia -H sed -i "s|^DAPP_SIZER_YOLO_US=.*|DAPP_SIZER_YOLO_US=$VAL|" $G/.env
say ".env $(sudo -n /usr/bin/grep -n '^DAPP_SIZER_YOLO' $G/.env | tr '\n' ' ')"
: > $W/$NAME.times
say "up $NAME"; nv "./all.sh up $NAME" > $W/$NAME.up.log 2>&1; say "up rc=$?"
docker ps --format '{{.Names}} {{.Status}}' | /usr/bin/grep -E '^dgist-(l1|gnb) ' | tr '\n' ';'; echo
slp 50
nv "./all.sh status" > $W/$NAME.status0.txt 2>&1; chmod 600 $W/$NAME.status0.txt
sed -n '2p;9,10p' $W/$NAME.status0.txt | cut -c1-200
echo "vcam_start_utc $(date -u +%T)" >> $W/$NAME.times
nv "./vcam.sh start 2 15" > $W/$NAME.vcam.log 2>&1; say "vcam started: $(tail -1 $W/$NAME.vcam.log | cut -c1-150)"
slp $SEC
echo "vcam_stop_utc $(date -u +%T)" >> $W/$NAME.times
nv "./vcam.sh status" > $W/$NAME.vcam_status.txt 2>&1
nv "cp runs/vcam/status.json runs/$NAME/vcam_status.json"
nv "./vcam.sh stop" > /dev/null 2>&1; say "vcam stopped"
nv "./all.sh status" > $W/$NAME.status1.txt 2>&1; chmod 600 $W/$NAME.status1.txt
sed -n '2p;9,10p' $W/$NAME.status1.txt | cut -c1-200
nv "./all.sh down" > $W/$NAME.down.log 2>&1; say "down rc=$?"
/usr/bin/grep -a -E 'Err.ind|late slots' $W/$NAME.down.log | cut -c1-230
cat $W/$NAME.times
