#!/bin/bash
# cots_core.sh up | down | status
#   Open5GS core network of this host (native systemd services, the researchers' setup: AMF 127.0.0.5, PLMN 999/70, TAC 1, UE subnet 10.45.0.0/16)
#   up     = start the 18 open5gs services + NAT routing for the UE subnet (core/routing.sh, the researchers' script; needs sudo)
#   down   = remove the NAT rules (core/routing_restore.sh) and stop the services
#   status = active units / ogstun address / NAT rules
D=$(cd "$(dirname "$0")" && pwd); . $D/cots.env
UNITS="open5gs-amfd open5gs-ausfd open5gs-bsfd open5gs-hssd open5gs-mmed open5gs-nrfd open5gs-nssfd open5gs-pcfd open5gs-pcrfd open5gs-scpd open5gs-seppd open5gs-sgwcd open5gs-sgwud open5gs-smfd open5gs-udmd open5gs-udrd open5gs-upfd open5gs-webui"
case "${1:-status}" in
up)     sudo systemctl start $UNITS || exit 1; for i in $(seq 1 15); do ip -br addr show ogstun 2>/dev/null | grep -q '10.45.0.1' && break; sleep 1; done
        sudo UE_SUBNET=$UE_SUBNET bash $D/core/routing.sh || echo "!! routing.sh failed (UE internet access needs the NAT rules)"; "$0" status ;;
down)   sudo bash $D/core/routing_restore.sh 2>/dev/null || true; sudo systemctl stop $UNITS; "$0" status ;;
status) act=$(systemctl is-active $UNITS 2>/dev/null | grep -c '^active'); if sudo -n true 2>/dev/null; then nat=$(sudo -n iptables -t nat -S POSTROUTING 2>/dev/null | grep -c OPEN5GS); else nat="? (sudo needs a password: sudo iptables -t nat -S POSTROUTING | grep -c OPEN5GS)"; fi
        echo "Open5GS: $act/18 units active | ogstun: $(ip -br addr show ogstun 2>/dev/null | awk '{print $2, $3}' | tr '\n' ' ') | UE NAT rules: ${nat:-?} | WebUI http://<host>:9999 (subscribers)" ;;
*)      echo "usage: $0 up | down | status"; exit 2 ;;
esac
