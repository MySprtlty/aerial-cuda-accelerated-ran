#!/usr/bin/env bash
set -Eeuo pipefail

UE_SUBNET="${UE_SUBNET:-10.45.0.0/16}"
TUN_IF="${TUN_IF:-ogstun}"

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: root 권한이 필요합니다. sudo $0 로 실행하세요." >&2
  exit 1
fi

if ! ip link show "$TUN_IF" >/dev/null 2>&1; then
  echo "ERROR: 인터페이스 '$TUN_IF'가 없습니다. Open5GS UPF/ogstun 상태를 확인하세요." >&2
  exit 1
fi

WAN_IF="${WAN_IF:-$(ip -4 route get 8.8.8.8 2>/dev/null | awk '
  {
    for (i = 1; i <= NF; i++) {
      if ($i == "dev") {
        print $(i+1)
        exit
      }
    }
  }'
)}"

echo "[1/4] IPv4 forwarding 활성화"
sysctl -w net.ipv4.ip_forward=1 >/dev/null

echo "[2/4] UE NAT 규칙 추가"
if ! iptables -t nat -C POSTROUTING \
    -s "$UE_SUBNET" -o "$WAN_IF" \
    -m comment --comment OPEN5GS-UE-NAT \
    -j MASQUERADE 2>/dev/null; then
  iptables -t nat -A POSTROUTING \
    -s "$UE_SUBNET" -o "$WAN_IF" \
    -m comment --comment OPEN5GS-UE-NAT \
    -j MASQUERADE
fi

echo "[3/4] UE → WAN forwarding 허용"
if ! iptables -C FORWARD \
    -i "$TUN_IF" -o "$WAN_IF" -s "$UE_SUBNET" \
    -m comment --comment OPEN5GS-UE-FWD-OUT \
    -j ACCEPT 2>/dev/null; then
  iptables -I FORWARD 1 \
    -i "$TUN_IF" -o "$WAN_IF" -s "$UE_SUBNET" \
    -m comment --comment OPEN5GS-UE-FWD-OUT \
    -j ACCEPT
fi

echo "[4/4] WAN → UE 응답 forwarding 허용"
if ! iptables -C FORWARD \
    -i "$WAN_IF" -o "$TUN_IF" -d "$UE_SUBNET" \
    -m conntrack --ctstate RELATED,ESTABLISHED \
    -m comment --comment OPEN5GS-UE-FWD-IN \
    -j ACCEPT 2>/dev/null; then
  iptables -I FORWARD 1 \
    -i "$WAN_IF" -o "$TUN_IF" -d "$UE_SUBNET" \
    -m conntrack --ctstate RELATED,ESTABLISHED \
    -m comment --comment OPEN5GS-UE-FWD-IN \
    -j ACCEPT
fi

echo
echo "Open5GS 라우팅 설정 완료"
echo "  UE subnet : $UE_SUBNET"
echo "  TUN       : $TUN_IF"
echo "  WAN       : $WAN_IF"
echo "  ip_forward: $(sysctl -n net.ipv4.ip_forward)"
echo
