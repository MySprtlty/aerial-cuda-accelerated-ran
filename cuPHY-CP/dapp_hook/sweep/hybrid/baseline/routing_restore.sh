#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: root 권한이 필요합니다. sudo $0 로 실행하세요." >&2
  exit 1
fi
UE_SUBNET="${UE_SUBNET:-10.45.0.0/16}"
TUN_IF="${TUN_IF:-ogstun}"

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

delete_all_matching() {
  local table="$1"
  shift
  while iptables -t "$table" -C "$@" 2>/dev/null; do
    iptables -t "$table" -D "$@"
  done
}

echo "[1/3] Open5GS NAT 규칙 삭제"
delete_all_matching nat POSTROUTING \
  -s "$UE_SUBNET" -o "$WAN_IF" \
  -m comment --comment OPEN5GS-UE-NAT \
  -j MASQUERADE

echo "[2/3] UE → WAN forwarding 규칙 삭제"
delete_all_matching filter FORWARD \
  -i "$TUN_IF" -o "$WAN_IF" -s "$UE_SUBNET" \
  -m comment --comment OPEN5GS-UE-FWD-OUT \
  -j ACCEPT

echo "[3/3] WAN → UE 응답 forwarding 규칙 삭제"
delete_all_matching filter FORWARD \
  -i "$WAN_IF" -o "$TUN_IF" -d "$UE_SUBNET" \
  -m conntrack --ctstate RELATED,ESTABLISHED \
  -m comment --comment OPEN5GS-UE-FWD-IN \
  -j ACCEPT

echo "Open5GS 라우팅 설정 복원 완료"
