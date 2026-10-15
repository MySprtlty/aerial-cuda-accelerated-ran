#!/bin/bash
# e6_parse_cache.sh patch|revert|status — E6: enable testMAC's parsed-vector cache for PDSCH / PUSCH / PUCCH / PRACH
# (launch_pattern.cpp cache_channels block: uncomment the four channels), back up the test_mac binary and rebuild only the
# test_mac target in the container's build.aarch64 (ninja; 13 testMAC objects + link, no other target). revert = source back
# (git checkout) + the backed-up binary. Other binaries' mtimes are listed before/after so that "nothing else changed" is checkable.
R=$(cd "$(dirname "$0")/../../../.." && pwd); C=c_aerial_troy
SRC=cuPHY-CP/testMAC/testMAC/launch_pattern.cpp
BIN=/opt/nvidia/cuBB/build.aarch64/cuPHY-CP/testMAC/testMAC/test_mac
BAK=$BIN.bak_e6
others() { docker exec $C bash -c 'stat -c "%y %n" /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/ru-emulator/ru_emulator/ru_emulator /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/testMAC/testMAC/test_mac 2>&1' | cut -c1-120; }
cd $R
case ${1:-status} in
  patch)
    git diff --quiet -- $SRC || { echo "!! $SRC already modified"; exit 1; }
    python3 - <<'PY'
import re
p = "cuPHY-CP/testMAC/testMAC/launch_pattern.cpp"
s = open(p).read()
old = s[s.index("            bool cache_channels = ("):s.index("                                );", s.index("            bool cache_channels = ("))]
new = old
for ch in ("PDSCH", "PUSCH", "PUCCH", "PRACH"):
    new = re.sub(r"// (ch == channel_type_t::%s \|\|)" % ch, r"\1", new)
assert new != old and new.count("//") == 0, "pattern not found"
open(p, "w").write(s.replace(old, new, 1))
print("patched cache_channels: PDSCH, PUSCH, PUCCH, PRACH cached")
PY
    git diff -- $SRC | grep -E '^[-+] ' ; echo "-- before:"; others
    docker exec $C bash -c "cp -p $BIN $BAK && cd /opt/nvidia/cuBB/build.aarch64 && ninja test_mac 2>&1 | tail -3"; rc=$?
    echo "-- after (rc $rc):"; others; docker exec $C bash -c "ls -la $BAK $BIN | cut -c1-110"; exit $rc ;;
  revert)
    git checkout -- $SRC && echo "source reverted ($(git status --short -- $SRC | grep -c .) changes left)"
    docker exec $C bash -c "[ -f $BAK ] && cp -p $BAK $BIN && echo 'binary restored from backup' || echo 'no backup'"; others ;;
  status) git status --short -- $SRC; others; docker exec $C bash -c "ls -la $BAK 2>/dev/null | cut -c1-110" ;;
  *) echo "usage: $0 patch|revert|status"; exit 1 ;;
esac
