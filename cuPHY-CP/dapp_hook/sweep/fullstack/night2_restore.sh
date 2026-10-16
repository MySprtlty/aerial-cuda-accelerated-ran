#!/bin/bash
# night2_restore.sh — §8 of the 2026-10-16 night: verify / restore everything the X-grid night touched (read-only checks + the
# listed restores). Evidence is printed per item. Does not touch PTP, aerial00/01, sudoers or other users' processes.
S=$(cd "$(dirname "$0")" && pwd); R=$(cd "$S/../../../.." && pwd); O=${FS_OUT:-$R/prof/run65}; cd $R
echo "== night2 restore $(date '+%F %T')"
echo "-- MIG: $(nvidia-smi -i 0 --query-gpu=mig.mode.current,mig.mode.pending --format=csv,noheader) (expected Disabled, Disabled)"
sudo -n nvidia-smi -i 0 -rgc > /dev/null 2>&1 && echo "-- GPU clocks: -rgc applied; now $(nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.applications.graphics --format=csv,noheader)"
if pgrep -f 'nvidia-cuda-mps-control -d' > /dev/null; then echo "-- MPS control daemon alive ($(pgrep -c -f 'nvidia-cuda-mps-control -d'))"; else echo "-- MPS missing -> starting"; bash $S/mig_night.sh start-mps; fi
echo "-- GPU apps: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';') (expected only our mps server)"
echo "-- RAN processes alive: $(ps -eo stat,comm --no-headers | awk '$1 !~ /^Z/ && $2 ~ /^(phy_main|cuphycontroller|ru_emulator|test_mac|mac_main)/' | wc -l) (expected 0)"
LAB="cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml cuPHY-CP/ru-emulator/config/config.yaml cuPHY/nvlog/config/nvlog_config.yaml cuPHY-CP/testMAC/testMAC/test_mac_config.yaml"
ref=$(ls $O/pre_CLK_clkA_unlocked_10c.* 2>/dev/null | head -1)
for f in $LAB; do b=$O/pre_CLK_clkA_unlocked_10c.$(basename $f); printf -- "-- lab %-32s %s\n" "$(basename $f)" "$( [ -f $b ] && (cmp -s $b $f && echo 'identical to the 22:02 pre-snapshot' || echo 'DIFFERS from the 22:02 pre-snapshot') || echo 'no pre-snapshot')"; done
echo "-- lab files git status: $(git status --short -- $LAB | tr '\n' ' ')"
V=testBenches/phase4_test_scripts/valid_perf_patterns.sh
if [ -f prof/tmp/tmp_valid_perf_patterns.orig.sh ]; then cp prof/tmp/tmp_valid_perf_patterns.orig.sh $V && echo "-- valid_perf_patterns.sh restored ($(grep -c tmp_ddddddsuuu $V) tmp names left, git: '$(git status --short -- $V)')"; fi
rm -f testVectors/multi-cell/launch_pattern_F08_{9,10,11}C_tmp_ddddddsuuu_{peak,average}.yaml; echo "-- temp patterns in testVectors/multi-cell: $(ls testVectors/multi-cell | grep -c '_tmp_')"
for f in prof/tmp/tmp_rules_*c.yaml; do [ -f "$f" ] && cp "$f" $O/ && echo "-- kept a copy of $(basename $f) in $O"; done
rm -rf prof/tmp/tmp_* prof/tmp/__pycache__; echo "-- prof/tmp entries left: $(ls -A prof/tmp | wc -l)"
docker exec -u root c_aerial_troy bash -c 'rm -f /tmp/phy.log /tmp/ru.log /tmp/testmac.log' && echo "-- container /tmp raw logs removed"
for p in $(pgrep -x ptp4l) $(pgrep -x phc2sys); do echo "-- ptp $p $(cat /proc/$p/comm) allowed=$(awk '/^Cpus_allowed_list:/{print $2}' /proc/$p/status)"; done
[ -f timing_histogram.txt ] && echo "!! timing_histogram.txt at the repo root" || echo "-- no timing_histogram.txt at the repo root"
echo "-- ring dumps left in prof: $(ls prof/ring_16C_*.bin 2>/dev/null | tr '\n' ' ')"
df -h / | tail -1 | awk '{print "-- disk: "$4" free ("$5" used)"}'
echo "-- git status (tools/patterns/engines to commit): "; git status --short | grep -v '^?? \(testVectors/\|CONTEXT.md\|.*pptx\|.*nrSim\|testBenches/phase4\|.*yaml_template\)' | grep -v 'cuphycontroller_F08_CG1.yaml\|l2_adapter_config_F08_CG1.yaml\|ru-emulator/config/config.yaml\|nvlog_config.yaml\|test_mac_config.yaml\|cuPhyChEstCoeffs' | head -20
