#!/bin/bash
# alt10_run.sh [out_dir] - 10-cell slow alternation (Average-L3 60 s | Peak 60 s) with the YOLO table policy, the night-3 N4v2 "ours A2"
# recipe (2026-10-18, prof/run66), re-run and PASSed on 2026-10-27 (prof/run67_alt: B_abs 0, Err.ind/drop/CRC 0, E2E 143.2 ms, mAP 53.0).
#
#   pattern     tmp_alt_l3 = 10C DDDDDDSUUU peak (make_launch_pattern_tdd.py from launch_pattern_F08_10C_59c.yaml, 80 slots) with the UL vectors
#               of frames 0-1 swapped for their L3 PUSCH load copies (TVnr_ULMIX_<93000 + id - 4471>); frames 2-3 stay peak
#   alternation RU_LP_BLOCK_FRAMES=2 RU_LP_BLOCK_REPEAT=3000 (+ TESTMAC_*): each 2-frame block is replayed 3000 x 20 ms = 60 s
#               (opt-in block-repeat mapping, RU emulator utils.hpp / testMAC launch_pattern.hpp; needs binaries built with it)
#   policy      TENANT_POLICY=table, estimator/table_x10_A.yaml: average (PUSCH PRB per UL slot <= 1327) -> 26x@61, peak (>= 1491) -> 26s@43
#               (the user values 1464/1626 were adjusted in night 3 because the measured Peak minimum 1614 is below 1626)
#   tenant      dispatcher mode, Poisson 10 req/s, uhd_*_q60.jpg, engines yolo{26n,26s,26m,26l,26x}_1280_b8_e2e_fp16_sm{56,80,104,132}
#   run         r0_ladder.sh -> fullstack_run.sh (setup1/setup2/test_config.sh, run, log extraction, lab-file restore), 1,270,000 slots
#               (635 s), then tenmin_table.py (10-min per-minute gate) and n4_switch.py (switch analysis)
#
# Temporary changes, all undone on exit (trap): the block-repeat RU emulator / testMAC binaries are copied into build.aarch64 (originals
# come back from $ORIG_BIN and are md5-checked), the two tmp patterns are generated in testVectors/multi-cell and registered in
# valid_perf_patterns.sh. The lab yamls are edited and restored by fullstack_run.sh itself.
#
# Env: SLOTS (default 1270000), PATCHED_BIN / ORIG_BIN (default prof/run66/bin/{patched,orig}), TABLE (default estimator/table_x10_A.yaml),
#      RATE (10). Prerequisites: c_aerial_troy running, no other RAN stack (nv-cubb / OAI), >= 30 GB free on / (a 10-min run writes
#      ~13 GB of raw container logs before extraction), PTP up. If PATCHED_BIN is missing, build the binaries from the committed sources:
#        cp -p build.aarch64/cuPHY-CP/ru-emulator/ru_emulator/ru_emulator build.aarch64/cuPHY-CP/testMAC/testMAC/test_mac $ORIG_BIN/
#        docker exec c_aerial_troy bash -lc "cd /opt/nvidia/cuBB/build.aarch64 && ninja cuPHY-CP/ru-emulator/ru_emulator/ru_emulator cuPHY-CP/testMAC/testMAC/test_mac"
#        cp -p <the two rebuilt binaries> $PATCHED_BIN/ ; cp -p $ORIG_BIN/* back into build.aarch64
set -u
S=$(cd "$(dirname "$0")" && pwd); R=$(cd "$S/../../../.." && pwd); E=$R/cuPHY-CP/dapp_hook/estimator
O=${1:-$R/prof/run_alt10_$(date +%m%d_%H%M)}; mkdir -p $O
B=$R/build.aarch64; RUB=$B/cuPHY-CP/ru-emulator/ru_emulator/ru_emulator; TMB=$B/cuPHY-CP/testMAC/testMAC/test_mac
PATCHED_BIN=${PATCHED_BIN:-$R/prof/run66/bin/patched}; ORIG_BIN=${ORIG_BIN:-$R/prof/run66/bin/orig}
V=$R/testBenches/phase4_test_scripts/valid_perf_patterns.sh; MC=$R/testVectors/multi-cell
PEAK=$MC/launch_pattern_F08_10C_tmp_ddddddsuuu_peak.yaml; ALT=$MC/launch_pattern_F08_10C_tmp_alt_l3.yaml
SLOTS=${SLOTS:-1270000}; TSECS=$(( SLOTS / 2000 ))
cd $R
echo "== alt10 $(date '+%F %T') -> $O ($SLOTS slots)"
# ---- prechecks (nothing is changed before they pass)
pgrep -x 'phy_main|cuphycontroller_scf|ru_emulator|test_mac|nr-softmodem' > /dev/null && { echo "!! a RAN process is alive - abort"; exit 1; }
docker ps --format '{{.Names}}' | grep -qE '^(nv-cubb|oai-gnb-aerial_hsw|oai-gnb-aerial-hybrid)$' && { echo "!! a foreign compose stack is running - abort"; exit 1; }
[ "$(docker inspect -f '{{.State.Running}}' c_aerial_troy 2>/dev/null)" = true ] || { echo "!! c_aerial_troy is not running (cuPHY-CP/container/run_aerial.sh -d)"; exit 1; }
for f in $PATCHED_BIN/ru_emulator $PATCHED_BIN/test_mac $ORIG_BIN/ru_emulator $ORIG_BIN/test_mac; do [ -s $f ] || { echo "!! missing $f (see the header for the build steps)"; exit 1; }; done
cmp -s $RUB $ORIG_BIN/ru_emulator && cmp -s $TMB $ORIG_BIN/test_mac || { echo "!! build.aarch64 binaries differ from $ORIG_BIN - abort (restore the lab binaries first)"; exit 1; }
[ -e $PEAK ] || [ -e $ALT ] && { echo "!! a tmp pattern is already present ($PEAK / $ALT) - abort"; exit 1; }
free=$(df --output=avail -BG / | tail -1 | tr -dc '0-9'); [ "$free" -ge 30 ] || { echo "!! only ${free} GB free on / (need >= 30)"; exit 1; }
cp -p $V $O/valid_perf_patterns.orig.sh
restore() {
  echo "== restore $(date +%T)"
  cp -p $ORIG_BIN/ru_emulator $RUB; cp -p $ORIG_BIN/test_mac $TMB
  cmp -s $RUB $ORIG_BIN/ru_emulator && cmp -s $TMB $ORIG_BIN/test_mac && echo "   binaries: originals back" || echo "!! binaries NOT restored"
  cp -p $O/valid_perf_patterns.orig.sh $V && echo "   valid_perf_patterns.sh restored"
  rm -f $PEAK $ALT; echo "   tmp patterns removed ($(ls $MC | grep -c _tmp_) left)"
  pkill -TERM -f "estimator/table_policy.py --table" 2>/dev/null && echo "   leftover table_policy stopped"
  echo "ALT10_DONE $(date +%T)"
}
trap restore EXIT
# ---- patterns
python3 cuPHY-CP/dapp_hook/tools/make_launch_pattern_tdd.py --tdd DDDDDDSUUU --src $MC/launch_pattern_F08_10C_59c.yaml --slots 80 --out $PEAK || exit 1
python3 - "$PEAK" "$ALT" "$V" <<'PY' || exit 1
import re, os, sys
src, dst, reg = sys.argv[1:4]; txt = open(src).read()
ids = sorted({int(x) for x in re.findall(r"ULMIX_(\d+)_gNB", txt)}); l3 = {i: 93000 + (i - 4471) for i in ids}
missing = [l3[i] for i in ids if not os.path.exists("testVectors/TVnr_ULMIX_%d_gNB_FAPI_s0.h5" % l3[i])]
assert not missing, "missing L3 vectors %s" % missing
i = txt.find("SCHED:"); head, sched = txt[:i], txt[i:]; blocks = sched.split("- config:"); out = [blocks[0]]
for b in blocks[1:]:
    slot = int(re.search(r"^\s*slot:\s*(\d+)", b, re.M).group(1))
    if slot < 40:
        b = re.sub(r"ULMIX_(\d+)_gNB", lambda mm: "ULMIX_%d_gNB" % l3[int(mm.group(1))], b)
    out.append(b)
assert not os.path.exists(dst); open(dst, "w").write(head + "- config:".join(out))
print("written", dst, "| UL ids slots 0-39:", sorted({int(x) for x in re.findall(r"ULMIX_(\d+)_gNB", "- config:".join(out[1:41]))})[:3],
      "slots 40-79:", sorted({int(x) for x in re.findall(r"ULMIX_(\d+)_gNB", "- config:".join(out[41:]))})[:3])
s = open(reg).read()
if "tmp_alt_l3" not in s:
    s = s.replace('valid_perf_patterns=(', 'valid_perf_patterns=("tmp_alt_l3" ', 1).replace('peak_patterns=(', 'peak_patterns=("tmp_alt_l3" ', 1)
    open(reg, "w").write(s)
print("registered tmp_alt_l3:", s.count("tmp_alt_l3"))
PY
# ---- block-repeat binaries (opt-in; without the env variables they run the original code path)
cp -p $PATCHED_BIN/ru_emulator $RUB; cp -p $PATCHED_BIN/test_mac $TMB
echo "   block-repeat binaries in: RU $(strings $RUB | grep -c RU_LP_BLOCK_REPEAT), testMAC $(strings $TMB | grep -c TESTMAC_LP_BLOCK_REPEAT)"
# ---- run
echo "ALT_ours_A pattern=tmp_alt_l3 ncells=10 slots=$SLOTS tags=311,223 keep_raw=2 tenant=dispatcher arrivals=poisson rate=${RATE:-10} tdelay=25 tsecs=$TSECS" > $O/list_alt10.txt
export TENANT_MODELS="26n 26s 26m 26l 26x" TENANT_CLASSES="56 80 104 132" TENANT_GLOB='uhd_*_q60.jpg'
RUN_ENV="RU_LP_BLOCK_FRAMES=2 RU_LP_BLOCK_REPEAT=3000 TESTMAC_LP_BLOCK_FRAMES=2 TESTMAC_LP_BLOCK_REPEAT=3000" TENANT_POLICY=table TENANT_TABLE=${TABLE:-$E/table_x10_A.yaml} \
  FS_OUT=$O bash $S/r0_ladder.sh $O/list_alt10.txt ALT10 > $O/driver_ALT10.log 2>&1; echo "   r0_ladder exit $? $(date +%T)"
n=ALT10_ALT_ours_A_10c
python3 $S/tenmin_table.py --dir $O --name $n --csv $O/tenmin_summary.csv > $O/tenmin_$n.txt 2>&1; echo "   10-min gate: $(head -1 $O/tenmin_$n.txt | cut -c1-160)"
python3 $S/n4_switch.py --dir $O --name $n > $O/n4sw_$n.txt 2>&1; echo "   switch analysis: $O/n4sw_$n.txt"
grep -E '^[0-9:]+: (peak|average) ->' $O/table_policy_$n.log | sed 's/^/   /'
