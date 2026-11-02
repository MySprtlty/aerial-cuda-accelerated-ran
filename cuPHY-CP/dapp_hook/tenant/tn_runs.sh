#!/bin/bash
# tn_runs.sh [name:pattern:mode ...]  (2026-11-01): one-process tenant runs on 8 RU-emulator cells, DDDDDDSUUU, green contexts,
# build.aarch64_f1002, 2 min of traffic each (SLOTS env, default 240000), outputs in prof/tenant (FS_OUT env).
#   mode = tenant   : PUSCH/PDSCH profiles + sizer + YOLO tenant thread (TENANT=1)
#          profiles : profiles + sizer, no tenant thread (PROFILES_ONLY=1)
#          base     : plain green contexts, yaml partitions, no profiles (the no-degradation baseline)
#          fixedNN  : tenant thread with a static class NN on a plain partition (TENANT_FIXED=NN)
# Extra knobs pass through the environment (PUSCH_PROFILES, PDSCH_PROFILES, TENANT_CLASSES, TENANT_ENV, SIZER_MODEL).
# NVTAGS="" (env) disables the INFO timing tags 311,223: no per-channel GPU timing (parse_phylog), but the L1 log stays small
# (with the tags a 10-min run writes ~13 GB into the container /tmp, which is on the host disk).
# After each run: tn_summary.py -> summary_<name>.txt; the big raw logs (phy_<name>.log, ru_<name>.log) are deleted.
set -u
R=/home/troy/build/aerial-cuda-accelerated-ran; F=$R/cuPHY-CP/dapp_hook/sweep/fullstack; T=$R/cuPHY-CP/dapp_hook/tenant
O=${FS_OUT:-$R/prof/tenant}; mkdir -p $O; cd $R
SLOTS=${SLOTS:-240000}
TENANT_ENV_CALLER=${TENANT_ENV:-}   # the caller's TENANT_ENV (e.g. -e DAPP_TENANT_UL_PAUSE=1) is appended to the mode's own
for spec in "$@"; do
  IFS=: read -r n pat mode <<< "$spec"
  extra=""
  case "$mode" in
    tenant)   envs="TENANT=1" ;;
    profiles) envs="PROFILES_ONLY=1" ;;
    base)     envs="" ;;
    fixed*)   envs="TENANT=1 TENANT_FIXED=${mode#fixed}" ;;
    static*)  envs="TENANT=1 TENANT_FIXED=${mode#static} TENANT_CLASSES=${mode#static}"; extra="-e DAPP_TENANT_DIR=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/full26n" ;;   # unsegmented b8 engine on a fixed partition (static baseline)
    ours)     envs="TENANT=1"; extra="-e DAPP_TENANT_BUSY_DIR=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/seg26n_b1" ;;   # two pipelines: quiet b8 + busy b1
    ours1)    envs="TENANT=1"; extra="-e DAPP_TENANT_BUSY_DIR=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/seg26n_b1 -e DAPP_TENANT_ONE_PER_GAP=1" ;;   # same with one context per gap (3 streams)
    stressspin5) envs="TENANT=1"; extra="-e DAPP_TENANT_STRESS=spin -e DAPP_TENANT_STRESS_US=5 -e DAPP_TENANT_ONE_PER_GAP=1" ;;   # diagnosis: launch rate only (5 us kernels)
    stressl2big) envs="TENANT=1"; extra="-e DAPP_TENANT_STRESS=l2 -e DAPP_TENANT_STRESS_MB=48 -e DAPP_TENANT_ONE_PER_GAP=1" ;;   # diagnosis: L2 at full capacity (48 MB)
    stresshbmlo) envs="TENANT=1"; extra="-e DAPP_TENANT_STRESS=hbm -e DAPP_TENANT_STRESS_BLOCKS=32 -e DAPP_TENANT_ONE_PER_GAP=1" ;;   # diagnosis: HBM streaming at reduced intensity (32 blocks)
    stressl2w)  envs="TENANT=1"; extra="-e DAPP_TENANT_STRESS=l2w -e DAPP_TENANT_ONE_PER_GAP=1" ;;   # diagnosis: L2-resident writes (dirty lines)
    stresshbmw) envs="TENANT=1"; extra="-e DAPP_TENANT_STRESS=hbmw -e DAPP_TENANT_STRESS_BLOCKS=32 -e DAPP_TENANT_ONE_PER_GAP=1" ;;   # diagnosis: HBM writes at reduced intensity
    stresscta)  envs="TENANT=1"; extra="-e DAPP_TENANT_STRESS=spin -e DAPP_TENANT_STRESS_BLOCKS=20000 -e DAPP_TENANT_ONE_PER_GAP=1" ;;   # diagnosis: many short CTAs (dispatcher load)
    stresshbm256)   envs="TENANT=1"; extra="-e DAPP_TENANT_STRESS=hbm -e DAPP_TENANT_STRESS_MB=256 -e DAPP_TENANT_ONE_PER_GAP=1" ;;   # diagnosis: HBM streaming, ~140 us launches (stay inside the slot)
    stresshbm256lo) envs="TENANT=1"; extra="-e DAPP_TENANT_STRESS=hbm -e DAPP_TENANT_STRESS_MB=256 -e DAPP_TENANT_STRESS_GAP_US=400 -e DAPP_TENANT_ONE_PER_GAP=1" ;;   # same at ~25 % duty
    stresstc)   envs="TENANT=1"; extra="-e DAPP_TENANT_STRESS=tc -e DAPP_TENANT_ONE_PER_GAP=1" ;;   # diagnosis: tensor cores busy, no memory traffic
    b1only)   envs="TENANT=1"; extra="-e DAPP_TENANT_DIR=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/seg26n_b1 -e DAPP_TENANT_BATCH=1 -e DAPP_TENANT_INPUT=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/seg26n_b1/input_uhd_b1_fp16.bin -e DAPP_TENANT_CLASSES=40,48" ;;   # diagnosis: batch-1 engine in every window (low DRAM traffic)
    stress*)  envs="TENANT=1"; extra="-e DAPP_TENANT_STRESS=${mode#stress} -e DAPP_TENANT_ONE_PER_GAP=1" ;;   # diagnosis: synthetic kernels (spin | l2 | hbm) on the gap contexts
    gap48)    envs="TENANT=1 TENANT_FIXED=48"; extra="-e DAPP_TENANT_FIXED_GAP=88,40" ;;   # fair static baseline: b8 on the Average busy-window gap, continuously
    *) echo "unknown mode $mode"; continue ;;
  esac
  echo "##### $n $(date +%T) pattern $pat mode $mode ($envs)"
  case "$pat" in tmp_*) pname=$pat ;; *) pname=ddddddsuuu_$pat ;; esac
  # TENANT_ENV (extra "-e K=V" args for the L1) is exported as one string: the mode's part plus whatever the caller set
  export TENANT_ENV="${extra}${TENANT_ENV_CALLER:+ $TENANT_ENV_CALLER}"
  env GREEN=1 NCELLS=${NCELLS:-8} PATTERN=$pname KEEP_RAW=2 MPSTAT=0 RECORD=0 FS_OUT=$O BUILD_DIR=build.aarch64_f1002 $envs bash $F/fullstack_run.sh $n $SLOTS "" "${NVTAGS-311,223}" async > $O/console_$n.log 2>&1
  grep -E "^== |L1 did not|profile fallbacks|Dropping" $O/console_$n.log | cut -c1-160
  python3 $T/tn_summary.py $n > $O/summary_$n.txt 2>&1
  grep -E "PUSCH completion|PDSCH compression|PDCCH end|RU: .*LATE SLOTS|L1 error|fallbacks|tenant last stats|^    pos" $O/summary_$n.txt | cut -c1-260
  rm -f $O/phy_$n.log $O/ru_$n.log $O/testmac_$n.log
  df -h --output=avail / | tail -1 | sed 's/^/   disk free/'
done
echo "TN_RUNS_DONE $(date +%T)"
