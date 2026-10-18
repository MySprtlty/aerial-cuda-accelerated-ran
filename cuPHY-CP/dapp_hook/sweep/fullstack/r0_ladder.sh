#!/bin/bash
# r0_ladder.sh <configs.txt> [prefix]      |  r0_ladder.sh --post <name> [name ...]  (post-process + report only)
# No-tenant baseline ladder (R0): 16-cell 59c full stack (RU emulator + cuphycontroller + testMAC, no YOLO), one run per
# line of <configs.txt>, 240,000 slots (120 s) each by default. Per run:
#   (a) leftover processes, MPS daemons, GPU compute apps / clocks, containers and the lab-file git state are recorded in
#       precheck_<name>.txt; a live RAN process of someone else (phy_main / ru_emulator / test_mac / nr-softmodem) aborts
#       the ladder unless FORCE=1
#   (b) the configuration is applied by fullstack_run.sh (hook line, nvlog INFO tags, RU core lists) and by test_config.sh
#       (TC_EXTRA: --ehq=N, --green-ctx=N, --BFP=N, --num-ports=N, --log-nic-timings) inside the container
#   (c) the run (fullstack_run.sh, KEEP_RAW=1: the 8-16 GB container logs are copied, parsed and deleted)
#   (d) post-processing: parse_phylog.py (phy_<name>.csv), L2A lines, WRN/ERR lines, RU late-packet tables (ru_late.py with
#       the RU.LATE_PACKETS INFO lines of tag 503), RU log trimmed to its summary lines
#   (e) the five lab files are snapshotted BEFORE the run (pre_<name>.*; fullstack_run.sh's own backup is taken after
#       test_config.sh, so it would keep --ehq/--BFP/--log-nic-timings edits), restored from that snapshot afterwards and
#       verified (restore_<name>.diff must be empty); applied_<name>.diff = what the configuration changed
#   (f) r0_ladder_report.py appends one row to $FS_OUT/ladder_table.csv and prints the accumulated table
# configs.txt: "<name> key=value ..." per line (# comments). Keys and defaults (= the lab configuration):
#   hook=async|off|summary|full|keep   eh=1 (early HARQ: testMAC uciIndPerSlot 2|0)   gc=0 (use_green_contexts; gc=1 needs
#   ALLOW_GC=1 because CONTEXT.md 5 says not to touch it)   bfp=9 (comp_meth 1 bit_width 9|14, both yamls)   ports=1
#   nicd=0 (--log-nic-timings: nvlog INFO for DRV/RU/FH timing tags + cuphydriver ul_rx_pkt_tracing_level 1)   mpstat=0
#   ru_hist=0 (RU timing_histogram)   pattern=59c   ncells=16   slots=$SLOTS   tags=311,223,503 (nvlog INFO tags; drop 503 for a run
#   without the RU.LATE_PACKETS lines)   record=0 record_secs=30 (hook ring dump prof/ring_16C_<name>.bin)   keep_phy=0 (1 keeps
#   the raw phy_<name>.log, 8-16 GB)   tc="extra test_config.sh arguments"
#   tenant=dispatcher (fs_tenant.sh dispatcher mode: dapp_sched + e2e/dispatcher.py + client)   model=26x   cap=43 (SM cap %)
#   arrivals=closed|poisson|fixed|jitter   rate=10 (req/s, open-loop modes)   noclient=0|1   tdelay=28 (s from testMAC start to the
#   first request)   tsecs=45 (client duration)   xclass=32 xmodels=26x (extra SM class engines, e.g. cap 24)
#   Before every run the leftover tenant processes of earlier runs are killed (dispatcher.py, e2e_client.py, dapp_sched, port 9007).
#   keep_raw=1|2 (2 = filtered phy log for long runs, fullstack_run.sh KEEP_RAW=2)   l1sed="<sed expressions>" (no spaces) or l1sedf=<file with the expressions>  applied to cuphycontroller_F08_CG1.yaml after the pre-snapshot (test-only yaml edits such as
#   split_ul_cuda_streams or the CQE tracer), reverted by the pre-snapshot restore and visible in applied_<name>.diff
# Before every run: precheck (exit 1 when another run is alive), disk rule (exit 3 when < 20 GB stays), PTP core record (ptp_<name>.txt).
# env: FS_OUT (default prof/run62), SLOTS (default 240000), FIRST_S (statistics skip, default 10), FORCE=1,
#      DRY=1 (apply + snapshot + restore + diff only, no RU / L1 / testMAC, no GPU; the report path is exercised on
#      DRY_REPORT_DIR / DRY_REPORT_NAME, default prof/run61 sc_R0_16c)
set -u
S=$(cd "$(dirname "$0")" && pwd); R=$(cd "$S/../../../.." && pwd)
CFG=$1; PREFIX=${2:-r0}; O=${FS_OUT:-$R/prof/run62}; SLOTS=${SLOTS:-240000}; FIRST_S=${FIRST_S:-10}; DRY=${DRY:-0}; FORCE=${FORCE:-0}
LAB="cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml cuPHY-CP/ru-emulator/config/config.yaml cuPHY/nvlog/config/nvlog_config.yaml cuPHY-CP/testMAC/testMAC/test_mac_config.yaml"
mkdir -p $O; cd $R

disk_check() {  # <name>: free space rule before a run. < 25 GB: delete prof/ring_16C_as_pilot.bin; < 20 GB: delete parsed raw logs of
  # earlier runs in $O (phy_/ru_/testmac_ *.log not listed in $O/keep_raw.txt) and re-check; still < 20 GB -> return 1 (skip to E7)
  local free; free=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  echo "   disk: ${free} GB free before $1"
  if [ "$free" -lt 25 ] && [ -f $R/prof/ring_16C_as_pilot.bin ]; then rm -f $R/prof/ring_16C_as_pilot.bin; echo "   disk: removed prof/ring_16C_as_pilot.bin"; free=$(df --output=avail -BG / | tail -1 | tr -dc '0-9'); fi
  if [ "$free" -lt 20 ]; then
    local f; for f in $O/phy_*.log $O/ru_*.log $O/testmac_*.log; do [ -f "$f" ] || continue; case "$(basename $f)" in phy_warn_*|ru_warn_*) continue;; esac; grep -qx "$(basename $f)" $O/keep_raw.txt 2>/dev/null && continue; rm -f "$f"; echo "   disk: removed $f"; done
    free=$(df --output=avail -BG / | tail -1 | tr -dc '0-9'); echo "   disk: ${free} GB free after cleanup"
  fi
  [ "$free" -lt 20 ] && { echo "!! disk below 20 GB -> no new run"; return 1; }
  return 0
}
ptp_record() {  # <name>: Cpus_allowed_list of ptp4l / phc2sys at run start -> ptp_<name>.txt
  local p; { for p in $(pgrep -x ptp4l) $(pgrep -x phc2sys); do echo "$p $(cat /proc/$p/comm) allowed=$(awk '/^Cpus_allowed_list:/{print $2}' /proc/$p/status) last_cpu=$(awk '{print $39}' /proc/$p/stat)"; done; } > $O/ptp_$1.txt 2>&1
  echo "   ptp: $(tr '\n' ';' < $O/ptp_$1.txt)"
}

tenant_cleanup() {  # kill leftover tenant processes of OUR earlier runs (host dispatcher / client, container dapp_sched), free port 9007
  local p n=0
  for p in $(pgrep -f 'e2e/dispatcher.py' ; pgrep -f 'e2e/e2e_client.py' ; pgrep -f 'e2e/infer_server.py' ; pgrep -f 'table_policy.py.*--ring'); do [ "$p" = "$$" ] && continue; kill -INT $p 2>/dev/null && n=$((n+1)); done
  sleep 1; for p in $(pgrep -f 'e2e/dispatcher.py' ; pgrep -f 'e2e/e2e_client.py' ; pgrep -f 'table_policy.py.*--ring'); do kill -KILL $p 2>/dev/null; done
  docker exec -u root c_aerial_troy bash -c 'for p in $(pgrep -x dapp_sched); do kill -INT $p 2>/dev/null; done; sleep 1; for p in $(pgrep -x dapp_sched); do kill -KILL $p 2>/dev/null; done; rm -f /dev/shm/dapp_sched.sock' 2>/dev/null
  local port; port=$(ss -ltnp 2>/dev/null | grep -c ':9007 ')
  echo "   tenant cleanup: $n process(es) signalled, port 9007 listeners now: $port"
}

precheck() {  # <name> -> precheck_<name>.txt; returns 1 when a RAN process of a foreign run is alive
  local name=$1 out=$O/precheck_$1.txt
  {
    echo "== precheck $name $(date '+%F %T') host $(hostname)"
    echo "-- RAN / tenant processes alive (non-zombie; this script's shell excluded):"
    ps -eo pid,ppid,stat,psr,etimes,comm,args | awk -v me=$$ -v pp=$PPID 'NR>1 && $1!=me && $1!=pp && $3 !~ /^Z/ && ($6 ~ /^(phy_main|cuphycontroller|ru_emulator|test_mac|mac_main|dapp_sched|nsys|nr-softmodem|infer_server)/ || $0 ~ /dispatcher\.py|e2e_client\.py|dapp_est|live\.py |replay\.py |dapp_ring_record/) && $0 !~ /awk -v me/' | cut -c1-160
    echo "-- MPS daemons (ppid 1 inside c_aerial_troy = $(docker inspect -f '{{.State.Pid}}' c_aerial_troy 2>/dev/null) is ours, left by run2 as usual):"
    ps -eo pid,ppid,stat,etimes,args | awk 'NR>1 && $3 !~ /^Z/ && /nvidia-cuda-mps-(control|server)/' | cut -c1-120
    echo "-- GPU compute apps:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | head -12
    echo "-- GPU state:"; nvidia-smi --query-gpu=persistence_mode,clocks.sm,clocks.mem,clocks.applications.graphics,clocks.applications.memory,clocks.max.sm,power.limit,power.draw,utilization.gpu,memory.used --format=csv 2>&1
    echo "-- containers:"; docker ps --format '{{.Names}}\t{{.Status}}\tcpuset={{.HostConfig.CpusetCpus}}' 2>/dev/null; for c in $(docker ps --format '{{.Names}}'); do printf '   %s cpuset=%s\n' $c "$(docker inspect -f '{{.HostConfig.CpusetCpus}}' $c)"; done
    echo "-- zombies in c_aerial_troy (harmless, PID 1 is tail -f): $(docker exec c_aerial_troy bash -c 'ps -eo stat | grep -c "^Z"' 2>/dev/null)"
    echo "-- non-kernel threads on cores 4-63 (core: thread(pid) ...):"
    ps -eLo psr,pid,comm --no-headers | awk '$1>=4 && $1<=63 && $3 !~ /^(kworker|ksoftirqd|migration|cpuhp|idle_inject|rcu|kthreadd|irq\/|watchdog)/ {c[$1]=c[$1]" "$3"("$2")"} END{for(k in c) print "   "k":"c[k]}' | sort -t: -k1,1n | cut -c1-200
    echo "-- PTP: $(ps -o pid=,psr=,comm= -p $(pgrep -d, -x 'ptp4l|phc2sys' 2>/dev/null || echo 1) 2>/dev/null | tr '\n' ';')"
    echo "-- lab files (git status --short, expected: all ' M' as in the lab):"; git -C $R status --short -- $LAB
    echo "-- hook line: $(grep -m1 '^dapp_hook:' cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml)"
    echo "-- L1 GPU settings: $(grep -E '^  (use_green_contexts|mps_sm_[a-z_]+|pusch_workCancelMode|pusch_deviceGraphLaunchEn|pmu_metrics):' cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml | tr -d ' ' | tr '\n' ' ')"
    echo "-- cells iq fmt: $(grep -m1 'dl_iq_data_fmt' cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml | tr -d ' ')  testMAC uciIndPerSlot: $(grep -A3 '^indicationPerSlot' cuPHY-CP/testMAC/testMAC/test_mac_config.yaml | grep uciIndPerSlot | tr -d ' ')"
  } > $out 2>&1
  local alive; alive=$(awk '/^-- RAN/{f=1;next} /^-- MPS/{f=0} f' $out | grep -c .)
  echo "   precheck: $alive RAN/tenant process line(s), see $out"
  [ "$alive" -gt 0 ] && [ "$FORCE" != "1" ] && { echo "!! RAN processes alive (another run?) -> abort; FORCE=1 overrides"; sed -n '/^-- RAN/,/^-- MPS/p' $out | head -8; return 1; }
  return 0
}

post() {  # <name> [keep_phy]: parse + trim as safecap_runs.sh, plus the RU late-packet tables
  local name=$1 keep=${2:-0}
  if [ -s $O/phy_$name.log ]; then
    python3 $S/parse_phylog.py $O/phy_$name.log $O/phy_$name.csv
    grep -F '[L2A.PROCESSING_TIMES]' $O/phy_$name.log > $O/l2a_$name.log
    grep -E ' (WRN|ERR|FATAL) |Send Err.ind|recovery|not responsive|No available|Successive Non-availability|Attempting PUSCH pipeline termination|Order kernel timeout|exit handler|pipeline failed|Dropping the slot command|dropped due to latency|SFN mismatch' $O/phy_$name.log | head -2000000 > $O/phy_warn_$name.log
    grep -E '\] Cell +[0-9]+ \| DL .* Slots CRC' $O/phy_$name.log > $O/l1_persec_$name.log      # the L1's per-second per-cell throughput / CRC line
    if [ "$keep" = "1" ]; then echo "phy_$name.log" >> $O/keep_raw.txt; echo "   raw phy log kept: $O/phy_$name.log ($(du -h $O/phy_$name.log | cut -f1))"; else rm -f $O/phy_$name.log; fi
  else
    echo "   no phy log for $name"
  fi
  [ -s $O/testmac_$name.log ] && { grep -E '\] Cell +[0-9]+ \| DL .* ERR ' $O/testmac_$name.log > $O/mac_persec_$name.log; rm -f $O/testmac_$name.log; }
  if [ -s $O/ru_$name.log ]; then
    python3 $S/ru_late.py $O/ru_$name.log $O/ru_$name > $O/ru_late_$name.txt 2>&1; rm -f $O/ru_${name}_slots.csv   # keep only the late rows (_late.csv)
    grep -E '\[RU\] \||PACKET_SUMMARY| (WRN|ERR) ' $O/ru_$name.log | head -400000 > $O/ru_warn_$name.log; rm -f $O/ru_$name.log
  fi
  [ -f $R/timing_histogram.txt ] && mv $R/timing_histogram.txt $O/ru_hist_$name.txt && echo "   RU timing histogram -> $O/ru_hist_$name.txt"   # ru_hist=1 writes it at the repo root
}

pre_snapshot() { local f; for f in $LAB; do cp $f $O/pre_$1.$(basename $f); done; }
restore_pre()  { local f; for f in $LAB; do [ -f $O/pre_$1.$(basename $f) ] && cp $O/pre_$1.$(basename $f) $f; done; }

verify_restore() {  # <name>: restore_<name>.diff (expected empty) and applied_<name>.diff (what the configuration changed)
  local name=$1 f b a
  : > $O/restore_$name.diff; : > $O/applied_$name.diff
  for f in $LAB; do
    b=$O/pre_$name.$(basename $f)
    [ -f $b ] || continue
    diff -u $b $f >> $O/restore_$name.diff
    case $(basename $f) in
      l2_adapter_config_F08_CG1.yaml) a=$O/l2_adapter_$name.yaml ;;
      cuphycontroller_F08_CG1.yaml)   a=$O/cuphycontroller_$name.yaml ;;
      config.yaml)                    a=$O/ru_config_$name.yaml ;;
      nvlog_config.yaml)              a=$O/nvlog_$name.yaml ;;
      test_mac_config.yaml)           a=$O/testmac_config_$name.yaml ;;
    esac
    [ -f "$a" ] && diff -u $b $a | grep -E '^[-+][^-+]' | sed "s|^|$(basename $f): |" >> $O/applied_$name.diff
  done
  if [ -s $O/restore_$name.diff ]; then echo "!! lab files NOT restored for $name (see restore_$name.diff)"; else echo "   lab files restored (restore_$name.diff empty); applied changes: $(grep -c . $O/applied_$name.diff) line(s)"; fi
}

if [ "$CFG" = "--post" ]; then      # r0_ladder.sh --post <name> [name ...]: post-process + report rows for finished runs (raw logs in $FS_OUT)
  shift; for name in "$@"; do post $name $( [ -f $O/keep_raw.txt ] && grep -qx "phy_$name.log" $O/keep_raw.txt && echo 1 || echo 0 ); python3 $S/r0_ladder_report.py --dir $O --first-s $FIRST_S --out $O/ladder_table.csv --append $name > $O/report_$name.txt 2>&1 && echo "   report row added for $name" || echo "!! report failed for $name"; done
  python3 $S/r0_ladder_report.py --dir $O --out $O/ladder_table.csv --table; exit 0
fi
[ -r "$CFG" ] || { echo "no config list $CFG"; exit 1; }
echo "== R0 ladder $PREFIX: $(grep -cE '^[^#[:space:]]' $CFG) run(s), $SLOTS slots each, out $O, dry $DRY $(date +%T)"
grep -E '^[^#[:space:]]' $CFG | while read -r line; do
  set -- $line; cname=$1; shift
  hook=async eh=1 gc=0 bfp=9 ports=1 nicd=0 mpstat=0 ru_hist=0 pattern=59c ncells=16 slots=$SLOTS tags="311,223,503" tc="" record=0 record_secs=30 keep_phy=0 tenant="" model="" cap=100 arrivals=closed rate=10 noclient=0 tdelay=28 tsecs=45 xclass="" xmodels="" l1sed="" l1sedf="" keep_raw=1
  for kv in "$@"; do k=${kv%%=*}; v=${kv#*=}; case $k in
    hook|eh|gc|bfp|ports|nicd|mpstat|ru_hist|pattern|ncells|slots|tags|tc|record|record_secs|keep_phy|tenant|model|cap|arrivals|rate|noclient|tdelay|tsecs|xclass|xmodels|l1sed|l1sedf|keep_raw) printf -v "$k" '%s' "$v" ;;
    *) echo "unknown key $k in '$line'"; exit 1 ;;
  esac; done
  name=${PREFIX}_${cname}_${ncells}c
  extra="$tc"
  [ "$eh" != "1" ] && extra="$extra --ehq=$eh"
  [ "$bfp" != "9" ] && extra="$extra --BFP=$bfp"
  [ "$ports" != "1" ] && extra="$extra --num-ports=$ports"
  [ "$nicd" = "1" ] && extra="$extra --log-nic-timings"
  if [ "$gc" != "0" ]; then
    if [ "${ALLOW_GC:-0}" = "1" ]; then extra="$extra --green-ctx=$gc"; echo "   NOTE $name: use_green_contexts=$gc (CONTEXT.md 5: GPU setting, report it)"; else echo "!! $name: gc=$gc skipped - CONTEXT.md 5 forbids use_green_contexts changes (ALLOW_GC=1 overrides)"; continue; fi
  fi
  { [ "$eh" != "1" ] || [ "$bfp" != "9" ] || [ "$nicd" = "1" ]; } && echo "   NOTE $name: changes a cuphycontroller/testMAC setting vs the lab (CONTEXT.md 9: report it): eh=$eh bfp=$bfp nicd=$nicd"
  echo "== $name: hook=$hook eh=$eh gc=$gc bfp=$bfp ports=$ports nicd=$nicd mpstat=$mpstat ru_hist=$ru_hist record=$record/$record_secs keep_phy=$keep_phy pattern=$pattern cells=$ncells slots=$slots tags=$tags tc_extra='$extra' tenant='$tenant' model=$model cap=$cap arrivals=$arrivals rate=$rate noclient=$noclient delay=$tdelay secs=$tsecs $(date +%T)"
  tenant_cleanup
  precheck $name || exit 1
  disk_check $name || exit 3
  ptp_record $name
  pre_snapshot $name
  [ -n "$l1sedf" ] && l1sed="$(cat $l1sedf)"
  if [ -n "$l1sed" ]; then sed -i -e "$l1sed" cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml; echo "   l1 yaml test edit: $(diff $O/pre_$name.cuphycontroller_F08_CG1.yaml cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml | grep -E '^[<>]' | tr -s ' ' | tr '\n' ' ')"; fi
  tenv=""
  if [ -n "$tenant" ]; then tenv="TENANT_MODE=$tenant TENANT_FORCE=$model TENANT_CAP=$cap TENANT_ARRIVALS=$arrivals TENANT_RATE=$rate TENANT_NOCLIENT=$noclient TENANT_DELAY=$tdelay TENANT_SECS=$tsecs"; [ -n "$xclass" ] && tenv="$tenv TENANT_EXTRA_CLASS=$xclass TENANT_EXTRA_MODELS=$xmodels"; fi
  env PATTERN=$pattern NCELLS=$ncells TC_EXTRA="$extra" MPSTAT=$mpstat RU_HIST=$ru_hist RECORD=$record RECORD_SECS=$record_secs KEEP_RAW=$keep_raw FS_OUT=$O DRY=$DRY $tenv bash $S/fullstack_run.sh $name $slots "" "$tags" $hook
  rc=$?
  restore_pre $name
  verify_restore $name
  if [ "$DRY" = "1" ]; then
    echo "   dry-run: report path on ${DRY_REPORT_DIR:-$R/prof/run61} ${DRY_REPORT_NAME:-sc_R0_16c}"
    python3 $S/r0_ladder_report.py --dir ${DRY_REPORT_DIR:-$R/prof/run61} --first-s $FIRST_S --out $O/ladder_table_dry.csv --append ${DRY_REPORT_NAME:-sc_R0_16c} || echo "!! report failed"
    continue
  fi
  [ $rc -ne 0 ] && echo "!! fullstack_run.sh returned $rc for $name"
  post $name $keep_phy
  python3 $S/r0_ladder_report.py --dir $O --first-s $FIRST_S --out $O/ladder_table.csv --append $name > $O/report_$name.txt 2>&1 && cat $O/report_$name.txt || echo "!! report failed for $name (see $O/report_$name.txt)"
  echo "== $name done $(date +%T)"
done
echo "LADDER_DONE_$PREFIX $(date +%T)"
