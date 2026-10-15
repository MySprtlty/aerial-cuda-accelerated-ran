#!/bin/bash
# b2p_run.sh — E2 B2': green contexts (gc=1) WITH the MPS daemon kept alive. Applies the prepared run2 copy with the flag-file
# guard (prof/tmp/tmp_run2_cuPHYcontroller.b2p.sh), touches the flag, runs ladder_E2_B2p.txt, then restores run2 from the
# saved original and removes the flag; records the run2 diff, the compute apps seen by nvidia-smi during traffic (MPS server +
# cuphycontroller = client attached), the MPS server log tail and the green-context SM counts.
R=$(cd "$(dirname "$0")/../../../.." && pwd); O=$R/prof/run62; cd $R
RUN2=testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh; ORIG=prof/tmp/tmp_run2_cuPHYcontroller.orig.sh; PATCHED=prof/tmp/tmp_run2_cuPHYcontroller.b2p.sh
FLAG=prof/tmp/tmp_gc_keep_mps
[ "$(sha256sum < $RUN2)" = "$(sha256sum < $ORIG)" ] || { echo "!! run2 differs from the saved original - not applying"; exit 1; }
cp $PATCHED $RUN2 && touch $FLAG && echo "-- run2 guard applied, flag $FLAG set $(date +%T)"
diff $ORIG $RUN2 > $O/run2_guard_applied.diff
( sleep 80; for i in 1 2 3; do echo "-- $(date +%T) compute apps:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; ps -eo pid,args --no-headers | grep '[n]vidia-cuda-mps-control -d' | cut -c1-80; sleep 20; done ) > $O/mps_clients_E2_B2p_16c.txt 2>&1 &
SAMP=$!
ALLOW_GC=1 FS_OUT=$O bash $R/cuPHY-CP/dapp_hook/sweep/fullstack/r0_ladder.sh $O/ladder_E2_B2p.txt E2; rc=$?
wait $SAMP 2>/dev/null
docker exec -u root c_aerial_troy bash -c 'echo "-- MPS server log tail:"; tail -15 /var/server.log 2>/dev/null; echo "-- MPS control log tail:"; tail -8 /var/control.log 2>/dev/null' > $O/mps_log_E2_B2p_16c.txt 2>&1
cp $ORIG $RUN2 && rm -f $FLAG && echo "-- run2 restored, flag removed $(date +%T)"
diff $ORIG $RUN2 > $O/run2_guard_restored.diff && echo "   run2 identical to the original (restore diff empty)" || echo "!! run2 still differs from the original"
grep -h 'green context with SM count\|GC_KEEP_MPS\|stopping MPS\|restarting MPS\|MPS is enabled' $O/run2_E2_B2p_16c.log 2>/dev/null | cut -c1-150
cat $O/mps_clients_E2_B2p_16c.txt; cat $O/mps_log_E2_B2p_16c.txt | cut -c1-150
exit $rc
