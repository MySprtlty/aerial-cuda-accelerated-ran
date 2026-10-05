#!/bin/bash
# Third MIG-off stage: full-stack reference ladder on the whole GPU (official 16-cell peak patterns, 2 min each),
# the counterpart of the same ladder inside MIG 4g that follows. Waits for E2E_PREP2_DONE.
cd "$(dirname "$0")/.."
LOG=runs/e2e_prep_resume.log
until grep -q "E2E_PREP2_DONE" $LOG 2>/dev/null; do pgrep -f "e2e_prep_resume2.sh" >/dev/null || { grep -q "E2E_PREP2_DONE" $LOG || { echo "$(date +%T) stage-2 driver gone without E2E_PREP2_DONE" | tee -a $LOG; exit 1; }; }; sleep 30; done
echo "$(date +%F_%T) == stage 11: full-stack reference ladder (whole GPU)" | tee -a $LOG
bash fullstack/ladder_runs.sh ref "59c 59e 65c 65a" > runs/ladder_ref.out 2>&1
grep -E "^== |L1 did not|testMAC did not|LADDER_DONE|Finished" runs/ladder_ref.out | tee -a $LOG
python3 fullstack/ladder_report.py ref_59c_16c ref_59e_16c ref_65c_16c ref_65a_16c > results/e2e_prep/ladder_ref.txt 2>&1
cat results/e2e_prep/ladder_ref.txt | cut -c1-200 | tee -a $LOG
echo "E2E_PREP3_DONE $(date +%T)" | tee -a $LOG
