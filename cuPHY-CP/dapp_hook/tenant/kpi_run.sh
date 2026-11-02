#!/bin/bash
# kpi_run.sh (2026-11-02): one KPI measurement = an L1 loopback run (tn_runs.sh, tenant + xApp agent on) with the RA
# stand-in sending 4K requests at a given rate and policy from the host while the traffic runs.
#   kpi_run.sh <name> <pattern average|peak> <ncells> <rate req/s> <policy> [deadline_ms=40] [slots=240000] [extra TENANT_ENV]
# e.g. kpi_run.sh k1_26x average 1 4 fixed:26x ; kpi_run.sh k1_map average 1 8 map 40
# Outputs in prof/tenant: the usual tn_runs artefacts (summary_<name>.txt, phy_<name>.csv, dapp_tenant_<name>.{log,csv}) plus
# ra_<name>.csv / ra_<name>.txt (per-request E2E, server time, model, SM; KPI summary).
set -u
R=$(cd "$(dirname "$0")/../../.." && pwd); cd $R
name=$1; pat=$2; ncells=$3; rate=$4; policy=$5; deadline=${6:-40}; slots=${7:-240000}; extra=${8:-}
O=prof/tenant; mkdir -p $O
export NCELLS=$ncells SLOTS=$slots
export PUSCH_PROFILES=${PUSCH_PROFILES:-16,24,32,40,56} PDSCH_PROFILES=${PDSCH_PROFILES:-24,40}
# tn_runs.sh reads the caller's extra L1 env from TENANT_ENV (appended after its own); TENANT_CLASSES is read by fullstack_run.sh
export TENANT_CLASSES=${CLASSES:-40,48,56,64,72,80,88,96,104,112,120}   # 2026-11-02: 8-SM ladder (classes without an engine file are skipped per model)
export TENANT_ENV="-e DAPP_AGENT=1 -e DAPP_AGENT_CELLS=$ncells ${extra}"
# 2026-11-02: at 1 cell the yaml mps_sm_ul_order (20 -> 24 SMs) is lowered to ORDER_SMS (default 8; the order kernel is one block
# per cell) so the union layout keeps 80 SMs for YOLO when the fallback guard is off; fullstack_run.sh restores the yaml afterwards.
if [ "$ncells" = "1" ] && [ "${ORDER_SMS:-8}" != "0" ]; then export SM_OVERRIDE="${SM_OVERRIDE:-} ul_order=${ORDER_SMS:-8}"; fi
echo "== kpi_run $name: $pat x $ncells cells, $rate req/s, policy $policy, deadline $deadline ms $(date +%T)"
bash cuPHY-CP/dapp_hook/tenant/tn_runs.sh $name:$pat:tenant > $O/kpi_console_$name.log 2>&1 &
RUN=$!
# wait for the agent port (the L1 must be up and the tenant started), then for the traffic to begin
for i in $(seq 1 240); do (echo > /dev/tcp/127.0.0.1/9810) 2>/dev/null && break; sleep 1; done
echo "   agent port open after $i s"
sleep ${KPI_WARMUP:-20}
dur=$(( slots / 2000 - ${KPI_WARMUP:-20} - 15 )); [ $dur -lt 10 ] && dur=10
docker exec c_aerial_troy bash -c 'ps -eLo psr,comm | grep -E "dapp_agent|dapp_tenant|dapp_export|msg_process|timer_thread"' | sort | uniq -c | tr "\n" ";"; echo
taskset -c ${RA_CORES:-65-71} python3 cuPHY-CP/dapp_hook/python/ra_stub.py --rate $rate --duration $dur --policy $policy --deadline-ms $deadline --csv $O/ra_$name.csv > $O/ra_$name.txt 2>&1
echo "   ra_stub done $(date +%T)"; grep -E "requests|E2E|models|^  26" $O/ra_$name.txt
wait $RUN
grep -E "^#####|L1 did not|PUSCH completion pos [789]|PDSCH compression done pos [02]|DL U LATE SLOTS|L1 error|tenant last|disk" $O/kpi_console_$name.log | cut -c1-220
