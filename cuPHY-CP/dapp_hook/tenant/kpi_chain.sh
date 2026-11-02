#!/bin/bash
# kpi_chain.sh (2026-11-02): the 1-cell KPI measurement set (loopback, lab yamls): fixed 26x / fixed 26l / map policy at 4 and 8 req/s,
# Average and Peak patterns. ~5 min per run. Results: prof/tenant/ra_<name>.txt (KPI summary) + summary_<name>.txt (cuPHY).
set -u
R=$(cd "$(dirname "$0")/../../.." && pwd); cd $R
K=cuPHY-CP/dapp_hook/tenant/kpi_run.sh
bash $K k1_26x_av4   average 1 4 fixed:26x 40
bash $K k1_26l_av4   average 1 4 fixed:26l 40
bash $K k1_map_av8   average 1 8 map 40
bash $K k1_dl_av8    average 1 8 deadline 30
bash $K k1_map_pk8   peak    1 8 map 40
bash $K k1_26x_av12  average 1 12 fixed:26x 40
bash $K k4_26x_av4   average 4 4 fixed:26x 40
bash $K k4_map_av8   average 4 8 map 40
echo "KPI CHAIN DONE $(date +%T)"
