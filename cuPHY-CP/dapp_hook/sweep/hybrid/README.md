# sweep/hybrid - one cuphycontroller for the Benetel RU cell (OAI gNB, COTS UE, Open5GS) + N RU-emulator cells (testMAC)

Principle (2026-10-29): the researchers' proven COTS-UE configuration is the BASELINE. The hybrid L1/L2-adapter yamls are generated from the
byte copies in `baseline/`; the emulator cells are appended to it. Every value that deviates from the baseline is listed in `CONFIG_MAP.md` §3.
Survey of the baseline (manual, files, values, live run): `BASELINE.md`. Session state and next steps: `HANDOFF.md`. History of the first
(lab-based) design and its measurements: `DESIGN.md`.

| file | role |
|---|---|
| `baseline/` | byte copies of the baseline: L1 yaml (`cuphycontroller_config_benetal.yaml` = `_all_success`, `_uplink_heavy`), `l2_adapter_config_P5G_GH.yaml`, OAI `gnb_original.txt` (+ DL/UL-centric variants), compose, entrypoint, RU cfg, core scripts, Open5GS yamls, `phyprach_aggr_nvidia.diff` |
| `benetel_cell.env` | the few values on top of the baseline: keys copied from the lab file (capacity/instrumentation/cores), prefixes, cores, image tag, PRACH threshold |
| `hybrid_yaml.py` | generator: baseline L1 + 9 lab cell blocks; baseline L2A + cores + nvipc file; two-transport `nvipc_multi_hybrid.yaml`; emulator/testMAC one-line edits. Testable on copies (`--out-*`). |
| `hybrid_config.sh` | `apply [--ncells 9] [--slots S]` (backup lab files -> pattern -> phase-4 setup/test_config in c_aerial_troy -> hybrid_yaml.py -> parse check -> OAI .env), `restore`, `status` |
| `hybrid_run.sh` | `start [name]`: prechecks -> RU emulator -> cuphycontroller (`build.aarch64_f1002`, `CUPHY_PRACH_FORCE_THR0=20.0`) -> OAI gNB container (cell 0) -> testMAC -> monitor; `stop`; `status`. Logs to `prof/hybrid_runs/<name>/`, lab files restored unless KEEP_CONFIG=1 |
| `oai_empty_uci.patch` | the one OAI change (image `hybrid-uci` = `latest` + this hunk) |
| `prach_buffer_map.patch` | the cuphydriver fix (2026-10-30, applied in the tree): `PhyPrachAggr::setup` maps PRACH occasions to stage-3 buffers only for cells that carry PRACH in the slot (upstream gives every cell of the slot a range -> a PUSCH-only cell in a PRACH slot shifts the map -> segfault / UL stall the moment a COTS UE is RRC-connected); + 2 guards in the PRACH task |
| `ue_watch.sh` | `start/stop/report/tail <run>`: COTS-UE attach recorder (OAI RA/RRC/NAS lines, L1 Err.ind/CRC of the Benetel cell, Open5GS amf/smf, 30 s STAT with UE ping) -> `prof/hybrid_runs/<run>/ue_timeline.log`; `report` = attach phases + failure signals + last UE stats |
| `hybrid_report.sh` | emulator-side KPI summary of a run (UL CRC, RU late slots, Err.ind, OAI cell SIB1 bytes) |
| `stall_bt.sh`, `msg4_trigger.sh`, `gdb/` | diagnosis helpers used 10-30: attach gdb inside the container on the first stall symptom; send SIGPWR to the L1 (`pgrep -f cuphycontroller_scf`: the process renames itself `phy_main`) after the UE's Msg4 ack so gdb (`gdb/gdbcmds_sig`, `gdb/gdbcmds_burst`) snapshots every thread; `gdb/gdbcmds` = crash backtrace for `L1_EXTRA_ARGS="--gdb_script 'gdb -q -batch -x /opt/nvidia/cuBB/prof/hybrid_runs/gdbcmds --args'"` (copy to prof/hybrid_runs) |

Cell map (= the verified full6 layout, `BENETEL_CELL_INDEX=last`): L1 cells 0..8 = emulator (nics[0] 0000:01:00.0, nvIPC `nvipc1`, testMAC wire ids 0..8),
L1 cell 9 = Benetel (nics[1] 0002:01:00.0, nvIPC `nvipc` = the baseline prefix, OAI unchanged: cell_id 0 on the wire). Putting the Benetel first (cell 0 / nics[0])
trips NIC-0/cell-0 assumptions in the cuphydriver (10-29 ue1: Debug-task nullptr; ue2: every emulator PUSCH CRC-failed).

Typical session (host, user troy; the core network as in the researchers' manual):
```
/home/nvidia/hsw/25_3_hsw/sh/start_cn.sh && sudo /home/nvidia/hsw/25_3_hsw/sh/routing.sh     # if not already up (10-29: up)
cuPHY-CP/container/run_aerial.sh -d                                                             # once: the --rm container c_aerial_troy
cuPHY-CP/dapp_hook/sweep/hybrid/hybrid_config.sh apply --ncells 9 --slots 7200000               # 1 h of slots; prints every deviation from the baseline
cuPHY-CP/dapp_hook/sweep/hybrid/hybrid_run.sh start ue1                                         # Ctrl-C or 'hybrid_run.sh stop' ends it and restores the lab files
```
Never start the researchers' compose stack (`sa_gnb_aerial`, nv-cubb) while c_aerial_troy runs an L1, and vice versa (same GPU, NIC, DPDK prefix).
To run their stack as troy for a baseline check: `cd .../sa_gnb_aerial && HOME=/home/nvidia docker compose up -d` (the compose file mounts `~/hsw/...`).

Known decisions baked in (details in CONFIG_MAP.md §3): mps_sm_* and the CPU cores from the lab, prach_aggr_per_ctx 4, use_batched_memcpy 1,
allowed_fapi_latency 1 (testMAC's 9-cell burst), DBG logging/tracing for the KPI tools; everything else of the L1/L2A = baseline (fix_beta_dl 0 -> emulator
payload_validation 0, ul_order_timeout_cpu 4 ms, debug_worker -1, enableTickDynamicSfnSlot 1, no work-cancel, prach_ta_offset 2.5); OAI poll core 62, cpuset 56-62, image hybrid-uci;
L1 + testMAC built in build.aarch64_f1002 with the baseline build's FAPI flags (SCF_FAPI_10_04=OFF, ENABLE_CONFORMANCE_TM_PDSCH_PDCCH=OFF, SCF_FAPI_10_04_SRS=ON; hybrid_run.sh checks CMakeCache.txt - CONFORMANCE ON inserts a testMode byte into DL_TTI.request and OAI's DL_TTI is misparsed: no SSB/SIB1) plus the researchers' PRACH threshold change (env-gated) and the staticSlotCellMask L2A patch (static slot numbers only for the emulator cells).
