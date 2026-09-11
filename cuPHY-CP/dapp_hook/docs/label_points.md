# cuphydriver 채널별 GPU 작업 시작/완료 계측 지점

Aerial 25-3, `cuPHY-CP/cuphydriver` 기준. 목적: 예측기 라벨용으로 채널별
(PUSCH, PDSCH, PDCCH, PUCCH, PRACH, SRS, + PBCH/CSI-RS) GPU 작업의
"시작"과 "완료"를 **추가 `cudaStreamSynchronize` 없이** 타임스탬프로 찍을 수
있는 지점. 줄 번호는 `feature/label` 분기 시점(develop `4ef8660`) 기준.

표기: **[sync-free]** = 그 자리에서 찍으면 GPU를 기다리지 않음.
**[poll]** = 이미 있는 non-blocking 폴링(`cudaEventQuery` / host flag) 결과가
참이 된 직후. **[blocking]** = 원래 코드가 이미 CPU를 멈추는 자리(추가 sync는
아니지만 폴 지연이 라벨에 섞임).

---

## 0. 먼저 알아야 할 공통 구조

### 0.1 스레드와 태스크

| 무엇 | 어디 | 비고 |
|---|---|---|
| 슬롯 커맨드 → 태스크 변환 | `l1_enqueue_phy_work` [cuphydriver_api.cpp:352](../../cuphydriver/src/common/cuphydriver_api.cpp#L352) | L2A `msg_processing` 스레드에서 호출됨. UL 태스크 생성 1229–1530, DL 1576–1900 |
| 태스크 실행 시각 | `task_ts_exec[]` [cuphydriver_api.cpp:1223](../../cuphydriver/src/common/cuphydriver_api.cpp#L1223) 이하 | UL은 `t0_slot ± UL_TASK*_OFFSET` (1265–1373), DL은 `tick_original` 즉시 (1664), DL 정리 태스크는 `t0_slot + 1 slot` (1882) |
| 워커 스레드 | `Worker::run` [worker.cpp:109](../../cuphydriver/src/common/worker.cpp#L109), 루프 379–430 | 이름 `UlPhyDriverNN` / `DlPhyDriverNN`, 코어는 yaml `workers_ul/workers_dl` (nrSim 90063: UL 6,7 / DL 8,9,10). SCHED_FIFO |
| 태스크 꺼내기 | `TaskList::get_task` [task.cpp:303](../../cuphydriver/src/common/task.cpp#L303) | `now ≥ ts_exec − threshold`일 때 우선순위 큐에서 pop → `Task::run` [task.cpp:197](../../cuphydriver/src/common/task.cpp#L197) |

한 슬롯의 UL은 보통 `TaskUL1AggrOrderKernel` → `TaskUL1AggrPucchPusch`(+Prach, +Srs) →
`TaskUL2Aggr` → `TaskUL3AggrEarlyUciInd` → `TaskUL3Aggr`(+`TaskUL3AggrSrs`) 순서,
DL은 `TaskDLFHCb` → `TaskDL1AggrPdsch` / `TaskDL1AggrControl` → `TaskDL1AggrCompression` →
`TaskDL2Aggr*` → `TaskDL3Aggr`(버퍼 정리) 순서다. 채널 GPU 작업의 **런치**는 `*1*`
태스크, **완료 감지**는 `*3*` 태스크(UL) / `TaskDL3Aggr`·`TaskDL2Aggr`(DL)에서 일어난다.

### 0.2 시계

| 시계 | 정의 | 쓰이는 곳 |
|---|---|---|
| `Time::nowNs()` | `std::chrono::system_clock` = CLOCK_REALTIME (ns) — [perf_metrics_utils.hpp:38](../../gt_common_libs/perf_metrics/include/perf_metrics/perf_metrics_utils.hpp#L38) 주석 | 모든 `timings.*`, `task_ts_exec`, L2A tick, dApp 링의 `ts_ns` |
| CUDA event | 두 이벤트 사이 경과시간만 (`cudaEventElapsedTime`) — [cuda_events.hpp](../../cuphydriver/include/cuda_events.hpp) `GetCudaEventElapsedTime` | `start_run`/`end_run` → `getGPURunTime()` |
| GPU `%globaltimer` | [cuphy_pti.hpp:110](../../../cuPHY/src/cuphy/cuphy_pti.hpp#L110) `__globaltimer()`; NIC PTP 레지스터와 보정 [cuphy_pti.cu:23–34](../../../cuPHY/src/cuphy/cuphy_pti.cu#L23) | DL GPU-comm의 PREPREP/PREP/TRIGGER 활동 스탬프 (`dh_gpu_start_times`) |

→ **절대시각 라벨**은 CPU `nowNs()`로 찍고, **GPU 순수 실행시간**은 CUDA event 경과로
얻는 조합이 이미 코드에 있다. GPU 쪽 절대시각이 필요하면 `%globaltimer`를 host-mapped
메모리에 쓰는 1-스레드 커널을 붙이면 되고(0.5절), Aerial이 같은 기법을 PTI에서 쓴다.

### 0.3 채널별 스트림과 우선순위 ([context.cpp](../../cuphydriver/src/common/context.cpp), 값이 작을수록 높음)

| 채널 | 스트림 변수 | 우선순위 | 줄 |
|---|---|---|---|
| UL order kernel (FH 패킷 대기) | `stream_order_pd`, `stream_order_srs_pd` | −5 | 721–722 |
| PUSCH phase1 / phase2 | `aggr_stream_pusch[PHASE1/2_SPLIT_STREAM1]` | −2 | 738–740 |
| PUSCH split 시 두 번째 세트 | `…_SPLIT_STREAM2` | 0 | 744–746 |
| UL BFW | `aggr_stream_ulbfw` | −2 | 758 |
| PUCCH | `aggr_stream_pucch[0]` / `[1]`(split) | −3 / −1 | 770 / 775 |
| PRACH | `aggr_stream_prach[0]` / `[1]`(split) | −3 / −1 | 783 / 788 |
| SRS | `aggr_stream_srs` | −2 | 799 |
| PDSCH, TB H2D copy | `aggr_stream_pdsch`, `H2D_TB_CPY_stream` | −4 | 807–808 |
| DL BFW | `aggr_stream_dlbfw` | −3 | 824 |
| PDCCH (DL/UL DCI 공용) | `aggr_stream_pdcch` | −4 | 836 |
| PBCH | `aggr_stream_pbch` | −4 | 845 |
| CSI-RS | `aggr_stream_csirs` | −4 | 852 |
| timing 스트림 | `stream_timing_dl/ul` | −1 | 863–865 |

nrSim 90063 yaml은 `enable_ul_cuphy_graphs: 1`, `enable_dl_cuphy_graphs: 1`이라 각
`cuphyRun*()`은 내부적으로 CUDA graph 런치다. 채널 객체는 `MpsCtx`(yaml `mps_sm_*`
SM 상한)를 `setCtx()`로 current로 두고 런치한다.

### 0.4 채널 객체의 공통 수명주기 (`PhyChannel`, [phychannel.hpp:190](../../cuphydriver/include/phychannel.hpp#L190))

```
setup(...)                       // CPU: 파라미터 → cuphySetup*(); start/end_setup 이벤트
[cudaStreamWaitEvent(order evt)] // GPU: 입력이 준비될 때까지 스트림 정지 (UL) / TB H2D 완료 대기 (PDSCH)
run()                            // cudaEventRecord(start_run) → cuphyRun*() → cudaEventRecord(end_run)
signalRunCompletionEvent()       // cudaEventRecord(run_completion)  [phychannel.cpp:399/415]
   또는 signalRunCompletion()    // 1-스레드 커널이 host flag(channel_complete_h)에 1을 씀 [phychannel.cpp:387]
```

완료 감지 원시연산 (모두 **추가 sync 없음**):

| 함수 | 방식 | 줄 |
|---|---|---|
| `waitRunCompletion(ns)` | host-pinned flag `channel_complete_h`를 busy-poll | [phychannel.cpp:429](../../cuphydriver/src/common/phychannel.cpp#L429) |
| `waitRunCompletionEventNonBlocking()` | `cudaEventQuery(run_completion)` | 456 → 359 |
| `waitStartRunEventNonBlocking()` | `cudaEventQuery(start_run)` — **"GPU가 실제로 시작했다"** 신호 | 380 |
| `PhyWaiter::checkAction()` | NOT_STARTED →(start_run done)→ STARTED →(run_completion done)→ COMPLETED | [task.cpp](../../cuphydriver/src/common/task.cpp) `PhyWaiter::checkAction` |
| `getGPURunTime()` | `elapsed(start_run, end_run)` ms — 두 이벤트가 끝난 뒤 부르면 즉시 반환 | [phychannel.cpp:560](../../cuphydriver/src/common/phychannel.cpp#L560) |

`start_run` 이벤트는 스트림 wait **뒤에** 기록되므로, 그 이벤트의 완료 시각이 곧
"입력(패킷/TB)이 갖춰져 커널이 돌기 시작한 GPU 시각"이다. 이것이 UL에서 특히 중요하다:
CPU 런치 시각(`start_t_ul_*_run`)은 T0 이전이고 실제 GPU 시작은 order kernel이 패킷을
다 받은 뒤다.

### 0.5 이미 있는 계측

| 계측 | 내용 | 켜는 법 / 출력 |
|---|---|---|
| `slot_map->timings` | 단계별 CPU `nowNs()` 스탬프. UL [slot_map_ul.hpp:44–92](../../cuphydriver/include/slot_map_ul.hpp#L44) (`start/end_t_ul_{pusch,pucch,prach,srs,bfw}_{cuda,run,compl,cb}`, order, rx_pkts), DL [slot_map_dl.hpp:44](../../cuphydriver/include/slot_map_dl.hpp#L44) (`…_{setup,run,compl}`, uprep, utx, compression, callback) | `SlotMapUl::printTimes` [slot_map_ul.cpp:528](../../cuphydriver/src/uplink/slot_map_ul.cpp#L528) ← `release(num_cells, true)` [ul_aggr_3:2731](../../cuphydriver/src/uplink/task_function_ul_aggr.cpp#L2731); `SlotMapDl::printTimes` [slot_map_dl.cpp:501](../../cuphydriver/src/downlink/slot_map_dl.cpp#L501) ← `release` 327→341. **NVLOGI** `[PHYDRV] SFN x.y PUSCH Aggr … times ===>` — nvlog에서 PHYDRV 태그를 INFO로 올리면 슬롯마다 나옴 |
| `getGPURunTime()` 계열 | 채널별 GPU 경과시간. PUSCH는 phase별 [phypusch_aggr.cpp:978–999](../../cuphydriver/src/uplink/phypusch_aggr.cpp#L978) (`getGPURunSubSlotTime`, `…PostSubSlotTime`, `…GapTime`, `getGPUPhaseRunTime`) | 계산만 있고 **소비처가 주석 처리됨**: [phypusch_aggr.cpp:1486](../../cuphydriver/src/uplink/phypusch_aggr.cpp#L1486), [phypucch_aggr.cpp:412](../../cuphydriver/src/uplink/phypucch_aggr.cpp#L412), [phypdsch_aggr.cpp:416](../../cuphydriver/src/downlink/phypdsch_aggr.cpp#L416). CSI-RS만 살아 있는데 이름이 틀림 [phycsirs_aggr.cpp:331](../../cuphydriver/src/downlink/phycsirs_aggr.cpp#L331) (`kPdschProcessingTime`에 넣음) |
| Prometheus 히스토그램 | `aerial_cuphycp_slot_processing_duration_us{channel=…}` [metrics.cpp:89–92](../../cuphydriver/src/common/metrics.cpp#L89) | `AERIAL_METRICS` 빌드 + 위 주석 해제 시 동작. 분포만 남고 슬롯별 값은 없음 |
| `TI_ADD` 태스크 계측 | 태스크 안 서브스텝별 `nowNs()` → `{TI} <task,sfn,slot,map,cpu> … name:ts,…` 한 줄 [task_instrumentation.hpp:133/148](../../cuphydriver/include/task_instrumentation.hpp#L133) | yaml `enable_cpu_task_tracing: 1` ([yamlparser.cpp:802](../../cuphycontroller/src/yamlparser.cpp#L802)), NVLOGI |
| PTI (`cuphy_pti`) | DL GPU-comm의 prepare/trigger 커널이 `%globaltimer`를 host 메모리에 기록 → `dh_gpu_start/stop_times` | yaml `enable_prepare_tracing: 1` → Debug 태스크 [task_function_dl_aggr.cpp:279–289](../../cuphydriver/src/downlink/task_function_dl_aggr.cpp#L279) (등록 [cuphydriver_api.cpp:1897](../../cuphydriver/src/common/cuphydriver_api.cpp#L1897)). 채널 단위가 아니라 슬롯 FH 전송 단위 |
| `cubb_gpu_test_bench` | [testBenches/cubb_gpu_test_bench/cubb_gpu_test_bench.cpp](../../../testBenches/cubb_gpu_test_bench/cubb_gpu_test_bench.cpp): cuPHY 파이프라인을 L2/FH 없이 직접 돌리는 독립 벤치 (`-u` 모드로 DDDSUUDDDD 패턴, `--U/--D/…` 컨텍스트 분리) | full stack에서는 **쓸 수 없음** (cuphydriver 태스크 체계를 안 거침). 대신 full stack에는 위 `start_run/end_run` 이벤트가 같은 자리에 있으므로 필요 없음 |
| CUPTI 커널 트레이스 (nsys `-t cuda`) | 커널 단위 시각 | **부적합**: 어제 실측에서 L1 `msg_processing`이 2초 만에 FATAL. 라벨 수집용으로는 쓰지 말 것 |

### 0.6 슬롯 기준 시각(tick) — SLOT_END와 같은 출처인가

```
tti_gen::slot_indication_thread_sleep_method            nv_tick_generator.cpp:207
  clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, next)   :271, next += window_nsec :280
  → PHY_module: current_tick_.store(tick)               nv_phy_module.cpp:1373
               current_tick_list_[slot%10] = tick       :1382   (명목 tick, 그리드 정렬)
               l1_slot_ind_tick_[slot%10]  = ts_now     :1383   (실제 깨어난 시각)
  → process_phy_commands:
      slot_cmd.tick_original = curr_tick − slot_interval × slot_ns     :1123
      SLOT_END.tick_original_ns / t0_ns(= tick + slot_advance × slot_ns) :1220–1223
  → cuphydriver l1_enqueue_phy_work:
      t0_slot = sfn_to_tai(sfn, slot, tick_original + TAI, gps_alpha, gps_beta) − TAI   cuphydriver_api.cpp:448
      correct_tick = t0_slot − slot_advance × slot_ns                                   :449
      slot_3gpp.t0_ = t0_slot, t0_valid_ = true                                         :450–451
      tick_original을 correct_tick으로 덮어씀 (다르면)                                   :462–465
      task_ts_exec[0] = tick_original  (= SlotMapUl::get_t0(), slot_map_ul.hpp:438)     :1223
```

결론: **같은 출처**(L2A tick, CLOCK_REALTIME)이되 cuphydriver가 SFN/slot 그리드에
스냅한다. 두 값의 차이는 NVLOGI `"SFN {}.{} L2A tick {} correct tick {} error {}"`
(:453)로 찍힌다. tick이 그리드에 정렬돼 있으면 SLOT_END의 `t0_ns` = cuphydriver의
`t0_slot`. UL 태스크 시각은 전부 `t0_slot` 상대값이고(order kernel `T0 −
UL_TASK1_ORDER_LAUNCH_OFFSET_FROM_T0_NS`, task2 `T0 + UL_TASK2_OFFSET…`, task3
`T0 + UL_TASK3_AGGR3_OFFSET…`), order kernel에는 `slot_start = t0 + TAI + (slotAhead−1)
× slot_ns`가 넘어가 GPU가 패킷 타임스탬프와 비교한다 ([ul_aggr_1_orderKernel:1855](../../cuphydriver/src/uplink/task_function_ul_aggr.cpp#L1855)).

---

## 1. PUSCH

| 단계 | 지점 | 스레드 / 동기화 | 표기 |
|---|---|---|---|
| **CPU 런치 시작** | `task_work_function_ul_aggr_1_pucch_pusch` [task_function_ul_aggr.cpp:144](../../cuphydriver/src/uplink/task_function_ul_aggr.cpp#L144) (split 모드면 `_1_pusch` :521). `timings.start_t_ul_pusch_cuda[0]` :260 → `pusch->setup()` :263 → order kernel 런치 대기 `waitOrderLaunched` :282–286 (CPU 폴, 1 ms 상한) → `cudaStreamWaitEvent(order run-completion)` :296 → `timings.start_t_ul_pusch_run[0]` :303 → `pusch->run(...)` :322/:414/:442/:467 → `signalRunCompletionEvent(phase2_stream)` :458/:483 → `timings.end_t_ul_pusch_cuda[0]` :495 (런치 **완료**이지 GPU 완료 아님) | UL 워커, 태스크 시각 `t0 − UL_TASK1_ORDER_LAUNCH_OFFSET` ([api:1268/1302/1320](../../cuphydriver/src/common/cuphydriver_api.cpp#L1268)) | [sync-free] |
| **GPU 시작 (실제)** | `cudaEventRecord(start_run, phase1Stream)` [phypusch_aggr.cpp:932/936](../../cuphydriver/src/uplink/phypusch_aggr.cpp#L932) — 위 :296의 스트림 wait 뒤에 큐잉되므로 **order kernel이 슬롯 패킷을 다 받은 시각**에 완료됨. CPU에서 감지: `pusch_waiter.checkAction()` → `WAIT_ACTION_STARTED` [ul_aggr_3:2473–2476](../../cuphydriver/src/uplink/task_function_ul_aggr.cpp#L2473) (`TI_ADD("Started PUSCH")`) | UL 워커(task 3), `cudaEventQuery` 폴 | [poll] |
| **GPU 완료** | `cuphyRunPuschRx` :949 뒤 `cudaEventRecord(end_run, phase2Stream)` :959/:970 (+ phase별 `end_run_ph1/ph2` :964/:969); `run_completion` 이벤트는 ul_aggr_1 :458/:483. 경과시간 `getGPURunTime()`, `getGPUPhaseRunTime(ph)`, `getGPURunSubSlotTime()`, `getGPURunGapTime()` :978–999 | 완료 뒤 어느 스레드에서든 즉시 읽힘 | [sync-free] |
| **CPU 완료 감지** | `task_work_function_ul_aggr_3` [:2185](../../cuphydriver/src/uplink/task_function_ul_aggr.cpp#L2185), 태스크 시각 `t0 + UL_TASK3_AGGR3_OFFSET` ([api:1366](../../cuphydriver/src/common/cuphydriver_api.cpp#L1366)). Wait Loop :2349 → `pusch_waiter.checkAction()` → `WAIT_ACTION_COMPLETED` :2478 → `timings.end_t_ul_pusch_compl[0] = nowNs()` :2479 → `pusch->validate()` :2482 → `start_t_ul_pusch_cb` :2485 → `pusch->callback()` :2486 → `end_t_ul_pusch_cb` :2487 | UL 워커, `cudaEventQuery` 폴 루프 (early-UCI 태스크 완료 후에만 검사 :2472) | [poll] |
| **CPU 완료 하한 = 지시 전송** | `PhyPuschAggr::callback` [phypusch_aggr.cpp:1340](../../cuphydriver/src/uplink/phypusch_aggr.cpp#L1340) → `ul_cb.callback_fn` :1368 → L2A 람다 [scf_5g_fapi_phy.cpp:4450](../../scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp#L4450) → `send_crc_indication` :4577 (`transport.tx_send` :4714 = nvIPC `tx_send_msg` [nv_phy_mac_transport.cpp:20/189](../../cuphyl2adapter/lib/nvPHY/nv_phy_mac_transport.cpp#L20), `notify` :4715) → `send_rx_data_indication` :4721 (tx_send :4897) → `send_uci_indication` :4474 / `send_rx_pe_noise_var_indication` :4478 | **UL 워커 스레드에서 동기 호출** (별도 지시 스레드 없음). CRC.ind가 nvIPC에 들어간 시각은 :4714 직전의 `nowNs()` 또는 `end_t_ul_pusch_cb` | [sync-free] |
| early-HARQ 경로 | `task_work_function_ul_aggr_3_early_uci_ind` :1517 — `waitCompletedSubSlotEvent`/`subSlotCompletedEvent` 폴 → `callback_fn_early_uci` :1613 → `send_early_uci_indication` [scf:4232](../../scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp#L4232) | UL 워커 | [poll] |

주의: PUSCH는 phase1(sub-slot, 심볼 단위 처리)과 phase2(full-slot, LDPC)로 나뉘고 사이에
GPU wait kernel(`Pre/Post Early Harq Wait kernel`)이 늦은 심볼을 기다린다. `getGPURunTime()`
= start_run→end_run에는 이 **대기 시간이 포함**된다. 순수 연산 라벨은
`getGPUPhaseRunTime(PH1)+PH2`, 대기는 `getGPURunGapTime()`으로 분리하는 게 맞다.
`validate()`의 `cudaStreamSynchronize` [:1259/1266](../../cuphydriver/src/uplink/phypusch_aggr.cpp#L1259)는 H5 덤프 트리거 시에만 돈다.

## 2. PUCCH

| 단계 | 지점 | 스레드 / 동기화 | 표기 |
|---|---|---|---|
| CPU 런치 | 같은 task 1 ([:144](../../cuphydriver/src/uplink/task_function_ul_aggr.cpp#L144); split이면 `_1_pucch` :783 → run :914). `start_t_ul_pucch_cuda` :241 → `setup` :244 → `waitToStartGPUEvent(order evt, pucch_stream)` :346 → `start_t_ul_pucch_run` :352 → (sub-slot 모드면 PUSCH `subSlotCompletedEvent` 대기 :360) → `run()` :365 → `signalRunCompletionEvent(pucch_stream)` :381 → `end_t_ul_pucch_cuda` :388 | UL 워커 | [sync-free] |
| GPU 시작/완료 | [phypucch_aggr.cpp:368](../../cuphydriver/src/uplink/phypucch_aggr.cpp#L368) `start_run` → :372 `cuphyRunPucchRx` → :382 `end_run`. `getGPURunTime()` | | [sync-free] |
| CPU 완료 | ul_aggr_3 `pucch_waiter.checkAction()` :2495 → `end_t_ul_pucch_compl` :2501 → `start_t_ul_pucch_cb` :2507 → `pucch->callback()` :2508 ([phypucch_aggr.cpp:400](../../cuphydriver/src/uplink/phypucch_aggr.cpp#L400)) | UL 워커, `cudaEventQuery` 폴 | [poll] |
| 지시 전송 | `ul_cb.uci_cb_fn2` 람다 [scf:4499](../../scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp#L4499) → `send_uci_indication` :4503 (:4008 오버로드) | UL 워커 동기 | [sync-free] |

## 3. PRACH

| 단계 | 지점 | 스레드 / 동기화 | 표기 |
|---|---|---|---|
| CPU 런치 | `task_work_function_ul_aggr_1_prach` [:965](../../cuphydriver/src/uplink/task_function_ul_aggr.cpp#L965) (`TaskUL1AggrPrach` [api:1499](../../cuphydriver/src/common/cuphydriver_api.cpp#L1499)). `start_t_ul_prach_cuda` :1064 → `waitToStartGPUEvent(order evt)` :1090 → `start_t_ul_prach_run` :1092 → `run()` :1095 → `signalRunCompletionEvent` :1112 → `end_t_ul_prach_cuda` :1120 | UL 워커 | [sync-free] |
| GPU 시작/완료 | [phyprach_aggr.cpp:516](../../cuphydriver/src/uplink/phyprach_aggr.cpp#L516) `start_run` → :520 `cuphyRunPrachRx` → :529 `end_run` → 결과 D2H 복사 :539 `start_copy` … :573 `end_copy` (복사까지가 채널 작업). `getGPURunTime()` + copy 구간 | | [sync-free] |
| CPU 완료 | ul_aggr_3 `prach_waiter` → `end_t_ul_prach_compl` :2522 → `prach->callback()` :2529 ([phyprach_aggr.cpp:800](../../cuphydriver/src/uplink/phyprach_aggr.cpp#L800)) | UL 워커, 폴 | [poll] |
| 지시 전송 | `ul_cb.prach_cb_fn` → [scf:4486](../../scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp#L4486) → `send_rach_indication` :3216 (호출 :4496) | UL 워커 동기 | [sync-free] |

`validate`의 `cudaStreamSynchronize` [:784/791](../../cuphydriver/src/uplink/phyprach_aggr.cpp#L784)는 디버그 덤프 전용.

## 4. SRS

| 단계 | 지점 | 스레드 / 동기화 | 표기 |
|---|---|---|---|
| CPU 런치 | `task_work_function_ul_aggr_1_srs` [:1145](../../cuphydriver/src/uplink/task_function_ul_aggr.cpp#L1145), 태스크 시각 `t0 + UL_TASK1_SRS_LAUNCH_OFFSET` ([api:1517](../../cuphydriver/src/common/cuphydriver_api.cpp#L1517)). `start_t_ul_srs_cuda` :1225 → `waitToStartGPUEvent(order / srs-order evt)` :1252/:1254 → `start_t_ul_srs_run` :1258 → `run()` :1261 → `signalRunCompletionEvent` :1278 → `end_t_ul_srs_cuda` :1286 | UL 워커 | [sync-free] |
| GPU 시작/완료 | [physrs_aggr.cpp:638](../../cuphydriver/src/uplink/physrs_aggr.cpp#L638) `start_run` → :641 `cuphyRunSrsRx` → :648 `end_run` | | [sync-free] |
| CPU 완료 | 전용 태스크 `task_work_function_ul_aggr_3_srs` [:2741](../../cuphydriver/src/uplink/task_function_ul_aggr.cpp#L2741): `srs_waiter.checkAction()` :2861 → `end_t_ul_srs_compl` :2867 → `srs->callback()` :2874 ([physrs_aggr.cpp:751](../../cuphydriver/src/uplink/physrs_aggr.cpp#L751)); ul_aggr_3 안에서 처리되는 구성이면 :2540–2552 | UL 워커, 폴 | [poll] |
| 지시 전송 | `ul_cb.srs_cb_fn` → [scf:4519](../../scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp#L4519) → `send_srs_indication` :5106 (호출 :4526) | UL 워커 동기 | [sync-free] |

## 5. PDSCH

nrSim 90063 yaml은 `gpu_init_comms_dl: 1` → DL U-plane 전송을 **GPU가 직접**(DOCA
GPUNetIO) 한다. 따라서 "처리된 슬롯을 FH로 넘기는" 지점이 CPU 함수 호출이 아니라
GPU 커널 체인 안에 있다. CPU 경로(`gpu_init_comms_dl: 0`)도 함께 적는다.

| 단계 | 지점 | 스레드 / 동기화 | 표기 |
|---|---|---|---|
| CPU 런치 | `task_work_function_dl_aggr_1_pdsch` [task_function_dl_aggr.cpp:459](../../cuphydriver/src/downlink/task_function_dl_aggr.cpp#L459) (`TaskDL1AggrPdsch` [api:1709](../../cuphydriver/src/common/cuphydriver_api.cpp#L1709), 태스크 시각 = `tick_original` 즉시 :1664 — **DL은 lead time 없음**). `start_t_dl_pdsch_setup` :501 → `pdsch->setup()` :506 (내부에서 스트림이 TB H2D 완료 이벤트를 wait [phypdsch_aggr.cpp:299](../../cuphydriver/src/downlink/phypdsch_aggr.cpp#L299)) → `end_t_dl_pdsch_setup`/`start_t_dl_pdsch_run` :516–517 → `pdsch->run()` :520 → `signalRunCompletionEvent(false)` 또는 `signalRunCompletion()` :536/:538 → `end_t_dl_pdsch_run` :547 → `addSlotEndTask` :553 | DL 워커 | [sync-free] |
| GPU 시작/완료 | [phypdsch_aggr.cpp:361](../../cuphydriver/src/downlink/phypdsch_aggr.cpp#L361) `start_run` (TB H2D wait 뒤) → :365 `cuphyRunPdschTx` → :374 `end_run`. `getGPURunTime()` | | [sync-free] |
| GPU → 압축 → FH (GPU-side) | `task_work_function_dl_aggr_1_compression` [:1033](../../cuphydriver/src/downlink/task_function_dl_aggr.cpp#L1033): 압축 스트림이 각 채널 완료를 **GPU에서** 기다림 `waitRunCompletionGPU[Event]` :1162–1200 → `cudaEventRecord(AllChannelsDoneEvt)` :1214 → `runCompression` :1241 ([dlbuffer.cpp:262/270](../../cuphydriver/src/downlink/dlbuffer.cpp#L262) `compression_start/stop_evt` → `getChannelToCompressionGap()` :446, `getCompressionExecutionTime()` :451) → GPU-comm 버퍼 ready 플래그 :1250–1252 | DL 워커가 큐잉만 함 | [sync-free] |
| FH 전송 (GPU-comm, 현재 설정) | `task_work_function_dl_aggr_2_gpu_comm` [:1588](../../cuphydriver/src/downlink/task_function_dl_aggr.cpp#L1588) / `_2_gpu_comm_tx` :2423: U-plane prepare :1666–1733 (`start/end_t_dl_uprep`) → `UserPlaneSendPacketsGpuComm` [fh.cpp:3455](../../cuphydriver/src/common/fh.cpp#L3455) :1806 (`start/end_t_dl_utx` :1788/:1815). 실제 패킷 송출 시각은 GPU 커널(PTI TRIGGER 활동)이 `%globaltimer`로 남김 → `enable_prepare_tracing` 시 :279–289에 출력 | DL 워커 | [sync-free] (GPU 스탬프) |
| FH 전송 (CPU-comm) | `task_work_function_dl_aggr_2` [:1312](../../cuphydriver/src/downlink/task_function_dl_aggr.cpp#L1312): `pdsch->waitRunCompletion(4 ms)` host-flag 폴 :1424 → `end_t_dl_pdsch_compl` :1429 (PDCCH_DL :1438/1443, PDCCH_UL :1452/1457, PBCH :1466/1471, CSI-RS :1480/1485) → 압축 대기 :1490–1506 → `UserPlaneSendPackets` [fh.cpp:3422](../../cuphydriver/src/common/fh.cpp#L3422) :1519 (`start/end_t_dl_utx` :1518/:1520) | DL 워커, host-flag 폴 | [poll] |
| CPU 완료 감지 (GPU-comm) | `task_work_function_dl_aggr_3_buf_cleanup` [:2008](../../cuphydriver/src/downlink/task_function_dl_aggr.cpp#L2008), 태스크 시각 `t0 + 1 slot` ([api:1882](../../cuphydriver/src/common/cuphydriver_api.cpp#L1882)). 채널별 `waitRunCompletionEventNonBlocking()` 폴: PDCCH_DL :2150→`end_t_dl_pdcchdl_compl` :2159, PDCCH_UL :2164→:2173, PBCH :2178→:2187, CSI-RS :2192→:2201, **PDSCH :2206→:2215**; FH 송출 완료 이벤트 `getTxEndEvt` :2234 (`non_blocking_event_wait_with_timeout` :47) | DL 워커, `cudaEventQuery` 폴 | [poll] |
| L2A 콜백 | `PhyPdschAggr::callback` [phypdsch_aggr.cpp:379](../../cuphydriver/src/downlink/phypdsch_aggr.cpp#L379) → `dl_cb.callback_fn` :406 → [scf:4385](../../scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp#L4385) → `on_dl_tb_processed` [nv_phy_module.cpp:1522](../../cuphyl2adapter/lib/nvPHY/nv_phy_module.cpp#L1522) (TX_DATA nvIPC 버퍼 반납). GPU-comm 경로에서는 TB H2D 완료만 확인하고 호출 :1828–1845 → **"TB 소비 완료"이지 "송출 완료" 아님** | DL 워커 | [poll] |

## 6. PDCCH (+ PBCH, CSI-RS: 같은 태스크·같은 완료 경로)

| 단계 | 지점 | 스레드 / 동기화 | 표기 |
|---|---|---|---|
| CPU 런치 | `task_work_function_dl_aggr_control` [:577](../../cuphydriver/src/downlink/task_function_dl_aggr.cpp#L577) (`TaskDL1AggrControl` [api:1733](../../cuphydriver/src/common/cuphydriver_api.cpp#L1733), 시각 `tick_original`). PDCCH_DL setup :622–630 / run :683(`start_t_dl_pdcchdl_run`) :684 :696/698(signal) :706(end); PDCCH_UL :637–645 / :712–735; PBCH :652–660 / :741–765; CSI-RS :667–675 / :771–790 | DL 워커 | [sync-free] |
| GPU 시작/완료 | PDCCH [phypdcch_aggr.cpp:255/258/266](../../cuphydriver/src/downlink/phypdcch_aggr.cpp#L255); PBCH [phypbch_aggr.cpp:215/219/228](../../cuphydriver/src/downlink/phypbch_aggr.cpp#L215); CSI-RS [phycsirs_aggr.cpp:289/293/301](../../cuphydriver/src/downlink/phycsirs_aggr.cpp#L289). 각각 `getGPURunTime()` | | [sync-free] |
| CPU 완료 | 5절의 buf_cleanup 폴(:2150–2201) 또는 CPU-comm 경로의 `waitRunCompletion` (:1438–1485) | DL 워커 | [poll] |
| L2A 콜백 | `PhyPdcchAggr::callback` :273, `PhyPbchAggr::callback` :233, `PhyCsiRsAggr::callback` :307 — DL 지시가 없으므로 메트릭/정리만 | | |

PDCCH는 DL_TTI의 DCI와 UL_DCI.request의 DCI를 **같은 슬롯 커맨드의 두 객체**
(`pdcch_dl`, `pdcch_ul`, [context.cpp:836–843](../../cuphydriver/src/common/context.cpp#L836))로 처리한다. 라벨을 채널 단위로 합칠 때 둘을 더해야 한다.

---

## 7. 라벨 수집 권장안 (우선순위 순)

### A. 새 GPU 작업 없이 (지금 바로 가능)

각 채널의 **CPU 완료 감지 지점 직후**(표의 `end_t_*_compl` 줄)에서

- `nowNs()` (이미 찍힘) = CPU가 완료를 인지한 시각(폴 지연 ≤ 수 µs)
- `chan->getGPURunTime()` (+ PUSCH는 phase/gap 시간) = GPU 순수 경과 (두 이벤트가
  끝난 뒤라 `cudaEventElapsedTime`은 즉시 반환)
- `timings.start_t_*_run[0]` = CPU 런치 시각, `slot_map->get_t0()` = T0

를 묶어 링 레코드로 내보내면 된다. 가장 깔끔한 단일 삽입점은 **슬롯 맵 release**
직전이다: UL [slot_map_ul.cpp:117–124](../../cuphydriver/src/uplink/slot_map_ul.cpp#L117) (`printTimes` 호출 자리, `aggr_pusch/pucch/prach/srs` 포인터와 `timings`가 모두 유효), DL [slot_map_dl.cpp:341](../../cuphydriver/src/downlink/slot_map_dl.cpp#L341). 채널 객체는 풀(`PHY_*_AGGR_X_CTX`개)에서 재사용되므로 release 전에 읽어야 이벤트가 다음 슬롯 것으로 바뀌지 않는다.

주의할 것 하나: 지금 dApp 링은 **단일 writer**(msg_processing)이다. UL/DL 워커에서
쓰려면 워커별 링(`/aerial_dapp_label_ul`, `_dl`)을 따로 만드는 게 가장 싸다
(atomic 예약을 넣는 MPSC보다 RT 경로 비용이 적음).

### B. GPU 절대시각이 필요할 때

`start_run` 직후와 `cuphyRun*()` 직후에 `%globaltimer`를 host-mapped 버퍼에 쓰는
1-스레드 커널을 큐잉한다. 원형은 [generic_cuda_kernels.cu:74–90](../../cuphydriver/src/common/generic_cuda_kernels.cu#L74) `launch_kernel_write`이고, globaltimer↔CLOCK_REALTIME 보정은 시작 시 한 번(cuPHY의 `cuphy_pti_calibrate_gpu_timer` [cuphy_pti.cu:23–34](../../../cuPHY/src/cuphy/cuphy_pti.cu#L23)와 같은 방식, PTP 레지스터 대신 `clock_gettime`으로도 충분). 비용은 채널당 슬롯당 커널 런치 2회(RT 스레드에서 각 2–3 µs)라 옵션으로 두어야 한다. 이 방식이면 "GPU가 언제 시작해 언제 끝났나"를 CPU 폴 지연 없이 얻는다.

### C. 먼저 눈으로 볼 때

코드 수정 없이: nvlog에서 `PHYDRV` 태그를 INFO로 올리면 슬롯마다
`[PHYDRV] SFN x.y PUSCH Aggr … times ===>` (setup/run/compl/cb 시각과 µs 차)가 찍히고,
yaml `enable_cpu_task_tracing: 1`이면 태스크별 `{TI}` 서브스텝 타임스탬프가 나온다.
둘 다 RT 경로에서 문자열 포맷을 하므로 측정용으로만 잠깐 켜는 것이 좋다.

---

## 8. 한눈에

| 채널 | CPU 런치 (worker, ts) | GPU 시작 이벤트 | GPU 완료 이벤트 | CPU 완료 감지 | CPU 완료 하한 (지시/FH) |
|---|---|---|---|---|---|
| PUSCH | ul_aggr_1 :303 (UL, T0−order offset) | phypusch :932/936 (order 완료 후) | :959/970 | ul_aggr_3 :2478–2479 | CRC.ind scf:4714, RX_Data.ind :4897 |
| PUCCH | ul_aggr_1 :352 | phypucch :368 | :382 | ul_aggr_3 :2495–2501 | UCI.ind (scf:4008) |
| PRACH | ul_aggr_1_prach :1092 | phyprach :516 | :529 (+copy :573) | ul_aggr_3 :2516–2522 | RACH.ind scf:3216 |
| SRS | ul_aggr_1_srs :1258 (T0+SRS offset) | physrs :638 | :648 | ul_aggr_3_srs :2861–2867 | SRS.ind scf:5106 |
| PDSCH | dl_aggr_1_pdsch :517 (DL, tick) | phypdsch :361 (TB H2D 후) | :374 | buf_cleanup :2206–2215 / dl_aggr_2 :1424–1429 | GPU-comm 송출 :1806 (GPU 스탬프) / CPU `UserPlaneSendPackets` :1519 |
| PDCCH | dl_aggr_control :683/:712 | phypdcch :255 | :266 | buf_cleanup :2150–2173 / dl_aggr_2 :1438–1457 | (PDSCH와 같은 슬롯 송출) |
| PBCH / CSI-RS | dl_aggr_control :741 / :771 | :215 / :289 | :228 / :301 | buf_cleanup :2178–2201 | 〃 |
