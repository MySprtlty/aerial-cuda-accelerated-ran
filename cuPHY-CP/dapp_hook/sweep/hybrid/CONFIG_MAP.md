# 하이브리드 설정 지도 (2026-10-29, 기준 반전판) — 기준 = 연구원들의 COTS UE 구성, 거기에 에뮬레이터 9셀을 덧붙인다

원칙(사용자 지침 10-29): 에뮬레이터 셀은 트래픽 생성용일 뿐이고, **COTS UE + Benetel RU + cuphycontroller + OAI MAC + corenet이 기준**이다.
그래서 하이브리드의 L1·L2 어댑터 yaml은 랩 파일을 고치는 것이 아니라 **`baseline/`의 실제 실행 파일에서 생성**하고, 랩 파일은 에뮬레이터 셀 블록과
§3에 열거한 소수의 키만 제공한다. 기준에서 벗어난 값은 전부 §3 표에 있고, 각각 (a)~(d) 분류와 이유가 붙어 있다. 기준 구성 자체의 조사 기록은 `BASELINE.md`.

## 1. 기준 구성의 실행 구조 (nvidia 계정, `~/hsw/25_3_hsw/openairinterface5g/ci-scripts/yaml_files/sa_gnb_aerial/`에서 `docker compose up -d`)

```
[호스트] Open5GS 네이티브 (sh/start_cn.sh = systemctl start …; sudo sh/routing.sh = ip_forward + NAT 10.45.0.0/16→enx705dccffd874) · WebUI :9999 · ptp4l/phc2sys · Benetel RU 8c:1f:64:d1:15:11 ↔ aerial00(0002:01:00.0)
[컨테이너 1] nv-cubb   25-3-cubb 이미지, privileged, net host, ipc shareable, GPU
     볼륨 ~/hsw/25_3_hsw/aerial-cuda-accelerated-ran → /opt/nvidia/cuBB (nvidia 트리: 소스 + build.aarch64(08-02, SCF_FAPI_10_04 OFF, phyprach_aggr.cpp 수정 포함) + config)
     entrypoint: MPS 재시작 → cuphycontroller_scf config_benetal → 읽는 파일 cuphycontroller_config_benetal.yaml (l2adapter_filename → l2_adapter_config_P5G_GH.yaml, nvlog_config.yaml) → /dev/shm/nvipc_* 생성
[컨테이너 2] oai-gnb-aerial_hsw   oai-gnb-aerial_hsw:latest, cpuset 13-18, ipc container:nv-cubb, net host
     볼륨 ./original.txt → /opt/oai-gnb/etc/gnb.conf ; nr-softmodem: nfapi AERIAL, tr_s_shm_prefix nvipc(secondary), tr_s_poll_core 18, PCI 0, 273 PRB, PRACH 152, TDD 6/6/3/0, AMF 127.0.0.5, e2_agent
```

## 2. 하이브리드가 읽는 파일과 그 출처

| 역할 | 하이브리드가 읽는 경로 | 만들어지는 방식 (`hybrid_config.sh apply` → `hybrid_yaml.py`) |
|---|---|---|
| L1 설정 | `cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml` (run2가 `cuphycontroller_scf F08_CG1`로 띄우므로 이 이름) | **`baseline/cuphycontroller_config_benetal.yaml` 전문** + cell_group_num 10 + 랩 nics 항목(nics[0]) + 랩 셀 블록 0..8(= L1 셀 0..8, 무변경) + 기준 셀 블록 = L1 셀 9 + §3.1의 키. (배치는 검증된 full6과 동일: `BENETEL_CELL_INDEX=last`) |
| L2 어댑터 | `…/config/l2_adapter_config_F08_CG1.yaml` | **`baseline/l2_adapter_config_P5G_GH.yaml` 전문** + §3.2의 3가지 |
| nvIPC transport 목록 | `…/config/nvipc_multi_hybrid.yaml` (생성, restore 시 삭제) | transport 0 `nvipc1` = 셀 [0..8](testMAC), transport 1 `nvipc` = 셀 [9](OAI, 기준 prefix); 풀 크기는 P5G_GH 값 |
| RU 에뮬레이터 | `cuPHY-CP/ru-emulator/config/config.yaml` (랩 파일, test_config.sh 처리 후) | `fix_beta_dl` 0(기준과 동일), `payload_validation 0` |
| testMAC | `cuPHY-CP/testMAC/testMAC/test_mac_config.yaml` (랩) + 패턴 `testVectors/multi-cell/launch_pattern_F08_9C_tmp_ddddddsuuu_peak.yaml`(생성·등록) | shm prefix `nvipc1` |
| OAI gNB | `openairinterface5g/ci-scripts/yaml_files/sa_gnb_aerial_hybrid/{docker-compose.yaml,.env,gnb_hybrid.conf}` | gnb_hybrid.conf = `baseline/gnb_original.txt` + poll core만 변경(§3.5) |
| L1/testMAC 바이너리 | `build.aarch64_f1002/…` (FAPI 레이아웃 플래그 3개 = 기준 빌드와 동일: SCF_FAPI_10_04 OFF, **ENABLE_CONFORMANCE_TM_PDSCH_PDCCH OFF**, SCF_FAPI_10_04_SRS ON; §3.6) | + `phyprach_aggr.cpp`의 env-gated 변경, L2A `staticSlotCellMask`(§3.6) |
| 값의 단일 출처 | `benetel_cell.env` | 기준에서 벗어나는 키 목록·코어·prefix·이미지 태그 |
| 코어망 | `/home/nvidia/hsw/25_3_hsw/sh/` 그대로 | 변경 없음 |

## 3. 기준에서 벗어난 값 전부 (수정값 | 기준값 | 랩값 | 분류 | 이유)
분류: (a) UE가 보는 프로토콜·RF·타이밍 → **기준값 유지**(벗어난 것 없음) · (b) 9+1셀을 위한 용량 · (c) KPI 도구가 읽는 계측 · (d) c_aerial_troy의 코어 배치(컨테이너 내부 자원, UE는 모름)

### 3.1 L1 `cuphycontroller_F08_CG1.yaml` (기준 `cuphycontroller_config_benetal.yaml`에서 생성; 아래 외 모든 전역 키는 기준값)
| 키 | 하이브리드 | 기준 | 랩(10셀) | 분류 | 이유 |
|---|---|---|---|---|---|
| `cell_group_num` | 10 | 1 | 10 | b | 1 + 9 |
| `nics` | `{0000:01:00.0, mtu 1514, …}`(랩, nics[0]) + 기준 항목(0002:01:00.0, mtu 9000, nics[1]) | 1개 | 1개(0000) | b | 에뮬레이터 NIC. **순서**: 기준 NIC을 nics[0]에 두면(ue1/ue2 10-29) cuphydriver의 NIC-0 가정에 걸린다(Debug task nullptr; 에뮬레이터 PUSCH 전부 CRC 실패) → 검증된 full6 순서 |
| `cells` | 랩 블록 0..8 그대로(L1 셀 0..8, cell_id 1..9) + 기준 블록 → `O-RU 9 (Benetel)`, cell_id 10 = L1 셀 9 | 1개 | 20개 | b | cell_id(M-plane id)는 중복 금지(yamlparser.cpp:366). Benetel을 셀 0에 두는 배치(`BENETEL_CELL_INDEX=first`)는 위 NIC 순서와 같은 이유로 실험용 |
| `mps_sm_pusch/pucch/prach/ul_order/pdsch` | 100 / 2 / 2 / 20 / 102 | 120 / 20 / 20 / 40 / 40 | 같음 | b | 10셀 랩 측정·SafeCap 표의 전제(CONTEXT §9 보고 대상). pdcch 10·pbch 2·gpu_comms 16은 양쪽 동일, srs는 기준 0 유지(enable_srs 0) |
| `prach_aggr_per_ctx` | 4 | 2 | 2 | b | OAI PRACH(슬롯 9/19)와 에뮬레이터 PRACH(슬롯 19)가 겹치면 2로는 "No available Aggr PRACH objects"(10-26 실측) |
| `use_batched_memcpy` | 1 | 없음(=0) | 1 | b | 10셀 DL H2D 배치 복사(랩 10셀 런의 전제) |
| `low_priority_core / dpdk_thread / h2d_copy_thread_cpu_affinity / workers_ul / workers_dl` | 4 / 4 / 5 / [6,7] / [8,9,10] | 10 / 10 / 11 / [4,5] / [6,7,8] | 같음 | d | 컨테이너 코어 플랜(L1 4-14, testMAC 15-18, 에뮬레이터 20-41, 훅 43, 세입자 44-55, OAI 56-62) |
| `debug_worker` | **-1 (기준)** | -1 | 13 | — | 랩값 13으로 첫 런(ue1) 즉시 segfault: 슬롯마다 도는 Debug task가 DL 셀이 없는 NIC의 `prepare_tx_dlbuf_per_nic`(nullptr)에 `waitPrePrepareStop`을 호출(task_function_dl_aggr.cpp, dlbuffer.cpp:345; gdb1 백트레이스). 2 NIC + 기준 NIC이 첫 항목인 구성에서만 발생. KPI 도구는 이 worker의 로그를 쓰지 않음 |
| `log_level / enable_cpu_task_tracing / enable_prepare_tracing / pmu_metrics / ul_order_timeout_gpu_log_enable / ul_order_timeout_log_interval_ns` | DBG / 1 / 1 / 3 / 1 / 0 | INFO / 0 / 0 / 없음 / 0 / 1e9 | 같음 | c | KPI 도구(tags 311/223, R0 분석)가 DBG 로그 행을 읽음 |
| `l2adapter_filename` | l2_adapter_config_F08_CG1.yaml | l2_adapter_config_P5G_GH.yaml | — | — | 파일 이름뿐(내용은 §3.2) |

**기준값을 그대로 둔 것 중 랩과 다른 것(= 이전 하이브리드와 달라진 점)**: `fix_beta_dl 0`(랩 1), **`ul_order_timeout_cpu_ns 4 ms`**(랩 8), `pusch_waitTimeOut* 5000`(랩 3000), `max_ru_unhealthy_ul_slots 0`(랩 100),
**`pusch_workCancelMode` 없음=0**(랩 2 — 연구 비교 시 유의), `ul_pcap_capture_enable` 없음=0(랩 1), `srs_aggr_per_ctx 2`/`ul_input_buffer_per_cell_srs 4`(랩 3/6), `mps_sm_srs 0`(랩 16), bfw_*/static_beam_* 키 없음(랩 있음), `pusch_nMaxLdpcHetConfigs 32`, `enable_csip2_v3 0`.

### 3.2 L2 어댑터 `l2_adapter_config_F08_CG1.yaml` (기준 `l2_adapter_config_P5G_GH.yaml`에서 생성)
| 키 | 하이브리드 | 기준 | 랩 | 분류 | 이유 |
|---|---|---|---|---|---|
| `timer_thread_config.cpu_affinity / message_thread_config.cpu_affinity` | 11 / 12 | 9 / 9 | 11 / 12 | d | 기준의 9는 랩 DL 워커 9와 충돌; 10셀 FAPI는 타이머와 분리 |
| `nvipc_config_file` | nvipc_multi_hybrid.yaml | null | 'null' | b | L2가 둘 |
| `dapp_hook: {…}` | 랩 행 그대로 | 없음 | 있음 | c | 연구 훅(링 /aerial_dapp_ring, export core 43) |
| `allowed_fapi_latency` | 1 (랩/NVIDIA 기본) | 0 | 1 | b | testMAC은 슬롯마다 9셀의 메시지를 한 번에 보내 마지막 셀이 한 tick 늦게 도착한다. 0이면 그 셀의 DL_TTI/UL_DCI/TX_DATA가 폐기(ue2: Err.ind 98 k, 0x03/0x06, 셀 9에 집중 + 'Dropping the slot command'). OAI 단일 셀 메시지는 지연 0이라 영향 없음 |
| `enable_precoding`, `enable_beam_forming` | **0 / 0 (기준)** | 0 / 0 | 1 / 1 | — | L2A 전역 스위치(ssb_pbch/pdsch 명령 빌더가 모든 셀에 적용 → Benetel 셀 C-plane에도 영향). ue4(10-29)에서 추측으로 1/1 + L1 bfw_*/beam-id 키 14개를 넣었으나 UL 실패의 원인은 static slot 키였음. **ue7(10-29 18:42)**: 0/0 + L1 키 제거로 에뮬레이터 UL CRC 0·OAI 셀 DL 0.04 Mbps 유지, Err.ind 0, RU DL U late 슬롯 88 → 24/셀로 감소 → 기준값으로 확정 |
| `staticPuschSlotNum / staticPucchSlotNum / staticPdcchSlotNum / staticPdschSlotNum / staticCsiRsSlotNum` | 0 (랩) **+ `staticSlotCellMask` = 에뮬레이터 셀(0x1FF)** | 없음(= -1, 실제 슬롯 번호) | 0 | b | **F08 TV는 전부 슬롯 0용(`_s0`)**: 랩은 이 키로 모든 PUSCH/PUCCH/PDSCH/PDCCH/CSI-RS 명령의 슬롯 번호를 0으로 고정해 어느 슬롯에서든 TV가 복호된다(scf_5g_slot_commands.cpp:158·:658, pdcch.cpp:79, pdsch_csirs.cpp:2989, scf_5g_fapi_phy.cpp:1153). 실제 슬롯 번호면 에뮬레이터 PUSCH 전부 CRC 실패(ue2~ue4), 모든 셀 0 고정이면 **실 UE는 슬롯 0 밖의 PDSCH/PUSCH를 못 쓴다**(final8 10-27 "RA 920회 WAIT_Msg3"의 진짜 원인). 키가 전역이라 **하이브리드 빌드에 셀 마스크를 추가**(§3.6): 마스크 안 셀만 고정, Benetel 셀은 실제 슬롯·동적 SFN |
| `staticSsbPcid -1, staticSsbSFN -1` | 랩 그대로(-1 = 비활성) | 없음 | -1 | — | 값 동일; 마스크 안 셀의 SSB SFN은 랩처럼 정적(0), Benetel 셀은 tick SFN |

기준값 유지(랩과 다름): **`enableTickDynamicSfnSlot 1`**(랩 0; Benetel 셀의 PBCH SFN을 tick에서 가져감, ssb_pbch.cpp:72; 에뮬레이터 셀은 마스크로 랩 동작),
`prach_ta_offset_usec 2.5` 20개 인스턴스 전부(랩 없음; RACH.ind TA 보정에만 쓰임, scf_5g_fapi_phy.cpp:3359), `timer_thread_wakeup_threshold` 없음=15000(랩 20000), `pcap_enable 1`(→ §3.3에서 OAI transport에만).

### 3.3 `nvipc_multi_hybrid.yaml` (생성)
| 값 | 이유 |
|---|---|
| `wait_all_cell_config 1` | 모든 셀 CONFIG 전까지 SLOT.ind 없음 → OAI를 testMAC보다 먼저 기동. 0 + 미구성 셀 = L1 segfault(실측) |
| transport 0: `prefix nvipc1`, `phy_cells [0..8]`, `pcap_enable 0` | testMAC wire id 0..8 → L1 셀 0..8 (full6과 같은 transport 0 = 에뮬레이터) |
| transport 1: `prefix nvipc`, `phy_cells [9]`, **`pcap_enable 1`**, pcap core 14 | OAI는 gnb.conf 그대로(`nvipc`, cell_id 0) → L1 셀 9 = Benetel. 기준의 FAPI pcap(실 셀 디버깅용, 2 GB 상한)을 OAI transport에만 유지; 코어는 기준 10(랩 DL 워커) 대신 14 |
| 풀 크기 | P5G_GH 값(랩과 동일) |

### 3.4 RU 에뮬레이터 `config.yaml` / testMAC `test_mac_config.yaml` (랩 파일 편집)
| 키 | 하이브리드 | 랩 | 이유 |
|---|---|---|---|
| ru `fix_beta_dl` | 0 | 1 | L1 전역과 동일해야 함(기준 0) |
| ru `payload_validation` | 0 | 1 | fix_beta_dl 0이면 L1의 계산 beta(146,561)로 압축된 바이트가 TV의 65,536 기준 바이트와 달라 매 DL 슬롯 memcmp 실패(10-27). DL 타이밍 검증·UL은 그대로 |
| testMAC `shm_config.prefix` | nvipc1 | nvipc | OAI가 기준 prefix `nvipc`를 가져감 |

### 3.5 OAI (`sa_gnb_aerial_hybrid/`)
| 항목 | 하이브리드 | 기준 | 이유 |
|---|---|---|---|
| `gnb_hybrid.conf` | `baseline/gnb_original.txt`와 **`tr_s_poll_core 62`만** 다름 | poll core 18 | 18 = testMAC builder 코어(15-18) |
| compose | OAI 서비스만, **ipc host**, cpuset **56-62**, GPU 예약 없음 | nv-cubb + OAI, ipc container:nv-cubb, cpuset 13-18 | L1은 c_aerial_troy(--ipc=host); 13-18은 L1 debug 13/pcap 14/testMAC 15-18과 충돌 |
| 이미지 | `oai-gnb-aerial_hsw:hybrid-uci` | `latest`(09-03) | latest + `oai_empty_uci.patch`(NR_IF_Module.c 1-hunk: num_ucis 0인 UCI.indication을 경고 후 폐기). 하이브리드 L1은 셀 그룹에 UCI-on-PUSCH가 있으면(에뮬레이터 셀) OAI 셀에도 빈 UCI.ind를 보내고 원본 OAI는 AssertFatal(10-27 final7). 단일 셀 기준에서는 발생하지 않음 |

### 3.6 빌드·실행
| 항목 | 하이브리드 | 기준 | 이유 |
|---|---|---|---|
| `SCF_FAPI_10_04` | OFF (`build.aarch64_f1002`) | OFF | 동일 |
| `ENABLE_CONFORMANCE_TM_PDSCH_PDCCH` | **OFF** (10-29 저녁 재구성·재빌드; 그 전 f1002와 랩 build.aarch64는 ON) | OFF | ON이면 `scf_fapi_dl_tti_req_t`에 `testMode` 1바이트가 끼어(scf_5g_fapi.h:636) DL_TTI.request 레이아웃이 SCF 222.10.02와 달라진다. testMAC도 같은 플래그로 빌드되어 랩에서는 양쪽이 일치했지만 OAI는 표준 레이아웃을 보내므로 L1이 nPDUs/nGroup을 밀려 읽음 → **ue1~ue5(그리고 10-26~27 full/final 전부)에서 OAI 셀 DL 0.00 Mbps = SSB·SIB1이 전파되지 않음**, 매 SIB1 슬롯 `PDU count mismatch DL_TTI=0 TX_DATA=1`(TX_DATA는 testMode 없음 → 정상 파싱). 기준 라이브 런은 DL 0.04 Mbps. 재빌드 뒤 ue6(10-29 18:00): OAI 셀 DL 0.04 Mbps, mismatch 0. `hybrid_run.sh`가 CMakeCache를 검사 |
| `SCF_FAPI_10_04_SRS` | ON (같은 재구성) | ON | 기준과 동일하게. SRS PDU/SRS.indication 형식만 바뀌고 `do_SRS 0`이라 메시지에는 안 나타남 |
| `phyprach_aggr.cpp` | `CUPHY_PRACH_FORCE_THR0` env가 있을 때만 모든 PRACH occasion `force_thr0`=그 값; `hybrid_run.sh`가 20.0 export | **무조건 20.0**(nvidia 작업 트리 수정, 빌드 포함) | 기준 L1 바이너리와 같은 PRACH 검출 동작. env 없으면 upstream과 동일 → 랩 빌드 무영향. cuPHY-CP 소스 변경이므로 보고 대상 |
| `log_level` | **INFO (기준)** — 10-30 ue21 이후. 랩 DBG는 10셀 L1에서 30분에 약 20 GB를 써 디스크를 가득 채움(teardown 중 restore 실패 → 재실행). KPI 도구용 짧은 런에서만 benetel_cell.env의 L1_KEYS_FROM_LAB에 다시 넣을 것 | INFO | DBG | 계측 | 디스크 |
| `phyprach_aggr.cpp` setup 버퍼 매핑 + `task_function_ul_aggr.cpp` PRACH 태스크 가드 2개 (`prach_buffer_map.patch`) | 이 슬롯에 PRACH PDU가 있는 셀만 stage-3 버퍼 범위를 받음(+범위 합 검사); 해제된 슬롯맵/launch 플래그 누락 시 PRACH 건너뜀 | 상류 코드(모든 셀에 정적 기회 수 범위) | **버그 수정**: PRACH 없는 셀(UE 셀)이 PRACH 슬롯(17)에 PUSCH/PUCCH로 공존하면 범위가 밀려 범위 밖 버퍼 → 세그폴트/UL 정지(ue8~ue20). 랩·1셀 기준에서는 미발현. CONTEXT §9 보고 |
| L2 어댑터 `staticSlotCellMask` (cuphyl2adapter nv_phy_config_option.hpp / nv_phy_module.cpp, scfl2adapter 4개 파일) | 새 yaml 키: static*SlotNum·staticSsb*가 마스크 안 셀에만 적용, 마스크 안 셀은 정적 SSB SFN(랩 동작) | 키 없음 | 키가 없으면 upstream과 동일(모든 셀 적용). 에뮬레이터 셀(TV 슬롯 0 고정)과 실 UE 셀(실제 슬롯)이 한 셀 그룹에 공존하기 위한 유일한 방법(TV 풀에 슬롯별 UL TV 없음: `_s0`뿐). cuPHY-CP 소스 변경, 보고 대상 |
| 기동 순서 | RU 에뮬레이터 → L1 → OAI(셀 0 CONFIG/START) → testMAC | nv-cubb → OAI | wait_all_cell_config 1 |
| 코어망 | 기준 그대로(start_cn.sh, routing.sh) | — | 변경 없음 |
