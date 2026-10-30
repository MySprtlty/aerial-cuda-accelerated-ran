# 하이브리드 L1 가능성 조사 — RU 에뮬레이터(testMAC) + Benetel RU(OAI gNB)를 한 cuphycontroller에 (2026-10-26, 읽기 전용 조사)

트리: A = /home/troy/build/aerial-cuda-accelerated-ran (Aerial 25-3-cubb), B = /home/troy/build/openairinterface5g (OAI, nvidia 계정 compose 프로젝트 사본).
이전 설계 초안(16셀, 10-16): prof/run64/hybrid_draft. 이 문서는 X=10 운영 설정(에뮬레이터 10셀 + Benetel 1셀 = 11셀)으로 다시 잘라 코드 사실을 재검증한 결과다.
조사 방식: 6개 축의 코드 읽기(L2 어댑터, OAI 연동부, testMAC, 인프라, Benetel 파라미터, FH 다중 NIC) + L2 어댑터 축의 핵심 주장 11건은 독립 재독자 2인이 반박 시도(11/11 유지).
GPU·컨테이너·바이너리는 일절 실행하지 않았다. 세부 근거(파일:행)는 research/*.json에 있다.

## 0. 질문 1·2에 대한 답
- B는 Aerial 코드가 아니다. OpenAirInterface 5G RAN 스택(gNB의 MAC/RLC/PDCP/RRC/NGAP + 자체 L1)이며, `nfapi/oai_integration/aerial/`가 Aerial L1(cuphycontroller)에 nvIPC/SCF FAPI로 붙는 L2 어댑터 역할을 한다. A는 L1(cuPHY, cuphycontroller, RU 에뮬레이터, testMAC). 즉 B = L2+, A = L1. B 안의 nvipc_src.2026.05.06은 A의 cuPHY-CP/gt_common_libs/nvIPC와 바이트 단위로 동일하다(라이브러리 버전 2530).
- B의 compose(sa_gnb_aerial)는 nv-cubb(Aerial 25-3-cubb 이미지, ~/hsw/25_3_hsw/aerial-cuda-accelerated-ran 마운트, config_benetal.yaml) + oai-gnb-aerial_hsw(gnb.conf = original.txt)를 올려 Benetel RU, COTS UE, 호스트의 Open5GS(AMF 127.0.0.5)와 연동한 구성이다. 현재도 nvidia 계정이 거의 매일 저녁 이 스택을 돌린다(10-25 22:14~22:15 실행, OAI exit 139로 종료).

## 1. 결론
**코드 수정 없이 설정만으로 가능하다(조건부).** 필요한 기능이 모두 25-3 트리에 있고 NVIDIA가 제공하는 테스트 경로(phase-4 Multi-L2, 2-port 템플릿)로 뒷받침된다.
단, 실제 시연 전에 리허설로 풀어야 하는 항목이 있다(3절). 가장 큰 것은 전역 키 `fix_beta_dl`(에뮬레이터 1 / Benetel 0) 충돌이며, 그 다음이 SLOT.indication 게이트·SLOT.response 지연 프로파일, OAI 재시작 시 전송 리셋 안전성, 두 NIC(CX7 + BF3) 혼합이다.

## 2. 코드 사실 (현 트리에서 확인; L2A 항목은 재독자 검증 통과)
| 항목 | 사실 | 근거 |
|---|---|---|
| L2 어댑터 다중 transport | `nvipc_config_file`이 가리키는 yaml의 `transport:` 목록, 항목마다 `transport_id`(= 인덱스), `phy_cells`(위치 = L2 셀 id, 값 = L1 셀), 자체 `prefix`. 합 = cell_group_num, 중복 금지, 위반 시 NVLOGF → L1 종료. 1셀 목록 허용 | A cuphyl2adapter/lib/nvPHY/nv_phy_mac_transport.cpp:383-486, :108-118 |
| 셀 id 재매핑 | rx: 범위 밖 셀 id는 로그 후 폐기, 아니면 cell_id·FAPI handle을 L1 셀로 재기록; tx는 역방향. OAI는 항상 cell_id/handle 0 → `phy_cells: [10]`이면 L1 셀 10 | nv_phy_mac_transport.cpp:181-194, :248-269; B nfapi/oai_integration/aerial/fapi_nvIPC.c:239,325,414 |
| SLOT.indication | tick 스레드 1개, STARTED transport마다 1통(해당 transport 첫 셀 → L2 id 0). **`wait_all_cell_config`(기본 1)면 L1의 11셀 전부 CONFIG 전까지 아무 transport에도 안 보냄**(latched) | nv_tick_generator.cpp:38-50; scfl2adapter/.../scf_5g_fapi_phy.cpp:672-692; nv_phy_module.cpp:1446-1451 |
| SLOT.response(ENABLE_L2_SLT_RSP ON) | 슬롯 처리 = 모든 active 셀의 0x8F 수신 또는 다음 loopback SLOT.ind 도착(타임아웃 없음). OAI·testMAC 모두 PDU 있는 슬롯에만 0x8F → OAI 셀이 비는 슬롯에는 10셀 처리가 다음 tick까지 미뤄짐(교착 아님, 지연 프로파일 변화) | nv_phy_module.cpp:718-746; scf_5g_fapi_phy.cpp:553-562; B nfapi_vnf.c:902-922; A testMAC/scf_fapi_handler.cpp:2831-2835 |
| CONFIG/START/STOP | 셀·transport 단위, 교차 순서 제약 없음(게이트 제외). 늦게 붙는 L2 허용 | scf_5g_fapi_phy.cpp:369-420, 2245-2268, 3077-3110 |
| transport 리셋 | L2 재접속 → 그 transport의 PHY 인스턴스만 리셋 + nvIPC ring/pool 리셋. L2A가 보유 중인 그 transport 버퍼 해제 시 pool 오염/예외 가능(위험) | nv_phy_module.cpp:91-132; nvIPC/common/nv_ipc_efd.c:279-309; shm/nv_ipc_shm_if.c:177-213 |
| 수치/TDD 가정 | mu·slot_advance·ss_curr·is_ul/is_dl이 모듈 전역(셀별 mu/TDD는 파싱만 함, 교차검사 없음) → OAI 셀도 mu 1·같은 슬롯 방향이어야 함(실제로 같음) | nv_phy_tick_updater.hpp:30-39; nv_phy_module.cpp:1461-1482, .hpp:532-543 |
| dapp_hook 링 | 셀 id 매핑 뒤 per-cell 핸들러에서 export → 두 transport 트래픽을 L1 셀 id로 모두 봄(cell 10 포함) | nv_phy_module.cpp:422-491; scf_5g_fapi_phy.cpp:1119-1123, 1413-1417 |
| OAI nvIPC secondary | prefix만 공급(tr_s_shm_prefix), ring/pool은 primary 복사, CUDA 불필요(libnvipc NVIPC_CUDA_ENABLE=OFF), L1 없으면 1 s 간격 무한 대기, SLOT.ind 지연에 abort 없음, L1 재시작 시 재접속 불가(OAI 재시작 필요), CONFIG.response 오류 시 로그 없이 멈춤, 프로세스당 셀 1개 | B fapi_nvIPC.c:483-574; nv_ipc_efd.c:352-365; B nfapi/open-nFAPI/vnf/src/vnf.c:490-526; gNB_scheduler.c:261 |
| OAI CONFIG.request | PCI 0, 273 PRB mu 1, TDD 주기 6(5 ms) 6/6/3/0 = DDDDDDSUUU(S = 6 DL 심볼) = 에뮬레이터 패턴, PRACH 152 = B4, 슬롯 9/19(UL 위치 9), 홀수 SFN만. 25-3 L1이 PCI 0을 수락한 로그 있음(길이 597 vs 589 경고만) | B config.c:395-580; nr_mac_common_tdd.c:62-130; nr_mac_common.c:913-914; B ci-scripts/.../log:1108-1116 |
| testMAC | `--cells` 비트 위치로 패턴 셀 선택 후 0..k-1로 재번호(와이어 id). 첫 SLOT.ind 대기에 타임아웃 없음. L1 yaml·cell_group_num을 읽지 않음. 자기 셀만 STOP | A testMAC/main.cpp:239-247; launch_pattern.cpp:2271-2291; scf_fapi_handler.cpp:145-203; test_mac.cpp:346-377 |
| Multi-L2 선례 | test_config.sh --ml2(반반 분할, test_mac_config_1.yaml 생성), run3_testMAC.sh --ml2 0/1 → `--cells <mask>`. **단일 L2 분기는 매 실행마다 nvipc_config_file을 'null'로 되돌림** | testBenches/phase4_test_scripts/test_config.sh:949-1026; README.md:144-205 |
| 셀별 NIC·2 NIC | 셀의 `nic` 문자열 = nics[].nic 항목; 2개 nics 항목은 NVIDIA 템플릿·스크립트가 지원(cuphycontroller_F08_CG1_2_PORT.yaml:49-60, test_config.sh --num-ports=2); NIC 객체마다 GpuComm 생성 | A cuphycontroller/src/yamlparser.cpp:235-254, 375-394, 1715-1736; cuphydriver/src/common/fh.cpp:178-180, 324; aerial-fh-driver/lib/nic.cpp:91 |
| ru_type | 1 = SINGLE_SECT_MODE(DL C-plane 1통/슬롯, SRS C-plane 없음, payload 길이 검사 생략), 3 = OTHER(에뮬레이터). 셀별 속성 → 혼용 가능 | aerial-fh-driver/include/.../oran.hpp:1839-1843; cuphydriver/src/common/fh.cpp:1327-1332; order_cuda_kernels.cu:1210-1215 |
| fix_beta_dl | **전역** 키. 1이면 BFP-9 셀 전부 beta_dl 65536. 에뮬레이터 yaml 전부 1, 실 RU yaml 전부 0(Benetel 런도 0). RU 에뮬레이터 config.yaml:349도 fix_beta_dl 1 | yamlparser.cpp:741-747; cuphydriver/src/common/cell.cpp:640-658; ru-emulator/config/config.yaml:349 |
| 전역 키 차이(Benetel 런 vs F08_CG1) | mps_sm_pusch 120/100, mps_sm_pdsch 40/102, ul_order_timeout_cpu 4 ms/8 ms, max_ru_unhealthy_ul_slots 0/100, prach_aggr_per_ctx 1/2 → 하나만 택해야 함(에뮬레이터 값 유지 권고) | B ci-scripts/.../log:103-122; A cuphycontroller_F08_CG1.yaml:27-52 |

## 3. 설계 (10 + 1)
| 구성요소 | 변경 | 비고 |
|---|---|---|
| nvIPC transport 맵 | 새 파일 `nvipc_multi_hybrid_11.yaml`(이 디렉터리의 초안): transport 0 = nvipc, phy_cells [0..9]; transport 1 = nvipc1, phy_cells [10] | l2_adapter_config_F08_CG1.yaml에 `nvipc_config_file: nvipc_multi_hybrid_11.yaml` 한 줄. test_config.sh 뒤에 적용 |
| cuphycontroller yaml | cell_group_num 11; nics 2개(0000:01:00.0, 0002:01:00.0); 셀 10 = Benetel(`cuphycontroller_hybrid_cell10.yaml` 초안: ru_type 1, vlan 1, dst 8c:1f:64:d1:15:11, eAxC [0,1,2,3]/[4,5,6,7], BFP 9, ul_gain 78.68, section_3 484, 273 PRB) | T1a/Ta4/Tcp_adv는 로그에 안 찍혀 FXN 템플릿 값으로 가정 → config_benetal.yaml 사본 필요 |
| L2 어댑터 instance 10 | `prach_ta_offset_usec: 2.5`(Benetel 런의 P5G_GH 어댑터가 쓴 값, 인스턴스별 키) | instances 20개는 그대로(≥ 11) |
| testMAC | `--cells 0x3FF`(와이어 id 0..9). yaml 변경 없음(prefix nvipc, 코어 15-18) | 10C/11C DDDDDDSUUU 패턴은 10-19 복원 때 지웠으므로 make_launch_pattern_tdd.py로 재생성·등록 |
| OAI gnb.conf(original.txt 기반) | `tr_s_shm_prefix = "nvipc1"`, `tr_s_poll_core = 62`; physCellId 0 유지 가능(에뮬레이터 셀 PCI 41..50) | TDD 6/6/3/0·PRACH 152·주파수(SSB 650016)는 그대로 |
| compose(OAI 서비스만) | `ipc: host`(c_aerial_troy가 --ipc=host), `depends_on nv-cubb` 제거, nv-cubb 서비스 기동 금지, cpuset "56-62", GPU 예약 제거 가능, 이미지 태그 명시 | 또는 `docker run --network host --ipc host --cpuset-cpus 56-62 --cap-drop ALL --cap-add SYS_NICE --cap-add IPC_LOCK -v conf:/opt/oai-gnb/etc/gnb.conf oai-gnb-aerial_hsw:<tag>` |
| 러너 | fullstack_run.sh는 test_config.sh 후 l2_adapter를 'null'로 되돌리고, 끝에 kill_ran이 phy_main을 죽임 → 하이브리드 전용 시작/정지 스크립트 필요 | 시작 순서: Open5GS → RU 에뮬레이터(10셀) → cuphycontroller(11셀) → OAI gNB → testMAC(--cells 0x3FF) → UE → 테넌트 |
| 코어 | L1 4-14, testMAC 15-18, 에뮬레이터 20-41(PTP 20), hook 43, dapp_sched 44-47, dispatcher 48-55, **OAI 56-62(poll 62)**, 63 여유, 64 phc2sys-cx7, 65-71 하우스키핑(Open5GS·분석) | compose의 13-18/poll 18은 L1 13·testMAC 15-18과 충돌 |
| 시계 | 변경 없음: ptp4l slave(aerial00, ptp1) → CLOCK_REALTIME, phc2sys-cx7가 ptp1→ptp0(에뮬레이터 NIC) 동기 | 실행 전 offset < 100 ns 확인 |
| 분류기/표 정책 | 11셀 합산(천장 3003 PRB) → X=10 임계 재적합 또는 load_class에 cell 10 제외 필터(소폭 코드) | 링은 코드 변경 없이 cell 10을 포함 |

## 4. 미해결·위험 (우선순위)
1. **fix_beta_dl 전역 충돌.** 에뮬레이터 셀 파라미터(fs_offset_dl 0/exp 4/ref 0)의 계산 beta ≈ 146,561(고정 65,536보다 7 dB 큼), Benetel 템플릿(fs_offset_dl 15)의 계산 beta ≈ 810 → 고정 65,536이면 **+38 dB** → RU DL 포화 가능. 권고: L1과 RU 에뮬레이터 config.yaml 모두 `fix_beta_dl 0`으로 두고(에뮬레이터는 같은 식으로 beta를 스스로 계산: csirs.cpp:194-197, pdcch.cpp:193-196) R4에서 DL 검증 카운터 확인. 실패하면 cell.cpp:640-658에 셀별 override(소규모 코드 변경).
2. **wait_all_cell_config 1**: 11셀 전부 CONFIG 전까지 SLOT.ind 없음(testMAC·OAI 모두 무한 대기하므로 교착은 아님). 시연에선 OAI를 testMAC보다 먼저 올리거나 0으로. 0일 때 "모든 셀이 초기에 설정돼야 한다"는 템플릿 주석(nvipc_multi_instances.yaml:18)의 cuphydriver 제약은 미확인(R2).
3. **SLOT.response 지연 프로파일**: OAI 셀이 PDU 없는 슬롯 → 10셀이 loopback tick까지 대기(+~1 슬롯 L2A 지연) → SafeCap 런과 다른 지연·RU late 분포. R2b(두 번째 testMAC을 희소 패턴으로)로 측정.
4. **transport 리셋(OAI 재시작)**: nvIPC pool 리셋과 L2A 보유 버퍼 충돌로 L1 전체 다운 가능 → R3 필수. L1 재시작 시 OAI는 재접속 못 함(재시작 필요).
5. **두 NIC(CX7 + BF3-내장 CX7) + DOCA GPU DL**: 템플릿·스크립트는 2 포트를 지원하나 서로 다른 어댑터 혼합은 미검증(R4). NIC당 DOCA GPU TX 버퍼(우리 F08 런 4 GiB, Benetel 런 32 GiB) → HBM 여유 확인. 에뮬레이터 NIC가 nics 첫 항목이라 tick PHC 선택 영향은 phc2sys-cx7로 상쇄(두 PHC 동일 시계).
6. **Benetel 윈도우 값 미확인**(T1a_max_up/cp_ul, Ta4, Tcp_adv, pcp, nics mtu): config_benetal.yaml은 읽기 불가 트리(/home/nvidia/hsw/25_3_hsw)에 있음 → 사본 요청이 가장 빠름. 참고: 생략 시 T1a_min_cp_ul_ns 기본값이 285 ms로 들어가는 파서 버그(yamlparser.cpp:520-528) → 명시 필요.
7. **운영 환경**: c_aerial_troy 현재 없음(--rm, run_aerial.sh로 재생성); 다른 사용자 컨테이너 icnl-aerial-26-1-kim(cpuset 0-17,21-35,65-71, GPU all) 상주; nvidia 계정 compose 스택과 GPU·DPDK prefix·MPS 파이프가 배타적 → 시간 창 조율. 10-25 런은 OAI segfault(exit 139)·L1 'NVIPC memory pool is full' → **단일 셀 OAI+Benetel 기준선(R5) 안정화가 선행 조건**. Open5GS는 유닛 전부 disabled(수동 기동, ogstun 10.45.0.1/16). ptp4l-bf3.service가 5 s마다 실패(코어 60) → disable 권고. OAI CONFIG 거부 시 OAI는 무증상으로 멈추므로 L1 로그의 CONFIG.response error_code 감시.
8. 전역 키(mps_sm_*, ul_order_timeout_cpu, max_ru_unhealthy_ul_slots)는 에뮬레이터 값 유지 → Benetel 셀은 RU 헬스체크(100 슬롯) 아래서 돎(오히려 10셀 보호에 유리), PRACH 파이프라인 실패 종료는 미커버.

## 5. 리허설 순서 (UE·Benetel 없이 → Benetel → 전체)
- R1 정적 검증: 하이브리드 yaml 3종을 파서 규칙으로 점검(transport 합 11, 셀 10 필수 키 전부, ru_type 1..3, BFP 9..16, dst MAC 유일, PCI 유일 41..50 + 0).
- R2 루프백 Multi-L2: cuphycontroller 11셀(에뮬레이터 11셀) + testMAC A `--cells 0x7FF`→실제로는 `0x3FF`(nvipc) + testMAC B `--cells 0x400 --config test_mac_config_1.yaml`(nvipc1, 코어 64-67 대신 63/65-67 수기 지정), wait_all_cell_config 1 → 0. 두 transport 모두 SLOT.ind 수신, Err.ind 0, RU late가 단일 L2 11셀 R0와 같은지.
- R2b 두 번째 testMAC을 희소 패턴으로 → SLOT.response 부재 시 10셀 지연·late 변화 측정(L2A.PROCESSING_TIMES, SLOT_END.slot_end_rcvd).
- R3 두 번째 L2를 중간에 kill -9/재시작, 깨끗한 STOP/START → L1 'PHY reset initiated', rx_release 예외, FATAL 유무.
- R4 2 NIC + fix_beta_dl 0: nics 2개·셀 10이 Benetel MAC을 가리키되 OAI 없이(또는 셀 10 CONFIG 없이 wait_all_cell_config 0) 10셀 R0 → DOCA 2 NIC 수락, HBM/hugepage 사용량, 에뮬레이터 DL 검증(beta 계산값) 통과 확인.
- R5 OAI 단독 273 PRB(cell_group_num 1, Benetel, UE 1): attach, PRACH(슬롯 9/19), iperf 97 Mbps UL, fix_beta_dl 0/1 비교(UE RSRP·PDSCH BLER) → 안정화(10-25 segfault 원인 포함).
- R6 전체 하이브리드(테넌트 없음) 10분: CRC/Err.ind 0, 셀 0..9 late vs R0; 셀 10은 L1 카운터만(Benetel엔 RU 표 없음).
- R7 테넌트(dapp_sched·table_policy(11셀 임계)·dispatcher 0.0.0.0:9007) + UE 클라이언트(10.45.0.1:9007, MTU 1400) → join.

## 6. 이번 조사의 한계
- 반박 검증은 L2 어댑터 축(11건)만 완료; OAI·testMAC·인프라·Benetel 축은 1인 조사(근거 행은 모두 기록됨). FH 다중 NIC 축은 전용 조사 대신 템플릿·스크립트·init 코드 확인으로 대체.
- 실행 검증 0건(모두 코드·설정·로그 읽기). 2 NIC DOCA, fix_beta_dl 0의 에뮬레이터 DL 검증, OAI 셀 273 PRB 11셀 그룹 CONFIG 수락은 리허설 항목.

## 8. 실측 기록 (2026-10-26, GPU 사용 허가 후; 로그 prof/hybrid_runs/)
| 시험 | 결과 | 조치 |
|---|---|---|
| ml2 (두 testMAC, 에뮬레이터 11셀, nvipc/nvipc1) | L1 두 transport 생성·매핑 정상, 11셀 CRC 0, FATAL 0. 두 번째 testMAC을 코어 65-67(비격리)에 두면 셀 10 SFN mismatch Err.ind ~2/s; 56-59(격리)면 시작 7건 외 0 | ML2_CORES=56,57,58,59 |
| wait_all_cell_config 0 + 미구성 셀 (단일 NIC에서도) | 마지막 START 직후 L1 segfault | 모든 셀에 L2 필수, wait_all 1 유지; nic2 모드 대신 nic2ml2 |
| 두 NIC (CX7 0000:01:00.0 + BF3 0002:01:00.0) | L1 기동, DOCA TX 버퍼 2개, DL U-plane 송출 확인(aerial00 tx_packets_phy) | - |
| Benetel UL 수신 | 전부 rx_steer_missed → 세 가지 원인: 포트 MTU 1514(점보 UL 프레임 4-8 kB oversize 드롭), VLAN PCP 7(스티어링 규칙이 TCI 전체 매칭 peer.cpp:550; RU는 PCP 0), FXN 창 280/405 µs(RU 창 밖) | BENETEL_NICS_MTU=8192, BENETEL_PCP=0, T1a_max_up 345000 / T1a_max_cp_ul 336000 → UL 프레임 수신 확인(16.8k pkt/s, steer_missed 증가 0) |
| testMAC이 Benetel 셀 구동(nic2ml2/benetel1) | PUSCH order-kernel 타임아웃 지속(UE 없음, 단일 섹션 RU에 다중 UE TV) → 에뮬레이터 셀까지 UL C-plane 지연 연쇄 | 실 RU는 OAI로만 구동; 진단 용도로만 사용 |
| full (OAI gNB → 셀 10) | CONFIG.request 거부(INVALID_CONFIG, TLV 34개 중 33개 Unknown): pcap으로 OAI 메시지는 정상(SCF 10.02, TLV 길이 u16) 확인. 우리 L1 빌드는 SCF_FAPI_10_04=ON(기본 OFF)으로 TLV 길이 u32(scf_5g_fapi.h:374-378) | build.aarch64_f1002 = SCF_FAPI_10_04=OFF로 cuphycontroller_scf + test_mac 별도 빌드, hybrid_run.sh L1_BUILD_DIR=build.aarch64_f1002 |
| 디스크 | CRC 실패 슬롯의 UL pcap 덤프 방지 | hybrid_yaml이 ul_pcap_capture_enable 0 강제 |
| full (OAI gNB, 10.02 L1 build.aarch64_f1002) | CONFIG/START 수락. 처음엔 OAI PRACH(슬롯 9/19)와 에뮬레이터 PRACH(슬롯 19)가 겹치는 슬롯에서 "No available Aggr PRACH objects" → 전 셀 UL_TTI에 Err.ind 0x33 | prach_aggr_per_ctx 2 → 4 (PRACH_AGGR_PER_CTX) 후 Err.ind 시작 2건 외 0, 에뮬레이터 10셀 DL 1332.69/UL 295.05 Mbps CRC 0, OAI 셀 SSB/SIB1 50 슬롯/s 송출, Benetel PRACH UL ~9.6k pkt/s 수신 (full5_f1002, 2026-10-26 08:57) |
| **full6_f1002 (최종, 09:06-09:12, 348 s)** | **Err.ind 0, FATAL 0.** 에뮬레이터 10셀 DL 1332.69 / UL 295.05 Mbps, UL 600 슬롯/s, CRC 0(런 전체); RU 에뮬레이터 DL late 슬롯 30/셀(단일 L2 런과 같은 수준). OAI 셀 10: CONFIG/START 수락, DL 50 슬롯/s(SSB/SIB1), OAI Err.ind 0, 프레임 진행 정상; Benetel PRACH UL 수신(steer_missed 증가 없음). UE·코어망 없음(NO_CORE=1). 종료 후 에뮬레이터 기존 yaml 5개 + valid_perf_patterns.sh 원복 확인 | 시연 구성 확정: L1_BUILD_DIR=build.aarch64_f1002, benetel_cell.env (mtu 8192, pcp 0, 345/336 µs, prach_aggr 4, fix_beta_dl 0) |
| **에뮬레이터 DL 페이로드 검증 (2026-10-27, full6 로그 재분석)** | RU 에뮬레이터 로그(1.8 GB/348 s)에서 에뮬레이터 10셀의 **모든 DL 슬롯이 Payload Validation ERROR**(PDSCH·PDCCH_DL·PDCCH_UL 각 491만, CSI-RS 245만 행). fix_beta_dl 1 런(r2_ml2 22건, r2b 0건)은 정상, 랩 빌드 + fix_beta_dl 0(testC) 386만 → 빌드 무관, fix_beta_dl 0이 원인. 메커니즘: `dl_approx_validation 0`에서 에뮬레이터는 수신 BFP-9 바이트를 TV의 `X_tf_cSamples_bfp9`(beta 65,536로 압축된 것)와 memcmp(pdsch.cpp:275, pdcch/csirs/pbch 동일); L1이 계산 beta 146,561(fs_offset 0/exp 4/273 PRB)로 압축하면 바이트가 달라 전부 실패. 정수 fs_offset_dl/ref_dl로는 65,536이 정확히 안 나오고(최근접 fs 2/ref −1 → 65,311), fix_beta_dl 1은 Benetel DL +17 dB(계산 9,160 대비). 검증 분기는 로그·카운터만 바꾸므로 L1 Err.ind·UL CRC·testMAC 지표는 유효하나 로그가 초당 ~5 MB(1시간 18 GB → 디스크 고갈) | hybrid_yaml이 에뮬레이터 `payload_validation 0`(benetel_cell.env RU_PAYLOAD_VALIDATION) 적용: DL 페이로드 검증만 끄고 DL 타이밍 검증(validate_dl_timing)·UL은 유지. full6 결론(Err.ind 0, FATAL 0, CRC 0)은 유지되나 "에뮬레이터 DL 검증 통과"는 주장 불가; 재런으로 로그 크기·OK 카운터 확인 |
| **final7_f1002 (2026-10-27 13:04-13:11, 최종 env값 + payload_validation 0, NO_CORE)** | L1 Err.ind 0, FATAL 0, 에뮬레이터 CRC 0, RU 로그 0.8 MB(검증 ERROR 0건), DL late 슬롯 25-27/셀. 셀 10 CONFIG/START 수락. **UE의 RA 시도가 실제로 들어옴**(223.9 preamble 48, TA 22, TC-RNTI 4c53; 코어 없이도 PRACH→RAR까지 진행) → Msg3 직후 OAI가 `Assertion (uci_ind->num_ucis > 0) failed` (NR_IF_Module.c:426)로 종료, 이후 L1 transport 1 NVIPC pool full. 원인(코드): L1의 PUSCH UCI.indication은 `out->totNumUciSegs > 0`(셀 그룹 전체)일 때 호출되고(scf_5g_fapi_phy.cpp:4488) 그 슬롯에 PUSCH가 있는 모든 셀에 메시지를 보내며 UCI가 없는 셀도 num_ucis 0으로 송신(4052-4275). 에뮬레이터 셀의 ULMIX TV가 UCI를 실으므로 플래그가 항상 참 → OAI 셀의 Msg3(UCI 없음)에 빈 UCI.ind. 단일 셀 nvidia 구성은 플래그가 거짓이라 발생 안 함 → **하이브리드 고유**, UE가 붙을 때마다 재발 | OAI 패치(cuPHY 무수정): NR_IF_Module.c에서 num_ucis 0인 UCI.ind는 경고 후 버림(`oai_empty_uci.patch`), nr-softmodem 재빌드(oai-build-dev, 2분) → 이미지 `oai-gnb-aerial_hsw:hybrid-uci`, benetel_cell.env OAI_TAG. 대안(미적용): L1 어댑터에서 num_ucis 0이면 tx_release — cuphycontroller 코드 변경이라 CONTEXT §9 보고 사안 |

## 9. 원본 config_benetal.yaml 대조 (2026-10-26, /home/troy/build/benetel_orig/)
| 키 | 원본 파일 | 하이브리드 값 | 비고 |
|---|---|---|---|
| nics mtu | 9216 | 9216 (8192으로도 동작 확인) | 점보 UL 프레임 |
| T1a_max_up / T1a_max_cp_ul | 345000 / 336000 | 동일 | 리허설에서 FXN 값(280/405) 폐기 후 일치 |
| Ta4_min / Ta4_max | 0 / 200000 | 0 / 200000 (수정) | 이전 50000/331000은 표준 템플릿 값 |
| Tcp_adv_dl, fh_len_range | 125000, 0 | 동일 | |
| fs_offset_dl / exponent_dl / ref_dl | 8 / 4 / 0 | 8 / 4 / 0 (수정) | fix_beta_dl 0에서 계산 beta ≈ 9.2k (273 PRB) |
| fs_offset_ul / max_amp_ul | 8 / 65504 | 8 / 65504 (수정) | |
| vlan / pcp | **3 / 7** | **1 / 0 유지** | 어제 실제 런 로그(/var/log/aerial/phy.log)는 VLAN 1·src MAC 00:..; RU의 UL 프레임은 TCI 0x0001에서만 매칭됨(수신 규칙이 TCI 전체 비교). 복사된 파일은 실행된 파일(cuphycontroller_config_benetal.yaml)과 다른 변형(51 PRB 스트라이드 포함) |
| src_mac_addr | d8:94:24:57:7f:f2 | 동일(자동과 같음) | |
| ru_type / eAxC / BFP / section_3 / ul_gain / lower_guard | 1 / [0-3],[4-7] / 9 / 484 / 78.68 / 845 | 동일 | 로그 복구값과 일치 |
| pusch_prb_stride, nMaxPrb | 51 (20 MHz 변형) | 273 | OAI gnb.conf 273 PRB |
| 전역: fix_beta_dl 0, prach_aggr 2, max_ru_unhealthy 0, ul_order_timeout_cpu 4 ms, mps_sm 120/40 | | fix_beta_dl 0, prach_aggr 4, 나머지 에뮬레이터 기존 값 유지 | 에뮬레이터 10셀 SafeCap 근거 유지 |
| L2 adapter (P5G_GH): prach_ta_offset_usec 2.5(전 인스턴스), allowed_fapi_latency 0, 스레드 코어 9 | | instance 10에만 2.5, 에뮬레이터 기존 값 유지 | |

## 10. 실 셀 우선 정합 (2026-10-26) — 셀 10의 경로에서 검증된 OAI+Benetel 설정과 다른 것
| 영역 | 검증된 설정(nvidia 계정 단일 셀) | 하이브리드 | 판단 |
|---|---|---|---|
| gnb.conf | original.txt | 동일 + prefix nvipc1, poll core 62 | UE attach에 영향 없음(전송 prefix·코어만) |
| OAI 컨테이너 | compose, ipc container:nv-cubb, cpuset 13-18, 이미지 TAG(.env latest; 최근 런은 ppo-273) | ipc host, cpuset 56-62, TAG 선택(OAI_TAG) | 같은 nvIPC 공유; **현재 UE로 검증된 이미지 태그를 OAI_TAG에 넣을 것** |
| 코어망 | 호스트 Open5GS(AMF 127.0.0.5), gNB network host | 동일 경로 | 리허설은 NO_CORE=1로만 돌렸음(UE 미시험) |
| L1 바이너리 | 25-3 기본 빌드(SCF_FAPI_10_04 OFF) | build.aarch64_f1002(OFF) | 동일 FAPI 레이아웃 |
| 셀 블록 | config_benetal.yaml (+ 로그 복구값) | ru_type/eAxC/BFP/MAC/창(345/336, Ta4 0-200, Tcp 125)/fs_offset 8/gain/guard/mtu 9216 동일; vlan 1·pcp 0(실제 런 로그·RU 프레임 기준; 복사 파일의 3/7과 다름); 273 PRB | 동일 |
| L1 전역 | fix_beta_dl 0, prach_aggr 2, pusch wait 5000, max_ru_unhealthy 0, ul_order_cpu 4 ms, mps_sm 120/40, allowed_fapi_latency 0 | fix_beta_dl 0, **prach_aggr 4**(충돌 해소), **pusch wait 5000, max_ru_unhealthy 0**(L1_GLOBAL_OVERRIDES), ul_order_cpu 8 ms(더 관대), mps_sm 100/102(10셀 SafeCap 근거), allowed_fapi_latency 1(더 관대) | 실 셀 UL 복조·RA에 닿는 키는 검증값, 나머지는 에뮬레이터 기존 값 |
| L2 어댑터 | prach_ta_offset_usec 2.5(전 인스턴스) | instance 10에 2.5 | 동일 효과 |
남은 검증: Open5GS 기동 + COTS UE attach + iperf를 하이브리드에서 1회(사용자 입회). 전역 키까지 전부 config_benetal 기준으로 맞추는 선택지(에뮬레이터 셀 SafeCap 수치 재측정 필요)도 가능.

## 11. 실행 파일 cuphycontroller_config_benetal.yaml 대조 (2026-10-26, 두 번째 전달) — 최종 채택값
이 파일이 L1이 실제 읽은 설정이다(로그의 "Config file: …/cuphycontroller_config_benetal.yaml"). 셀 블록을 그대로 채택했고 §9의 "원본" 열(config_benetal.yaml = UL-heavy sed 변형)은 폐기한다.
| 키 | 실행 파일 | 하이브리드 (benetel_cell.env) | 비고 |
|---|---|---|---|
| vlan / pcp | 1 / 0 (주석 "#7") | 1 / 0 | 리허설 결과와 일치: 수신 규칙이 TCI 전체 매칭 |
| nics mtu | 9000 | 9000 | 점보 UL 프레임 |
| src_mac_addr | 00:00:00:00:00:00 | 동일 | |
| T1a_max_up / T1a_max_cp_ul | 280000 / 495000 | 280000 / 495000 | **정정**: 리허설 C/C2의 UL 실패 원인은 창이 아니라 MTU 1514·pcp 7이었음(창 405→336 변경 후에도 steer-missed 지속). 405 µs 추정에 대한 "RU가 무시" 해석은 철회 |
| Ta4_min / Ta4_max / Tcp_adv_dl | 50000 / 331000 / 125000 | 동일 | |
| fs_offset_dl/ul, exponent, ref, max_amp_ul | 8/8, 4, 0, 65504 | 동일 | |
| pusch_ldpc_n_iterations | 7 | 7 (에뮬레이터 기존 셀 블록에 없어 삽입) | |
| strides / nMaxPrb | 273 | 273 | |
| 전역 | fix_beta_dl 0, pusch wait 5000, max_ru_unhealthy 0, prach_aggr 2, pusch_aggr 3, ul_order_cpu 4 ms, mps_sm 120/20/20/40/40, workers 4-5 / 6-8, low_priority 10 | fix_beta_dl 0, pusch wait 5000, max_ru_unhealthy 0, **prach_aggr 4**, pusch_aggr 3, ul_order_cpu 8 ms, mps_sm 100/2/2/20/102(에뮬레이터 기존), 에뮬레이터 기존 코어 플랜 | prach_aggr만 실측 필요값; mps_sm·코어는 10셀 에뮬레이터 기존 환경 유지 |
| 어댑터 | l2_adapter_config_P5G_GH.yaml: prach_ta_offset_usec 2.5 전 인스턴스, allowed_fapi_latency 0 | instance 10에 2.5, allowed_fapi_latency 1(에뮬레이터 기존) | |

## 12. 기준 반전 (2026-10-29) — 이 문서의 §3/§10/§11은 "랩 파일에 Benetel 셀을 끼워 넣는" 첫 설계의 기록이다
사용자 지침에 따라 기준을 뒤집었다: 하이브리드 L1/L2A yaml은 연구원들의 실행 파일(`baseline/`)에서 생성하고 에뮬레이터 9셀을 덧붙인다(Benetel = L1 셀 0, OAI는 gnb.conf의 prefix `nvipc` 그대로).
기준에서 벗어난 값의 전체 목록은 `CONFIG_MAP.md` §3, 기준 구성의 조사 기록은 `BASELINE.md`. 새로 알게 된 사실: nvidia 트리의 L1 빌드에는
`phyprach_aggr.cpp`의 PRACH 검출 임계값 강제(force_thr0 20.0)가 들어 있다 — final8의 "RA 920회 모두 WAIT_Msg3 실패"는 임계값 없이 잡음이 PRACH로 검출된 결과일 가능성이 크다(미검증).
