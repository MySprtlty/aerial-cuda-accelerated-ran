# HANDOFF — Hybrid L1 (Benetel COTS-UE 셀 1 + RU 에뮬레이터 9셀, cuphycontroller 1개) — 2026-10-29 기준 (기준 반전판, 브랜치 `hybrid`)

작성 목적: 다음 세션(사람 또는 에이전트)이 이 문서 하나로 ① 무엇이 어디까지 됐는지, ② 지금 호스트가 어떤 상태인지, ③ 다음에 무엇을 어떤 순서로
해야 하는지 알 수 있게 한다. 상세 근거는 §9의 문서 지도로.

## 1. 한 줄 결론
- **기준이 바뀌었다.** 하이브리드의 L1/L2A yaml은 이제 연구원들의 COTS UE 구성 파일(`baseline/`, 10-27 밤 UE가 붙어 통신한 그 파일들)에서 **생성**되고,
  랩(에뮬레이터 10셀) 파일은 에뮬레이터 셀 블록 9개와 소수의 용량·계측·코어 키만 제공한다. Benetel = L1 셀 0(nvIPC `nvipc`, OAI gnb.conf는 poll core 한 줄 외 무변경), 에뮬레이터 = 셀 1..9(`nvipc1`).
- 기준에서 벗어난 값은 **전부** `CONFIG_MAP.md` §3에 분류(a~d)와 이유가 붙어 있다. (a) UE가 보는 프로토콜·RF·타이밍 값 중 벗어난 것은 **없다**.
- 새 사실: nvidia 트리의 L1 빌드에는 `phyprach_aggr.cpp`에 **PRACH 검출 임계값 강제(force_thr0 20.0)** 소스 수정이 들어 있다. env-gated로 이식해 `build.aarch64_f1002`를 다시 빌드했다(10-29 15:23).
- **GPU 런은 아직 안 했다.** apply → 생성 → 파싱 검사 → restore 왕복만 검증(10-29 15:26). 기준 스택 자체는 10-29 14:41-14:49 troy가 매뉴얼대로 돌려 NG Setup·셀 기동 확인(UE 없음).

## 2. 산출물 위치
| 무엇 | 경로 | 비고 |
|---|---|---|
| 실행 패키지 | `cuPHY-CP/dapp_hook/sweep/hybrid/` | `README.md`(사용법) · `baseline/`(기준 파일 사본 + md5) · `benetel_cell.env` · `hybrid_yaml.py`(생성기) · `hybrid_config.sh apply/status/restore` · `hybrid_run.sh start/stop/status` · `oai_empty_uci.patch` |
| 기준 구성 조사 | `BASELINE.md` | 매뉴얼 각 단계의 실제 동작, 읽히는 파일·값, DL/UL-centric 차이, nvidia 트리 소스 수정, 라이브 런 기록, 접근 불가 항목 |
| 변경값 표 | `CONFIG_MAP.md` | §1 기준 실행 구조, §2 하이브리드가 읽는 파일, §3 기준에서 벗어난 값 전부 |
| 첫 설계 기록 | `DESIGN.md` | 랩 파일 기반 10+1 설계와 10-26/27 실측(full6, final7/8). §12에 반전 메모 |
| OAI 측 | `/home/troy/build/openairinterface5g/ci-scripts/yaml_files/sa_gnb_aerial_hybrid/` | compose(OAI만, ipc host, cpuset 56-62), `gnb_hybrid.conf`(= baseline/gnb_original.txt + poll core 62), `.env`(apply가 씀) |
| 기준 원본 | `/home/nvidia/hsw/25_3_hsw/{sh, openairinterface5g/ci-scripts/yaml_files/sa_gnb_aerial, aerial-cuda-accelerated-ran/cuPHY-CP/cuphycontroller/config}` | 읽기만. 손대지 않음 |
| L1 빌드 | `build.aarch64_f1002/` | SCF_FAPI_10_04 OFF(기준과 동일) + PRACH 변경. root로 `ninja -C build.aarch64_f1002 cuphycontroller_scf`(컨테이너 안). 랩 `build.aarch64`는 그대로(10-03) |
| 런 로그 | `prof/hybrid_runs/<name>/`, `lab_backup/`, `applied_<ts>/`, `hybrid_yaml.log` | 10-29 `applied_20261029_152615`가 새 생성기의 첫 산출물 |
| 메모리 | `~/.claude/projects/-home-troy-build-aerial-cuda-accelerated-ran/memory/hybrid-l1-feasibility-2026-10-26.md` | 갱신됨 |

## 3. 검증 상태
### 확인됨 (10-29, GPU 없이)
- `hybrid_config.sh apply --ncells 9`: 9C 패턴 생성·등록, phase-4 설정(컨테이너), 생성기 실행(결정 로그 30여 줄), PyYAML 파싱·셀 수 일치 검사 통과, OAI .env 기록. `restore`로 랩 파일 6개 바이트 일치 복원, 생성 파일·패턴 제거.
- 생성된 L1 yaml: 기준 파일 전문 + cell_group_num 10 + nics 2개 + 셀 10개(0 = 기준 블록 그대로, 1..9 = 랩 블록 renamed/cell_id 2..10) + §3.1의 키. L2A: P5G_GH 전문 + 코어 11/12 + nvipc 파일 + 훅 행.
- 기준 스택 라이브 런(14:41-14:49): `HOME=/home/nvidia docker compose up -d` → NGSetupResponse, CONFIG/START 셀 0, DL 50 slots/s, UL 0, Err.ind 0. UE 없음(연구원이 켜야 함).
- 기준 트리의 OAI 소스 = 우리 복사본(차이는 우리 UCI 패치뿐). 기준 L1 소스 = upstream + phyprach_aggr.cpp 1곳.
- **첫 GPU 런(ue1, 10-29 15:32)**: RU 에뮬레이터·L1(10셀)·OAI(셀 0 CONFIG/START·NGSetup)·testMAC(9셀 CONFIG/START) 모두 기동 → 첫 슬롯에서 L1 SIGSEGV. gdb 런(gdb1)으로 확정:
  `DebugWorker13` → `task_work_function_debug` → `DLOutputBuffer::waitPrePrepareStop(this=0x0)`(dlbuffer.cpp:345): Debug task가 `prepare_tx_dlbuf_per_nic[0]`를 null 검사 없이
  호출하는데 NIC 0 = 기준 nics 첫 항목(Benetel)이고 OAI 셀이 그 슬롯에 DL이 없으면 null. 랩에서 가져온 `debug_worker 13`이 원인 → **기준값 -1로 복귀**(CONFIG_MAP §3.1).
- **ue2~ue4 (10-29 15:40-16:09)**: L1 생존, 에뮬레이터 DL 1332 Mbps, UL 패킷 100 % 정시 도착인데 **에뮬레이터 PUSCH 전부 CRC 실패**(600/600 슬롯·초, 9셀 전부), HARQ pool 고갈 연쇄.
  배치 변경(Benetel을 마지막 셀/NIC 1로, ue3)·allowed_fapi_latency 1(ue3: Err.ind 98 k → 60)·빔포밍/프리코딩 1(ue4)로도 UL은 그대로.
- **replica10 (16:15)**: 10-27 full6 생성 파일(랩 기반 10+1)을 재빌드 바이너리로 그대로 돌림 → UL 295 Mbps·CRC 0·Err.ind 3·batched copy 오류 0. **바이너리·패턴은 정상, 기준 파생 설정이 원인.**
- **근본 원인(코드로 확정)**: 랩 L2A의 `staticPuschSlotNum/staticPucchSlotNum/staticPdschSlotNum/staticPdcchSlotNum/staticCsiRsSlotNum: 0`. F08 TV는 전부 슬롯 0용(`_s0`)이고
  L2A는 이 키로 모든 채널 명령의 슬롯 번호를 0으로 고정한다(`scf_5g_slot_commands.cpp:158` `staticPuschSlotNum > -1 ? static : slot`). 기준(P5G_GH)에는 키가 없어(-1) 실제 슬롯 번호를 쓰므로
  TV의 DMRS와 어긋나 전부 CRC 실패. 반대로 모든 셀을 0으로 고정하면 실 UE는 슬롯 0 밖의 PDSCH/PUSCH를 못 쓴다 → **10-27 final8 "RA 920회 모두 WAIT_Msg3"의 진짜 원인**(Msg2/Msg3 슬롯 번호 불일치).
  키가 전역이라 설정만으로는 해결 불가(TV 풀에 슬롯별 UL TV 없음) → L2A에 `staticSlotCellMask`(yaml, 에뮬레이터 셀 비트) 추가: 마스크 안 셀만 정적 슬롯/정적 SSB SFN, 나머지는 실제 슬롯·tick SFN.
  cuPHY-CP 소스 변경 6파일(cuphyl2adapter 2, scfl2adapter 4; 키가 없으면 upstream 동작), `build.aarch64_f1002` 재빌드. CONFIG_MAP §3.2·§3.6.

- **ue5 (10-29 17:36-17:41, 마스크 패치 빌드, 9+1)**: 에뮬레이터 9셀 **UL 295.05 Mbps·600 슬롯/s·CRC 0**(런 전체), DL 1332.69 Mbps, Err.ind 0, FATAL 0, testMAC ERR 0·UCI ONTIME 100 %,
  RU DL U late 슬롯 73-75/셀(358 k 슬롯; replica10 13, ue4 40), payload 오류 0. Benetel = L1 셀 9 CONFIG/START, OAI NGSetupResponse, Assert 0. **마스크 패치가 에뮬레이터 UL을 고쳤다.**
  남은 경고: `early received next slot message`(셀 9, 0x80/0x81, 26 k — OAI가 SLOT.ind에 답하는 사이 msg 스레드가 testMAC 9셀을 처리하느라 ss_curr가 늦게 넘어가 캐시 후 재생; 손실 없음),
  `PDU count mismatch DL_TTI=0 TX_DATA=1` 11.9 k(= 매 SIB1 슬롯), batched copy 오류 2.9 k(testMAC 정지 뒤 OAI 단독 구간 50/s). 그리고 **OAI 셀 DL 0.00 Mbps**(기준 라이브 런은 0.04 = SIB1 바이트).
- **근본 원인 2(코드로 확정, 10-29 18:30)**: 모든 하이브리드 런(10-26 full1 ~ ue5)에서 OAI 셀 DL이 0.00이었다 = **SSB·SIB1이 한 번도 전파되지 않았다**. 원인은 빌드 플래그:
  우리 빌드(`build.aarch64`·`build.aarch64_f1002`)는 `ENABLE_CONFORMANCE_TM_PDSCH_PDCCH=ON`, 기준(nvidia build.aarch64)은 OFF. ON이면 `scf_fapi_dl_tti_req_t`에 `testMode` 1바이트가
  들어가(scf_5g_fapi.h:636, slot 다음) DL_TTI.request 레이아웃이 SCF 222.10.02와 달라진다. testMAC도 같은 플래그라 랩에서는 무증상, OAI는 표준 레이아웃 → L1이 nPDUs 자리를 testMode로,
  nGroup을 nPDUs로 읽어 SSB-only 슬롯(nGroup 0)은 PDU 0개, SIB1 슬롯은 밀린 오프셋의 PDU 1개 → PDSCH 미등록 → TX_DATA(testMode 없음, 정상 파싱) 1 PDU와 불일치. L1의 "50 Slots"는
  TX_DATA 도착 수(`dl_slot++`는 on_phy_dl_tx_request)라서 SSB 전송의 증거가 아니었다. final8의 "RA 920회"도 SSB 없이 UE가 동기할 수 없으므로 PRACH 오검출(연구원 force_thr0 수정의 이유).
  조치: `cmake -DENABLE_CONFORMANCE_TM_PDSCH_PDCCH=OFF -DSCF_FAPI_10_04_SRS=ON build.aarch64_f1002`(기준 빌드 플래그와 동일; 전역 정의라 276 스텝 재빌드) → `ninja cuphycontroller_scf test_mac`.
  `hybrid_run.sh`가 CMakeCache 플래그 3개를 사전검사, `hybrid_report.sh`가 OAI 셀 DL 바이트를 판정. CONFIG_MAP §3.6. 랩 build.aarch64는 그대로(랩 testMAC과 일치해야 함).

- **ue6 (10-29 17:59-18:04, 플래그 재빌드 + 마스크 패치, 9+1)**: **OAI 셀 9 DL 0.04 Mbps·50 슬롯/s(237/239 샘플, 기준 라이브 런과 동일 = SIB1 전송)**, `PDU count mismatch` 0, batched copy 오류 0,
  Err.ind 0, FATAL 0; 에뮬레이터 9셀 UL 295.05 Mbps·CRC 0·DL 1332.69 Mbps, testMAC ERR 0, RU payload 오류 0; NGSetupResponse, UE 없을 때 OAI RA 이벤트 0(PRACH 오검출 없음).
  남은 관찰: `early received next slot message` 25 k(셀 9; 캐시 후 재생, 손실 없음), RU DL U late 슬롯 87-89/셀(358 k 슬롯 = 0.025 %; replica10 13 → ue4 40 → ue5 75 → ue6 88로 증가 추세, OAI pcap/msg 스레드 지연 의심, 미조사).
  모든 이전 하이브리드 런(full6·final7·final8·replica10·ue2~ue5)은 OAI 셀 DL 0.00이었음을 로그로 재확인(hybrid_report "OAI cell N: DL bytes seen" 판정 추가).
- **ue7 (10-29 18:42-18:47, enable_precoding/enable_beam_forming 0/0 = 기준, L1 bfw_*/beam-id 키 제거)**: 에뮬레이터 9셀 UL 295.05 Mbps·CRC 0, DL 1332.69 Mbps; OAI 셀 DL 0.04 Mbps(237/239),
  mismatch 0, batched copy 0, Err.ind 0, FATAL 0, RA 이벤트 0; **RU DL U late 슬롯 24/셀(ue6 88)** → BF/프리코딩 키가 DL 지연을 키우고 있었음. 이 편차는 제거(benetel_cell.env). 남은 편차 = 용량(cell_group_num, nics, prach_aggr_per_ctx, use_batched_memcpy, allowed_fapi_latency, static slot 키+마스크), GPU SM 배분(mps_sm_*), 코어, 계측(CONFIG_MAP §3).

- **ue8~ue20 (10-30 12:40~13:56, Pixel 접속 시험)**: 매 런 PRACH → Msg3 → RRC Setup → Msg4 ACK("CBRA procedure succeeded")까지는 성공하고, 그 직후 첫 x.17 슬롯에서
  L1 세그폴트(또는 UL 워커 정지 → Aggr PUSCH 객체 고갈 → L1 종료). gdb/마커 계측 13회 재현으로 **근본 원인 3** 확정: cuphydriver `PhyPrachAggr::setup`의
  occasion→stage-3 버퍼 매핑이 슬롯의 모든 셀에 정적 PRACH 기회 수만큼 범위를 주지만, 버퍼는 그 슬롯에 PRACH가 있는 셀에만 할당됨(`SlotMapUl::addCell`,
  rach_occasion=0). UE가 RRC 연결되면 OAI가 매 UL 슬롯에 PUSCH/PUCCH를 주므로 슬롯 17(에뮬레이터 PRACH 8/17/19, Benetel PRACH 8/9/18/19)에 Benetel 셀이
  "PRACH 없이" 들어오고, 에뮬레이터 범위가 한 칸 밀려 마지막 기회가 범위 밖 버퍼를 가리킴(null → 세그폴트, stale → GPU 대기로 UL 워커 정지). 랩(동일 셀)과
  1셀 기준에서는 절대 안 밟는 경로. TDD 패턴 무관.
- **수정(10-30, CONTEXT §9 보고 대상, `prach_buffer_map.patch`)**: setup이 이 슬롯에 PRACH PDU가 있는 셀만 범위를 받고 범위 합≠버퍼 수면 PRACH를 건너뜀;
  PRACH 태스크 가드 2개(해제된 슬롯맵 → 건너뜀, waitOrderLaunched 실패 → 종료 대신 건너뜀). 진단용 마커·복구 한계 변경은 제거.
- **ue21 (10-30 14:00~, 수정 빌드, 설정 = ue7과 동일)**: 14:01:58 RA → Msg4 → RRCSetupComplete → AMF `Registration complete`(imsi-999700000177998) →
  SecurityMode/UECapability → RRCReconfigurationComplete → SMF IPv4 10.45.0.2, 호스트 ping 5/5(평균 60 ms). UE in-sync, RSRP -58, PH 48 dB, UL MCS 27 SNR 27 dB
  BLER 0, MAC RX 34 MB/2분. L1 Err.ind 0 FATAL 0, Benetel 셀 UL 2 Mbps CRC 0(접속 직후 15초만 CRC 수십/초), 에뮬레이터 9셀 1332/295 Mbps CRC 0 병행.
  **= 하이브리드 L1에서 COTS UE 접속 성공.**
  **30분 완주(14:00:35~14:31:44)**: UE는 한 번의 RA 뒤 끝까지 RRC_CONNECTED(재접속·release·out-of-sync 0), ping 48/48 응답(마지막 1회는 teardown 후).
  UE 트래픽 UL 397.6 MB / DL 75.5 MB(MAC 카운터; UL 평균 1.76 Mbps = L1 셀 9 집계와 일치). UL 링크 적응: MCS 22~27, SNR 20~24 dB, 1차 재전송 7.5 %,
  최종 오류 734/668,685 TB(0.11 %), DL 17,370 TB 오류 0. L1 Err.ind 0·FATAL 0 전 구간, 에뮬레이터 9셀 CRC 0·testMAC ERR 0 전 구간.
  L1 셀 9 초당 CRC 실패 평균 29/420 슬롯(HARQ 1차 BLER, 정상). **함정**: 랩 log_level DBG의 L1 로그가 30분에 ~20 GB → teardown 중 디스크 가득 →
  lab 파일 restore 실패(재실행으로 복구). 이후 log_level은 기준 INFO(benetel_cell.env).

### 아직 안 됨 (다음 세션, GPU 창 + 사용자 허가)
1. ~~UE 접속~~ → ue21에서 확인됨(위). 남은 것: 장시간 안정성(30분+), UE 트래픽(iperf/영상) 중 에뮬레이터 KPI, 접속 직후 15초 CRC 원인(UL 링크 적응으로 추정). 확인 도구: `ue_watch.sh start/report`, UE ping 10.45.0.x, L1 셀 9 UL 슬롯/CRC.
   UE 없을 때 RA 이벤트 0은 ue6에서 확인됨. UE가 붙으면 e2e/ue_client.py로 E2E 지연 재개.
2. 에뮬레이터 셀 상태: Err.ind 0, `ru_payload_errors.txt` 0, DL late 슬롯 수(night-3와 비교). 기준값으로 바꾼 전역(ul_order_timeout_cpu 4 ms, work-cancel 0, allowed_fapi_latency 0)이 에뮬레이터 9셀에 문제를 만들면 → CONFIG_MAP §3의 "기준값 유지" 항목을 하나씩 랩값으로 되돌리며 보고.
3. UE 통신 후: UE E2E 클라이언트(e2e/ue_client.py)와 YOLO 세입자·table policy를 이 구성 위에서 재개.

## 4. 호스트 상태 스냅샷 (2026-10-29 18:10 KST)
| 항목 | 상태 |
|---|---|
| git | 브랜치 `hybrid`(develop e4d9fcc에서 생성). 미커밋: `phyprach_aggr.cpp`(M), L2A 마스크 패치 6파일(M: cuphyl2adapter 2, scfl2adapter 4), `cuPHY-CP/dapp_hook/sweep/hybrid/`(untracked), `testVectors/cuPhyChEstCoeffs.h5`(LFS 포인터 artefact, 커밋 금지). `build.aarch64_f1002`는 CONFORMANCE OFF/SRS ON으로 재구성됨(빌드 디렉터리, git 밖). `coloring` 브랜치는 사용자 것, 건드리지 않음 |
| 랩 yaml | restore 완료, `status` = not applied |
| 프로세스/컨테이너 | RAN 프로세스 없음. `c_aerial_troy` Up 3 days(--rm). `icnl-aerial-26-1-kim` 상주(건드리지 말 것). nvidia 스택 down |
| Open5GS | 18개 유닛 모두 active(disabled: 부팅 시 안 뜸), routing.sh 규칙 적용 상태, ogstun 10.45.0.1 |
| 디스크 | 32 GB 여유(97%). 런 전 확인; prof/hybrid_runs의 오래된 런 로그 정리 필요 |
| sudo | troy `sudo -n` 가능. RU ssh 불가 |
| nvidia 계정 | 10-27 22:22-23:47 KST UE 런, 10-27 23:46 yolo/video/xApp 서비스 로그 → 저녁 GPU 창 조율 |

## 5. 다음 세션 실행 절차 (GPU 필요, 사용자 허가 후)
```bash
docker ps --format '{{.Names}}'; pgrep -a phy_main; df -h / | tail -1         # nv-cubb/oai-gnb 없어야 함, 여유 15 GB 이상
pgrep -x open5gs-amfd && ip -br a show ogstun                                   # 아니면 /home/nvidia/hsw/25_3_hsw/sh/start_cn.sh && sudo .../routing.sh
cuPHY-CP/dapp_hook/sweep/hybrid/hybrid_config.sh apply --ncells 9 --slots 7200000
cuPHY-CP/dapp_hook/sweep/hybrid/hybrid_run.sh start ue1          # L1 = build.aarch64_f1002, CUPHY_PRACH_FORCE_THR0=20.0, OAI hybrid-uci
# 종료: Ctrl-C 또는 hybrid_run.sh stop (랩 파일 자동 복원)
```
분기: NGSetup 실패 → 코어(L1 무관). PRACH 보이는데 RRC 안 됨 → mps_sm_pucch/prach(랩 2/2 vs 기준 20/20)를 먼저 기준값으로. 에뮬레이터 CRC/Err.ind → ul_order_timeout_cpu 8 ms, allowed_fapi_latency 1 순으로 랩값 복귀(각각 보고).

## 6. 함정·교훈 (반복 금지)
- compose 파일의 `~/hsw/...`는 `$HOME` 기준: troy가 nvidia 스택을 돌릴 땐 `HOME=/home/nvidia docker compose …`. 안 그러면 /home/troy/hsw가 비어 있는 채로 마운트된다.
- `config_benetal.yaml` ≠ `cuphycontroller_config_benetal.yaml`(전자는 entrypoint sed 대상, L1이 읽지 않음). 후자 = `_all_success`.
- `build.aarch64_f1002`는 root 소유 → ninja는 `docker exec -u root`. 브랜치 전환으로 소스 mtime이 바뀌면 cmake 재생성 + cuPHY 일부 재컴파일(10-29: 90 스텝).
- 기준 L2A 파일의 `h2d_copy_thread_cpu_affinity : 11`처럼 콜론 앞 공백이 있는 키가 있다(생성기 정규식에 반영).
- `test_config.sh`는 매 런 랩 yaml을 다시 쓰므로 생성은 그 뒤(apply 순서에 반영). `restore`는 git HEAD가 아니라 `lab_backup/`으로 되돌린다.
- `wait_all_cell_config 1`: OAI를 testMAC보다 먼저. 0 + 미구성 셀 = segfault.
- 빈 UCI.indication은 하이브리드 고유(이미지 `hybrid-uci` 필수). testMAC CRC 0 ≠ 에뮬레이터 DL 검증 통과(`ru_payload_errors.txt` 확인).
- 디스크: `ul_pcap_capture_enable`은 기준에 없음(=0) — 그대로 둘 것. nvipc pcap은 OAI transport만(2 GB 상한).
- `c_aerial_troy`는 `--rm`. 다른 테넌트 컨테이너 건드리지 말 것. 26x-1280 + 16셀 조합은 사용자 확인 후.
- `debug_worker`(랩 13)는 2-NIC 구성에서 nics[0]에 DL 셀이 없는 슬롯에 segfault(위 §3). 기준값 -1 유지. 코어 덤프는 호스트 apport 파이프가 컨테이너 크래시를 버리므로 못 건진다 →
  백트레이스는 `L1_EXTRA_ARGS="--gdb_script 'gdb -q -batch -x /opt/nvidia/cuBB/prof/hybrid_runs/gdbcmds --args'" hybrid_run.sh start <name>`(gdb 아래에서는 Err.ind 0x81/0x03이 늘어남, 타이밍 영향).

## 7. git
- 커밋 없음(사용자 결정). 커밋 대상: `cuPHY-CP/dapp_hook/sweep/hybrid/`(baseline/ 포함), `cuPHY-CP/cuphydriver/src/uplink/phyprach_aggr.cpp`. OAI 트리 쪽 `sa_gnb_aerial_hybrid/`는 별도 트리.
- `cuPhyChEstCoeffs.h5`의 M은 LFS 포인터 artefact — 커밋 금지.

## 8. 문서 지도
- 사용법: `README.md` · 기준 조사: `BASELINE.md` · 변경값 표: `CONFIG_MAP.md` · 첫 설계·실측: `DESIGN.md` · OAI 측: `sa_gnb_aerial_hybrid/README.md`
- 세션 간 메모리: `hybrid-l1-feasibility-2026-10-26.md`
