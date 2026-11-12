# 1셀 COTS UE 데모 실행 매뉴얼 — Benetel RU + OAI gNB + Open5GS 위의 우리 cuphycontroller(green context + YOLO dApp)

작성 2026-11-05. 이 디렉터리(`cots/`) 하나로 데모 스택을 켜고 끕니다. 호스트 쪽에 필요한 것은 전부 이 안에 있고(yaml, OAI compose, 코어 스크립트),
L1은 Aerial 컨테이너 `c_aerial_troy` 안에서 `docker exec`로 돕니다. docker 그룹 사용자면 누구나(troy, nvidia) 실행할 수 있습니다.

## 1. 무엇이 어디서 도는가

| 구성 요소 | 어디서 | 어떻게 |
|---|---|---|
| cuphycontroller_scf (L1, phy_main) + YOLO 테넌트 + xApp 에이전트 | 컨테이너 `c_aerial_troy`(host IPC/네트워크, SDK 트리 = `/opt/nvidia/cuBB`) | `cots_run.sh start`가 `docker exec -u root`로 `build.aarch64_f1002/.../cuphycontroller_scf dgist` 실행. **MPS 없음**, CUDA green context로 SM 분할 |
| OAI gNB (nr-softmodem, MAC/RLC/PDCP/RRC/NGAP) | 컨테이너 `oai-gnb-dgist`(`oai/docker-compose.yaml`, ipc host) | L1과 nvIPC 공유메모리 prefix `nvipc`로 FAPI 통신. `oai/gnb.conf` = 연구실 `gnb_original.txt` + `tr_s_poll_core 62` |
| Open5GS 5G 코어 (AMF 127.0.0.5, PLMN 999/70, TAC 1, UE 10.45.0.0/16) | 호스트 systemd 서비스 18개 | `cots_core.sh up/down/status` (연구실 `start_cn.sh`/`routing.sh`와 동일) |
| Benetel RAN550 RU | 외부 장비, 프론트홀 `aerial00`(0002:01:00.0), 관리 IP 10.10.0.100 | 설정 변경 없음(연구실 그대로). 켜져 있고 PTP가 잠겨 있으면 됨 |
| COTS UE (Pixel 등, 가입자 99970000017799x) | 무선 | 셀이 뜨면 자동 접속 |

L1 설정 `config/cuphycontroller_dgist.yaml`, `config/l2_adapter_dgist.yaml` = 연구실의 UE 접속 성공 기준 파일(`cuphycontroller_config_benetal.yaml`,
`l2_adapter_config_P5G_GH.yaml`)에 다음만 더한 것. UE가 보는 값(RF, TDD, PRACH, 셀 파라미터, nvIPC, 타이밍)은 전부 기준 그대로.

| 키 | 기준 | 우리 | 이유 |
|---|---|---|---|
| `use_green_contexts` | 0 | 1 | SM 분할을 MPS 대신 green context로 (dApp과 GPU 공유) |
| `mps_sm_pusch / pdsch` | 120 / 40 | 60 / 40 | 시작 시 `DAPP_PUSCH_PROFILES`(16,24,32,40,56 → 20~60 SM), `DAPP_PDSCH_PROFILES`(24,40)의 최대값으로 대체됨. 슬롯마다 사이저가 프로파일 선택 |
| `mps_sm_pucch / prach / ul_order` | 20 / 20 / 40 | 2 / 2 / 8 | UL 제어·order 커널의 green context 크기(1셀이면 충분; 10-30 UE 접속 런도 2/2) → YOLO에 최대 80 SM |
| L2A `dapp_hook` 줄 | 없음 | enable 1, export_cpu 43 | FAPI UL_TTI/DL_TTI를 사이저·테넌트로 내보내는 훅 |

환경변수(`cots.env`): `CUPHY_PRACH_FORCE_THR0=20.0`(연구실 바이너리에 하드코딩된 PRACH 임계값; 우리 빌드는 env로 읽음), dApp 클래스 `24,40,48,56,64,72,80`(4K), FHD는 `...,80,88`(2026-11-11: 88 = (P132,N24) 틈, UL 작업이 4슬롯 안에 없는 D 4~7 슬롯에만 생겨 추론 전체가 그 창에 들어갈 때만 선택됨)
(= 1셀 배치가 YOLO에 남길 수 있는 자유 SM 값 P−N−8; 32는 나오지 않음. 모델별 엔진 `models/full26<m>/seg00_sm<c>.engine` (4K 프로파일, batch 8; `DAPP_INPUT=fhd`면 `models/fhd26<m>/`, batch 3: 1280×1080 크롭 2 + 전체 1, 2026-11-09; fhd의 클래스는 `DAPP_TENANT_CLASSES_FHD`(기본 24,…,80, 4K와 같음; 2026-11-10: 모든 엔진을 primary context에서 역직렬화하고 fhd26s/fhd26l의 sm80 엔진은 sm72 timing cache로 빌드함 — tenant/dapp_tenant.cu load_set 주석), 엔진당 GPU 메모리 약 1 GB → 35개 로드 시
L1 포함 약 78 GB / 97 GB 사용), 테넌트/에이전트 코어 45/46, xApp 포트 9810.

**실 L2(OAI)와 testMAC의 차이 (2026-11-05 발견·수정):** OAI는 내용이 있는 슬롯(SSB/SIB1, PRACH 기회, 스케줄된 UE)에만 DL_TTI/UL_TTI를 보내고 빈 슬롯은 아무
메시지도 보내지 않는다(UE 없을 때 전체 슬롯의 약 12 %만 메시지 있음). 11-04까지의 테넌트는 "계획 없는 슬롯 = 모름 = 최악(PUSCH 60 + PDSCH 40)"으로 보아 실 gNB
아래에서는 YOLO에 항상 24 SM만 주었다(26x 95 ms). 수정: 슬롯의 L2 메시지 창이 열린 뒤 `DAPP_TENANT_EMPTY_US`(기본 400 µs)가 지나도 계획이 없으면 그 슬롯은
유휴로 본다(지난 프레임의 같은 슬롯도 동일 규칙). 사이저가 계획 도착 선행 시간(`g_plan_lead_ns`)을 기록해 창 시각을 구한다. testMAC(매 슬롯 메시지)에서는 동작 변화 없음.

## 2. 사전 조건 (한 번만)

1. 컨테이너 `c_aerial_troy`가 떠 있어야 함(`docker ps`). 없으면 SDK 트리에서 `cuPHY-CP/container/run_aerial.sh -d`(`--rm` 컨테이너라 호스트 재부팅 뒤에는 다시).
2. L1 빌드 `build.aarch64_f1002`(플래그 SCF_FAPI_10_04 OFF, ENABLE_CONFORMANCE_TM_PDSCH_PDCCH OFF, SCF_FAPI_10_04_SRS ON, ENABLE_DAPP_HOOK ON)와
   YOLO 엔진 5모델 × 8클래스가 있어야 함. `cots_run.sh start`가 둘 다 검사함.
3. PTP(`ptp4l`, `phc2sys`) 동작, `aerial00` 링크 up, RU 전원. hugepages 20개 이상 여유.
4. **연구실의 `sa_gnb_aerial` compose 스택(nv-cubb, oai-gnb-aerial_hsw)과 동시에 띄우지 말 것.** 같은 GPU·NIC·nvIPC prefix를 씀. `cots_run.sh start`는 그 컨테이너가 있으면 거절.
5. GPU MIG 꺼진 상태(`nvidia-smi` mig.mode Disabled). 저녁 연구원 GPU 시간과 겹치지 않게.

## 3. 명령어

| 명령 | 하는 일 |
|---|---|
| `./cots_core.sh up` | Open5GS 18개 서비스 시작 + UE 서브넷 NAT(`core/routing.sh`, sudo). 이미 떠 있으면 그대로 |
| `./cots_core.sh status` | 활성 유닛 수, `ogstun 10.45.0.1`, NAT 규칙 수 |
| `./cots_core.sh down` | NAT 제거 + 서비스 중지 (보통 안 내림) |
| `./cots_run.sh start [이름]` | ① 사전검사 → ② 컨테이너 안 MPS 끄기 → ③ 우리 yaml 2개를 컨테이너 config 디렉터리로 복사(새 파일명, 연구실 파일 무변경) → ④ L1 시작, `L1 is ready` 대기 → ⑤ dApp 테넌트 준비(엔진 40개 로드, 포트 9810 열림) 대기 → ⑥ OAI gNB 컨테이너 시작, 셀 0 CONFIG/START·NG Setup 확인 → ⑦ 10초마다 상태 한 줄 출력. 종료 신호까지(또는 `DURATION_S=초`) |
| `./cots_run.sh stop` | 실행 중인 start에 STOP 신호 → start가 역순으로 정리(OAI → L1), 로그를 `runs/<이름>/`에 모음. start 프로세스가 없으면 `stop --now` |
| `./cots_run.sh status` | 컨테이너/MPS/RAN 프로세스/코어/xApp 포트/GPU 한눈에 |
| `./cots_run.sh ue [이름]` | UE 접속 단계 보고(NG Setup, RA, Msg3, RRCSetupComplete, RRCReconfiguration, PDU 세션, 실패 신호). 실행 중이면 라이브 로그로 |
| `./cots_run.sh test [모델] [req/s] [초]` | xApp 인터페이스 점검: 컨테이너 안에서 `ra_stub.py`가 4K JPEG를 에이전트에 보내고 지연 통계 출력(기본 26x, 2 req/s, 15초) |
| `./install_nvidia.sh` | 이 패키지를 `/home/nvidia/dgist_cots`로 복사하고 소유자를 nvidia로 (sudo) |

## 4. 표준 절차

```bash
cd <이 디렉터리>                 # troy: cuPHY-CP/dapp_hook/cots   nvidia: /home/nvidia/dgist_cots
./cots_core.sh status || ./cots_core.sh up      # 코어가 내려가 있으면 올림 (sudo)
./cots_run.sh start demo1                       # 약 2~3분 뒤 "running - a COTS UE can attach now"
#   ... UE가 붙는지 확인: 모니터 줄의 "UE: RA n RRC n PDU n" 또는 다른 터미널에서 ./cots_run.sh ue
#   ... dApp 확인: 다른 터미널에서 ./cots_run.sh test 26x 2 15
./cots_run.sh stop                              # 또는 start 터미널에서 Ctrl-C
```
백그라운드로 돌리려면 `DURATION_S=1800 ./cots_run.sh start demo1 > runs/demo1_console.log 2>&1 &` 뒤 `./cots_run.sh stop`.

모니터 줄 읽는 법: `L1 Err.ind 0 FATAL 0`(0이어야 정상) `| cell 0: DL 0.04 Mbps UL 0.00 Mbps 0 Slots CRC 0`(UE 없을 때 DL 0.04 = SSB/SIB1 전파 중; UE가 붙으면 UL 슬롯 수와 Mbps가 올라감)
`| UE: RA 1 RRC 1 PDU 1 RA-fail 0`(접속 완료) `| dApp: agent ok n fail n queue n`(xApp 요청 처리 수).

## 5. 로그 (`runs/<이름>/`)

L1 로그는 컨테이너 안 `/tmp/phy.log`에 쌓이고(기준 INFO 레벨에서 **분당 약 45 MB**, 시간당 2.7 GB) `stop` 때 발췌만 남기고 지운다. 여러 시간 데모라면 디스크 여유(`df -h /`)를 먼저 확인.

`l1_console.log`(L1 콘솔), `phy_warn.log`(WRN/ERR/FATAL/Err.ind/RACH/사이저 통계), `l1_persec.log`(셀 0 초당 DL/UL/CRC), `phy_tail.log`(L1 로그 끝 200만 줄),
`oai.log`(gNB 전체 로그: RA/RRC/NAS), `dapp_tenant.log`(테넌트 10초 통계, 배치표), `dapp_tenant_slots.csv`, `l1_env.txt`(L1 환경변수), `oai_compose.log`.

## 6. 문제 분기

| 증상 | 확인 | 조치 |
|---|---|---|
| `precheck failed` | 출력의 `!!` 줄 | 메시지대로(컨테이너, 다른 스택, RAN 프로세스 잔존, PTP, 링크, 코어, 디스크) |
| L1 did not come up | `runs/<이름>/l1_console.log`의 FATAL/ERR | NIC/hugepages/PTP; RU 전원; 다른 L1이 NIC를 잡고 있는지 |
| 테넌트가 안 뜸 | `dapp_tenant.log` 끝 | 엔진 파일 누락(`!! missing`), GPU 메모리 |
| NGSetupResponse 없음 | `./cots_core.sh status`, `oai.log`의 SCTP/NGAP | 코어 재시작(`cots_core.sh down && up`) |
| RA는 보이는데 RRC 안 됨 | `cots_run.sh ue`의 Msg3/RA-fail | 10-30 기준: UE 셀 설정은 기준과 동일하므로 UE 재부팅·비행기 모드 토글 먼저 |
| UE는 붙었는데 인터넷 안 됨 | `cots_core.sh status`의 NAT 규칙 0 | `sudo bash core/routing.sh` |
| L1 Err.ind 증가 | `phy_warn.log` | 연구원 스택과 동시 실행 여부, GPU에 다른 프로세스 |
| `WRN ... Incorrect msg length cell_id=0 msg_id=0x02/0x04` 2줄 | 기동 시 1회씩 | 정상. OAI의 CONFIG/START.request가 8바이트 길지만 L1이 받아들임(10-30 UE 접속 런 포함 모든 OAI 런에서 동일) |
| `YAML invalid key: ul_*` WRN 5줄 | 기동 시 | 정상. 기준 yaml에 있는 키를 이 빌드가 모름(기본값 사용) |
| 26x가 90 ms대, `sm mean 24` | `cots_run.sh test` 출력 | 유휴 슬롯 규칙이 꺼졌거나(`DAPP_TENANT_EMPTY_US=-1`) 계획 선행 시간이 아직 안 잡힘. 기동 1분 뒤 다시 |

## 7. 이 패키지가 건드리는 것 / 안 건드리는 것

- 건드림: 컨테이너 안 `/opt/nvidia/cuBB/cuPHY-CP/cuphycontroller/config/{cuphycontroller_dgist,l2_adapter_dgist}.yaml`(새 파일), `/tmp/phy.log` 등 실행 로그, MPS 상태(끔), OAI 컨테이너 `oai-gnb-dgist`, 코어 서비스/NAT(`cots_core.sh`만).
- 안 건드림: 연구실 yaml(`cuphycontroller_config_benetal.yaml`, `F08_CG1` 등), 연구실 compose/이미지, RU 설정, 연구실 `build.aarch64`, 코어 가입자 DB.

## 8. 검증 기록 (2026-11-05, troy, GPU 낮 시간)

| 런 | 결과 |
|---|---|
| val1 10:01–10:13 (8분) | L1 ready 4 s, 테넌트 ready +18 s(엔진 40개, GPU 83.7 GB), OAI 셀 0 CONFIG/START, NGSetupResponse, SSB/SIB1 전파(DL 0.04 Mbps·50 slots/s = 연구실 기준 런과 동일), **Err.ind 0, FATAL 0**. UE 없음(RA 0, 프리앰블 검출 0). xApp test: 26x **95 ms, sm 24** ← 아래 문제 |
| val2 10:18–10:24 | 클래스 32 제외(엔진 35개, GPU 78.5 GB). 유휴 슬롯 규칙 1차: 여전히 sm 24 (지난 프레임 조회가 "계획 없음"을 유휴로 안 봄) |
| val3 10:24–10:35 | 규칙 2차: 26n/26m은 80 SM, 26x는 24(지평선 220슬롯 > 예측 8프레임). 그런데 26n 추론 11.0 ms(루프백 5.6)의 2배: 테넌트 스레드가 매 슬롯 모델 5개 × 200슬롯 지평선 조회에 묶여 조각 발사가 늦어짐 |
| val4 10:37– (15분) | 지평선 캐시 + 예측 12프레임: **26n 11.4 ms(추론 5.7) / 26m 21.0(14.9) / 26x 38.2(32.3), 모두 sm 80** = 11-04 루프백·스윕 값과 일치. Err.ind 0, FATAL 0 |

바뀐 코드(모두 우리 파일, NVIDIA 소스 변경 없음): `dapp_hook/include/dapp_hook/dapp_sizer.hpp`(계획 선행 시간 `g_plan_lead_ns`),
`dapp_hook/tenant/dapp_tenant.cu`(`plan_of` 유휴 슬롯 규칙 `DAPP_TENANT_EMPTY_US`, 예측 프레임 기본 12, `horizon_cache`). `build.aarch64_f1002` 재빌드 10:37.

UE 접속 자체는 이날 UE가 꺼져 있어 확인하지 못했다. UE가 보는 모든 설정은 10-30 접속 성공 런(ue21)과 동일하므로, UE를 켜고 `./cots_run.sh ue`로 단계를 확인하면 된다.
