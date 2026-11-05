# CONTEXT.md — FAPI-Aware GPU Scheduler 연구 컨텍스트

> Claude Code는 PDF(논문·Aerial 공식문서)를 읽지 못하므로, 이 파일이 그 문서들의 근거 요약본이다.
> 작업 전에 반드시 읽고, 여기 적힌 값·결정과 충돌하는 것을 발견하면 진행하지 말고 보고할 것.
> 마지막 갱신: 2026-10-01 (v5 — ⑥-0 조사 결과 반영. **서버 사본이 v2에 머물러 있었음: 이 파일로 교체할 것**)

## 1. 연구 목표 (한 문장)

GH200 위에서 inline cuPHY(Aerial CUDA-Accelerated RAN 25-3)와 AI 추론(YOLO26n/26x, 이후 LLM 등)을 MPS로 공존시키되,
FAPI 메시지로 확정되는 슬롯별 cuPHY 부하를 보고 AI 세입자의 SM 상한(cap)과 투입 허가(gate)를 조절하여,
cuPHY 채널별 deadline 신뢰도 99.9999%(6-nines)를 지키면서 AI 처리 용량을 최대화한다.

## 2. 시스템 구조 (구현 중)

| 컴포넌트 | 위치 | 역할 | 주기 |
|---|---|---|---|
| A. cuPHY 훅 | cuphycontroller 내부 (scfl2adapter, `scf_5g_fapi_phy.cpp` 부근) | FAPI 메시지 요약을 `aerial_dapp_ring`(공유메모리)에 레코드로 발행 (UL_TTI, DL_TTI, UL_PDU, DL_PDU, SLOT_END). 채널별 완료 시각(CHAN_DONE)은 **아직 없음** | 슬롯(0.5ms) |
| B. 스케줄러+예측기 (dApp) | 별도 실시간 프로세스, isolated Grace 코어 | 채널별 latency 상한 예측 → gate, cap 결정 | gate 0.5ms / cap T |
| C1. 디스패처 | AI 쪽 sidecar | 요청 큐, gate·cap 적용, 동적 배치, 워커 선택 | 요청마다 |
| C2. AI 워커 | MPS 클라이언트 | 추론 실행. **액추에이터 = 한 프로세스 안의 SM 개수 컨텍스트 풀(`dapp_sched`, cuCtxCreate_v3 exec affinity)로 확정.** percentage 방식 워커는 벤치 스윕용 | 추론마다 |

- **cuPHY는 훅 외에 수정하지 않는다.** cuPHY 채널별 cap(`mps_sm_*`)은 초기화 때 고정, 바꾸려면 L1 재시작(=셀 다운).
  재구성 대상은 항상 AI 세입자뿐. CPU fallback 없음(inline PHY라 불가).
- **GPU에서 도는 스케줄러 코드 없음.** 스케줄러는 정수 두 개 `{gate_slots, cap_pct}`를 공유메모리에 쓸 뿐.
- 현재 저장소: `dapp_sched`는 한 프로세스 안의 **SM 개수 컨텍스트 풀** 방식(= Phase 2 액추에이터, 인정). estimator는 규칙 기반·deadline 임시값·최근 구간 백분위로 cap 결정(= §8 목표 설계의 이전 단계). §8은 목표이고 코드가 아직 거기 있지 않음을 전제로 작업.

## 3. 용어 (이 이름으로 통일)

- **cap (SM 상한)**: AI 워커 커널이 한 순간에 점유할 수 있는 SM 최대 %. CUDA 컨텍스트에 묶여 있어 실행 중 변경 불가.
  변경 = 미리 만든 컨텍스트 풀에서 선택(확정) 또는 워커 재시작(수백 ms). 후보 레벨 {10,20,…,100}.
  **엔진 제약**: percentage 방식은 빌드 시점의 cap에서만 TensorRT 엔진이 동작(현재 40개 엔진 = cap 10단계 × 모델 2 × 배치 2). SM 개수 컨텍스트 방식은 작은 SM에서 빌드한 엔진 하나가 더 큰 컨텍스트에서도 동작 → 풀 방식이 실용적.
- **gate (투입 허가)**: 지금부터 안전한 슬롯 수 k. 0이면 새 추론을 던지지 말라. 비용 0, 슬롯마다 갱신.
- **SafeCap(π)**: 최악 부하(φ_peak)에서 모든 채널이 deadline을 지키는 최대 cap. 미래 예측 없이 안전한 바닥.
- **regime**: 슬롯 부하가 아니라 수십 ms~초 단위로 변하는 부하 수준(저/고). SafeCap 위로 느슨하게 갈 때만 사용.
- **π (세입자 프로필)**: 모델×배치×해상도별 특성(dur(batch,cap), sm_fill, kernel_max_us, HBM 대역폭).
- **잔차(residual)**: 실제 latency − 예측 latency. 컨텍스트 버킷별 ring buffer에 모아 최댓값을 상한에 더함(Concordia 방식).
- **horizon**: 아는 미래의 길이. **UL과 DL이 다르다(링 실측).** UL_TTI.request는 슬롯 N의 GPU 처리(N+1~N+2)보다 약 1.5~1.9ms 앞서 도착 → UL 채널은 3~4슬롯 선행. DL_TTI.request는 도착 57µs 뒤 바로 enqueue → **DL 채널의 선행 시간은 사실상 0.** (testMAC 기준, 실 RU 미검증)
- **일감(work unit)**: FAPI 필드를 cuPHY 커널별 처리량으로 변환한 feature. 채널마다 Σ(총량)/max(단일 PDU 임계경로)/count(런치 수).

## 4. 선행 연구 핵심 사실 (PDF에서 확인됨)

### Concordia (Foukas & Radunovic, ACM SIGCOMM 2021) — CPU, Intel FlexRAN v20.02
- 유저스페이스 deadline 스케줄러, **20µs마다** 코어 yield/wake. vRAN 최우선, 나머지 best-effort. 목표 **99.999%**. 유휴 CPU 최대 70% 회수.
- 예측기: **task(DAG 노드, 40개+)마다 quantile decision tree** + leaf별 ring buffer(5K) **최댓값**을 WCET로. 온라인은 leaf 샘플만 갱신, 트리 고정.
- feature = TTI별 기지국 상태(스케줄된 UE 수, TB 크기, 레이어, MCS 등) = 사실상 FAPI 내용. Algorithm 1: 손 선택 + distance correlation 상위 N + backward elimination.
- 학습 데이터: TTI마다 파라미터 무작위(UE 0~16, TBS, MCS), **500K 샘플**, 격리 상태 프로파일링.
- Fig 14: 선형 회귀는 deadline miss 수십 %로 실패, gradient boosting ≈ quantile DT(miss율), QDT가 tightest(평균 오차 43µs).
- 관찰: 셀 부하 peak 대비 평균 CPU 이용률 ≤42%(Fig 4a); 합산 트래픽 변동은 √n로 증가(§2.2); 캐시 간섭으로 runtime 분포 변함(KS p≪0.001).
- baseline: pWCET(EVT), Shenango 변형, utilization 기반(burst 못 잡아 <99.99%). 동거 워크로드 Redis/Nginx/TPCC/MLPerf. 15분 런 기본, 5-nines는 8시간.
- §7: 가속기(GPU/FPGA) 확장은 future work로 명시.

### YinYangRAN (Lo Schiavo et al., 2024) — GPU A100, FEC(LDPC 디코딩) look-aside만
- DU의 **FEC만** GPU(BBDev/O-RAN AAL). 나머지 PHY는 CPU. 재구성 중 DU는 **CPU FlexRAN으로 fallback**.
- MPS vs MIG 실측: 재구성 **MPS 0.266s, MIG 6.9s**, 그동안 GPU 작업 불가 → MIG 부적합.
- SMO/O2에서 **δ=1s** 주기 결정. 상태 = 지난 1초의 (TB 크기, SNR) 2D 히스토그램(D=5). CNN 기반 분위수(10개) 신뢰도 추정기, τ1=0.005.
- YYR-Full(CEC 동적계획, 재구성 비용 r 고려) / YYR-Lite(greedy). 신뢰도 목표 **ε=0.01~0.25 (75~99%)**. naive 공유 대비 신뢰도 50%↑.
- 컨텍스트별 최소 SM 비율 **1~80%**(Fig 4). Fig 5: 단일 100MHz TB, MCS 0~28 × SNR 0~30dB × SM 15/50/100%.
- 온라인 학습 없음. 셀 1개.

### 2025~26 경쟁 연구
- AoRA (SIGCOMM'25 short, KAIST/SNU): AI-on-RAN, GPU 사용률 임계값 admission + MEC fallback. MPS/SM 분할 언급 없음.
- Northeastern+NVIDIA (arXiv 2512.06493): GH200에서 dApp 추론을 cuPHY와 동거, MIG 없이 공유 시 메모리 경합으로 latency 변동 관측, "스케줄링 메커니즘 탐색 중".
- INFOCOM WS 2026 (arXiv 2602.23722): LLM+DU를 MIG 정적 분할로 안전 동거. 오케스트레이터 제안 없음. → 현재 state of practice = 정적 MIG.

## 5. Aerial 25-3 공식문서에서 확인한 사실

- **TDD만 지원**(FAPI CONFIG FrameDuplex: "Only TDD is supported").
- **SLOT.indication은 슬롯 N−3에 발행**(3 slot advance), L2 응답 예산 500µs → UL_TTI/DL_TTI.request는 약 N−2에 L1 도착.
- cuPHY driver는 채널별 **MPS 서브컨텍스트**를 만들고 SM 상한을 건다: `mps_sm_{pusch,pdsch,pdcch,pucch,prach,srs,pbch,ul_order,gpu_comms}`.
  MIG 4g(64 SM) 예제에서 상한 합이 190 → **MPS cap은 겹칠 수 있는 상한이지 예약이 아님.** 채널별 stream priority 존재.
- `use_green_contexts`: experimental, 문서상 "끄라". 건드리지 않는다.
- E3 Agent(`e3_agent_enable`): 슬롯 파이프라인 완료 후 indication을 공유메모리로 dApp에 발행. NVIDIA 샘플 dApp 공개(aerial-sample-apps).
- FAPI 스누핑: OAM `GetFAPIStream` gRPC, nvIPC pcap 로거(25-3), Data Lake(`fapi`, `fh` 테이블, ClickHouse).
- SINR: cuPHY가 PUSCH PDU마다 계산해 **CRC.indication `ul_cqi`**로 보고(`pusch_sinr` 1=PostEq/2=PreEq). UL_TTI.request에는 없음.
- 위반 신호: `ul_order_timeout_gpu_ns`, PUSCH work-cancel 모드, ERROR.indication late.
- 플랫폼: CUDA 12.8/12.9(드라이버 575). GH200 4T4R 100MHz **peak cell 20개**(F08_20C). MU-MIMO UL 8 / DL 16 레이어, MCS 0~27, 273 PRB.
- NVIDIA AI-RAN 검증 구성: cuBB on **MIG 4g.48gb(64 SM, peak cell 8개)** + LLM on MIG 3g. (정적 MIG baseline)
  9/28 MIG 측정은 임의 SM 배분(상한 합 130)이었음 → baseline으로 쓰려면 **NVIDIA 예제 값(상한 합 190)** 으로 재측정.
- CUDA 문서: MPS active thread percentage 하에서 green context는 provision보다 많은 SM에서 돌 수 있음(상한 누수 가능).

## 6. 테스트베드

- 환경은 세 가지이며 섞지 말 것:

| 환경 | FAPI·훅 | 슬롯 타이밍 | 셀 수 | 용도 |
|---|---|---|---|---|
| 벤치 (cubb_gpu_test_bench) | 없음 (TV 규격 직접 읽음, 자체 yaml, launch pattern 안 씀) | 패턴 안에서만 | 16 | 귀속 실험(①②), 간섭 곡선, 규모 |
| full stack loopback (testMAC + cuphycontroller + RU 에뮬레이터, launch pattern 사용) | 있음, 훅 검증됨 | 있음 | **16 (주 실험)**, 8 (MIG 비교 지점) | 신뢰도·SafeCap 최종값, 시스템 평가, MIG baseline |
| 실 RU 1셀 (Benetel RU + Falcon 스위치 + GH200) | 있음 | 있음 (미검증) | 1 | E2E·훅·타이밍 검증, 보정항 |

- 실 테스트베드 TDD는 **DSUUUDSUUU**(UL 중심). full stack loopback도 이 패턴으로.
- **주 실험은 16셀이다.** 8셀은 GPU 여유가 커서 어떤 세입자·cap이든 안전해 스케줄링 문제가 생기지 않는다(벤치: 16셀 고부하에서야 PUSCH 여유 ~30 %, 26x b8 SafeCap 20). 8셀은 NVIDIA MIG 4g 검증 구성과 같은 셀 수이므로 **정적 MIG와의 정면 비교 지점**으로만.
- **9/11 16셀 격리 런(40,000슬롯, DDDSUUDDDD, testMAC) 조사 결과(⑥-0 완료)**:
  - A. "드롭 2회" = 첫 tick 워밍업(SFN 0 슬롯 5·6, L2 어댑터 처리 600µs 초과 시 슬롯 명령 폐기, NVIDIA 코드가 첫 UL·DL 슬롯용으로 둔 분기). GPU 유휴 상태에서 발생, 이후 19초 동안 추가 없음. → **통계에서 런 시작 첫 1초 제외.**
  - C. +19.05s의 ~2ms 정지 = nsys GPU 메트릭 수집(10kHz, 60초) 종료가 만든 것(4셀에서는 수집 시작 시점에 2회). → **측정 구간 안에서 nsys 시작·종료 금지. GPU 개요는 DCGM 1초로.**
  - 벤치 격리 max 3ms 아웃라이어 = 추적 도구가 만든 커널 발행 공백 2.09ms(두 반복에서 같은 패턴 7351, 같은 레코드 수 직후). 경합 아님. 해소.
  - **D. 진짜 경계 신호**: RU 에뮬레이터 기준 DL U-plane 지연이 셀당 72슬롯(0.24%), 그중 ~62슬롯이 **TDD 위치 6·7(U 슬롯 직후 DL)**. L1 로그엔 오류 없음. 8셀에서도 셀당 5슬롯(~10⁻⁴), nsys 없어도 동일. 주기 내 2.5~3.0ms 구간에서 표본 19%가 SM ≥95%로 GPU 포화와 시각이 맞지만, DL 워커가 CPU 3코어뿐이라 CPU 원인 미배제. **원인(GPU vs CPU) 판별이 다음 과제.**
  - 함의: (i) 절대 6-nines는 8셀에서도 격리 baseline이 못 미침 → no-degradation이 주 지표(§8-9). (ii) U→D 전이 위치는 격리에서도 채널 간 경합이 드러나는 자리 → **TDD 슬롯 위치는 결정론적 feature이자 regime**. 스케줄러가 위치 6·7 앞에서 gate를 선제적으로 닫는 정책 후보.
  - 기록 정정: 9/11의 "testMAC 전 셀 ERR 0"은 16셀에서 거짓(첫 1초 오류 통보 22건). early HARQ는 이 구성에서 켜져 있음.
  - 드롭 판정 기준(코드): tick 이후 L2 + L2 어댑터 처리 > 600µs(500 + 허용 100) → 슬롯 명령 폐기.
- 벤치 feature는 TV 규격에서 뽑되 열 이름은 훅 필드와 동일하게.
- 벤치 TDD 모드는 DDDSU / DDDSUUDDDD / mMIMO 세 가지뿐. DSUUU는 yaml만으로 불가, `-u 4` 범용 모드 확인 → 안 되면 **벤치 코드 수정 허용**(벤치·testMAC 패턴은 테스트 도구이므로 §9의 cuPHY 무수정 규칙에 해당하지 않음). 이전 측정치는 DDDSUUDDDD 기준.
- 벤치는 **패턴 안 pacing만 있고 패턴 간 이월이 없으며**, **ul_order 커널이 없다.** 따라서 벤치 = 귀속 실험(①②)과 간섭 곡선용, full stack(cuphycontroller + testMAC/RU 에뮬레이터 또는 실 RU) = 신뢰도·SafeCap·타이밍용.
- 슬롯 타이밍 실측(DL 57µs 등)은 모두 testMAC + RU 에뮬레이터 기준. 실 RU 1셀 미검증.
- UL_DCI.request는 훅하지 않음(UL grant DCI가 feature에서 빠짐). MIG 켜면 nsys 시작 실패(설정 지시자로 해결).
- 16셀 빌드 제약: 셀 그룹당 TB 128개 상한 → **셀당 최대 8 UE**.
- 채널 deadline: `testBenches/perf/measure/TDD/check_cell_capacity.py` 60~101행. PUSCH/PUCCH 1500, PRACH 1250(PUSCH 직렬 옵션 off), SRS 500, PDSCH/PDCCH 375 µs. **DDDSUUDDDD 벤치 타임라인에 대한 판정값이라 패턴을 바꾸면 재확인.**

## 7. 지금까지 측정된 사실 (16셀, DDDSUUDDDD 벤치, 런당 1.7~1.9만 슬롯 → p99까지만 유효)

- 세입자 순위: 26x b8 ≫ 26x b1 ≈ 26n b8 > 26n b1. **배치가 모델 크기보다 큰 변수.**
- SafeCap 판정 규칙(고정): **임시 = 모든 채널 p99 ≤ deadline의 85 %**; **최종 = 런당 ≥10⁶ 슬롯에서 p99.9 + margin_c ≤ deadline**. 판정은 **"아래 cap이 모두 통과하는 최대 cap"**(단조 일관, 보수적). 비단조 점은 노이즈 후보로 3회 재측정 중앙값.
  85 % 규칙 적용(26x b8 최악 채널 p99 사용률, 고부하: cap10 74.0 / 20 78.4 / 30 85.9 / 40 97.8 / 50 101.8 %; 저부하: 60 80.0 / 70 92.9 / 80 84.2 / 90 83.3 / 100 89.7 %):
  **26x b8 SafeCap = 고부하 20, 저부하 60**(70이 비단조 실패, 재측정 대상). 26n b1/b8, 26x b1은 고부하에서도 최대 81.5 % → 100. 모두 DDDSUUDDDD 벤치 기준 provisional.
  **이전에 p99.99 기준으로 보고된 값(20/90, 부하 인지 이득 +157 %, trade-off 그림, README)은 철회. p99 provisional로 재표기할 것.**
- 채널 민감도(고부하 26x b8): PRACH > PDSCH > PDCCH > PUSCH > SRS > PUCCH. 저부하는 PDSCH·PDCCH가 가장 민감. PUCCH는 둔감.
- 세입자 실행시간(슬롯, 고부하 p99): 26n b1 5.6@cap100 / 26n b8 7.7@100 / 26x b1 9.5@100 / 26x b8 27@100, 45@30.
  동거 세금: 단독 대비 p99 배율 1.2~3.0×(26n b1만 2.1~3.0×, p50은 1.2~2.0×). **cap과 무관한 것은 배율이 아니라 늘어난 절대 시간**(cap 20~100 구간). RAN 부하의 영향은 **7~20%뿐** → 실행시간은 (모델, 배치, cap)으로 예측 가능.
- throughput은 cap보다 배치가 좌우: 이미지당 26n b8@100 1.0슬롯 vs b1@100 5.6; 26x b8@30 5.6 vs b1@100 9.5.
- 격리 고부하 PUSCH p99 1013µs(deadline의 69%) → 여유 ~30%. 격리 **max 2976µs 아웃라이어** 존재(원인 미규명, 6-nines 전제 위협).
- **cap 10 이상 현상**: 26x에서 cap 10의 꼬리가 cap 20~40보다 나쁨(미검증). 단조성 가정에 영향.
- 무거운 세입자에서 런당 패턴 수가 최대 12% 감소하는 것은 유실이 아님. 벤치는 **패턴 안에서는 pacing이 있고**(DL 슬롯 500µs 절대 시각, UL 고정 지연 → 패턴 안 적체는 재현) **패턴 간(5ms 주기)에는 이월이 없다**(앞 패턴이 끝나야 다음 시작). 채널 표본 비율 6:2:1 정확. 함의: 주기를 넘는 적체가 재현되지 않아 **위반율을 과소평가할 수 있음** → 신뢰도·SafeCap 최종값은 full stack에서.
- 기존 데이터로 FAPI 필드 귀속은 불가(설계점이 lowest/highest 2개뿐). 8셀 one-at-a-time 스캔 31점, 16셀 사다리 10점 있음(YOLO 없음).

## 8. 설계 결정 (변경 시 보고)

1. **안전은 예측이 아니라 상한에**: cap의 바닥 = SafeCap(π) (최악 부하 기준, horizon 무관). regime은 그 위로 느슨하게 갈 때만, 오판율 ε(W,D)를 트레이스로 측정해 신뢰도 예산 안에서.
2. **두 시계**: gate는 매 슬롯, cap은 T마다(Phase 1 재시작 ~1s → Phase 2 컨텍스트/워커 풀 ~10ms).
   **gate의 확정값 보호는 UL 채널(PUSCH·PUCCH·PRACH·SRS)에만 해당.** DL 채널(PDSCH·PDCCH)은 선행 시간이 0이라 SafeCap과 regime으로만 보호. "FAPI lookahead는 UL에만 존재"는 논문의 관찰 항목.
3. **예측 대상은 채널별 latency 상한** Q̂_c(φ, X, π) = f_c + 버킷 잔차 최댓값 + margin_c. 최소 SM은 이를 뒤집어 탐색. 채널 6개(PUSCH, PDSCH, PDCCH, PUCCH, PRACH, SRS). PBCH·CSI-RS는 존재 feature + 상수. ul_order는 PUSCH 안에 접음(**벤치에는 없으므로 full stack에서만 관측**). gpu_comms는 감시만.
4. **feature는 원시 필드가 아니라 커널 일감**(Σ/max/count, 셀 합산 + 인접 ±2슬롯 동시 부하 + 간섭 π + **TDD 주기 내 슬롯 위치**). 훅이 읽은 값으로 만들고 생성기 설정값은 SNR·검증용으로만.
5. **모델**: 슬롯 단위 대량 데이터엔 LightGBM(+monotone 제약, 잔차 버킷, Concordia QDT 비교군). **설계점이 수십 개인 정적 실험엔 트리 금지** → log-linear 회귀 + Sobol 분산 분해.
6. 동시 실행: Phase 1은 "동시에 하나만". 이후 Σcap ≤ SafeCap 회계. 요청 폭주는 동시 실행이 아니라 동적 배치로.
7. 평가 구조: 벤치 16셀 = 귀속·간섭 곡선, **full stack 16셀 = 주 실험(신뢰도·SafeCap·시스템 평가)**, full stack 8셀 = 정적 MIG 비교, 실 RU 1셀 = E2E·훅 검증. baseline = 정적 MIG(NVIDIA 구성, 상한 합 190), 정적 SafeCap, YYR-Lite 재구현, naive 공유(cap 100).
8. **margin_c 정의(임시)**: margin_c = (격리 런 ≥10⁶슬롯의 max − p99.9) + 세입자 drain δ_c(π, X) + 지터 20µs. δ_c는 스윕에서 채널별로 (L_k − L_iso)의 상위 꼬리로 추정. 값이 확정되면 여기 기록.
9. **신뢰도 지표는 두 층**:
   - **주 지표 = no-degradation**: 동거 시 채널·슬롯 miss율이 같은 조건의 격리 miss율을 넘지 않는다(추가 miss 0건, ≥3×10⁶ 슬롯, rule of three로 추가 miss율 95 % 상한 10⁻⁶). 격리 baseline이 완벽하지 않아도 성립하며 테스트베드 결함을 우리 시스템 탓으로 떠안지 않는다(Concordia도 isolated FlexRAN 대비 비교).
   - **절대 6-nines**: 격리 baseline이 ≥3×10⁶ 슬롯 위반 0건인 조건에서만 주장. 16셀 격리 드롭 원인이 규명·제거되면 16셀에서, 아니면 그 조건이 성립하는 셀 수·부하에서.
   - 위반이 나오면 원인(GPU 경합 / fronthaul / CPU / 워밍업)을 분류해 보고. 분류 없이 합산 금지.

## 9. 금지·주의 (위반하려 하면 멈추고 보고)

- cuPHY 라이브러리·cuphycontroller의 GPU 관련 설정(`mps_sm_*`, LDPC 반복, priority, `use_green_contexts`) 변경 금지. 훅 레코드 형식 변경 금지. **테스트 도구(벤치, testMAC 패턴, TV 생성기)와 CPU 코어 배정(DL/UL 워커 코어 수, isolcpus)은 수정 가능**하되 변경 내용과 전후 비교를 보고.
- 측정 런 안에서 nsys 시작·종료 금지. 통계는 런 시작 첫 1초 제외.
- 벤치(패턴 간 이월 없음)에서 얻은 latency로 신뢰도·SafeCap 최종값 주장 금지. 벤치는 귀속·간섭 곡선용.
- 훅 필드를 코드가 아니라 가정으로 정의 금지. 파서는 훅 헤더를 import/생성.
- 파일럿 없이 본 런 금지. 런 간 슬롯 컨텍스트가 다르면 "대략" 짝짓기 금지.
- 런당 슬롯 ≥10⁶ 미만이면 p99.99·6-nines 주장 금지(1.8M 슬롯은 p99.9까지).
- 설계점 수십 개 데이터에 트리 모델 금지.
- 생성기 설정값을 feature로 사용 금지(SNR 제외).

## 10. 알려진 버그·미해결

- YOLO fps 과대 집계 (수정 필요). 추론 단위 latency 샘플 미저장.
- 벤치에서 UL/DL이 시간상 거의 안 겹침 → 채널 간 경합 과소평가 (DSUUUDSUUU 배치로 수정 중).
- CHAN_DONE(채널별 완료 시각) 훅 없음. 현재 label 출처를 hook_schema.md에 명시할 것.
- **D(위치 6·7 DL 지연)의 원인: GPU 포화 vs DL 워커 CPU 부족** — 미판별. cap 10 이상 현상, 저부하 26x b8 cap 70 비단조 점: 미규명(재측정 대상). 격리 max 아웃라이어·드롭 2회·2ms 정지는 해소.
- deadline 상수: 출처 확인됨(check_cell_capacity.py). 패턴 변경 시 재확인.
- DL 커널 실제 시작 시각(트레이스) 미확인. UL_DCI 훅 없음. 커널 로그(dmesg)·runqlat은 root 권한 필요.

## 11. 다음 단계 (짧은 프롬프트로 하나씩)

①-a 벤치 `-u 4` 범용 모드로 DSUUUDSUUU 가능한지 확인, 불가면 벤치 코드 수정안 보고. 벤치 실시간 pacing 옵션 유무 확인.
①-b 이전 p99.99 기반 산출물을 p99 provisional로 재표기, SafeCap을 85 % 규칙으로 재계산
② fps 수정, 추론 샘플 저장
③ 정적 설계점 40~60개 + 시간표 (실행 금지)
④ 실행 {없음, 26x b8 cap30, cap100} + 10점 × {cap60, 26n b1 cap100}  (벤치)
⑤ 분석: ① log-linear, ② Sobol(field×cap), Σ/max 전환, PRACH R²
⑥-0 (완료) 드롭 2회 = 워밍업, 2ms 정지 = nsys, 벤치 3ms = 도구. 남은 것은 D.
⑥-1 D 원인 판별: 파일럿 2분(DDDSUUDDDD 16셀, nsys 없음) → A 재현·C 소멸·D 유지 확인. 이어서 DL 워커 코어 수를 3→6으로 늘린 런과 비교. 판정: GPU 완료가 늦으면 GPU 경합, GPU 정상인데 압축·송신 태스크 시작이 늦고 runqlat 꼬리가 크면 CPU, L1 송신 정상인데 RU만 늦으면 fronthaul/에뮬레이터.
⑥ full stack **16셀** DSUUU에서 SafeCap·no-degradation 재측정(구성당 ≥3×10⁶ 슬롯) + 8셀 MIG baseline(NVIDIA 상한 합 190) + 벤치↔full stack 보정항 5점 + 실 RU 1셀 훅·타이밍 검증
