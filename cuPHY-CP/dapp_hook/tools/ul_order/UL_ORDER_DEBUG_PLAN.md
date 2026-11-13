# UL 결과 순서 뒤바뀜 → HARQ 폭주 → 폰 송신 전력 고착: 원인과 디버그 빌드 계획 (2026-11-14)

대상: `_dgist` Ours 모드(aerial b02a0a9, 6셀 = 폰 셀 1 + 발생기 셀 5), 실행 `ours_tg_auto_q90_1114_0510`.
표기: [측정] 로그에서 센 것, [코드] 소스를 읽은 것, [가설] 아직 재지 않은 것. 1~4절은 수정 전(05:54)에 쓴 계획이고, 수정과 검증 결과는 5절에 있다.

## 1. 원인 사슬

1. [측정] L1이 이웃한 두 UL 슬롯의 PUSCH 결과를 gNB에 **순서를 바꿔** 올린다. UL 슬롯마다 다른 워커(UlPhyDriver06/07)가
   끝을 기다렸다가 콜백을 부르는데(`cuphydriver/src/uplink/task_function_ul_aggr.cpp:2488-2498`), 슬롯 간격은 0.5 ms이고
   PUSCH 한 슬롯은 1.8~2.6 ms 걸린다. 앞 슬롯이 뒤 슬롯보다 0.5 ms 넘게 오래 걸리면 뒤 슬롯이 먼저 보고된다.
   첫 사례: SFN 959.18(2587 us)이 959.19(1987 us)보다 늦게 보고, 87 us 뒤 gNB 첫 경고.
2. [코드, 연구실 OAI] `handle_nr_ul_harq`(`openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_ulsch.c:668-725`)는 CRC 결과가 보낸 순서대로
   온다고 가정한다. 다르면 그 앞의 HARQ 프로세스를 전부 재전송 목록으로 넘긴다(아직 L1이 디코딩 중인 것 포함).
   경고 `Unexpected ULSCH HARQ PID N (have M)`.
3. [코드+측정] 그 재전송이 아직 쓰이는 HARQ 버퍼에 도착 → `harq_pool.cpp:389` "MAC reusing hqbuf … in flight" →
   `PhyPuschAggr::setup`이 첫 실패에서 `return -1`(`phypusch_aggr.cpp:800-810`) → **6셀 묶음 슬롯 전체**를 디코딩하지 않는다
   (`task_function_ul_aggr.cpp:258-275, 628-635`).
4. [측정] 자기 유지 고리: 실패한 슬롯은 GPU를 기다리지 않아 시작 후 약 1.5 ms에 보고된다(정상 1.8~2.6 ms). 그래서 바로 앞의
   정상 슬롯을 23~31 % 확률로 추월한다 → 2번이 또 일어난다. 이 실행의 순서 뒤바뀜 6,263쌍 중 약 5,900쌍이 이 형태다.
5. [측정+코드+가설] 폭주가 끝난 뒤에도 폰 4대가 최대 송신 전력에 고착된다. PH(gNB 출력값 = 폰 보고값 + 10·log10(PRB·2))가
   48 → 5~9 / -22, SNR 20 → 3~8 dB, MCS 0, RSRP는 그대로(-65~-75 dBm). OAI는 SNR < 목표-9 dB이면 +3 dB 명령을 내고
   (`gNB_scheduler_primitives.c:3118`), RSSI로 막는 장치는 꺼져 있다(`pusch_RSSI_Threshold` 기본값 0,
   `GNB_APP/MACRLC_nr_paramdef.h:150`; gnb.conf에 줄 없음). L1 어댑터는 파이프라인이 돌았는지 확인하지 않고
   `out.pSinrPostEq[ue_index]`를 그대로 보고한다(`scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp:3984-4005`).
   [가설] 폭주 중의 낮은/낡은 SNR 보고가 전력을 끝까지 올리고, 최대 전력에서는 RU 수신단이 찌그러져 SNR이 낮게 유지된다.

## 2. 무엇이 1번을 일으키나 [측정]

PUSCH 슬롯 시간(us, p50 / p99 / max), sizer가 준 PUSCH SM 계획 × YOLO 실행 여부:

| 계획 SM | YOLO 쉼 | YOLO 실행 중 |
|---|---|---|
| 20 | 1838 / 2330 / 2644 (637,640) | 1985 / 3157 / 3582 (3,499) |
| 28 | 1859 / 2479 / 2694 (144,733) | 2034 / 2970 / 3456 (1,953) |
| 36 | 1835 / 2484 / 2644 (104,890) | 2063 / 2743 / 3215 (2,151) |
| 44 | 1763 / 2010 / 2136 (41,924) | 1923 / 2268 / 2342 (684) |
| 52 | 1969 / 2130 / 2164 (6,383) | 1998 / 2374 / 2581 (410) |
| 108 | 1760 / 1839 / 1932 (1,539) | 1826 / 1947 / 1947 (18) |

- 정상/정상 쌍의 뒤바뀜 비율: YOLO 실행 중 2.3~4.7 %, 쉼 0.03~0.07 % (약 100배).
- Baseline + auto(`base_tg_auto_q90_1114`): 3,736,002 슬롯, 실패 0, 뒤바뀜 0, p99 1.92~2.07 ms.
- gNB 경고 수: Ours low 0 / 0 / 0, Baseline 0~2, Ours high 111,040, Ours auto 137,007 / 244,831 / 218,230.
- 시작 시점: 20:20:05 VA 폰 2대 스트리밍 시작(폰 셀 UL 30 → 65 Mbps, YOLO 시작) → p99 2.2~2.5 ms → 15 초 뒤 폭주.
- 열림: YOLO가 다른 SM에서 도는데 왜 PUSCH가 느려지는지, 폰 TB 크기 효과와의 분리, `mid` 레벨.

## 3. 빌드 항목 (우선순위 순; L1 소스 변경은 CONTEXT.md 9절에 따라 보고 대상)

- **D0 계측 (먼저, 동작 변화 없음)**
  - 콜백 순서 검사: UL 슬롯에 일련번호를 주고 콜백 직전에 "직전 보고 번호+1"이 아니면 카운터 + 한 줄(슬롯, 시간 차).
  - 실패 슬롯이 올려 보내는 값 한 줄: PDU별 CRC, ul_cqi, rssi, TA (낡은 값인지 확인).
  - "MAC reusing hqbuf"에 버퍼를 잡고 있는 슬롯과 경과 시간 추가, 초당 건수 제한.
  - 셀별 버려진 슬롯 카운터를 초당 줄에.
- **F1 순서 보장 게이트 (핵심)**: `task_function_ul_aggr.cpp:2488-2498`에서 앞 UL 슬롯의 콜백이 끝날 때까지 뒤 슬롯의
  validate/callback을 붙잡는다. 실패 슬롯도 같은 문을 지나야 한다. 앞 슬롯이 콜백 없이 끝나는 모든 경로에서 번호를
  넘기거나, 시간 제한(예: 1 ms)으로 풀고 경고를 남긴다.

  ```cpp
  // sketch: slot_map->ul_seq is assigned in slot-command order; ctx->ul_seq_reported is atomic
  const uint64_t my = slot_map->ul_seq; const t_ns t0 = Time::nowNs();
  while(ctx->ul_seq_reported.load() + 1 != my && Time::nowNs() - t0 < UL_ORDER_GATE_TIMEOUT_NS) { /* spin */ }
  if(ctx->ul_seq_reported.load() + 1 != my) { ctx->ul_order_gate_timeouts++; /* NVLOGW once per second */ }
  pusch->validate(...); pusch->callback(...);
  ctx->ul_seq_reported.store(my);
  ```
  비용: 뒤 슬롯 보고가 앞 슬롯이 늦은 만큼(수백 us) 늦어진다.
- **F3a 슬롯 전체 포기 금지**: `phypusch_aggr.cpp:800-810`에서 HARQ 버퍼 충돌이 난 UE만 빼거나(그 UE는 CRC 실패 보고)
  임시 버퍼로 결합 없이 디코딩하고, 나머지 UE와 5개 셀은 정상 처리.
- **F3b 실패 슬롯의 보고 내용 정의**: 파이프라인이 돌지 않은 슬롯은 측정값을 "없음"으로 보고(ul_cqi 무효, rssi 0xFFFF,
  TA 0xFFFF). OAI가 무효값을 어떻게 다루는지(`nr_rx_sdu` :936-1001) 확인하고 맞춘다.
- **연구실 OAI 쪽 제안(우리가 고치지 않음)**: F2 `handle_nr_ul_harq`가 목록 앞을 비우지 말고 HARQ id로 찾기,
  F5 `pusch_RSSI_Threshold` 설정으로 전력 상승 제한(값은 RSSI를 먼저 로그로 보고 정한다).
- **F4 설정만 바꾸는 실험(빌드 없음, GPU 설정 변경 = 보고 대상, 미측정)**: `.env DAPP_PUSCH_PROFILES`의 최소를 40으로
  (계획 ≥ 44에서 p99 ≤ 2.1~2.4 ms). YOLO가 받는 SM이 준다.

## 4. 재현과 판정

1. Ours, 발생기 `low`로 기동, 폰 4대 접속, xApp 시작, VA 2대 + non-VA 2대 스트리밍.
2. `./tgen.sh load auto`(또는 `high`), 10분.
3. 스캔:
   ```bash
   sudo -n ionice -c3 nice -n 19 taskset -c 0-3 /usr/bin/grep -a 'PUSCH Aggr' /data/dgist_tgen/log/<run>/phy.log \
     | nice -n 19 taskset -c 65-67 python3 scan_pusch.py > rows.tsv
   python3 inv.py rows.tsv [gnb_unexpected.log]          # 분당: 슬롯, 실패, 뒤바뀜, p50/p99/max
   python3 join.py rows.tsv /data/dgist_tgen/log/<run>/dapp_tenant_slots.csv   # SM 계획 × YOLO 상태
   python3 tl.py rows.tsv runs/<run>/l1_console.log HH:MM:SS HH:MM:SS 5        # 시간축
   ```
4. 합격: 뒤바뀜 0쌍, `PUSCH setup returned error` 0, gNB `Unexpected ULSCH HARQ PID` ≤ Baseline 수준(1시간 44분에 2건),
   폰 PH가 떨어지지 않음(gNB `UE RNTI … PH` 줄), Err.ind/FATAL 0, VA QoS 비교.

## 5. 결과 (2026-11-14 07:30 빌드, 같은 날 검증)

빌드한 것(cuphydriver 7개 파일, GPU 설정·훅 레코드 형식·연구실 OAI는 그대로):

- **F1 순서 게이트**: PUSCH가 있는 UL 슬롯에 슬롯 명령 순서대로 순번을 주고(`cuphydriver_api.cpp`), `task_work_function_ul_aggr_3`이
  앞 슬롯의 콜백이 돌아온 뒤에만 validate/callback을 부른다. 기다림 한도 기본 3000 us(`DAPP_PUSCH_INORDER_US`),
  `DAPP_PUSCH_INORDER=0|1`(dApp PUSCH 프로파일이 있으면 기본 1, 없으면 0 = 종전 동작).
- **D0 일부**: 슬롯·보류·시간 초과·순서 역전·setup 실패 계수기와 콘솔 요약 줄 `PUSCH in-order gate: slots N held N ...`
  (게이트가 꺼져 있어도 센다).
- **F3b 일부**: setup이 실패한 슬롯은 모든 TB를 CRC 실패로 보고한다(`PhyPuschAggr::validate`).
- 빌드하지 않은 것: D0의 나머지 줄, F3a, F3b의 측정값 무효 표시, F2/F5(연구실 OAI 쪽 제안), F4.

[측정] 6셀, Ours, 발생기 auto, YOLO 30 req/s(가상 카메라 2대 × 15 fps, 모델 비율 10:10:60:10:10), 폰 4대는 접속만 한 유휴 상태,
10분씩. 두 실행의 차이는 게이트뿐이다(끈 쪽은 `DAPP_PUSCH_INORDER=0`).

| 10분 구간 | 게이트 ON `ours_tg_auto_inorder_1114_0731` | 게이트 OFF `ours_tg_auto_gateoff_1114_0745` |
|---|---|---|
| UL 슬롯 | 360,600 | 360,600 |
| `PUSCH setup returned error` | 0 | 2,901 |
| 순서 역전: L1 계수기 / `inv.py`(두 슬롯 모두 디코딩된 쌍) | 0 / 0 | 444 / 256 |
| gNB `Unexpected ULSCH HARQ PID` | 0 | 20,660 (YOLO 시작 전 0) |
| 발생기 셀 CRC 오류 | 0 | 14,505 (= 2,901 × 5셀) |
| 게이트 보류 | 823슬롯(0.23 %), 평균 81 us, 최대 326 us, 시간 초과 0 | - |
| PUSCH 슬롯 시간 p50 / p99 / max (us) | 약 1,775 / 2,100 / 2,674 | 약 1,780 / 2,100 / 2,801 |
| Err.ind / FATAL / RU on-time | 0 / 0 / 100 % | 0 / 0 / 100 % |

- 게이트가 없으면 폰이 유휴여도 YOLO가 돌기 시작한 지 1초 안에 1절의 사슬이 시작된다.
- 재지 않은 것: 4절의 조건(폰 4대 스트리밍 10분), 보류 중인 슬롯이 UL 워커 2개 중 하나를 붙잡는 효과, 다른 부하 레벨.
- 열림: sizer가 최대 프로파일을 고른 PUSCH 슬롯 비율이 두 실행에서 다르다(0.53 % 대 0.03 %; 게이트 없는 이전 실행들도
  0.03~0.36 %). 1절 5번의 전력 고착 가설과 2절의 "왜 PUSCH가 느려지는지"도 그대로 열려 있다.
- 한 실행 점검: `verify.sh <run> [from UTC] [to UTC]`, 초당 콘솔 줄 요약 `runsum.py`, sizer 계수기 `infeas.py`.
