# sweep: cuPHY 채널 실행시간 vs 채널 부하 × YOLO SM cap

Aerial cuBB 25-3 / GH200에서 cuPHY 채널 파이프라인(PUSCH, PDSCH, PDCCH, PUCCH, PRACH, SRS)의
GPU 실행시간이 (a) 채널 파라미터와 (b) 같은 GPU에서 MPS로 도는 YOLO26(n, x)의 SM 상한(cap)에
따라 어떻게 변하는지 스윕하는 하네스. 1단계는 `cubb_gpu_test_bench`(프론트홀·L2 불필요),
2단계는 같은 인터페이스로 cuphycontroller full stack을 붙인다(`runner.py`의 `FullStackRunner`, 미구현).

```
matrix.yaml ──► sweep.py ──► runs/<manifest_id>/manifest.yaml (큐·상태, resume)
                   │
                   └─ run_one.py (조합 하나)  ──► runs/<mid>/<run_id>/
                        ├ runner.py  TestBenchRunner: bench.yaml 생성, 벤치 명령
                        ├ yolo_worker.py  세입자(TensorRT, MPS 클라이언트, cap/priority env, taskset)
                        ├ nsys -t cuda (창 N초)  +  dcgmi dmon 1 s  +  nvidia-smi -l 1
                        └ collect.py  → bench_avg.csv, chan_samples.{csv,parquet}, dcgm_parsed.csv, summary.json
plot.py ──► runs/<mid>/plots/*.png + aggregate.csv + deadline_table.csv
```

## 사용법

호스트(`/home/troy/build/aerial-cuda-accelerated-ran`)에서 실행한다. 벤치·세입자·nsys·collect는
`docker exec`로 컨테이너 `c_aerial_troy` 안에서 root로 돌고(MPS pipe `/var`), DCGM은 호스트에서 돈다.
사전 조건: 컨테이너 안에 MPS 데몬(run2 스크립트가 띄움), 호스트에 `sudo nv-hostengine`,
`models/`에 YOLO26 엔진(아래), `testVectors/GPU_test_input/`에 스윕 TV(`sweep_tv_manifest.yaml`).

```bash
cd cuPHY-CP/dapp_hook/sweep
python3 sweep.py --profile pilot --out runs/pilot --dry-run      # 큐만 출력
python3 sweep.py --profile pilot --out runs/pilot                # 실행 (같은 --out으로 다시 돌리면 이어서 함)
python3 sweep.py --profile sweep --out runs/sweep --skip-tenant yolo26x --only pusch_   # 부분 실행
docker exec -u root c_aerial_troy bash -lc 'cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/sweep && python3 plot.py --runs runs/pilot'
python3 run_one.py --spec runs/pilot/default__none0/spec.yaml --out runs/tmp/x     # 조합 하나만
```

프로필(`matrix.yaml: profiles`)

| 프로필 | 내용 | 런 수 | 런당 | 예상 시간 |
|---|---|---|---|---|
| `pilot` | 기본점 × {none×2, 26n, 26x} × cap {100, 40} + 26x cap40 priority + 세입자 단독 4개 | 11 | 2분 (+워밍업 30 s, nsys 60 s 창) | 0.7 h |
| `interference` | 기본점 × {26n, 26x} × cap {100,80,60,40,20,10} × batch {1,8} × priority {0,1} + none + 단독 | 49 + 48 | 15분 | ~14 h |
| `sweep` | 1단계 점 84개 × {none, 26n cap100/40, 26x cap100/40}, batch 1 | 420 | 2분 | ~19 h |
| `grids` | 특성화 격자 (i) MCS×SNR 196점, (ii) CB 17점, 세입자 없음 | 213 | 1분 | ~5 h |

전체 매트릭스를 15분 런으로 다 돌리면(점 84 × 세입자 조합 49 × 15분) 43일이 걸리므로,
스윕 점은 2분, 간섭 격자는 기본점에서만 15분으로 나눴다. 런 시간은 `profiles.*.duration_s`로 바꾼다.

## 8셀 최소·최대 부하 비교 (채널별 파이프라인 latency)

```bash
python3 sweep.py --profile c8_scan --out runs/c8_scan                       # 8셀, 세입자 없음, 채널별 가벼운/무거운 후보 31점, 1분 런 (~1.3 h)
python3 pick_minmax.py --scan runs/c8_scan --stat mean                      # 채널마다 측정 latency 최소/최대 TV -> points/c8_minmax_points.yaml (c8_min, c8_max)
python3 sweep.py --profile c8_minmax --out runs/c8_minmax --points-file points/c8_minmax_points.yaml   # none x2, 26n/26x x cap {100,40,10}, 단독 기준선, 5분 런 (~2 h)
docker exec -u root c_aerial_troy bash -lc 'cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/sweep && python3 plot.py --runs runs/c8_minmax'
```
```bash
# cap 격자: c8_min / c8_max x {26n, 26x} x cap 10..100 (10 단위) x batch {1, 8}, 2분 런 124개 (~5 h). 엔진이 cap마다 필요하다: models/build_engines.sh 90 70 50 30
python3 sweep.py --profile c8_capgrid --out runs/c8_capgrid --points-file points/c8_minmax_points.yaml
docker exec -u root c_aerial_troy bash -lc 'cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/sweep && python3 plot.py --runs runs/c8_capgrid'
```
`plots/capgrid_latency.csv`는 (부하, 채널, 세입자, batch, cap)마다 mean/p50/p99/p99.99/max, budget 대비 %, 격리 대비 배수, 세입자 fps와 단독 대비 %를 담고,
`plots/capgrid_p99_<tenant>_b<batch>.png` 4장은 채널별 패널에 최소 부하(파랑)와 최대 부하(주황)의 p99를 cap에 대해 그린다(점선 = 같은 부하의 세입자 없음 p99, 회색 실선 = 채널별 deadline = NVIDIA GPU-time budget). `..._zoom.png`는 y축을 데이터에 맞춰 확대한 판으로, deadline은 p99가 75 % 이상 접근한 패널에만 그린다.

점(`points:`)의 채널 값은 스윕 TV id(정수) 대신 NVIDIA가 배포한 라이브러리 TV의 파일 이름(`testVectors/TVnr_..._gNB_CUPHY_....h5`, `tv_index.csv`에 색인)도 받는다.
예: `points/c8_peak_points.yaml`의 `c8_peak`(PUSCH 1 UE 273 PRB 4 layer MCS 27, PDSCH 동일)와 `c8_peak_mu`(PUSCH 2 UE, PDSCH = NVIDIA 20셀 DLMIX peak 6 UE × 4 layer MCS 27), 프로필 `c8_peak_probe`.
주의: 라이브러리 TV는 SNR이 42~46 dB라 20 dB로 만든 스윕 TV와 LDPC 반복 횟수가 다를 수 있다.

### 16셀 부하 사다리 (2026-09-24, x86 ladder TV 68000+/68100+)
```bash
python3 sweep.py --profile c16_scan --out runs/c16_scan --points-file points/c16_points.yaml   # 16셀, 세입자 없음, L1~L5 + peak 변형, 각 2회 (20런, 30분)
```
결과(p99.99 / deadline): PUSCH L1 16 % → L3 40 % → L4 69 % → L5 99 %(+UCI 117 %), PDSCH 14 → 58 %, PDCCH 20 → 53 %, SRS(최소 TV) 57~65 %.
L5에 제어 채널까지 최대로 두면 PUSCH·PRACH·SRS 모두 초과(용량 초과). 세입자 실험의 작업점은 최소 = L1, 최대 = L4(전 채널 59~69 %), L5는 용량 한계 기준점.
비64C 빌드 한계(셀그룹당 TB·PDSCH UE 128개) 때문에 16셀에서는 셀당 8 UE까지, 68105(16 UE)는 불가. 그림 `runs/c16_scan/plots/ladder_16c.png`.

### 16셀 cap 격자 (2026-09-25, `c16_capgrid`: 최소 = c16_L1, 최대 = c16_L4)
```bash
python3 sweep.py --profile c16_capgrid --out runs/c16_capgrid --points-file points/c16_points.yaml   # 124런, ~5 h
docker exec -u root c_aerial_troy bash -lc 'cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/sweep && python3 plot.py --runs runs/c16_capgrid'   # 최소/최대 점은 자동 탐지 (--lo/--hi로 지정 가능)
```
p99.99 기준 안전 cap: 26n b1/b8 두 부하 모두 100(최대 부하에서 PDSCH 91 %까지 접근), 26x b1 100(단 cap 10은 초과: 저 cap 꼬리 이상), 26x b8 최소 부하 90 / 최대 부하 20(cap 30 이상은 PDSCH·PUSCH 초과, cap 10도 초과).
**저 cap 꼬리 이상**: cap을 10까지 줄이면 세입자 커널 하나가 10배 길어져(40 → 300 µs 이상) cuPHY 파티션 안의 SM을 오래 점유하므로, p99.99가 cap 20~100보다 나빠진다. 세입자마다 cap 하한이 필요하고 estimator의 kernel_max_us는 cap별 표여야 한다. `tradeoff_summary.csv`의 `unsafe_low_caps_*` 열이 이를 표시한다.

`plots/tradeoff_p99_99.png`(과 `tradeoff_p99.png`)은 같은 데이터를 트레이드오프로 다시 그린 것이다: 패널 = (모델, batch), 곡선 = 부하, 점 = cap 하나,
x = cuPHY 옆에서의 YOLO 처리량(img/s), y = budget에 가장 가까운 채널의 latency/budget(%). 원 표시는 budget을 한 번도 넘지 않고 도달한 최대 cap이고,
두 부하의 안전 cap이 다르면 부하를 모르는 정적 cap(둘 중 작은 값)이 포기하는 처리량을 화살표로 적는다. 숫자는 `tradeoff.csv`, `tradeoff_summary.csv`(p99.99 기준).

`plots/minmax_latency_isolation.png`이 "부하 최소 vs 최대에서 채널별 파이프라인 latency"(막대 = 평균, 수염 = p99.99, 점선 = NVIDIA budget),
`minmax_latency_tenants.png`이 세입자·cap별 p99 증가율, `minmax_latency.csv`가 원본 숫자다.

### 16셀 cap 격자, YOLO26s (2026-10-03, `c16_capgrid_26s`, `results/c16_capgrid_26s/`)
26n·26x와 같은 격자(최저 = c16_L1, 최고 = c16_L4, cap 10~100, batch 1·8, 2분 런, 같은 벤치 슬롯 배치)를 YOLO26s로 64런(none ×2 재측정, 단독 20). 엔진은 `models/build_engines.sh`에 `MODELS=yolo26s`로 20개(cap마다 4.5~5.5분). 파일럿 7런 뒤 본 런. 세입자 없는 기준선은 9/25 값과 채널별 p99 1~3 µs 안에서 같다.
결과(최악 채널 p99 / deadline, 85 % 규칙의 임시 SafeCap): b1은 두 부하 모두 cap 100까지 안전(최고 부하 PUSCH 73~75 %, 26n과 같고 26x 76~80 %보다 낮음). b8 최저 부하는 SRS 59~74 %로 cap 100(90에서 64 %로 내려가는 비단조 점). **b8 최고 부하는 cap 30(82 %)까지 통과, 40·60 86 %, 100 89 %(PUSCH)** — 26n b8(100, 81 %)과 26x b8(20, 100에서 166 %) 사이. 50이 84 %로 다시 통과하는 비단조 점은 재측정 대상. YOLO26s latency(p99, 슬롯): b1 단독 2.2 → 동거 5.0(최저)/6.1(최고) @cap 100, cap 10에서 7.0/7.7; b8 단독 4.5 → 7.5/9.1 @100, cap 10에서 26/28. 처리량(동거, 부하 중 창만): b1 572/482 img/s @100 (단독 895), b8 2381/1948 @100 (단독 3530).
```bash
python3 sweep.py --profile c16_capgrid_26s --out runs/c16_capgrid_26s --points-file points/c16_points.yaml
python3 tenant_latency.py --runs runs/c16_capgrid_26s --out results/c16_capgrid_26s --slot-ms 0.5; python3 capgrid_overview.py --dir results/c16_capgrid_26s
```

### 16셀 MIG 분할 (2026-09-28, `c16_mig`: cuPHY = 4g.48gb 64 SM, 세입자 = 3g.48gb 60 SM)
MIG 전환·복구는 GPU 위에 프로세스가 없어야 하고 서버의 다른 사용자에게 영향을 주므로 사람이 직접 한다(컨테이너 안 root).
```bash
echo quit | nvidia-cuda-mps-control                 # 기존 MPS 데몬(CUDA_MPS_PIPE_DIRECTORY=/var) 종료
nvidia-smi -i 0 -mig 1 && nvidia-smi mig -cgi 4g.48gb,3g.48gb -C && nvidia-smi -L     # UUID를 matrix.yaml `mig:`에 적는다(재생성하면 바뀜)
# 4g 인스턴스 전용 MPS 데몬: CUDA_VISIBLE_DEVICES=<4g UUID> CUDA_MPS_PIPE_DIRECTORY=/var/mig_ran CUDA_MPS_LOG_DIRECTORY=/var/mig_ran nvidia-cuda-mps-control -d
bash ../models/build_engines_mig.sh <3g UUID>      # 세입자 엔진 4개(모델 2 x 배치 2), 약 22분
python3 sweep.py --profile c16_mig --out runs/c16_mig --points-file points/c16_points.yaml          # 16런, 약 40분
python3 sweep.py --profile c16_mig_ladder --out runs/c16_mig_ladder --points-file points/c16_points.yaml   # L2, L3 (세입자 없음)
python3 sweep.py --profile c16_mig_split --out runs/c16_mig_split --points-file points/c16_points.yaml     # 64 SM의 다른 배분
python3 mig_report.py --runs runs/c16_mig --out results/c16_mig --ref results/c16_capgrid/capgrid_latency.csv
```
- nsys 2025.3.1은 MIG가 켜지면 "Unexpected TH500 socket configuration"으로 시작하지 못한다(MIG 장치를 GPU 개수로 세어 Grace 플랫폼 판별 실패). `NSYS_CONFIG_DIRECTIVES=LinuxPerfTH500Config=1`로 해결하며 프로파일의 `mig.nsys_directives`가 넣어 준다.
- 기준을 넘어도 p99는 측정된다: 벤치에는 deadline 초과 시 중단·폐기가 없다(그건 실제 L1 cuphydriver의 동작). `mig_validation.csv`가 런마다 종료 코드, 컨텍스트→채널 매핑, 표본 완전성(채널 간 비율과 창 길이 대비)을 확인한다. 18런 모두 통과, 완전성 0.9989 이상.
- 64 SM 안에서 실제로 부여된 SM(요청 2,5,2,49,48,8): PRACH 8, PDCCH 8, PUCCH 8, PDSCH 50, PUSCH 48, SRS 8.

결과(p99 / deadline, 세입자 없음; 세입자가 있어도 1 % 이내로 같다):

| 부하 | PUSCH | PUCCH | PRACH | SRS | PDSCH | PDCCH |
|---|---|---|---|---|---|---|
| MIG 4g, L1(최저) | 332 us (22 %) | 113 (8 %) | 202 (16 %) | 532 (106 %) | 54 (14 %) | 74 (20 %) |
| MIG 4g, L4(최고) | 2319 (155 %) | 113 (8 %) | 249 (20 %) | 572 (114 %) | 370 (99 %) | 136 (36 %) |
| 공유 GPU 132 SM, L1 | 237 (16 %) | 115 (8 %) | 245 (20 %) | 286 (57 %) | 51 (14 %) | 71 (19 %) |
| 공유 GPU 132 SM, L4 | 1013 (68 %) | 131 (9 %) | 306 (24 %) | 285 (57 %) | 215 (57 %) | 147 (39 %) |

- 격리: 세입자(26n/26x, 배치 1/8)를 3g에서 돌려도 cuPHY p99는 변하지 않고(x0.999~1.011), 세입자도 cuPHY의 영향을 받지 않는다(26x b8 p50 12.86 ms, 단독 12.85 ms). 공유 GPU에서는 같은 조합이 L4에서 PUSCH 126 %, PRACH 145 %, PDSCH 166 %까지 올라갔다.
- 용량: 64 SM으로는 L4를 감당하지 못한다. 부하 사다리(`mig_ladder.csv`)에서 배분 A(PUSCH 48, SRS 8)는 PUSCH가 L3까지 83 %로 들어오지만 SRS가 모든 단계에서 106~114 %, 배분 B(PUSCH 40, SRS 16, PDCCH 10)는 SRS 55~62 %로 들어오지만 PUSCH가 L3에서 112 %. 두 배분 모두 통과하는 최대 단계는 배분 B의 L2(셀당 2 UE)이고, 공유 GPU는 L4(셀당 6 UE)까지 통과한다.
- 세입자 성능(3g, 60 SM, p50): 26n b1 1.08 ms, 26n b8 3.39 ms, 26x b1 3.00 ms, 26x b8 12.86 ms. 부하와 무관하게 일정하다.
- 한계: 벤치 슬롯 배치(UL 0~1 ms, DL 1~4 ms)가 NVIDIA 배치와 달라 UL·DL 시간 겹침이 적다는 점은 이 측정에도 그대로다. MPS의 SM 수는 상한이지 분할이 아니어서 64 SM 안의 컨텍스트들은 SM을 공유한다.

### 세입자 latency와 한 장 요약 (2026-09-27)
```bash
python3 tenant_latency.py --runs runs/c16_capgrid --out results/c16_capgrid      # YOLO p50/p99 vs cap, 최저·최고 부하, 점선 = YOLO 단독
python3 capgrid_overview.py --dir results/c16_capgrid [--stat p99_99_us]         # cuPHY 채널 latency, 4개 구성 통합, 슬롯 단위
```
둘 다 기존 기록만 읽는다(GPU 불필요). 세입자 통계는 추론별 표본이 없으므로 1초 창 p50/p99의 중앙값이다.
주의: `collect.py`의 세입자 요약(`tenant_fps`, tradeoff의 fps 축)은 벤치 시작 전·종료 후의 창까지 평균해 부하 중 fps를 과대평가한다(c16_capgrid에서 중앙값 +14 %, 범위 +4~+24 %). 부하 중 값은 `tenant_latency.csv`의 `fps`를 쓴다.
결과(16셀, p50): cuPHY와 함께 돌면 YOLO는 1.2~2.0배 느려지고, 늘어나는 시간은 cap 20~100에서 거의 일정하다(26x b8 +5~6 ms, 26x b1 +2.2 ms, 26n b8 +1.5~1.8 ms, 26n b1 +0.6~0.7 ms). cap 10에서만 더 크다.

### 16셀 full stack 격리 런 (2026-10-01, `fullstack/`, 결과 `results/c16_fullstack/`)
RU 에뮬레이터 + cuphycontroller + testMAC(59c, DDDSUUDDDD, F08_CG1 루프백), nsys 없음. 벤치가 아니라 FAPI·훅·타이밍이 실제로 도는 조건이다.
```bash
bash fullstack/fullstack_run.sh <name> <slots> [dl_cores] [nvlog_tag_ids] [async|full|summary|off|legacy|keep]   # 기본 async(코어 43), full = 인라인; RECORD=1 링 기록, RU_HIST=1 RU 히스토그램, NCELLS, KEEP_RAW=0, MPSTAT=1
python3 fullstack/ring_compare.py prof/ring_16C_a.bin prof/ring_16C_<name>.bin --slots 200000   # 두 링의 레코드 수·슬롯 내 순서·내용 비교
python3 fullstack/e_analysis.py e_on e_sum e_off      # tag 311(L2A.PROCESSING_TIMES) → tick 기준 L2A end/enqueue 지연, 드롭, 위치별
python3 fullstack/parse_phylog.py prof/run61/phy_a.log prof/run61/phy_a.csv && python3 fullstack/compare_ab.py prof/run61/phy_a.csv prof/run61/phy_b.csv
python3 fullstack/analyze_run.py a                    # L1 WRN/ERR, 드롭, testMAC Err.ind, RU 늦은 패킷(TDD 위치별), 링 SLOT_END, DCGM
```
통계는 첫 1초를 제외한다. 드롭 기준은 `nv_phy_module.cpp` `check_time_threshold`: `l2a_end − l1_slot_ind_tick > 500 + l2a_allowed_latency(100) µs`.
- ⑥-1 (a) DL 워커 3코어 vs (b) 6코어, 각 10분: CPU 단계 동일, RU 늦은 패킷 비율(위치 6/7) 변화 없음 → DL 워커 CPU는 원인에서 제외 (`analysis_a.txt`, `analysis_b.txt`).
- E 훅 비용 (`e_hook_cost.txt`, 각 5분 = 600k 슬롯): 훅 on(PDU 포함) − off = L2A end − tick p50 +20.2 / p99 +25.5 / p99.9 +28.8 / max +34.7 µs; UL 슬롯(위치 4·5) +38~45, DL 슬롯 +15~30. 훅 on은 600 µs 선까지 여유 5 µs(드롭 1), off는 40 µs(드롭 0). 레코드당 ~100 ns × 슬롯당 133~457 레코드(UL 슬롯 UL_PDU 384~440)라 비용은 PDU 수에 비례한다. → PDU 내보내기를 L2A 스레드 밖(디스크립터 큐 + 복사 스레드)으로 옮기는 안을 제안, 구현 전까지 full-stack 신뢰성 런은 `export_pdus: 0`.
- 주의: `e_sum`(요약 모드)과 `d` 런은 스크립트가 직전 런의 `enable: 0`을 되돌리지 않아 훅 off로 돌았다(이후 모드마다 훅 라인 전체를 다시 쓰도록 수정). `e_sum`은 훅-off 반복 런으로만 쓴다(런 간 차이 ≤ 1.5 µs).
- MIG 4g full stack (`mig_fullstack.txt`, 2026-10-04; `fullstack_run.sh`의 `MIG_UUID=<4g uuid> MIG_SM=1`, `mig_runs.sh`): 59c 16셀(UL 6 UE × 2레이어 = 벤치 사다리 L2~L3급, 벤치 최고 부하 L4가 아님)을 64 SM에서 돌리면 멈추지 않고 느려진다 — 풀 고갈·recovery·종료 0, CRC 0, testMAC ERR 0. UL 파이프라인만 늦어져 U4 PUSCH 완료 1.54 → 2.14 ms, U5 1.92 → 3.19 ms(슬롯 시작 기준 p50). U5 인디케이션이 testMAC 허용창(8슬롯 = t0 + 2.5 ms)을 1슬롯 넘겨 `ul_slot=9 > 8`로 전부 INV 처리(UL 181.95 vs 196.70 Mbps). 8·12셀은 창 안. E는 공유 GPU와 같고 D는 0.03 %. 벤치 L4(4레이어 UL)에서 예측한 L1_RECOVERY는 59c(2레이어 UL)에서는 일어나지 않는다.
- 비동기 내보내기 검증 (`async_export.txt`, `async_ring_compare.txt`, `async_report.txt`; 2026-10-03, `async_runs.sh`·`async_runs2.sh`, 훅 모드 `async`): l2 adapter yaml `dapp_hook`에 `async_export: 1, export_cpu: 43`을 주면 L2A 스레드는 TTI 요청당 디스크립터 하나만 큐에 넣고 `dapp_export` 스레드(코어 43)가 같은 레코드를 같은 순서로 쓴다. 16셀 5분 런(off×3, async×3 + 10분 soak, inline×1): L2A end − tick p99.9 off 554.8 / async 557.3 / inline 584.3 µs → async − off = +2.5 µs(인라인 +29.5), 드롭 async 0(300만 슬롯) vs inline 122(전부 UL 위치 4·5). `ring_compare.py`로 인라인 링과 200,000슬롯(4,088만 레코드) 비교: 레코드 수·순서·내용 동일. 추정기 replay 결정 동일. RU DL 늦은 패킷: off 0.026~0.074 %, async 0.063~0.091 %(범위 겹침), inline 0.426 % — 인라인 훅의 20~40 µs 지연이 D도 키운다. 이 검증 뒤 기본값을 비동기로 전환했다(코드 기본 `async_export: 1`, 단 `export_cpu`가 없으면 경고 후 인라인; 저장소의 `l2_adapter_config_F08_CG1.yaml`에 `async_export: 1, export_cpu: 43`). `fullstack_run.sh`의 기본 훅 모드도 `async`이고 인라인은 `full`로 명시한다.
- CPU 원인 판별 (`cpu_cause.txt`, `cpu_report.txt`; `cpu_runs.sh` → `cpu_report.py`, 2026-10-02): 코어 맵·IRQ·PTP 겹침, 16셀 mpstat, 8/12/16셀 스케일링, RU 코어 2배. E는 L2A 단일 스레드(코어 12)의 셀당 ~6 µs 직렬 처리(testMAC은 16셀 메시지를 17 µs 안에 보냄, L2A는 97 µs에 걸쳐 처리; 링 (a)의 송신·처리 시각 차이로 확인)라 CPU 원인이 맞고 셀 수에 선형. D는 포화 코어 없음·RU HW 타임스탬프라 CPU 아님; RU 에뮬레이터 스레드를 2배로 하면 D가 8배 악화(0.047 → 0.374 %)되어 D의 절대값은 에뮬레이터 송신 패턴에 좌우되는 테스트베드 산물. `setup1_DU.sh`가 tag 216/217/224/225/238/243/311/312/409를 INFO로 켜므로 이 랩의 "기본 로깅"은 이미 311을 포함한다.
- D 확정 (`d_ru_late.txt`, `d_match.txt`, 10분, RU.LATE_PACKETS INFO + 히스토그램): `ru_late.py`로 RU 슬롯·셀별 도착 창과 늦은 패킷의 심볼 분포를 뽑고 `d_match.py`로 L1 슬롯과 t0(500 µs 격자, SFN mod 256·슬롯 확인)로 1:1 대응. 늦은 슬롯 294개(위치 6: 0.073 %, 7: 0.059 %, 9: 0.042 %)에서 L1 CPU 단계(압축 완료, U-plane 준비·tx 종료 ≤ t0 −587 µs)는 정상 슬롯과 같고, 늦은 패킷은 심볼 0~1 집중(20 %)이 아니라 위치별 패턴(7 = 머리 0~3, 6/8/9 = 꼬리 12~13)으로 퍼진다. 송신은 GPU 커널(`gpu_init_comms_dl: 1`, `mps_sm_gpu_comms: 16`)이 하며, 늦은 슬롯의 60~80 %는 같은 패턴의 UL GPU 작업(order/PUSCH)이 p99를 넘긴 경우와 겹친다(정상 슬롯 3.8 %). → D = GPU 발신 커널의 지연·늘어짐(다른 cuPHY GPU 작업과의 경합), CPU·선로 아님.

### E2E 시나리오 준비 — 1280 입력, open-loop 요청률, nvJPEG (2026-10-04~05, `results/e2e_prep/`, `results/c16_capgrid_1280/`, `results/c16_openloop_1280/`)
UE가 4K JPEG을 보내는 E2E 시나리오에서 정적 MIG 대비 ≥10 % 지연 감소가 나오는 조건을 찾기 위한 UE 없는 측정. 4K는 640으로 6배 축소하면 의미가 없으므로 **1280 입력**(`models/export_1280.py`, 연산량 4배 = SM-bound)으로 26n/26s/26x를 다시 export하고 엔진을 cap 100/50/30에 빌드한다. 세입자 이름은 `yolo26n1280` 등(`matrix.yaml tenants`의 `model`·`input` 키로 엔진 stem과 입력 npy 지정).
```bash
~/.venvs/dapp_yolo/bin/python ../models/export_1280.py                                   # *_1280_b{1,8}.onnx + bus_1280.npy
docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var -e MODELS="yolo26n_1280 yolo26s_1280 yolo26x_1280" c_aerial_troy bash /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/build_engines.sh 100 50 30
python3 sweep.py --profile c16_capgrid_1280 --out runs/c16_capgrid_1280 --points-file runs/c16_points.yaml       # 56런(none 2, 세입자 36, 단독 18), ~3 h
python3 sweep.py --profile c16_openloop_1280 --out runs/c16_openloop_1280 --points-file runs/c16_points.yaml     # open-loop 12런, ~1.6 h
docker exec -u root c_aerial_troy bash -lc 'cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/sweep && python3 tenant_latency.py --runs runs/c16_capgrid_1280 --out results/c16_capgrid_1280 --slot-ms 0.5 && python3 openloop_report.py --runs runs/c16_openloop_1280 --out results/c16_openloop_1280'
bash ../e2e/gh200_segment.sh                                                             # 실제 파이프라인(TCP → NVJPG → TensorRT → 응답) 루프백 측정
# MIG 3g 쪽(사람이 MIG를 켠 뒤): MODELS="yolo26n_1280 yolo26s_1280 yolo26x_1280" bash ../models/build_engines_mig.sh <3g UUID>; 프로필 c16_mig_1280, c16_openloop_mig_1280
```
- `openloop_worker.py`: 도착 시각을 미리 고정한 **open-loop** 스트림(FIFO, batch 1). 느린 서버는 요청이 줄지 않고 큐가 쌓이므로 정적 MIG와 공유 GPU에 같은 스케줄을 넣어 공정하게 비교할 수 있다. 요청률별 served/backlog/latency/queue/busy를 `worker_summary.csv`에 남기고 `openloop_report.py`가 rate → latency 곡선과 용량표를 만든다. `run_one.py`는 `tenant.worker`/`tenant.args`로 이 워커를 띄운다.
- nvJPEG(`results/e2e_prep/nvjpeg.txt`, `tools/nvjpeg_probe.cu`): GH200의 NVJPG 하드웨어 디코더 7기가 MPS 클라이언트에서 그대로 보이고, 4K q85 4:2:0 한 장 디코드 + 1280 letterbox = **5.1 ms**(p99 = p50, cap 10에서도 5.4 ms → SM 미사용). progressive JPEG은 미지원(CPU 22 ms), 4:4:4는 8.6 ms. MPS는 한 번에 한 사용자만 서비스하므로 세입자 프로세스는 cuPHY와 같은 사용자(root)로 띄운다.
- 실제 파이프라인 도구는 `../e2e/`(`infer_server.py`, `e2e_client.py`, `libnvjpeg_dec.so`): 응답에 서버 측 시각을 실어 UE RTT를 공중/전송 구간과 GH200 구간으로 분해한다. batch 8 엔진은 **타일 추론**(4K → 1280 원본 해상도 타일 6장 + 전체 뷰, NMS로 병합)으로 쓴다.
- 요청률 제한 세입자(`c16_ratelimit_1280`, `c16_ratelimit_1280_b8`: Poisson 10/20 req/s, nsys 창 켬)로 실제 duty cycle에서의 cuPHY 채널 p99를 본다. `collect.py`는 open-loop 워커의 요청별 `worker.csv`도 요약한다.
- full-stack 부하 사다리: `fullstack/fullstack_run.sh`에 `PATTERN`(기본 59c) 추가, `fullstack/ladder_runs.sh <prefix> "59c 59e 65c 65a"` → `fullstack/ladder_report.py <names>` (슬롯 위치별 PUSCH GPU 완료, 과부하 문자열, L1/testMAC 카운터). MIG 안에서는 `MIG_SM=1 MIG_UUID=...`를 그대로 넘긴다.
- 실행 순서를 담은 드라이버는 `e2e_prep/`(`e2e_prep_resume{,2,3}.sh` = MIG off, `e2e_prep_mig.sh` = MIG 전환 후; `runs/`에 복사해 실행한 것의 사본).

**MIG off 결과 요약(`results/e2e_prep/summary_mig_off.txt`, 2026-10-05; MIG 쪽은 미측정)**: cuPHY를 지키는 cap(85 % 규칙)에서 1280 세입자의 지연은 26x b1 6.4(최저)/13.9 ms(최고, cap 30), 타일 26x 68 ms(cap 30)/안전 cap 없음으로, 격리된 3g 추정치(~7 / ~42 ms)보다 빠르지 않다. cap 100은 타일 26x에서 PDSCH p99를 deadline의 410~471 %로 만들고, 10 req/s로 요청률을 낮춰도 297~388 %다. 실제 파이프라인 서버 지연(p50)은 26x b1 13.5~18.6 ms(cap 100), 타일 26x 36~49 ms이며 디코드는 4.7 ms로 일정하다. 전체 GPU의 full-stack 사다리(59c/59e/65c/65a)는 모두 오류 없이 U5 PUSCH 완료 1.69~2.07 ms. → E2E 이득은 RAN 쪽(MIG 4g의 UL 지연)과 모델 선택에서 찾아야 하며, 3g 실측과 4g 사다리가 남아 있다. 10-05 00:46의 호스트 비정상 재부팅(26x-1280 cap 100 + L4 첫 런 중)은 같은 조합 재실행에서 재현되지 않았다.

## YOLO26 엔진

```bash
# 호스트 venv: 가중치 → ONNX (batch 1, 8)
cd cuPHY-CP/dapp_hook/models
~/.venvs/dapp_yolo/bin/python -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); YOLO('yolo26x.pt')"
for m in yolo26n yolo26x; do for b in 1 8; do ~/.venvs/dapp_yolo/bin/yolo export model=$m.pt format=onnx imgsz=640 batch=$b device=cpu; mv $m.onnx ${m}_b$b.onnx; done; done
~/.venvs/dapp_yolo/bin/python export_1280.py          # 1280 입력판: yolo26{n,s,x}_1280_b{1,8}.onnx, bus_1280.npy (4K 입력용, FLOPs 4배)
# 컨테이너: FP16 엔진. 반드시 작은 cap 아래서 빌드 (전체 GPU에서 빌드한 엔진은 48 SM 이하 파티션에서 실행 실패)
docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10 c_aerial_troy \
  /usr/src/tensorrt/bin/trtexec --onnx=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/yolo26n_b1.onnx --fp16 \
  --saveEngine=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/yolo26n_b1_fp16_atp10.engine --skipInference
```

`bus_640.npy`는 `bus.jpg`를 letterbox한 1×3×640×640 float32 입력이다. 워커는 매 추론마다 H2D → 네트워크 → D2H를
동기로 돌려(사진 항상 대기) 1초마다 fps와 latency p50/p99/max를 `worker.csv`에 남긴다.

## label 소스와 한계

| 소스 | 무엇 | 한계 |
|---|---|---|
| `bench.out` (stdout) | 패턴당 채널·슬롯별 평균 실행시간. `Average PUSCH run time: END from START`의 END−START | `-r N` 전체 평균 하나뿐이라 분포가 없다. 런치 측 대기(START 오프셋 포함)라 nsys보다 10~20 µs 길다 |
| `nsys -t cuda` 커널 트레이스 | 채널별 컨텍스트의 슬롯 실행 하나 = 표본 하나. 실행 경계는 패턴 시작(메인 컨텍스트 delay 커널의 끝)과 스트림 소유(PUSCH1/PUSCH2처럼 실행마다 자기 스트림 세트, PDSCH·PDCCH처럼 한 스트림을 순서대로 공유)로 나눈다. 슬롯이 500 µs를 넘겨 다음 슬롯과 겹쳐도 분리된다 (`summary.json: nsys.segmentation`이 `pattern`이어야 함, `gap`이면 예전 간격 방식으로 떨어진 것). `chan_samples.csv`에 실행마다 `dur_us` → p99, p99.99, max | 런 안의 `nsys_window_s` 창(기본 60 s ≈ 13k 패턴)만. 처음 10 s는 워밍업으로 버림(`--drop-first-s`). 컨텍스트→채널은 커널 이름으로 매핑한다(이름이 기대 채널 집합을 1:1로 덮을 때; `nsys.mapping_source` = `names`). 벤치가 채널 컨텍스트를 여러 스레드에서 만들어 생성 순서가 `--M` 순서와 어긋날 수 있기 때문이다(2026-09-21 6채널 런 16개 중 4개에서 PDSCH/PUSCH 또는 PUSCH/SRS가 뒤바뀜, `nsys.order_mismatch`에 기록). 이름으로 못 가리면 생성 순서로 떨어진다(`order`). `nsys.mapping_mismatch`는 최종 매핑과 커널 이름이 다른 컨텍스트로, 항상 []여야 한다 |
| DCGM 1 s (`dcgm_parsed.csv`) | SMACT/SMOCC/DRAMA(0~1), GPUTL %, POWER W | 벤치가 슬롯 타이밍용 delay 커널로 GPU를 계속 점유해 GPUTL은 무의미하고 SMACT도 이를 포함한다. 세입자 없음 런과의 차이로만 해석 |
| full stack (2단계) | cuphydriver `[PHYDRV] ... times ===> [GPU Run] DURATION` NVLOGI 라인 (nvlog PHYDRV INFO) | 코드 수정 없이 얻을 수 있는 유일한 채널별 GPU 시간. 아직 `FullStackRunner` 미구현 |

벤치의 시간 의미: 패턴(8슬롯) 안에서 슬롯 k는 k×500 µs에 시작하도록 스케줄되고, 채널 시간은 그 슬롯의
첫 커널 시작부터 마지막 커널 끝까지다. 패턴 사이에는 `-w`(1 ms) 간격이 있어 실제 TDD 10슬롯 주기(5 ms)와 비슷하다.
현재 설정에서 패턴 주기는 약 4.4 ms(`bench.pattern_ms`, `-r` 계산용).

`D_c`(마감)는 NVIDIA 셀 용량 판정 코드에 하드코딩된 채널별 latency constraint(`testBenches/perf/measure/TDD/check_cell_capacity.py:60-101`: PUSCH·PUCCH 1500, SRS 500, PDSCH·PDCCH·SSB 375, PRACH는 Parallel 모드 1250 / Sequential(-B) 1500 µs)를 기본으로 쓴다. 이 하네스는 `-B` 없이 돌리므로 PRACH는 1250이다. 판정 기준은 '모든 채널의 표본 100 %가 constraint 이하'(같은 파일 :186-191)이고, 재는 양은 UL·DL 모두 슬롯 실행의 END−START(`parser/standard.py`, `parser/common.py`)라 `dur_us`와 같다. `plot.py --deadline-source rules`면 estimator의 provisional 값, `--deadline PUSCH=2000`으로 개별 덮어쓰기. 1차 통계 `dur_us`는 한 슬롯 실행의 첫 커널 시작부터 마지막 커널 끝까지이고(벤치의 END−START, budget과 같은 양), `lat_us`는 예정된 슬롯 시작부터 끝까지다(PRACH·SRS는 벤치가 약 500 µs 늦게 스케줄하므로 그만큼 크다). max는 단발 stall 하나에 좌우되니 p99.99를 본다.
`deadline_table.csv`에 채널·런별 초과 비율이 나온다.

## 산출물 (`runs/<manifest_id>/<run_id>/`)

`spec.yaml`(조합), `snapshot.yaml`(GPU 클럭·전력·MPS 기본값·TV manifest 항목·벤치 명령), `bench.yaml`, `bench.out`,
`nsys_trace.nsys-rep/.sqlite`, `worker.csv/.out`, `dcgm.csv`, `smi.csv`, `bench_avg.csv`, `chan_samples.csv/.parquet`,
`dcgm_parsed.csv`, `summary.json`, `status.yaml`(ok / failed+사유 / skipped+사유), `run_one.log`.
매니페스트 `runs/<mid>/manifest.yaml`은 런마다 상태를 갱신하며, 실패·타임아웃은 다음 조합으로 넘어간다.

## 건너뛴 조합

- SRS 8UE × 4port (`65015~65017`): comb 2 cyclic shift 한계로 TV 생성 불가 → 매니페스트 `skipped`.
- CB 격자 3CB/4UE (`67001`): TV 생성 불가.
- PUCCH F0/F1 11·22 bit: 포맷상 2 bit 초과 불가라 격자에서 제외(F2/F3만).
- PRACH format 0 TV는 2슬롯짜리(`_s1` 파일)라 full stack에서는 testMAC 슬롯 검증 확인 필요.
- 재전송 TV(`60052`)의 UE0는 HARQ 버퍼 없이 단독 디코드라 CRC 실패가 정상.
- priority 축: 2026-09-17 프로브에서 `CUDA_MPS_CLIENT_PRIORITY`의 효과가 측정되지 않았다. 파일럿에 1런만 포함.

## 파일

| 파일 | 역할 |
|---|---|
| `matrix.yaml` | 기본점, 점(축별 TV id), 격자, 간섭 격자, 프로필, 세입자(26n/26s/26x) |
| `sweep.py` | 매트릭스 → 런 큐, resume, 순서(none → 26n → 26x) |
| `run_one.py` | 조합 하나: 스냅샷, 텔레메트리, 세입자, nsys+벤치, collect |
| `runner.py` | `Runner` 인터페이스, `TestBenchRunner`, `FullStackRunner`(stub) |
| `yolo_worker.py` | 세입자 워커 |
| `collect.py` | 원시 로그 → 표 |
| `plot.py` | 그림·집계 CSV |
| `pick_minmax.py` | 스캔 결과에서 채널별 최소/최대 latency TV를 골라 c8_min/c8_max 점 생성 |
| `tenant_latency.py` | 세입자(YOLO) latency vs SM cap. `worker.csv`의 1초 창 중 벤치가 실제로 돌던 구간만 집계 → `tenant_latency.csv`, 모델·배치별 그림, 요약 그림(ms, 슬롯) |
| `mig_report.py` | MIG 스윕 검증(`mig_validation.csv`)과 집계(`mig_latency.csv`), 공유 GPU 대비 그림(`mig_p99_channels.png`) |
| `capgrid_overview.py` | `capgrid_latency.csv`를 읽어 세입자 구성 전체를 한 장에(행 = 구성, 열 = 채널), y축 슬롯 단위 |
| `design_points.py` | ③ 정적 설계점: 사다리(A) + 단일축(B) + 2^(6−2) 분할요인(C) → `points/design_points.csv`(feature 열 = 훅 필드 이름), `design_points.yaml`(sweep 점), `design_schedule.csv`(실행 시간표) |
| `fullstack/` | 16셀 full-stack 격리 런 스크립트와 분석(위 절) |
| `tv_index.py`, `tv_pick.py` | 기존 TV 라이브러리 인덱스와 선택기 (TV 생성 전에 쓰던 도구) |

## 저장소에 남기는 것 / 남기지 않는 것

- 커밋: 하네스 코드, `matrix.yaml`, `points/`(점 파일; `pick_minmax.py`의 출력은 `runs/<scan>/minmax_points.yaml`에 생기므로 쓸 것은 `points/`로 복사), `results/<run>/`(그림·집계 CSV 사본), `../models/build_engines.sh`, `testVectors/sweep_tv_manifest.yaml`·`sweep_expanded.yaml`(TV 규격).
- 커밋하지 않음(.gitignore): `runs/`(nsys 트레이스 포함, 수십 GB), `tv_index*.csv`, `testVectors/GPU_test_input/`(TV h5, 5 GB)와 그 FAPI 심링크, 엔진·ONNX·가중치, 전달용 tgz.
