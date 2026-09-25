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

## YOLO26 엔진

```bash
# 호스트 venv: 가중치 → ONNX (batch 1, 8)
cd cuPHY-CP/dapp_hook/models
~/.venvs/dapp_yolo/bin/python -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); YOLO('yolo26x.pt')"
for m in yolo26n yolo26x; do for b in 1 8; do ~/.venvs/dapp_yolo/bin/yolo export model=$m.pt format=onnx imgsz=640 batch=$b device=cpu; mv $m.onnx ${m}_b$b.onnx; done; done
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
| `matrix.yaml` | 기본점, 점(축별 TV id), 격자, 간섭 격자, 프로필 |
| `sweep.py` | 매트릭스 → 런 큐, resume, 순서(none → 26n → 26x) |
| `run_one.py` | 조합 하나: 스냅샷, 텔레메트리, 세입자, nsys+벤치, collect |
| `runner.py` | `Runner` 인터페이스, `TestBenchRunner`, `FullStackRunner`(stub) |
| `yolo_worker.py` | 세입자 워커 |
| `collect.py` | 원시 로그 → 표 |
| `plot.py` | 그림·집계 CSV |
| `pick_minmax.py` | 스캔 결과에서 채널별 최소/최대 latency TV를 골라 c8_min/c8_max 점 생성 |
| `tv_index.py`, `tv_pick.py` | 기존 TV 라이브러리 인덱스와 선택기 (TV 생성 전에 쓰던 도구) |

## 저장소에 남기는 것 / 남기지 않는 것

- 커밋: 하네스 코드, `matrix.yaml`, `points/`(점 파일; `pick_minmax.py`의 출력은 `runs/<scan>/minmax_points.yaml`에 생기므로 쓸 것은 `points/`로 복사), `results/<run>/`(그림·집계 CSV 사본), `../models/build_engines.sh`, `testVectors/sweep_tv_manifest.yaml`·`sweep_expanded.yaml`(TV 규격).
- 커밋하지 않음(.gitignore): `runs/`(nsys 트레이스 포함, 수십 GB), `tv_index*.csv`, `testVectors/GPU_test_input/`(TV h5, 5 GB)와 그 FAPI 심링크, 엔진·ONNX·가중치, 전달용 tgz.
