# 실행 매뉴얼 (코드 기준, 2026-10-02 확인)

모든 명령은 호스트의 저장소 루트(`/home/troy/build/aerial-cuda-accelerated-ran`)에서 실행한다.
이 디렉터리는 컨테이너 `c_aerial_troy` 안에서 `/opt/nvidia/cuBB`로 보인다.
Aerial 바이너리와 TensorRT는 컨테이너 안에만 있으므로 그쪽 명령은 `docker exec`로 실행한다.

각 명령 옆의 표시: **[확인]** = 2026-10-02에 실제로 실행해 본 것, **[스크립트]** = 10/01 런에 쓰인
`sweep/fullstack/fullstack_run.sh`에 들어 있는 명령(오늘 다시 돌리지는 않음), **[미실행]** = 코드만 읽은 것.

## Setup (이 서버의 실제 값)

| 항목 | 값 |
|---|---|
| 서버 | GH200 (GPU 132 SM), 호스트 Ubuntu 22.04.5, 커널 6.8.0-1025-nvidia-64k |
| 부팅 파라미터 | `isolcpus=managed_irq,domain,4-64 nohz_full=4-64 rcu_nocbs=4-64`, hugepage 512M × 48 |
| GPU 드라이버 | 575.57.08 |
| 컨테이너 | `nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb` (Ubuntu 22.04.5, CUDA 12.9, TensorRT 10.3.0, Python 3.10.12, cmake 3.26.6 + ninja) |
| MPS | 컨테이너 안, root, 파이프 `/var` (`run2_cuPHYcontroller.sh`가 띄움) |
| 호스트 도구 | Python 3.10.12 + PyYAML, `dcgmi`(`sudo nv-hostengine` 필요), `~/.venvs/dapp_yolo`(ultralytics: ONNX 내보내기, replay 그림) |

코어 배정(`testBenches/phase4_test_scripts/physical_cores_du.yaml`, `physical_cores_ru.yaml`):
L1 4–14, testMAC 15–18, RU 에뮬레이터 19–42. **43–64가 비어 있다.** 스케줄러·세입자는 이 범위에 둔다
(`isolcpus=domain`이라 `taskset -c 44-47`처럼 범위를 줘도 첫 코어에만 머문다. 코어 하나를 지정할 것).

## 1) 훅 모듈 빌드

CMake 옵션 이름은 `ENABLE_DAPP_HOOK`이고 기본값이 ON이다(`CMakeLists.txt:151`). 빌드 트리는 `build.aarch64`.

```bash
# 컨테이너가 없으면 먼저 띄운다 (c_aerial_$USER 생성, 저장소를 /opt/nvidia/cuBB에 마운트)
./cuPHY-CP/container/run_aerial.sh

# [미실행: ninja -n 으로 타깃만 확인] 기존 트리 재구성 + 훅·도구 빌드
docker exec c_aerial_troy bash -lc 'cd /opt/nvidia/cuBB && cmake -B build.aarch64 -S . -DENABLE_DAPP_HOOK=ON && \
  ninja -C build.aarch64 -j $(nproc) cuphycontroller_scf dapp_ring_dump dapp_ring_fakel1 dapp_ring_selftest dapp_export_test dapp_sched'

# [확인] 빌드 뒤 자체 검사 (L1·GPU 불필요). "EXPORT TEST PASS (0 failures)", "SELFTEST PASS"
docker exec c_aerial_troy /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/dapp_hook/dapp_export_test
docker exec c_aerial_troy /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/dapp_hook/dapp_ring_selftest --records 200000 --ring-len 4096
```

빌드 트리가 아예 없을 때는 `testBenches/phase4_test_scripts/build_aerial_sdk.sh`(기본: `build.aarch64`,
toolchain `grace-cross`, preset `perf`)로 전체를 빌드한다 **[미실행]**.

훅을 컴파일해 넣는 것만으로는 아무 일도 일어나지 않는다. 링은 L2 어댑터 yaml에 `dapp_hook:` 블록이 있을 때만
만들어진다(파싱: `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_module.cpp:425-443`).

## 2) 실험 절차

### 2-1. 코어 격리, MPS 데몬

코어 격리는 부팅 파라미터로 이미 되어 있다. 따로 할 일은 없고 확인만 한다.

```bash
cat /proc/cmdline                                   # [확인] isolcpus=managed_irq,domain,4-64 ...
```

MPS는 손으로 띄우지 않는다. `run2_cuPHYcontroller.sh`가 L1을 시작할 때마다 `/var` 파이프로 **재시작**한다
(`run2_cuPHYcontroller.sh:222-239`). 그래서 세입자는 반드시 "L1 is ready" 이후에, root로,
`CUDA_MPS_PIPE_DIRECTORY=/var`를 주고 시작해야 한다(그 전에 띄운 MPS 클라이언트는 끊긴다).

```bash
# [확인] 데몬 상태 (서버 pid가 찍히면 정상)
docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var c_aerial_troy bash -c 'echo get_server_list | nvidia-cuda-mps-control'

# [미실행: run2가 하는 일과 같은 명령] L1 없이 엔진 빌드·프로필만 할 때 수동 기동
docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var -e CUDA_MPS_LOG_DIRECTORY=/var c_aerial_troy \
  bash -c 'nvidia-cuda-mps-control -d && echo start_server -uid 0 | nvidia-cuda-mps-control'
```

### 2-2. 훅을 켠 L1 실행 (RU 에뮬레이터 + cuphycontroller + testMAC)

훅 스위치는 `cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml:15`의 한 줄이다.

```yaml
dapp_hook: {enable: 1, shm_name: /aerial_dapp_ring, ring_len: 65536, mlock: 1, export_ul: 1, export_dl: 1, export_pdus: 1}
```

스크립트 한 번으로 실행 **[스크립트]** — 설정 생성, 훅 줄 기록, RU → L1 → testMAC 기동, DCGM 1초 기록, 종료,
로그 복사, 랩 설정 원복까지 한다. 산출물은 `prof/run61/`(`FS_OUT`으로 변경).

```bash
# fullstack_run.sh <name> <num_slots> [dl_worker_cores] [nvlog_tag_ids] [full|summary|off]
#   600000 슬롯 = 300 s. 훅 모드: full = UL+DL+PDU, summary = export_pdus 0, off = enable 0
#   환경변수: NCELLS(기본 16), RECORD=1(링을 prof/ring_16C_<name>.bin에 기록), KEEP_RAW=0(8~16 GB 원시 로그 생략)
bash cuPHY-CP/dapp_hook/sweep/fullstack/fullstack_run.sh myrun 600000 "" "311" full
```

같은 일을 손으로 할 때 **[스크립트]** (`fullstack_run.sh:39-41, 87-98`이 실행하는 명령):

```bash
docker exec -u root c_aerial_troy bash -lc "cd /opt/nvidia/cuBB/testBenches/phase4_test_scripts && ./setup1_DU.sh --ru-host-type=_LOOPBACK --du-eth0=aerial02 -y F08_CG1 && ./setup2_RU.sh --ru-eth0=aerial03"
docker exec -u root c_aerial_troy bash -lc "cd /opt/nvidia/cuBB/testBenches/phase4_test_scripts && ./test_config.sh 59c --num-cells=16 --num-slots=600000"
# (여기서 위 dapp_hook 줄을 원하는 값으로 고친다)
docker exec c_aerial_troy bash -lc "/opt/nvidia/cuBB/testBenches/phase4_test_scripts/run1_RU.sh --timeout 630"                  # 터미널 A
docker exec -u root c_aerial_troy bash -lc "/opt/nvidia/cuBB/testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh --timeout 580"   # 터미널 B, "L1 is ready" 대기
docker exec c_aerial_troy bash -lc "/opt/nvidia/cuBB/testBenches/phase4_test_scripts/run3_testMAC.sh --timeout 420"               # 터미널 C
```

- `test_config.sh`는 두 번째부터 "has already been run"으로 멈춘다(`test_config.sh:429`). 스크립트는
  `test_config_summary.sh`에서 이전 값을 지우는 방식으로 넘긴다(`fullstack_run.sh:24-35`). `--force` 옵션도 있으나 이 랩에서 써 보지 않았다.
- 로그는 컨테이너의 `/tmp/phy.log`, `/tmp/ru.log`, `/tmp/testmac.log`.
- testMAC은 시작 후 약 35 s 뒤부터 FAPI를 보낸다.

링 확인 **[확인]** (호스트에서 읽기 전용):

```bash
build.aarch64/cuPHY-CP/dapp_hook/dapp_ring_dump --stats
build.aarch64/cuPHY-CP/dapp_hook/dapp_ring_dump --type UL_TTI,SLOT_END --max 50
```

### 2-3. 세입자 프로필 측정

엔진이 먼저 있어야 한다. `dapp_sched`(SM 개수 컨텍스트 풀)에는 **작은 cap에서 빌드한 batch 1 엔진** 하나를 쓴다
(`models/yolo26n_b1_fp16_atp10.engine`; 엔진 빌드는 `models/build_engines.sh`와 `sweep/README.md`의 "YOLO26 엔진").

```bash
# [확인] 격리 상태(L1 트래픽 없음)에서. 호스트에는 libnvinfer가 없어 컨테이너 안에서 돌려야 한다.
docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var -w /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/estimator c_aerial_troy \
  python3 profile_tenant.py --name yolo26n_b1 --batch 1 --caps 10,20,40,60,80,100 \
  --engine ../models/yolo26n_b1_fp16_atp10.engine --image ../models/bus.jpg \
  --sched /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/dapp_hook/dapp_sched --out profiles/yolo26n_b1.yaml
```

- 오늘 결과: `dur_ms = {10: 1.44, 20: 1.25, 40: 1.14, 60: 1.11, 80: 1.09, 100: 1.07}`.
  `sm_fill`, `kernel_max_us`는 `--nsys`를 줘야 측정되고, 없으면 기본값(0.5, 100 µs)이 들어간다.
- `--batch`는 한 번에 하나, `--caps`는 콤마 구분이다. root로 돌리므로 출력 파일 소유자가 root가 된다.
- **batch 8은 이 경로로 못 잰다.** `dapp_sched`는 입력 `[1,3,H,W]`만 받는다(batch 8 엔진은
  `unexpected input shape for images`로 거부, 오늘 확인). batch 8은 `--pytorch --weights yolo26n.pt`(미실행)이거나
  벤치 스윕의 `sweep/yolo_worker.py`(percentage cap, cap마다 엔진 하나)로만 돈다.

### 2-4. 스케줄러(예측기) 실행

슬롯마다 gate·cap을 정하는 프로세스는 `dapp_sched`가 아니라 **`estimator/live.py`** 다. 링을 읽어
제어 블록 `/dev/shm/aerial_dapp_ctrl`에 `{gate_slots, cap_pct, slot_id}`를 쓴다.

```bash
# [확인: 가짜 L1 + yolov8n_b1 프로필로] 호스트, 빈 격리 코어 하나에 고정
cd cuPHY-CP/dapp_hook/estimator
taskset -c 44 python3 live.py --profile profiles/yolo26n_b1.yaml --rules rules.yaml \
  --log out/decisions_myrun.csv --timing out/timing_myrun.csv
```

- 옵션: `--ring /aerial_dapp_ring`, `--ctrl /aerial_dapp_ctrl`, `--cells cells.yaml`, `--seconds N`, `--spin`, `--quiet`.
- 초안의 코어 40은 루프백 구성에서 RU 에뮬레이터 DL 스레드(31–40)가 쓰므로 43 이상을 쓴다.
- 순수 Python이라 50 µs 예산을 못 지킨다(오늘 가짜 L1 8셀×6UE: p50 36 µs, p99 317 µs, 40 % 초과, 유실 0).

### 2-5. 세입자 서버와 요청 보내기

`dispatcher.py`는 저장소에 없다(요청 큐·동적 배치 미구현). 지금 있는 것은 세입자 서버 `dapp_sched --ctrl`과
요청 클라이언트 `tools/dapp_sched_client.py`다.

```bash
# [확인: 임시 링/제어 블록 이름으로] 컨테이너, "L1 is ready" 이후에 시작
docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var c_aerial_troy taskset -c 45 \
  /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/dapp_hook/dapp_sched \
  --engine /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/yolo26n_b1_fp16_atp10.engine \
  --image  /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/bus.jpg \
  -s /dev/shm/dapp_sched.sock --ctrl --cells 16

# [확인] 호스트: 요청 1000건, 10 ms 간격 (이미지 경로는 컨테이너 안 경로, 생략하면 --image)
python3 cuPHY-CP/dapp_hook/tools/dapp_sched_client.py -s /dev/shm/dapp_sched.sock -n 1000 -i 0.01 RUN
```

- 시작 로그에서 `SM capping=active`와 `budget source: estimator control block ... (attached ...)`를 확인한다
  (같은 줄에 `MPS=off`로 찍혀도 `SM capping=active`면 상한이 걸린 것이다. 오늘 출력이 그랬다).
- 응답: `OK sm=106 granted=96 cap=80 gate=40 slot_id=... age_us=... ms=1.09 dets=5 ... source=ctrl`
  또는 `HOLD gate= cap= ... reason=`(gate 0, 제어 블록이 20 ms보다 오래됨, 아직 결정 없음).
- cap → SM: `cap/100 × 132` 이하의 가장 큰 클래스(기본 16/32/48/64/96, `--classes`로 변경).

L1 없이 2-4, 2-5만 맞물려 보려면 가짜 L1을 쓴다 **[확인]**. 실제 링을 덮어쓰지 않도록 이름을 따로 준다.

```bash
docker exec c_aerial_troy /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/dapp_hook/dapp_ring_fakel1 -n /aerial_dapp_ring_test --slots 50000 --cells 8 --ue-per-cell 6
# live.py에는 --ring /aerial_dapp_ring_test --ctrl /aerial_dapp_ctrl_test, dapp_sched에는 -n /aerial_dapp_ring_test --ctrl /aerial_dapp_ctrl_test
```

## 3) 결과 확인

`analyze.py`는 없다. 무엇을 보느냐에 따라 셋 중 하나를 쓴다.

```bash
# [확인: 기존 런 cpu16으로] full stack 런 요약: L1 WRN/ERR, 드롭, testMAC Err.ind, RU 늦은 패킷, 링 SLOT_END, DCGM
cd cuPHY-CP/dapp_hook/sweep && python3 fullstack/analyze_run.py myrun

# [확인: 기존 결정 로그로] estimator 결정 로그: cap 분포, gate 닫힘 비율, 채널별 miss율(label이 있을 때)
cd cuPHY-CP/dapp_hook/estimator && python3 evaluate.py --decisions out/decisions_myrun.csv --nvlog ../../../prof/run61/phy_myrun.log

# [미실행 오늘] 벤치 스윕(runs/<manifest_id>/): 집계 CSV와 그림
docker exec -u root c_aerial_troy bash -lc 'cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/sweep && python3 plot.py --runs runs/<manifest_id>'
```

- `evaluate.py`는 채널 완료 시각 label(IND_SENT/CHAN_DONE)이 없으면 채널을 `unmeasured`로 두고, 관측 슬롯이
  10⁶ 미만이면 `statistically unconfirmed`로 표시한다. 지금은 label 훅이 없어 miss율 칸이 비어 나온다.
- 그 밖의 full stack 분석: `fullstack/e_analysis.py <names...>`(훅 비용), `parse_phylog.py <phy.log> <out.csv>`,
  `ru_late.py <ru.log> <prefix>`, `d_match.py`, `cpu_report.py <names...>`.

## 4) 코드를 수정한 뒤

```bash
# C++ (훅, dapp_sched, 링 도구): 다시 빌드 [미실행]
docker exec c_aerial_troy bash -lc 'cd /opt/nvidia/cuBB && ninja -C build.aarch64 -j $(nproc) cuphycontroller_scf dapp_ring_dump dapp_ring_fakel1 dapp_ring_selftest dapp_export_test dapp_sched'
# 그 다음 1)의 dapp_export_test, dapp_ring_selftest [확인]

# Python (estimator): 단위 테스트 [확인: 24개 통과]
cd cuPHY-CP/dapp_hook/estimator && python3 -m unittest discover -s tests -p 'test_*.py'

# 기록해 둔 링 덤프로 결정 재현 [미실행 오늘] (out/의 같은 이름 CSV를 덮어쓴다)
cd cuPHY-CP/dapp_hook/estimator && ~/.venvs/dapp_yolo/bin/python replay.py --dump ../../../prof/ring_59c_8C.bin --profile profiles/yolov8n_b1.yaml --out-dir out --plot
```

- 훅을 고쳤으면 L1을 다시 띄워야 반영된다(셀 다운). `dapp_sched`를 고쳤으면 그 프로세스만 다시 띄운다.
- `rules.yaml`, `profiles/*.yaml`, `cells.yaml`만 바꿨으면 빌드 없이 `live.py`만 다시 시작한다.
- 훅 레코드 형식(`include/dapp_hook/dapp_ring_abi.h`)과 cuPHY GPU 설정(`mps_sm_*` 등)은 바꾸지 않는다(CONTEXT.md §9).
- 2026-10-02 현재 `ninja -n`은 17스텝을 다시 빌드하겠다고 한다. 9/10 19:50 빌드 직후 브랜치 전환(20:07 커밋·머지)으로
  소스 mtime만 바뀐 것으로 보이며, 다음 빌드 때 `nvphy`, `scf_5g_fapi`, `cuphycontroller_scf`가 다시 링크된다.

## 알아둘 것

- 16셀에서 훅 `full` 모드는 600 µs 드롭 선까지 여유가 5 µs였다(10/01, 600k 슬롯에 드롭 1). 신뢰성 런은
  `summary`(`export_pdus: 0`)로 돌리기로 되어 있고, 이때 estimator는 요약 합계만으로 일감을 추정한다(`summary_only`).
- 측정 런 안에서 nsys를 시작·종료하지 않는다. 통계는 첫 1초를 뺀다(CONTEXT.md §9).
- `fullstack_run.sh` + `live.py` + `dapp_sched`를 한꺼번에 돌린 16셀 런은 아직 없다. 9/11에 8셀에서 손으로 같은 순서로 확인한 것이 전부다.
