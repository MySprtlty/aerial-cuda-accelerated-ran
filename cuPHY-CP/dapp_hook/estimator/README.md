# dapp_est — FAPI 기반 규칙형 estimator (cuPHY + YOLO / MPS 공존)

매 슬롯 L1 훅 링(`/dev/shm/aerial_dapp_ring`, ABI v2)에 기록된 UL/DL_TTI.request
내용을 보고, 같은 GPU에서 MPS로 돌아가는 YOLO(세입자)에게
**SM 상한 `cap_pct`** 와 **투입 허가 `gate_slots`** 를 정하는 규칙 기반 추정기다.
ML은 쓰지 않고, cuPHY와 링 레코드 형식은 건드리지 않는다.
출력은 공유메모리 제어 블록 `/dev/shm/aerial_dapp_ctrl`
(`{gate_slots, cap_pct, slot_id}`, 레이아웃은 `include/dapp_hook/dapp_ctrl_abi.h`)에 쓴다.

```
ring records ─► slot_context (셀 합산 GPU 컨텍스트) ─► estimator (규칙 a~g) ─► control block
      ▲                 work_formulas.py                 rules.yaml, cells.yaml,      ▲
      │                                                  profiles/*.yaml              │
   replay.py (덤프 파일)  /  live.py (실시간 폴링, 순수 Python)                   dapp_sched (세입자)
```

## 구성

| 파일 | 역할 |
|---|---|
| `dapp_est/ringio.py` | 링 헤더/레코드 디코딩(`struct`, numpy 없음), `RecordFile`(덤프), `LiveRing`(shm 폴링, seqlock) |
| `dapp_est/work_formulas.py` | PDU 하나의 채널별·커널 그룹별 "work" 식 (**provisional**) |
| `dapp_est/slot_context.py` | (sfn, slot) 단위로 모든 셀의 PDU를 합산: total / critical(최대 단일 PDU) / count / RNTI 수, 셀별 합계, SLOT_END로 닫힘 |
| `dapp_est/estimator.py` | `predict_latency_us(ctx, cap_pct, profile, rules, cells)`, `decide(ctx_now, ctx_future, profile, state, cells)` |
| `dapp_est/config.py` | `rules.yaml`, `cells.yaml`, `profiles/*.yaml` 로더 |
| `dapp_est/pipeline.py` | replay/live 공용 결정 경로(`Runner`) — 같은 입력이면 같은 출력 |
| `dapp_est/control_block.py` | 제어 블록 writer/reader (seqlock) |
| `dapp_est/decision_log.py` | 결정 로그 레코드 형식(CSV) |
| `dapp_est/labels.py` | label 인터페이스: IND_SENT / SLOT_VIOLATION(nvlog 파서) / CHAN_DONE 어댑터 |
| `replay.py` | 덤프 → 결정 로그 CSV, 요약표, 그림(cap/gate/latency vs D_c) |
| `live.py` | 링 폴링 → 매 슬롯 decide → 제어 블록. 슬롯당 처리시간 로그, 예산 초과 경고 |
| `profile_tenant.py` | 세입자 프로필 측정 (TensorRT `dapp_sched --sweep` 우선, PyTorch fallback, nsys 선택) |
| `calibrate.py` | 측정된 커널 시간으로 `w_ck` 최소자승 적합, R² 게이트, `--write` |
| `evaluate.py` | 결정 로그 + label 조인 → 채널별 miss율/신뢰도/tightness, cap 분포, gate 닫힘 비율 |
| `tests/` | 레이아웃, LDPC 분할, 규칙, replay==live 동일성, calibrate, label/evaluate 단위 테스트 |

## 실행

```bash
cd cuPHY-CP/dapp_hook/estimator
python3 -m unittest discover -s tests -p 'test_*.py'

# replay: 덤프 여러 개 × 프로필 여러 개 (그림은 matplotlib이 있는 venv로)
~/.venvs/dapp_yolo/bin/python replay.py --dump ../../../prof/ring_59c_8C.bin \
    --profile profiles/yolov8n_b1.yaml --profile profiles/yolov8m_b8.yaml --out-dir out --plot

# live: L1이 떠 있을 때 (순수 Python)
python3 live.py --profile profiles/yolov8n_b1.yaml --log out/decisions_live.csv --timing out/timing_live.csv

# 프로필 측정 (MPS 필요)
python3 profile_tenant.py --name yolov8n_b1 --batch 1 --engine ../models/yolov8n_fp16_sm16.engine \
    --image ../tools/bus.jpg --sched ../../../build.aarch64/cuPHY-CP/dapp_hook/dapp_sched --out profiles/yolov8n_b1.yaml

# 가중치 보정 (nsys 커널 시간 CSV가 생기면)
python3 calibrate.py --measure kernels.csv --dump-dir ../../../prof --write

# 평가
python3 evaluate.py --decisions out/decisions_ring_59c_8C_yolov8n_b1.csv --nvlog /path/phy.log --ind ind_sent.csv
```

덤프는 `tools/dapp_ring_record.py`로 만든다 (`--out prof/ring_x.bin --seconds 80`,
헤더는 `.hdr`에 같이 저장). shm 이미지를 그대로 복사한 파일도 읽는다.

## 시간축과 "now"

L2는 슬롯 s의 FAPI를 약 `slot_advance`(=3) 슬롯 앞서 보내고, L1은 s를 enqueue한 뒤
SLOT_END(s)를 링에 쓴다. 따라서 SLOT_END(s) 시점에 s-2, s-1, s는 확정됐지만 GPU에는
아직 안 올라간 상태다(DL(s)는 곧 시작, UL(s)는 약 2 ms 뒤). 결정은 SLOT_END마다 한 번
내리며, 마지막 `known_slots`(3)개 확정 슬롯 중 가장 오래된 것이 `ctx_now`, 나머지가
`ctx_future`다. 세입자는 제어 블록의 최신 값을 읽는다.

## 규칙 a~g와 근거

- **a. base_c = fixed_c + Σ_k w_ck · work_ck · scale(cell)**
  채널 c의 GPU 시간을 커널 그룹 k의 work에 선형이라고 본다. `total`(모든 셀·PDU 합)과
  `critical`(커널별 최대 단일 PDU, 병렬화로도 못 줄이는 체인)을 따로 계산한다.
  work 식은 `work_formulas.py` (PUSCH chest/eq/llr/ldpc/uci, PDSCH enc/mod/prec, PDCCH agg/bits,
  PUCCH f01/f234, PRACH ocas, SRS ports, SSB present, CSIRS count). LDPC는 38.212 분할로
  `num_cb·Zc`, 실패하면 `num_cb`. FAPI code rate 단위는 /10240.
- **b. sm_avail = 1 − cap/100 · sm_fill,  latency_c = max(base_crit, base_total / sm_avail)**
  세입자가 cap의 sm_fill만큼 SM을 실제로 점유한다고 보고 cuPHY 몫이 그만큼 줄어든다고
  가정(처리량 모델). critical 체인은 SM이 남아도 못 줄어드니 하한으로 둔다.
- **c. margin_c = profile.kernel_max_us + jitter_us(30)**
  이미 실행 중인 세입자 커널 하나는 선점되지 않으므로 그 최대 길이만큼, 그리고 타이머/큐
  지터만큼 여유를 둔다.
- **d. gate**: 확정 슬롯(now, +1, +2)에 대해 `latency_c + margin_c ≤ D_c`를 검사하며
  연속 안전 슬롯 수 k를 센다. 셋 다 안전하고 최근 W초(1.0)의 최대 컨텍스트도 안전하면
  k = `max_gate_slots`(40). `k·slot_us < dur_ms[batch][cap]`이면 세입자 한 번이 안전 구간을
  넘기므로 `gate_slots = 0`.
- **e. cap**: 최근 T초(1.0)의 p퍼센타일(100 = 최악) 컨텍스트에서 모든 채널이 안전한 최대
  cap ∈ {20,40,60,80}. 올릴 때는 N회(3) 연속 후보가 나와야 하고 내릴 때는 즉시 —
  잘못 올리면 마감 위반, 잘못 내리면 처리량 손실뿐이므로 비대칭.
- **f. 안전 기본값**: 컨텍스트 없음/미완성/링 정체(`stale_after_ms`) → gate 0, cap 유지.
  프로필 없음 → cap 0.
- **g. 셀 스케일**: `cells.yaml`의 셀별 정적 계수(BW·안테나). 지금은 모두 1.0.

`D_c`는 채널 GPU 작업이 시작될 수 있는 시점부터의 시간 예산이다(UL: T0 + 1 슬롯,
DL: tick = T0 − slot_advance 슬롯). `deployment_constants.md`가 생기면 그 값으로 바꾼다.

## 결정 로그 레코드 (`decision_log.py`, CSV 한 줄 = 슬롯 하나)

`ts_ns, sfn, slot, slot_id, t0_ns, n_cells, ul, dl, {ch}_n, {ch}_rnti, {ch}_{kernel}(work 합),
base_{ch}(us), lat_{ch}(선택된 cap에서의 추정 us), gate_slots, cap_pct, safe_k, need_slots, reason, proc_us`.
replay와 live는 `proc_us`를 뺀 모든 열이 같다(`tests/test_pipeline.py`).

## label 인터페이스 (`dapp_est/labels.py`)

- `IND_SENT(sfn, slot, cell, channel, ts_ns)` — CSV. L2 어댑터 쪽 훅이 만들어야 한다(미구현).
- `SLOT_VIOLATION(sfn, slot, cell, reason, ts_ns, side)` — Aerial nvlog 텍스트에서 파싱:
  `ul_order_timeout`, `srs_order_timeout`, `pusch_wait_timeout`, `work_cancel`, `ulc_task_timeout`,
  `error_ind_<code>`("Send Err.ind for SFN x.y cell_id=..."), 그 외 ERR 레벨은 `nvlog_error`.
  sfn/slot이 없는 줄은 "unattributed"로 따로 센다.
- `CHAN_DONE(sfn, slot, cell, channel, start_ns, done_ns, source)` — `docs/label_points.md`의
  지점에서 나올 미래 데이터의 어댑터. 지금은 CSV 로더만 있다.

`evaluate.py`는 label이 없는 채널을 "unmeasured"로 두고(0으로 채우지 않음), 관측 슬롯이
`--min-slots`(기본 10⁶) 미만이면 "statistically unconfirmed"로 표시한다.

## provisional 값 전체 목록 (측정 전)

| 값 | 위치 | 현재 | 교체 방법 |
|---|---|---|---|
| work 식 | `work_formulas.py` | 필드 곱 근사 | `kernel_map.md`(커널별 측정)로 식 수정 후 calibrate 재실행 |
| `w_ck` (커널 가중치) | `rules.yaml: weights_us_per_work` | 8셀 덤프에서 UL 슬롯 PUSCH ≈ 400 us(ldpc ≈ 60 %), DL 슬롯 PDSCH ≈ 360 us(enc ≈ 40 %)가 되도록 잡은 값 | `calibrate.py --measure kernels.csv --write` (R² ≥ 0.8인 커널만) |
| `fixed_us` | `rules.yaml` | 채널별 15~60 us | calibrate에 절편 추가 또는 nsys 최소 슬롯 시간 |
| `D_c` | `rules.yaml: deadline_us` | PUSCH 2000, PUCCH 1500, PRACH 3000, SRS 4000, DL 900 | `deployment_constants.md` |
| `jitter_us` | `rules.yaml: margin` | 30 | 세입자 시작 타이밍 분포 측정 |
| `stale_after_ms` | `rules.yaml: safety` | 20 | 링 heartbeat 분포 |
| 셀 스케일 | `cells.yaml` | 1.0 | 셀 설정이 달라질 때 (bw/100)·(ant/4) |
| `dur_ms` yolov8n b1 | `profiles/yolov8n_b1.yaml` | dapp_sched sweep 측정값(status: measured) | `profile_tenant.py` 재측정 |
| `dur_ms` yolov8m b8 | `profiles/yolov8m_b8.yaml` | **추정** | `profile_tenant.py --pytorch` 또는 TensorRT 엔진 빌드 후 sweep |
| `sm_fill` | profiles | default batch1 0.5 / batch8 1.0 | `profile_tenant.py --nsys` (time-fill 근사) |
| `kernel_max_us` | profiles | default 100 | `profile_tenant.py --nsys` |
| label 시작 오프셋 | `rules.yaml: label` | UL +500 us, DL −1500 us | `label_points.md` 지점의 실제 타임스탬프 |
| live 예산 | `rules.yaml: live.budget_us` | 50 | 요구사항 |

## 교체 체크리스트

1. work 식 → `kernel_map.md`가 생기면 `work_formulas.py`를 고치고 `tests/test_ringio.py`의 기대값 갱신.
2. `w_ck` → nsys 커널 시간 CSV(`tv,sfn,slot,channel,kernel,duration_us`)를 만들고 `calibrate.py --write`.
   R² < 0.8 커널은 경고만 나오고 값은 유지된다.
3. `D_c` → `deployment_constants.md`의 값으로 `rules.yaml: deadline_us` 교체.
4. 프로필 → `profile_tenant.py`로 `profiles/*.yaml` 재생성 (`status:` 줄의 default/provisional이 사라져야 함).
5. 전부 끝나면 `rules.yaml: provisional: false`.

## 한계 (현재)

- live 모드는 순수 Python이다. 가짜 L1(`dapp_ring_fakel1`, 8셀×6UE, 실시간) 기준
  슬롯당 decide p50 ≈ 25 us, 레코드 1개인 슬롯은 총 35 us지만 PDU 레코드가 36개 들어오는
  슬롯은 디코딩까지 ≈ 250 us(p99 316 us)로 50 us 예산을 넘긴다(전체 슬롯의 약 40 %가 경고).
  예산을 지키려면 `pipeline.Runner`와 같은 순서로 동작하는 C++ 포팅이 필요하다.
  (`--log`의 CSV 기록 ≈ 19 us는 예산 계산에서 뺀다.)
- 채널별 D_c는 채널이 서로 독립적으로 스트림을 쓴다고 가정한다. 실제로는 한 슬롯의
  DL 채널들이 같은 task chain에서 처리되므로 DL 합산 예산이 더 맞을 수 있다(label로 확인).
- 셀 간 병렬 실행(멀티 스트림)은 `total` 합산에 반영되지 않는다. 8셀 이상에서 과대추정
  가능성이 있고, calibrate가 `w_ck`를 낮추는 식으로 흡수한다.
