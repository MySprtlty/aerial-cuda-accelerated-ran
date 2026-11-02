# dapp_tenant: cuphycontroller + YOLO in one process, PUSCH : PDSCH : YOLO SM split chosen every slot (2026-11-01)

Goal: the segmented YOLO 26n (tools/seg_yolo, 30 segments, batch 8 × 1280², the tiled-engine ONNX) runs as a thread of the
cuphycontroller process; every slot the hooked FAPI content (UL_TTI / DL_TTI) sizes the PUSCH and PDSCH green-context
partitions, and the tenant runs its next segment on the largest SM partition that is disjoint from everything cuPHY has in
flight. Objective: cuPHY deadlines first, as many SMs as possible for YOLO.

## Pieces

| piece | where | what |
|---|---|---|
| cuphydriver profiles | `cuphydriver/include/{cuphydriver_api,context,phypusch_aggr}.hpp`, `src/common/{context,cuphydriver_api}.cpp`, `src/uplink/{phypusch_aggr,task_function_ul_aggr}.cpp` | `DAPP_PUSCH_PROFILES="40,56,72,96"` / `DAPP_PDSCH_PROFILES="40,56"` (SM counts): the largest replaces the yaml `mps_sm_pusch` / `mps_sm_pdsch` (regular split), every smaller one gets its own green context, stream set and pipeline-object pool (3 PUSCH objects × 86 MiB, 10 PDSCH objects × 373 MiB per profile). The DL context (cell DL streams, compression, buffer memsets) is the smallest PDSCH profile. `getNextPuschAggr/getNextPdschAggr` take the pool of the profile the L2A requested for the slot (`cuphydriver_dapp_set_profile`), falling back to larger, then smaller profiles (never dropping a slot; logged as WRN "profile X requested, Y used"). |
| sizer | `include/dapp_hook/dapp_sizer.hpp` (header-only), `scfl2adapter/.../scf_5g_fapi_dapp_export.cpp` (`sizer_add_ul/dl`, `sizer_done`), `scf_5g_fapi_phy.cpp` (`sizer_expect`), `cuphyl2adapter/.../nv_phy_module.cpp` (`sizer_init`, `sizer_plan` before `l1_enqueue_phy_work`, 10-s statistics line "dApp sizer:") | Work units = `estimator/dapp_est/work_formulas.py` ported 1:1 from the decoded PDU records; model = `estimator/sm_model.txt` (`export_cmodel.py` from `sm_model_fit.yaml`): PUSCH completion model `start_us[slot] + queue + Σ work·(w_ins + w_par·s_ref/s) ≤ 2100 − margin`, PDSCH duration model `fixed + Σ … ≤ 375 − margin`; the smallest configured profile that satisfies it, the largest when none does. Bounded wait (`DAPP_SIZER_WAIT_US`, 40) for the async export thread. Writes `g_slot_plan` (t0, profiles, UL/DL presence) for the tenant. The ring record format is untouched. |
| tenant | `tenant/dapp_tenant.{hpp,cu}` (static lib `dapp_tenant`, TensorRT), started from `cuphycontroller_scf.cpp` after "L1 is ready" | Green contexts per (P, N) gap (below), engines `models/seg26n_fine/seg<k>_sm<c>.engine` for classes 40/48/64/88, slot clock from the newest plan (t0 = tick_original + slot_advance·500 µs, CLOCK_REALTIME), busy window = PUSCH of slots w−2..w−4 (+ UL control of w−1), PDSCH of w+1..w+2, guard near the slot end (the next segment must also fit the next slot), pause when no class fits. Log `/tmp/dapp_tenant_<run>.log` (10-s statistics incl. sizer and cuphydriver counters), csv per wall-clock slot. Thread pinned to core 45. |
| tools | `tenant/tn_runs.sh`, `tenant/tn_summary.py`, `tools/gc_bench/gc_nest.cu` | run driver / per-run summary / SM-layout verification |

## SM layout (measured, `tools/gc_bench/gc_nest.cu`, no MPS so `%smid` is physical)

`cuDevSmResourceSplitByCount(result, …, input, remainder, 0, count)` is deterministic but the SM ids are GPC-interleaved, not
linear. Results of `split(all, N)` are nested prefixes (PDSCH 40 ⊂ 56; DL ctrl 24, comms 16 and the order kernel's 8 inside
PDSCH 40); remainders of `split(all, 132 − Y)` are nested suffixes (PUSCH 44 ⊂ 60 ⊂ 76 ⊂ 100 for requested 40/56/72/96;
PUCCH 8 + PRACH 12 inside PUSCH 44; ORDER disjoint from every PUSCH). The gap between PDSCH_N and PUSCH_Y has to be built as
the **remainder of `split(prefix(P), N)`** with `prefix(P)` = the result of the PUSCH split (P = 132 − Y'): that set is disjoint
from PDSCH_N, PUSCH_Y and ORDER, and the first N of prefix(P) equals PDSCH_N. The naive "first c of the remainder of
`split(all, N)`" overlaps PUSCH. Gap sizes: (P 88, N 40) 48, (88, 56) 32, (72, 40) 32, (72, 56) 16, (56, 40) 16; without PUSCH:
92 (N 40), 76 (N 56). So YOLO can run at 40/48 SMs only while PUSCH 44 + PDSCH 40 are active (Average traffic), at 88/64 in
DL-only or quiet windows, and pauses in the busy window of Peak traffic (PUSCH ≥ 72 needed).

## Running

```
# 2 min each, 8 emulator cells, DDDDDDSUUU, build.aarch64_f1002 (lab build untouched); outputs prof/tenant
bash cuPHY-CP/dapp_hook/tenant/tn_runs.sh tn_av:average:tenant base_av:average:base tn_pk:peak:tenant base_pk:peak:base
python3 cuPHY-CP/dapp_hook/tenant/tn_summary.py tn_av
```
`fullstack_run.sh` knobs: `BUILD_DIR`, `TENANT=1`, `PROFILES_ONLY=1`, `PUSCH_PROFILES`, `PDSCH_PROFILES`, `TENANT_CLASSES`,
`TENANT_FIXED`, `TENANT_ENV` (extra `-e K=V`), `SIZER_MODEL`. Needs `GREEN=1` and the async hook (export_dl 1).
Engines for a new class c: `trtexec` under `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` of the class (31 % → 40, 37 % → 48, 49 % → 64,
67 % → 88 SMs), see tools/seg_yolo; do not build engines while an L1 run restarts the MPS daemon (the clients die).

## Results (2026-11-01, prof/tenant, 2 min each, build.aarch64_f1002, lab F08_CG1 config, 8 emulator cells; `summary_<run>.txt`)

| run | mode | sizer choice | PUSCH completion p99 pos 7/8/9 (us after t0) | DL compression done after −470 us, pos 0/1/2 | RU DL U late slots / cell | tenant |
|---|---|---|---|---|---|---|
| base_av | static green (yaml 100/104), Average | – | 1140 / 1127 / 1138 | 7.8 / 7.9 / 0.0 % | 3736 | – |
| prof_av | profiles + sizer, no tenant | PUSCH 44 (all U slots), PDSCH 40 (all) | 1249 / 1240 / 1241 | 7.6 / 7.7 / 0.03 % | 3647 | – |
| tn_av | + tenant, default policy | same | 1467 / 1512 / 1509 (max 1639) | **12.2 / 14.6 / 2.9 %** | **6274** | 107 inf/s, mean 39 SMs, class 48 in the busy window |
| tn_av56 | + tenant, PDSCH floor 56 (busy window has no class → pause) | PUSCH 44, PDSCH 56 | 1346 / 1315 / 1272 | 8.7 / 8.3 / 0.07 % | 4049 | 95 inf/s, mean 26 SMs, class 64 at pos 4–7 |
| base_pk | static green, Peak | – | 1269 / 1290 / 1271 | 3.4 / 3.4 / 0.02 % | 1669 | – |
| tn_pk | + tenant, default policy | PUSCH 76 (pos 7, 8) / 100 (pos 9), PDSCH 40 | 1443 / 1544 / 1305 (max 1785) | 2.5 / 2.3 / 0.0 % | 1089 | 133 inf/s, mean 35 SMs, 88/64 at pos 4–7, 48 at pos 8, paused elsewhere |

No PUSCH completion over 2100 us in any run (0 misses of 72k U slots each); the sizer never found a slot infeasible and
cuphydriver never had to fall back to another profile (pool exhaustion 0). The lab config itself has ~8 % of DL slots at
positions 0/1 finishing compression after −470 us even without the tenant (the known lab-config DL-late issue; the hybrid
config does not show it). Attribution at Average traffic: the profiles alone change nothing on the DL side (prof_av = base);
the tenant running at 48 SMs inside the busy window (PUSCH 44 + PDSCH 40 in flight, wall-clock positions 8, 9, 0–3) adds
+4.5 / +6.7 / +2.9 points of late DL slots and +68 % RU late packets although its SMs are disjoint from both channels: DCGM DRAM
bandwidth utilisation goes from 1.9 % to 13 % (memory / L2 contention, the effect quantified earlier with hbm_flood). Keeping
the tenant out of the busy window (tn_av56; knob `DAPP_TENANT_UL_PAUSE=1` does the same with the PDSCH-40 profile) leaves the
DL side within +1 point of the baseline and still gives YOLO 64–88 SMs in the DL-only and quiet windows (positions 4–7).
The f1002 static-green baseline completes PUSCH ~330 us earlier than the lab-build ladder (1140 vs 1478 us): build-related,
not explained; comparisons are within f1002 only.

Policy knobs (environment of the L1): `DAPP_TENANT_UL_PAUSE=1` (pause while PUSCH / UL control is in flight),
`DAPP_TENANT_DL_PAUSE=1` (pause while PDSCH is in flight), `DAPP_TENANT_DL_MAX_CLASS=<c>` (class cap while PDSCH is in flight).

### 10-minute runs (1.2M slots each, NVTAGS="", 2026-11-01)

The L1 log grows ~0.9 GB/min even without the INFO timing tags (lab nvlog config), so 10-min runs fill the 8.6 GB that are
free and lose the RU / phy excerpts (and `restore_lab` then truncates the lab yamls: now guarded, falls back to `git checkout`).
Salvaged from the console logs: base_av10 / tn_av10 (UL_PAUSE=1) / tn_pk10 (UL_PAUSE=1) / base_pk10: Err.ind 1 each (the
startup slot 0.3), slot drops 0, profile fallbacks 0, testMAC CRC errors 0 and UCI on time 100 % for all 8 cells; sizer:
tn_av10 PUSCH 44 for every U slot and PDSCH 40 for every DL slot, tn_pk10 PUSCH 76 (positions 7, 8) / 100 (position 9); tenant
~146 inferences/s in the traffic windows (classes 64 / 88 only). tn_pk10 logged one "Order kernel timeout error (exit condition
3)" event (SFN 703.18, cells 2–7); base_pk10 none — a single event, not attributable yet. Longer runs with the GPU timing
need ≥ 10 GB free (the 6 GB of ring dumps in prof/ are the obvious candidates).

Slot-clock fix after these runs: plan ids are now monotonic across SFN wraps and the clock reference never moves backwards;
a missing plan in the UL look-back is treated as "PUSCH at the largest profile" (pause) instead of "no UL" — before the fix
~9 % of the busy-window slots ran a 64-SM segment because the look-back found no plan (SFN wrap / late reference).

## 2026-11-02: xApp 에이전트, 모델별 요청 경로, horizon 규칙

- **xApp 에이전트**(`dapp_agent.{hpp,cpp}`, 프로토콜 `docs/XAPP_INTERFACE.md`): `DAPP_AGENT=1`이면 테넌트가 TCP 9810에서 Resource
  Allocator / Network Slicing xApp을 받는다. RUN(JPEG) → NVJPG 하드웨어 디코드·타일(e2e/nvjpeg_dec.cu) → 선택 모델의 엔진을 틈
  컨텍스트에서 실행 → e2e 헤드 파싱 + 타일 간 NMS → `result`. 슬롯마다 모델별 `est_ms`(큐 + 디코드 EMA + 클래스별 시간 EMA)를 푸시하고,
  주기적으로 `sm_report`(완료 요청의 평균 SM, 시간 가중 YOLO/RAN SM)를 보낸다. 모델 집합은 `DAPP_TENANT_MODELS=26n=/dir,...`
  (기본 `models/full26<m>`: e2e 엔진 심볼릭 링크 `seg00_sm{56,80,104,132}.engine`).
- **horizon 규칙**(`choose_part`): 다음 조각(전체 엔진이면 추론 전체)이 끝날 때까지 걸치는 모든 슬롯의 바쁜 집합(FAPI 계획, 없으면 이전
  프레임)에 맞는 가장 큰 클래스를 고른다. 26m/26l/26x 전체 엔진(12~40 ms)이 U 창에 걸쳐도 PUSCH SM을 침범하지 않는다. 1셀에서는 80(U 창
  포함)·104(DL 창만)가 된다. `DAPP_TENANT_HORIZON_MAX`(기본 200슬롯).
- **독립 실행**: `dapp_tenant_main`(cuphydriver 스텁, `DAPP_TENANT_FREERUN=1` = 벽시계 슬롯·계획 미지 = 유휴)로 L1 없이 요청 경로를
  시험한다. `dapp_agent_test`는 CUDA 없는 프로토콜 시험용 가짜 테넌트.
- **KPI 런**: `tenant/kpi_run.sh <name> <average|peak> <ncells> <rate> <policy> [deadline_ms]` = L1 루프백 런(tn_runs.sh, 에이전트 on) +
  호스트에서 `python/ra_stub.py`가 4K 요청을 보내고 `ra_<name>.csv/txt`(요청별 E2E·서버 시간·모델·SM, 요청 가중 mAP)를 남긴다.
- 모델 사다리(타일 b8, GPU ms, 56/80/104/132 SM): 26n 6.2/4.9/4.1/3.8, 26s 10.2/8.0/6.7/6.1, 26m 18.0/13.9/11.7/–, 26l 22.9/17.8/14.9/–,
  26x 40.3/30.7/26.5/24.6; 디코드 5.7~6.2 ms. 전체이미지 b1은 검출 −35 %라 제외. 결정 기록: `docs/PLAN_2026-11-02.md`.
- 1셀 루프백(Average): 사이저는 PUSCH 20 SM을 고르고(66000/66000, fallback 0) PUSCH GPU 실행은 243 us(100 SM일 때 233) → 1셀 PUSCH는
  SM이 아니라 고정 단계 지연에 묶여 있어 YOLO에 108(DL 창)/88(U 창) SM을 내줄 수 있다. lab yaml의 DL pos 0/1 지연 병리는 1셀에서 32 %
  (base도 동일)로, 실 UE 데모에는 hybrid yaml 묶음을 써야 한다.

### 2026-11-02 아침: 1셀 stall 원인과 새 손잡이
- 원인·조치는 `docs/PLAN_2026-11-02.md` §9. 요약: 1셀 사다리에서 cuphydriver의 UL order 녹색 컨텍스트가 [48,72)에 놓여 테넌트 틈 안에 들어간다 → 요청마다 L1 recovery. 회피 설정: `PUSCH_PROFILES=16,24,32,40,96 PDSCH_PROFILES=40 CLASSES=56`.
- `DAPP_AGENT_DEC_PART=1`(기본): 디코드 커널·D2H를 파티션 스트림에서 실행. 0이면 예전처럼 프라이머리 비차단 스트림(dec_s) 사용 — cuphycontroller 안에서는 쓰지 말 것.
- 이분법 손잡이: `DAPP_AGENT_SKIP_DECODE=1`(디코드 생략, 입력 버퍼 그대로 추론), `DAPP_AGENT_SKIP_INFER=1`(TRT enqueue 생략, 이벤트만 기록). kpi_run.sh의 8번째 인자로 넘긴다: `kpi_run.sh b_nodec average 1 4 fixed:26x 40 160000 "-e DAPP_AGENT_SKIP_DECODE=1"`.
- `dapp_tenant_main`: `DAPP_VICTIM=1`이면 cuPHY 역할 스레드(DAPP_VICTIM_CPU=47)가 400 us마다 프라이머리/8-SM 녹색 스트림에 memset+이벤트 왕복을 재고 DAPP_VICTIM_WARN_US(2000) 초과를 stderr에 기록(컨텍스트 락·디바이스 동기화 검출용; `DAPP_VICTIM_BLOCKING=1`이면 blocking 스트림).
- 운영 교훈: 셸 대기 조건에 `pgrep -f <문자열>`을 쓰면 자기 명령줄과 매칭된다(체인10이 3시간 공회전). 중단된 런 뒤에는 `pgrep -af dcgmi`로 로거 잔류를 확인.

### 2026-11-02 낮: 합집합 파티션(union layout) — order 조각을 비운 YOLO 컨텍스트
- **원인 요약**: 1셀 사다리에서 cuphydriver의 UL order 녹색 컨텍스트(yaml 20 → 24 SM, 분할 순서 위치 [48,72))가 테넌트 틈 공식
  "prefix(P) \ prefix(N)"의 한가운데 놓였다. 틈 공식은 선택된 PUSCH·PDSCH prefix만 제외했지 order·UL 제어·폴백 프로파일을 제외한 적이 없다.
  8셀(yaml PUSCH 100, order가 PDSCH 40 prefix 안)에서는 우연히 안 보였다. 긴 커널 엔진(26m/26l/26x)이 그 SM을 점유하면 패킷 대기 order
  커널이 못 떠서 UL AGGR 대기 → recovery → 320슬롯이면 "L1 not responsive" FATAL(scf_5g_fapi_phy.cpp:277). `docs/PLAN_2026-11-02.md` §9·§10.
- **구성(`DAPP_TENANT_LAYOUT=union`, 기본)**: 장치를 8-SM 그룹 15개(+나머지 12)로 한 번 split하고(한 split 호출의 출력끼리만
  `cuDevResourceGenerateDesc`로 합칠 수 있음), (PUSCH 프로파일 i, PDSCH N)마다 다음과 겹치지 않는 그룹들의 합집합을 클래스 컨텍스트로 만든다:
  PUSCH i와 그 위 `DAPP_TENANT_FB_GUARD`(기본 1)개의 큰 프로파일(pool busy 폴백), PUCCH+PRACH, PDSCH N, UL order. 폴백 영역에서 먼 그룹부터 쓴다.
  모든 cuphydriver 컨텍스트는 분할 순서의 prefix/suffix라서(PDSCH N = [0,N), PUSCH y = [132−y,132), order = [C−order, C), C = 132 − yaml PUSCH)
  집합은 위치로 계산한다. cuphycontroller는 MPS 클라이언트라 `%smid`가 컨텍스트 안 가상 번호(0..n−1)로 나오므로 측정은 쓰지 못하고,
  MPS가 없는 독립 실행(`dapp_tenant_main`)에서는 모든 집합·클래스 컨텍스트를 물리 smid로 자기검증한다(로그 "%smid-verified").
  오프라인 검증 도구: `tools/gc_bench/gc_union.cu`(`GC_ORDER=8 GC_GUARD=1` 등), 결과 `prof/tenant/gc_union_o8g{0,1}.log`, `gc_union_o20g1.log`.
- **cuphydriver export(API만 추가, CONTEXT.md §9 보고)**: `cuphydriver_dapp_profile_sms(2, …)` = {UL order 실제 SM 수, mode(0 = PUSCH 여집합의
  나머지, 1 = 예외 prefix)}, `(3, …)` = {PUCCH, PRACH 실제 SM 수}. context.hpp/cpp에 값 저장만 추가, 분할 자체는 그대로.
- **1셀 yaml order**: `kpi_run.sh`가 ncells=1이면 `SM_OVERRIDE="ul_order=${ORDER_SMS:-8}"`로 `mps_sm_ul_order` 20 → 8(order 커널은 셀당 1블록;
  fullstack_run.sh가 런 뒤 yaml 복원). 1셀 결과: guard 1 → 72 SM 자유 → class 56(사용 중인 cuPHY 컨텍스트와 공유 0, 폴백 28·36·44와도 0);
  guard 0 → 80 SM → class 80(폴백 PUSCH 28과 8 SM 공유, pool busy일 때만). order 20(24 SM)이면 guard 1에서 56, guard 0에서도 80 불가.
- **QoS(`DAPP_AGENT_QOS=1`, 기본)**: RUN의 `qos.deadline_ms`가 있으면 시작 시점 추정(대기 + 디코드 EMA + 클래스 시간) > deadline이면 즉시
  `fail/overload`(docs/XAPP_INTERFACE.md). 고정 모델 KPI 런은 deadline 150으로 돌린다(`kpi_run.sh <name> average 1 4 fixed:26x 150`).
- **런 이름**: `u1_<model>_av4`(guard 1), `u0_<model>_av4`(guard 0), `u1_map_av8`/`u0_map_av8`(RA map 정책, deadline 40). 체인: `prof/tenant/chain18.log`.

### 2026-11-02 오후: 진짜 원인(채널 큐 기아)과 수정(조각 발사) — 요청 경로의 모든 모델이 cuPHY를 건드리지 않게 됨
- 합집합 배치(공유 SM 0, §위)로도 26m/26l/26x는 recovery가 남았다(u1_26m 433, u1_26l 772; MPS 아래에서는 FATAL). 스트림 수(s1_26m: 보조 스트림 0·class 56만)도 무관.
- **nsys(f26m_av4)**: 멈춘 것은 DL 압축 스트림(kernel_compress → kernel_write = 전송 ready 플래그)의 디스패치였고, DOCA 전송 트리거 커널은 그 플래그를
  17 ms 스핀했다(`docs/PLAN_2026-11-02.md` §11). 같은 컨텍스트의 다른 스트림은 멀쩡했고 두 사건 모두 YOLO 그래프가 끝나는 순간에 풀렸다. 예전 8셀 스트레스
  (hbm 350 us 런치 무사 / hbmlo 2 ms 런치 붕괴)와 합치면: **테넌트 채널(하드웨어 작업 큐)에 쌓여 있는 일의 양이 cuPHY 스트림의 최대 대기 시간이 된다.**
  전체 엔진 그래프(26m 18 ms, 26x 31 ms)를 한 번에 발사하는 것이 문제였고, 8셀 조용한 경로(조각당 ≤ 0.5 ms, lookahead 1)가 멀쩡했던 이유도 이것이다.
- **수정 `agent_pump`**: 캡처한 그래프를 위상 순서로 `DAPP_AGENT_GRAPH_CHUNK`(기본 6)노드씩 조각(복제 후 나머지 노드 제거, `cudaGraphGetEdges_v2`: TRT 그래프는
  PDL 엣지 데이터가 있어 구형 API는 거부), 요청 때 `DAPP_AGENT_CHUNK_AHEAD`(기본 2)개까지만 큐에 두고 이벤트로 페이싱. 35개 그래프 → 2055조각, 0.7 s.
  26m class 56: 54조각, 조각 p50 0.24 / p90 0.75 / max 1.55 ms(가장 긴 단일 커널 0.6 ms 포함), 추론 18.5 ms(단일 그래프 18.4와 같음).
- **결과 u2_26m_av4**(1셀 Average 4 req/s, class 56, MPS 없음): 197/197, recovery 0, AGGR 대기 0, PrePrepare 0, 서버 26.0 ms(디코드 5.7 + 추론 18.5),
  PUSCH 완료 p99 1055 us. 같은 구성의 단일 그래프 u1_26m은 recovery 433. 나머지 모델·guard 0·map 런은 `prof/tenant/chain19.log`.
- 진단 도구: `DAPP_AGENT_PROFILE_S`(cudaProfilerStart/Stop; nsys는 프로세스 시작부터 켜야 L1이 산다: `L1_EXTRA_ARGS="--nsys --cuda-graph-trace=node -o /tmp/x"`),
  스크래치 `nsys_delay.py`. 운영 변경: fullstack_run.sh GREEN=1 → test_config `--green-ctx=1` → L1이 MPS 없이 실행(예전처럼 MPS 아래로 돌리려면 GREEN_MPS=1).

### 2026-11-02 14:00: 요청 경로의 완성 형태 (RA RUN → 모델 → SM → 실행), 검증 chain20
요청 한 건의 흐름:
1. RA가 TCP 9810으로 `run{model, qos.deadline_ms, image}`를 보낸다(`docs/XAPP_INTERFACE.md`). 큐가 가득 차면 즉시 `fail/overload`.
2. 테넌트 스레드가 큐에서 꺼내 모델 집합(26n~26x, `DAPP_TENANT_MODELS`)을 고르고, **horizon 규칙**으로 SM 파티션을 정한다: 이 추론이 걸칠 슬롯들의
   FAPI 계획(사이저가 매 슬롯 쓴 PUSCH/PDSCH 프로파일, 없으면 이전 프레임 예측)에서 바쁜 집합을 모으고, 그와 겹치지 않는 가장 큰 파티션(합집합 배치:
   UL order·PUCCH/PRACH·PDSCH·선택 PUSCH(+guard 단계 폴백) 제외)을 고른다. 1셀 Average: guard 0 → 80 SM, guard 1 → 72(엔진이 있으면)/56.
   추정(대기 + 디코드 EMA + 클래스 시간) > deadline이면 `fail/overload`(QoS).
3. NVJPG 하드웨어 디코드·타일(파티션 스트림) → 그 파티션용 엔진의 CUDA 그래프 조각(기본 3노드)을 **1개 선행**으로 발사(`agent_pump`) → e2e 헤드 파싱·
   타일 간 NMS → `result`(지연·SM·검출). 매 슬롯 `estimate`(모델별 est_ms, 현재/최소 SM), 주기적으로 `sm_report`.
cuPHY 보호 장치 세 겹: (a) SM: 합집합 배치로 cuPHY 컨텍스트와 공유 0(폴백 1단계는 guard로 선택), (b) 큐: 조각 발사로 테넌트 채널에 쌓인 일 ≤ 조각 1~2개,
(c) 멈춤: 맞는 파티션이 없으면 요청을 들고 기다림(pause). 검증 런: chain20(`v_*`, 3노드 조각, 80 SM 기본), 결과는 PLAN §14.
- **최종 결과(15:35, chain22, `docs/PLAN_2026-11-02.md` §15)**: 기본값 = 합집합 배치 + guard 0(80 SM) + 3노드 조각·선행 2 + QoS, MPS 없음. 1셀 Average에서 26m 21.8 ms
  (E2E −16 %, mAP −7.7 %), 26l 26.6(−13 %, −4.3 %), RA 정책 8 req/s 25.7(−13.6 %, −4.8 %)로 두 KPI 통과, 26x 42.4(−2.6 %) 미달; 모든 런 recovery 0.
