# FlexRIC xApp에서 dApp에 RUN 보내기 (`xapp_flexric/`, 2026-11-05)

## 핵심

E2로는 RUN을 보낼 수 없다. FlexRIC의 E2 경로는 `xApp —E42→ nearRT-RIC —E2→ gNB E2 agent(OAI L2)`에서 끝나고, L1(cuphycontroller)이나
GPU에 닿는 E2 서비스 모델이 없다. 그래서 xApp **프로세스 하나 안에서 인터페이스 두 개**를 쓴다.

| 인터페이스 | 상대 | 용도 |
|---|---|---|
| FlexRIC (E42/E2) | nearRT-RIC → OAI gNB | RAN 상태 읽기·제어 (KPM, MAC, RC, Slice SM) |
| dApp 에이전트 (TCP 9810, JSON 줄) | cuphycontroller 프로세스 안의 `tenant/dapp_agent.cpp` | 슬롯마다 오는 추정치(`estimate`) 받기, `run`(JPEG) 보내고 `result` 받기 |

프로토콜은 `docs/XAPP_INTERFACE.md`. 다른 연구실의 기존 FlexRIC xApp에는 **`dapp_client.h` 한 파일만 포함**하면 된다 (의존성 없음, C11).

```c
#include "dapp_client.h"
dapp_client_t c;
dapp_connect(&c, "127.0.0.1", 9810);
dapp_hello(&c, "ra", "my-xapp", 1, 1, 5000);             // role ra, estimate 구독(매 슬롯), welcome 5초 대기
dapp_estimate_t e; dapp_latest_estimate(&c, &e);         // e.est_ms["26n".."26x"], e.yolo_sm_now, e.queue
int m = dapp_pick_model(&e, 40.0);                        // deadline 40 ms 안에 드는 가장 큰 모델
dapp_result_t r; dapp_run(&c, jpeg, len, DAPP_MODELS[m], 40, 0, 5000, &r);
printf("%s sm %d infer %.1f latency %.1f\n", r.status, r.sm, r.infer_ms, r.latency_ms);
dapp_close(&c);
```

Python xApp이면 `python/xapp_client.py`(`DappAgent`)를 그대로 import 하면 된다.

## 파일

| 파일 | 내용 |
|---|---|
| `dapp_client.h` | 헤더 전용 C 클라이언트: 연결, hello/welcome, estimate 수신 스레드, run/result(req_id 매칭), sm_report, get |
| `xapp_dapp_run.c` | 예제 xApp: FlexRIC 등록 → E2 노드 목록 → MAC SM(142) 구독(UE 수·DL/UL 처리량·PUSCH SNR) + dApp에 RUN 반복, 결과 한 줄씩 출력·csv |
| `CMakeLists.txt` | `xapp_dapp_run`(FlexRIC + dApp)과 `xapp_dapp_run_noric`(dApp 경로만, FlexRIC 없이 빌드) |
| `flexric_xapp.conf` | FlexRIC xApp 설정 (`/usr/local/etc/flexric/flexric.conf`와 같음: RIC 127.0.0.1, DB /tmp/) |

## 빌드

FlexRIC는 gNB의 E2 agent와 nearRT-RIC를 빌드한 **같은 소스 트리**(같은 E2AP/KPM 버전)로 링크해야 한다. 이 호스트에서는
연구실 트리 `/home/nvidia/hsw/25_3_hsw/openairinterface5g/openair2/E2AP/flexric` (FlexRIC 2.0.0, E2AP_V2, KPM_V2_03)이고 CMake 기본값이다.

```bash
cd cuPHY-CP/dapp_hook/xapp_flexric
cmake -S . -B build            # 다른 트리면 -DFLEXRIC_DIR=/path/to/flexric (그 안에 build/ 필요) -DE2AP_VERSION=... -DKPM_VERSION=...
cmake --build build
# build/xapp_dapp_run        FlexRIC + dApp
# build/xapp_dapp_run_noric  dApp 경로만 (FlexRIC 설치 없이 프로토콜 시험)
```
링크: `libe42_xapp.a` + `-lsctp -ldl -pthread` + FlexRIC 예제가 만든 `RRC_MESSAGES/libasn1_nr_rrc.a`(sqlite 래퍼가 RRC를 디코드함) + `sqlite3`(있으면).

## 실행 순서

1. nearRT-RIC (연구실 빌드): `.../flexric/build/examples/ric/nearRT-RIC -c /usr/local/etc/flexric/flexric.conf -p /usr/local/lib/flexric/`
   → SCTP 36421(E2, gNB용) / 36422(E42, xApp용) 리슨.
2. gNB: 우리 1셀 스택 `cots/cots_run.sh start demo1` (OAI gnb.conf의 `e2_agent` 블록이 127.0.0.1의 RIC에 붙는다. 우리 L1의 dApp 에이전트는 TCP 9810).
3. xApp:
```bash
./build/xapp_dapp_run --agent 127.0.0.1:9810 --images ../models/img4k --model auto --deadline-ms 40 --count 20 --rate 2 -c flexric_xapp.conf
```
FlexRIC의 `init_xapp_api()`는 **RIC에 E2 노드(gNB)가 하나도 없으면 5초마다 E42 SETUP을 재시도하며 돌아오지 않는다**(FlexRIC 고유 동작).
gNB가 먼저 붙어 있어야 한다. dApp 쪽만 시험하려면 `--no-ric`(또는 `xapp_dapp_run_noric`).

| 옵션 | 뜻 (기본값) |
|---|---|
| `--agent HOST:PORT` | dApp 에이전트 (127.0.0.1:9810) |
| `--image FILE` / `--images DIR` | 보낼 JPEG (baseline JPEG만; `prog` 파일은 제외) |
| `--model auto\|26n\|26s\|26m\|26l\|26x` | auto = estimate의 est_ms가 deadline 안에 드는 가장 큰 모델 (auto) |
| `--deadline-ms N` | 정책 기준이자 `qos.deadline_ms` — 에이전트가 못 지킬 것이 확실하면 즉시 `fail/overload` (40) |
| `--rate R`, `--count N` | 요청 속도·개수 (2 req/s, 20) |
| `--want-dets` | 결과에 검출 목록 포함 |
| `--mac-period 1_ms\|2_ms\|5_ms\|10_ms` | MAC SM indication 주기 (10_ms). 이 네 값만 허용 — 다른 값은 FlexRIC 라이브러리 안에서 assert |
| `--csv FILE` | 요청별 기록 |
| `--no-ric` | FlexRIC 생략 |
| `-c conf`, `-p dir` | FlexRIC 인자 (우리 옵션 뒤에) |

출력 한 줄 = 요청 하나:
`#n ok 26x sm 80 (mean 80.0) wait 0.1 dec 6.3 inf 32.8 lat 38.7 e2e 39.8 ms det 59 | est 26x 37.9 ms yolo_sm 108 q 0 | RAN: ind 0 UEs 0 DL 0.00 UL 0.00 Mbps SNR 0.0 CQI 0`
— sm = 추론 시작 SM 수, wait/dec/inf/lat = 서버 쪽 대기·디코드·추론·수신→응답, e2e = 클라이언트 송신→결과, est = 보낼 때의 추정치, RAN = E2 MAC indication 집계.

## 검증 기록 (2026-11-05)

- **dApp 경로** (독립 테넌트 `dapp_tenant_main`, 80 SM, `xapp_dapp_run_noric`): 26x 고정 5건 + auto 10건 모두 성공, 서버 37 ms (추론 30.7, 디코드 ~6).
- **FlexRIC + 실 스택** (연구실 nearRT-RIC + OAI gNB E2 agent + 우리 L1 `cots_run.sh start xric`):
  - gNB가 RIC에 등록: `nb_id 3584, PLMN 999/70, ngran_gNB, RAN functions 2 3 142 142 143 144 145 146 148`
  - xApp E42 등록(xApp ID 59) → MAC SM 142 구독 `10_ms` 성공 → RUN 10/10 성공, 서버 평균 38.4 ms / E2E 39.0 ms (26x, 80 SM; estimate yolo_sm 108) → 구독 삭제·정지 정상.
  - MAC indication은 UE가 없어 0건 (OAI E2 agent는 UE 통계가 있을 때 보낸다). UE를 붙이면 `RAN:` 필드가 채워진다.
  - L1 Err.ind 0, FATAL 0 (스택 로그 `cots/runs/xric/`).
- 참고: RAN function 142가 두 번 보이는 것은 `/usr/local/lib/flexric/`에 `libmac_sm.so`와 백업 `libmac_sm.so.before_ul_wait`가 함께 있어 E2 agent와 RIC가 둘 다 dlopen 하기 때문이다(연구실 파일, 건드리지 않음).

## 대안: E2로 보내고 싶다면

가능은 하지만 큰 공사다: 커스텀 E2SM(또는 RC 제어 메시지에 실어)을 FlexRIC에 추가 → OAI gNB의 E2 agent가 제어를 받아 L2 → nvIPC 벤더 메시지 → L1의 에이전트로 전달. 세 코드베이스(FlexRIC·OAI·우리 L2A)를 고쳐야 하고
JPEG 같은 큰 페이로드에 E2가 맞지 않아, xApp이 TCP로 직접 보내는 지금 구조를 권한다.
