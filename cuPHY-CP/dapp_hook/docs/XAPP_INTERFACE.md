# dApp 에이전트 ↔ xApp 인터페이스 (v1, 2026-11-02)

우리 쪽(cuphycontroller 프로세스 안의 dApp 에이전트)이 **서버**, 두 xApp(Resource Allocator, DRL-based Network Slicing)이
**클라이언트**다. 전송은 TCP 위의 줄 단위 JSON(UTF-8, 한 메시지 = 한 줄, `\n` 종료)이고, 이미지 같은 바이너리는
헤더 줄 바로 뒤에 `bytes` 길이만큼 원시 바이트로 붙는다. 선택 사항으로 같은 JSON을 UDP 데이터그램으로도 보낼 수 있다
(슬롯마다 나가는 추정치를 연결 관리 없이 받고 싶을 때).

| 항목 | 값 |
|---|---|
| 포트 | TCP 9810 (`DAPP_AGENT_PORT`), 모든 인터페이스에 바인드 |
| UDP 추정치 스트림 (선택) | `DAPP_AGENT_UDP=host:port` 이면 매 슬롯 `estimate`를 그 주소로 전송 |
| 인코딩 | JSON 한 줄 + (있으면) 원시 바이트 `bytes`개 |
| 시각 | `t_ns` = CLOCK_REALTIME ns (UE·xApp 쪽 시각과 바로 비교 가능), `sfn`/`slot` = 공중 슬롯 |
| 모델 이름 | `"26n" "26s" "26m" "26l" "26x"` |
| 동시 연결 | xApp마다 하나. 같은 역할이 다시 접속하면 새 연결이 이전 것을 대체 |

모든 메시지는 `type` 필드를 갖는다.

## 1. 접속

xApp → 에이전트
```json
{"type":"hello","role":"ra","name":"resource-allocator","subscribe":["estimate"],"every":1}
{"type":"hello","role":"slicing","name":"drl-slicing","subscribe":["sm_report"],"period_ms":1000}
```
- `subscribe`: 푸시 받을 메시지 종류. `every` = 추정치를 몇 슬롯마다 받을지(1 = 매 슬롯, 0.5 ms), `period_ms` = SM 보고 주기.

에이전트 → xApp
```json
{"type":"welcome","version":1,"models":["26n","26s","26m","26l","26x"],"slot_us":500,"cells":1,"tdd":"DDDDDDSUUU","gpu_sm":132}
```

## 2. Resource Allocator 쪽

### 2-1. 추정치 (에이전트 → RA, 매 슬롯 푸시)
```json
{"type":"estimate","t_ns":1793550000123456789,"sfn":812,"slot":7,
 "ul_sm":16,"dl_sm":24,"yolo_sm_now":92,"yolo_sm_min":80,"queue":0,"inflight":1,
 "est_ms":{"26n":6.4,"26s":9.9,"26m":15.8,"26l":21.9,"26x":34.7},
 "est_sm":{"26n":104,"26s":104,"26m":80,"26l":80,"26x":80}}
```
- `est_ms[m]` = **지금 RUN을 보내면 모델 m의 결과가 돌아오기까지 서버에서 걸릴 시간의 추정**(ms):
  큐 대기 + JPEG 디코드 + 추론. 추론 시간은 FAPI로 확정된 앞으로의 슬롯 계획(어느 슬롯에 PUSCH·PDSCH가 몇 SM을
  쓰는지)에서 YOLO가 받을 SM 수를 구하고, 모델별 (SM 수 → 시간) 표로 환산한다.
- `est_sm[m]` = 그 추정의 근거가 된 SM 수(추론 구간의 최소 틈). `ul_sm`/`dl_sm` = 이 슬롯에 cuPHY가 쓰는 PUSCH/PDSCH SM.
- `yolo_sm_now` = 이 슬롯에 YOLO가 쓸 수 있는 SM, `yolo_sm_min` = 앞으로의 계획 구간에서 최소.
- `queue`/`inflight` = 대기 중/실행 중 요청 수. RA는 이 값과 QoS로 모델을 고른다.

### 2-2. 실행 요청 (RA → 에이전트)
```json
{"type":"run","req_id":1001,"model":"26m","qos":{"deadline_ms":150,"priority":0},"image":"jpeg","bytes":982341,"want_dets":false}
```
뒤에 JPEG 982341바이트. `req_id`는 RA가 정한 정수(응답에 그대로 돌아감). `qos.deadline_ms`가 있으면 그 안에 못 끝낼 것이
확실할 때(큐 상태로 판단) 바로 `fail/overload`로 거절한다. `want_dets: true`면 검출 결과를 응답에 넣는다.

### 2-3. 결과 (에이전트 → RA)
```json
{"type":"result","req_id":1001,"status":"success","model":"26m","sm":80,"sm_mean":84.2,
 "t_recv_ns":..., "t_start_ns":..., "t_end_ns":..., "wait_ms":0.3,"decode_ms":5.2,"infer_ms":14.9,"latency_ms":20.6,
 "detections":37,"tiles":8}
```
- `status`: `"success"` 또는 `"fail"`. 실패 사유 `reason`: `no_engine`(모델/클래스 엔진 없음), `decode_error`,
  `overload`(큐 가득·QoS 불가), `paused`(RAN 보호로 SM 0), `internal`.
- `sm` = 추론을 시작한 SM 수, `sm_mean` = 추론 동안의 평균(조각마다 바뀔 수 있음). `latency_ms` = 수신→응답.
- `want_dets`일 때 `"dets":[{"cls":0,"name":"person","conf":0.91,"xyxy":[x1,y1,x2,y2]},...]` (4K 원본 좌표).

### 2-4. 단발 조회
`{"type":"get","what":"estimate"}` → `estimate` 한 개. `{"type":"get","what":"models"}` → `welcome`과 같은 모델 목록.

## 3. DRL-based Network Slicing 쪽

### 3-1. SM 보고 (에이전트 → Slicing, 주기 푸시)
```json
{"type":"sm_report","t_ns":...,"window_ms":1000,"requests":9,
 "sm_mean":88.4,"sm_mean_per_request":[80,104,80,...],
 "sm_by_model":{"26x":{"n":5,"sm_mean":80.0},"26m":{"n":4,"sm_mean":99.0}},
 "yolo_sm_time_mean":93.1,"ran_sm_time_mean":38.9}
```
- `sm_mean` = 창 안에서 끝난 요청들에 **할당된 SM 수의 평균**(요청별 평균의 평균). 요청이 없으면 `requests: 0`, `sm_mean: null`.
- `yolo_sm_time_mean` = 시간 가중 평균(슬롯마다 YOLO 몫 SM의 평균, 요청 유무와 무관), `ran_sm_time_mean` = cuPHY 몫.

### 3-2. 단발 조회
`{"type":"get","what":"sm_report"}` → `sm_report` 한 개(최근 창).

## 4. 참고 구현
- `python/xapp_client.py`: 두 역할 공용 클라이언트 라이브러리(접속·구독·run·결과 파싱).
- `python/ra_stub.py`: Resource Allocator 대역. 추정치를 받아 간단한 규칙(QoS 안에서 가장 큰 모델)으로 고르고 `img4k`
  이미지를 정해진 속도로 보내며 결과를 csv로 남긴다. 타 연구실 모듈이 오기 전의 시험·시연용.
- `python/slicing_stub.py`: SM 보고를 받아 출력/csv.
- 서버 쪽 구현: `tenant/dapp_agent.{hpp,cpp}` (소켓·JSON), `tenant/dapp_tenant.cu` (요청 실행·추정).
