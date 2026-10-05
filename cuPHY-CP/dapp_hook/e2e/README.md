# e2e — UE → GH200 → UE 지연 측정 도구

E2E 지연 = UE가 4K JPEG 추론 요청을 보낸 순간 → 응답을 받은 순간 (UE → RU → gNB(Aerial L1 + OAI L2/코어) → **이 서버** → 역순).
이 디렉터리는 그중 GH200 구간(수신 → 디코드 → 추론 → 응답)을 실제 파이프라인으로 돌리고, 응답에 서버 측 시각을 실어
E2E를 **공중/전송 구간**과 **GH200 구간**으로 분해할 수 있게 한다. UE 없이도 루프백으로 GH200 구간을 측정한다.

| 파일 | 역할 |
|---|---|
| `nvjpeg_dec.cu` → `libnvjpeg_dec.so` | JPEG 바이트 → NVJPG **하드웨어 디코드**(SM 미사용) → 1280 letterbox float CHW를 디바이스에 직접 생성. ctypes용 C API |
| `infer_server.py` | TCP 서버. 요청마다 디코드 → TensorRT(YOLO26) → NMS → JSON 응답(검출 + 서버 측 시각). 엔진 여러 개 로드(모델 선택), GPU 워커 1개 FIFO |
| `e2e_client.py` | UE 쪽. **open-loop**(고정 요청률, 도착 시각 사전 확정) 송신, RTT·서버 시각 기록. 표준 라이브러리만 사용 → UE 노트북에서 그대로 실행 |
| `gh200_segment.sh` | 루프백에서 {부하 없음, 16셀 최저, 최고} × cap × 요청률별 GH200 구간 측정 (결과: `sweep/results/e2e_prep/gh200_segment*/`). `BATCH=8`이면 타일 요청, `MIG_TENANT_UUID`·`ENGINE_TAG=mig3g`이면 세입자를 3g 인스턴스에서 |

## 빌드·실행 (컨테이너 안, cuPHY의 MPS 클라이언트와 같은 사용자 = root)

```bash
docker exec c_aerial_troy bash -lc 'cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/e2e && nvcc -O2 -arch=sm_90 -Xcompiler -fPIC -shared -o libnvjpeg_dec.so nvjpeg_dec.cu -lnvjpeg'
docker exec -d -u root -e CUDA_MPS_PIPE_DIRECTORY=/var c_aerial_troy bash -lc 'cd /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/e2e && taskset -c 52-55 python3 infer_server.py \
   --engine 26x1280=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/yolo26x_1280_b1_fp16_atp100.engine \
   --engine 26n640=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/yolo26n_b1_fp16_atp100.engine --port 9000 --log /tmp/server.csv'
# UE(또는 호스트 루프백)에서
python3 e2e_client.py --server <GH200 N6/DN 주소>:9000 --images ../models/img4k --rate 10 --seconds 60 --log client.csv
```

- MPS는 **한 번에 한 사용자**만 서비스한다. 다른 uid의 클라이언트는 현재 사용자의 서버가 비워질 때까지 CUDA 초기화에서 대기한다
  (컨테이너 기본 사용자로 띄운 워커가 root trtexec 뒤에서 멈춘 것을 확인). 서버는 반드시 root로.
- cap은 MPS 환경변수(`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`)로 프로세스 단위, 엔진은 빌드된 cap의 것을 써야 한다.
- JPEG는 **baseline**(카메라 기본)이어야 하드웨어 디코드가 된다. progressive는 CPU 경로(22 ms)로 떨어진다.

## 타일 추론 (batch 8 엔진)

batch 1 엔진은 4K 전체를 입력 크기로 letterbox해서 본다. batch N 엔진을 주면 서버는 그 요청을 **타일**로 처리한다: 입력 크기의 원본 해상도 조각을
격자로 잘라(3840×2160, 1280 입력 → 3×2 = 6장, 세로 400 px 겹침) 앞 슬롯에 넣고 남는 슬롯에는 전체 뷰를 넣은 뒤, 모든 슬롯의 박스를 원본 좌표로 되돌려
NMS 한 번으로 합친다(`njd_decode_tiles`, `postprocess`). 같은 4K 거리 사진에서 검출 수 중앙값이 25(전체 letterbox) → 89(타일, 26x)였다.

## 프로토콜 (TCP, little endian)

```
요청: '<4sIII'  b'DAPQ', req_id, engine_sel(0xFFFFFFFF = 서버 정책, 아니면 엔진 index), n_bytes ; JPEG
응답: '<4sIII'  b'DAPR', req_id, status(0 = ok), n_json ; JSON
JSON: {req_id, status, engine, n_det, dets:[[x1,y1,x2,y2,conf,cls]...], img:[w,h],
       t_ms:{net_in, queue, parse, decode, letterbox, infer, post, server}}
```
`net_in` = 헤더 첫 바이트 수신 → 페이로드 마지막 바이트 수신(서버가 보는 상향 꼬리), `queue` = 수신 완료 → GPU 워커 집기,
`server` = 첫 바이트 수신 → 응답 소켓에 넘김. **UE RTT − server = 공중 + 전송 구간.**

## 측정 계획과의 관계

- 공정성: 기준선(정적 MIG)과 우리 시스템에 **같은 요청 스케줄**(같은 rate·seed·이미지)을 넣는다. 클라이언트가 open-loop인 이유.
- 서버의 `policy()`가 dApp 스케줄러 연결점이다(지금은 기본 엔진 고정). 모델 선택 ablation은 `--engine-sel`로 클라이언트가 강제한다.
- 디코드 ~5 ms(4K q85 4:2:0, NVJPG, cap 무관), letterbox 14 µs. numpy NMS는 1280 입력에서 3.5~5 ms(타일 7 ms)로 느리다 — 도구의 한계이며 GPU 쪽으로 옮길 대상.
  수치는 `sweep/results/e2e_prep/nvjpeg.txt`, `summary_mig_off.txt`.
