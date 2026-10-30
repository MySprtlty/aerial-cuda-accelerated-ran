# BASELINE — 연구원들의 COTS UE 구성(Benetel RU + cuphycontroller + OAI gNB + Open5GS) 조사 기록 (2026-10-29)

하이브리드의 **기준(基準)** 이 되는 구성이다. 모든 값은 `/home/nvidia/hsw/25_3_hsw/`의 실제 파일과 실행 로그에서 읽었고, 파일 자체는
`baseline/`에 바이트 그대로 복사해 두었다(md5는 §7). 매뉴얼(사용자 전달)의 각 단계가 실제로 무엇을 하는지도 확인했다.

## 1. 매뉴얼의 각 단계가 하는 일

| 매뉴얼 | 실제 동작 (확인) |
|---|---|
| corenet 경로 `sh/`, `./start_cn.sh` | `sudo systemctl start open5gs-{nrfd,scpd,amfd,smfd,upfd,ausfd,udmd,udrd,pcfd,nssfd,bsfd,hssd,mmed,pcrfd,sgwcd,sgwud,seppd,webui}` — **호스트 네이티브 Open5GS**(docker 아님). 유닛은 모두 `disabled`(부팅 시 안 뜸). 10-29 14시 현재 전부 `active`(누군가 수동 기동). |
| `sudo ./routing.sh` | `net.ipv4.ip_forward=1` + iptables: `POSTROUTING -s 10.45.0.0/16 -o <WAN> MASQUERADE`, `FORWARD ogstun→WAN ACCEPT`, `FORWARD WAN→ogstun ESTABLISHED ACCEPT`(주석 OPEN5GS-UE-*). WAN = 기본 경로 NIC `enx705dccffd874`(10.150.21.161, USB NIC). 10-29 현재 규칙 적용돼 있음. |
| `sudo ./routing_restore.sh`, `./stop_cn.sh` | 위 규칙 삭제 / 유닛 stop. |
| UE 등록 `http://10.150.21.161:9999/` | Open5GS WebUI(호스트 9999 포트). 가입자 DB = MongoDB `open5gs.subscribers`: IMSI **999700000177992 / …994 / …996 / …998 / 999700000000001**(5개, `mongosh`로 확인). K/OPc는 읽지 않음. |
| oai+cuPHY `docker compose up -d` (sa_gnb_aerial/) | 컨테이너 2개: `nv-cubb`(L1) + `oai-gnb-aerial_hsw`(OAI gNB). 자세한 구조는 CONFIG_MAP.md §1. **compose 파일의 `~/hsw/...` 마운트는 `$HOME` 기준**이므로 troy가 돌릴 때는 `HOME=/home/nvidia docker compose …`(§6). |
| DL centric: RU `ru_config.cfg_100MHZ_and_good`, L1 `…benetal.yaml_all_success`, OAI `original.txt_success_100MHZ` | OAI TDD **7D+S(6 DL sym)+2U** (`nrofDownlinkSlots 7, nrofUplinkSlots 2`), L1 = 현재 파일과 동일(md5 d328e249). RU 파일은 접근 불가(§5). |
| UL centric: RU `…_uplink_centric`, L1 `…_uplink_heavy`, OAI `…_uplink_heavy` | OAI TDD 주기 2.5 ms **1D(8 sym)+3U**(DSUUU), PRACH idx 159, CORESET0 12, min_rxtxtime 2 등; L1은 `pusch_aggr_per_ctx 6`, `ul_input_buffer_per_cell 16`만 다름. |
| **지금 상태 (= 우리가 쓸 것)** | 세 번째 조합: `original.txt`(10-15 15:42 편집) = **TDD 6/6/3/0 = DDDDDDSUUU**(주기 5 ms, S = DL 6 sym), PRACH idx 152, e2_agent(FlexRIC) 활성; L1 = `_all_success`; RU cfg(10-15 사본) = `tdd_pattern_1=DDDD tdd_pattern_2=DDSUUU, special DDDDDDGGGGGGGG`. 10-27 22:22-23:47 KST에 이 상태로 UE가 붙어 트래픽이 흐른 L1 로그가 있다(§6). |

## 2. 읽히는 설정 파일과 값

### 2.1 L1 `cuphycontroller_config_benetal.yaml` (= `_all_success`, 실행 파일. `config_benetal.yaml`은 entrypoint의 sed 대상일 뿐 L1이 읽지 않음)
- 전역: `l2adapter_filename l2_adapter_config_P5G_GH.yaml`, `low_priority_core 10`, `dpdk_thread 10`, `workers_ul [4,5] / workers_dl [6,7,8]`, `h2d core 11`, `debug_worker -1`,
  `mps_sm pusch 120 / pucch 20 / prach 20 / ul_order 40 / pdsch 40 / pdcch 10 / pbch 2 / gpu_comms 16 / srs 0`, `fix_beta_dl 0`, `cell_group_num 1`,
  `ul_order_timeout_cpu_ns 4000000 / gpu 3000000`, `pusch_waitTimeOutPre/PostEarlyHarqUs 5000`, `max_ru_unhealthy_ul_slots 0`, `pusch_aggr_per_ctx 3`, `prach_aggr_per_ctx 2`,
  `pucch_aggr_per_ctx 4`, `ul_input_buffer_per_cell 10`, `max_harq_pools 384`, `enable_srs 0`, `log_level INFO`, tracing 모두 0, `pusch_workCancelMode`·`ul_pcap_capture_enable`·`use_batched_memcpy` 키 없음(=0).
- nics: `0002:01:00.0`(aerial00) **mtu 9000**, cpu_mbufs 196608, txq 60/rxq 20, txq_size 8192, rxq_size 16384.
- 셀(1개, `O-RU 0`, cell_id 1): `ru_type 1`, src MAC 00:…(=NIC), **dst 8c:1f:64:d1:15:11**, `vlan 1`, `pcp 0`, eAxC [0,1,2,3] 전 채널 / PRACH [4,5,6,7], BFP 9/9, `section_3_time_offset 484`,
  `fs_offset_dl 8 / exponent 4 / ref 0`, `fs_offset_ul 8 / exponent 4 / max_amp 65504`, **T1a_max_up 280000, T1a_max_cp_ul 495000, Ta4 50000-331000**, Tcp_adv_dl 125000, `ul_u_plane_tx_offset 280000`,
  stride/nMaxPrb 273, `pusch_ldpc_n_iterations 7`, `ul_gain_calibration 78.68`, `lower_guard_bw 845`, `tv_pusch cuPhyChEstCoeffs.h5`.

### 2.2 L2 어댑터 `l2_adapter_config_P5G_GH.yaml`
`slot_advance 3`, `tick_generator_mode 1`, **`allowed_fapi_latency 0`**, `allowed_tick_error 10`, timer/msg 스레드 **둘 다 core 9**(99/95), `mu_highest 1`, `dl_tb_loc 1`,
instances 20개 **모두 `prach_ta_offset_usec 2.5`**, `nvipc_config_file null`, transport shm prefix `nvipc` ring 8192, pool cpu_msg 15000×4096 / cpu_data 576000×1024 / cpu_large 4096000×64 / cuda_data 307200×0 / gpu_data 576000×0,
**`pcap_enable 1`**(core 10, 512 MB 캐시, 2 GB 파일), `prepone_h2d_copy 1`, **`enable_precoding 0`, `enable_beam_forming 0`**, dtx thresholds pusch 1.0 / pucch [-100,-100,1,1,-100], **`enableTickDynamicSfnSlot 1`**.

### 2.3 OAI `original.txt` (gnb.conf, 컨테이너 안 /opt/oai-gnb/etc/gnb.conf)
`nfapi AERIAL`, gNB_ID 0xe00, TAC 1, **PLMN 999/70 sst 1**, nr_cellid 1, `pdsch_AntennaPorts_XP 2 / N1 2`, `pusch_AntennaPorts 4`, CSI-RS/SRS 0, `min_rxtxtime 3`,
**physCellId 0**, band n78, **absoluteFrequencySSB 650016**(3750.24 MHz), pointA 646724, SCS 30 kHz, **273 PRB**, BWP 1099, CORESET0 10, **PRACH idx 152**, zeroCorr 12, preambleTargetPower -100, **TDD periodicity 6(5 ms) / 6 D / 6 sym / 3 U / 0 sym**, ssPBCH_BlockPower -11,
AMF **127.0.0.5**, NG/NGU 127.0.0.1, MACRLCs `tr_s_preference aerial`, **`tr_s_shm_prefix nvipc`**, **`tr_s_poll_core 18`**, pusch/pucch TargetSNRx10 200, max MCS 28, 보안 **nea2 / nia2**, log ngap/f1ap debug, **e2_agent 127.0.0.1 + /usr/local/lib/flexric**(RIC가 없으면 E2 SETUP 재시도 로그만 남고 무해).

### 2.4 compose / entrypoint / 이미지
- `nv-cubb`: 이미지 `nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb`, privileged, network host, **ipc shareable**, GPU all, 볼륨 `~/hsw/25_3_hsw/aerial-cuda-accelerated-ran → /opt/nvidia/cuBB`, `/var/log/aerial`, `../../../cmake_targets/share → /opt/cuBB/share`.
  entrypoint: MPS 재시작(파이프 /var, 컨테이너 내부) → `build.aarch64/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf config_benetal` → 종료 후 `pcap_collect nvipc` → `/var/log/aerial/nvipc.pcap`. healthcheck = `/tmp/phy.log`의 "L1 is ready!".
- `oai-gnb-aerial_hsw`: 이미지 `oai-gnb-aerial_hsw:${TAG}`(.env **TAG=latest**, 2026-09-03 빌드; 10-23 `ppo-273` 등 연구원 실험 태그 다수), **cpuset 13-18**, cap SYS_NICE/IPC_LOCK, **ipc container:nv-cubb**, network host, 볼륨 `./original.txt → gnb.conf`, `/usr/local/lib/flexric`(ro). healthcheck = softmodem 프로세스.
- nvidia 트리 빌드(`build.aarch64`, 2026-08-02): **`SCF_FAPI_10_04 OFF`**, `SCF_FAPI_10_04_SRS ON`, `ENABLE_L2_SLT_RSP ON`, Release. (우리 랩 빌드는 10_04 ON → OAI와 TLV 길이 불일치; `build.aarch64_f1002`가 OFF.)
- **nvidia 트리의 유일한 소스 수정**: `cuPHY-CP/cuphydriver/src/uplink/phyprach_aggr.cpp`(07-30 00:17) — 모든 PRACH occasion에 `force_thr0 = 20.0f`(cuPHY 계산 검출 임계값을 20으로 강제, `cuphy_api.h:3386`). 빌드(08-02)에 포함돼 있다. 나머지 `M`은 LFS 바이너리·노트북 데이터뿐. `baseline/phyprach_aggr_nvidia.diff`.

### 2.5 RU (`/home/troy/ru_config.cfg`, 10-15 사본; RU ssh root@10.10.0.100은 키/비밀번호 없어 **접근 불가**, ping은 됨)
`mimo_mode 1_2_3_4_4x4`, `compression static_compressed`, `prach_format short`, `prach_freq_offset_dynamic true`, DU MAC(C/U) **d8:94:24:57:7f:f2**(= aerial00), VLAN C/U-UL/U-DL **1**, `bandwidth_hz 100000000`,
**`centre_frequency_hz 3750000000`**, `tx_power_dbm 22`, **`tdd_pattern_1 DDDD`, `tdd_pattern_2 DDSUUU`, `special_slots_symbols DDDDDDGGGGGGGG`**(= DDDDDDSUUU, S = DL 6 + guard 8), `tdd_switching 1500`, `dl_tuning_special_slot 0x13b6`.

### 2.6 Open5GS (`/etc/open5gs/*.yaml`, 주석 제거본 `baseline/open5gs_{amf,smf,upf}.yaml`)
AMF: sbi 127.0.0.5:7777, **ngap 127.0.0.5**, PLMN 999/70, TAC 1, sst 1, NIA2/NEA2, 이름 open5gs-amf0. SMF/UPF: **UE 서브넷 10.45.0.0/16**(ogstun 10.45.0.1), IPv6 2001:db8:cafe::/48, **MTU 1400**, UPF GTP-U 127.0.0.7. 로그 `/var/log/open5gs/*.log`는 open5gs 그룹 전용(troy는 `sudo tail`).

## 3. DL-centric / UL-centric / 현재 상태의 차이 요약

| | DL centric | UL centric | 현재(10-15 편집, 우리 기준) |
|---|---|---|---|
| OAI TDD | 7D / S(6 DL sym) / 2U, 5 ms | 1D / S(8 DL sym) / 3U, 2.5 ms | **6D / S(6 DL sym) / 3U, 5 ms = DDDDDDSUUU** |
| OAI PRACH idx / CORESET0 / min_rxtxtime | 152 / 10 / 3 | 159 / 12 / 2 | 152 / 10 / 3 |
| OAI e2_agent | 주석 | 주석 | **활성** |
| L1 yaml | `_all_success` | `_uplink_heavy`(pusch_aggr 6, ul_input_buffer 16) | `_all_success` |
| RU cfg | `_100MHZ_and_good` | `_100MHZ_uplink_centric` | 10-15 사본 = DDDDDDSUUU |

## 4. 호스트 접근·상태 사실 (10-29)
- troy: `/home/nvidia` 읽기 가능, `sudo -n` 가능, docker 가능. nvidia 트리의 `git status`는 LFS 필터 권한 때문에 `-c filter.lfs.*=cat`로만 가능.
- `/var/log/aerial/phy.log`(root, 644)는 **nvidia 스택의 마지막 L1 로그**(UTC 표기). 10-27 13:22-14:47 UTC(22:22-23:47 KST) 런: `CONFIG.req cell_id=0 phy_cell_id=0` 수락, UE **rnti 50189 / 8190**의 HARQ pool 경고·UL 트래픽(초당 UL 수백 슬롯, CRC 수십) → **UE가 붙어 통신한 증거**; 끝은 `NVIPC memory pool is full`(OAI 쪽이 먼저 죽음).
- nvidia 계정은 10-27 저녁에도 `yolo_service`/`video_stream_service`/xApp 로그를 남겼다(`/home/nvidia/hsw/25_3_hsw/log/*` 23:46). GPU 창은 계속 조율 필요.

## 5. 접근 불가·미확인
- RU 설정 변경·열람(ssh) 불가 → RU가 지금 어느 TDD인지 라이브로는 **UL 성공 여부로만** 알 수 있다(10-15 사본과 10-27 UE 성공 런이 DDDDDDSUUU를 가리킴).
- UE 자체(기종·전원·위치)는 우리가 제어하지 못한다. `original_pixel10.txt`(10-01)로 보아 Pixel 10이 쓰였을 가능성.

## 6. 라이브 런 (2026-10-29 14:41-14:49 KST, troy가 매뉴얼대로 nvidia compose 스택 실행; 로그 scratchpad `baseline_run/`)
`HOME=/home/nvidia docker compose up -d` → nv-cubb healthy 30 s, OAI healthy. OAI: F1 Setup, **NGSetupResponse from AMF**(13:41:21 UTC), `nvIPC_Init prefix nvipc` OK, CONFIG_REQ 34 TLV, 시작 직후 1회 `sfn slot doesn't match unpacked one! L2->L1 0.0 vs L1->L2 0.3`(무해), 이후 1.28 s마다 `Frame.Slot` 출력, E2 SETUP REQUEST timeout 반복(RIC 없음).
L1: `CONFIG.req cell_id=0` → `CONFIG.response 0x0` → `START.request`; 8분 동안 **DL 50 slots/s(SSB/SIB1), UL 0 slots, PRACH 검출 0** → UE 없음(전원 꺼짐/부재). Err.ind 0, FATAL 0. `docker compose down`으로 정상 종료, nvidia 파일 무변경.
결론: 코어·L1·OAI의 기동 경로는 지금도 그대로 동작한다. UE attach는 UE를 켜 줄 사람이 필요하다.

## 7. baseline/ 스냅샷 (md5)
| 파일 | 원본 | md5 |
|---|---|---|
| cuphycontroller_config_benetal.yaml | nvidia 트리 config/ (= _all_success = /home/troy/build/benetel_orig 사본) | d328e2492645b305c55cbf19c979052f |
| cuphycontroller_config_benetal.yaml_uplink_heavy | 〃 | 8e966c7f4fed365c5ccd5d0c57b2a9e0 |
| l2_adapter_config_P5G_GH.yaml | 〃 | b1379cad26df62e56008e017c935684c |
| gnb_original.txt (현재 original.txt) / _success_100MHZ / _uplink_heavy | sa_gnb_aerial/ | 31432112… / b2cec853… / e8e14456… |
| docker-compose.yaml, aerial_l1_entrypoint.sh, compose.env(.env) | sa_gnb_aerial/ | d5a6bdcf… / ed65da7d… / 24e7a413… |
| ru_config.cfg | /home/troy/ru_config.cfg (10-15) | f369472c6966c2a0916b3a3c6a579617 |
| start_cn.sh, stop_cn.sh, routing.sh, routing_restore.sh | sh/ | deceaa1a… / cbc21055… / b2d8a65b… / 72a2562d… |
| open5gs_{amf,smf,upf}.yaml | /etc/open5gs (주석 제거) | 3a0c6b87… / f72214bb… / 355552b8… |
| phyprach_aggr_nvidia.diff | diff(우리 HEAD, nvidia 작업 트리) | 53f64673… |
