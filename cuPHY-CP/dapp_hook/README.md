# dApp FAPI hook: in-process hook + shared-memory ring

Exports every FAPI `UL_TTI.request` (optionally `DL_TTI.request`) that L2 sends
to L1, plus a per-slot outcome record, into a POSIX shared-memory ring that any
other process can read. Intended for dApps that must react to the scheduling
decision *before* the corresponding cuPHY GPU work runs, e.g. a GPU/MPS
scheduler that predicts cuPHY load and sizes the SM budget of a co-located
workload.

The hook is **observation only**. It never modifies the FAPI message, never
blocks, and takes no lock. When disabled it costs one predictable branch per
message.

## Where the hook runs

| Record | Hook site | Thread |
|---|---|---|
| `UL_PDU`, `UL_TTI` | `scf_5g_fapi::phy::on_ul_tti_request`, before the PDU→slot-command walk | `msg_processing` (SCHED_FIFO) |
| `DL_PDU`, `DL_TTI` | `scf_5g_fapi::phy::on_dl_tti_request`, after the state/validate guards | `msg_processing` |
| `SLOT_END` | `nv::PHY_module::process_phy_commands`, after `l1_enqueue_phy_work` and on the drop path | `msg_processing` |

By default (asynchronous export) the hook sites only hand the message over and
the records are written by the `dapp_export` thread; the table above is where
the message is observed (see *Inline or asynchronous export*).

### What each record carries (ABI v2)

| Record | Fields |
|---|---|
| `UL_PDU` PUSCH | rnti, handle, rb_start/size, symbols, layers, mcs_index/table, qam, code rate, tb_size, num_cb, rv, harq id, ndi, DMRS, transform precoding, BWP, UCI harq/csi1 bit lengths |
| `UL_PDU` PUCCH / PRACH / SRS | format, prb_start/size, symbols, harq/csi1/csi2 bit lengths, sr / prach format, occasions, num_ra / ant ports, symbols, repetitions, comb, bandwidth index |
| `UL_TTI` | per-cell counts and sums: PRB, layers, PRB x layers, TB bytes, code blocks, PUCCH PRB, SRS ports |
| `DL_PDU` PDSCH | rnti, fapi pdu_index, rb, symbols, layers, codewords, mcs/qam/code rate, tb_size (cw0, cw1), DMRS, **num_prgs, prg_size, dig_bf_interfaces, prg_bf_sum** |
| `DL_PDU` PDCCH | num_dl_dci, **agg_level_sum, agg_level_max, dci_payload_bits_sum, prg_bf_sum** (over its DCIs), dci_truncated |
| `DL_TTI` | per-cell counts and sums incl. **tot_pdsch_prg_bf, tot_dci_agg_level, max_dci_agg_level, tot_dci_payload_bits, tot_pdcch_prg_bf, max_pdsch_mcs** |
| `SLOT_END` | enqueued/dropped, trigger, UL/DL flags, tick_original, T0, L2+L2A latency |

The bold DL fields live in the variable-length tail of the FAPI PDU (the
precoding/beamforming block after the optional PTRS section for PDSCH; one
block per DCI for PDCCH) and were added in ABI v2. `num_cb` is exported as
received; the testMAC L2 leaves it 0, so derive it from `tb_size` when needed.
`tools/dapp_export_test.cpp` builds synthetic FAPI messages and checks all of
this, including truncated inputs, without an L1.

`UL_TTI` arrives roughly `slot_advance` slots (1.5 ms at mu=1, slot_advance 3)
before the slot's air time T0. The UL PUSCH GPU pipeline runs around T0, so a
consumer has a few milliseconds of lead time. DL GPU work is enqueued
immediately at the hook, so `DL_TTI` carries almost no lead time; it is off by
default.

`SLOT_END` reports whether the slot command actually reached cuphydriver.
L1 drops the whole slot when L2+L2A exceed `mu_to_ns(mu) + l2a_allowed_latency`
(600 us in the F08_CG1 config), in which case no GPU work happens for it.

## Build

Aerial only builds inside its container: the third-party dependencies
(fmtlog, HDF5, gRPC/protobuf, DPDK, DOCA, CUDA, ZeroMQ, ClickHouse,
prometheus-cpp) and the pinned GCC 12.3 toolchain at `/usr/local/gnu/bin`
selected by the `grace-cross` toolchain file live there, not on the host.

```bash
./cuPHY-CP/container/run_aerial.sh        # host; mounts the repo at /opt/nvidia/cuBB
# then, inside the container:
cmake -B build.aarch64 -S .               # reuses the cached toolchain and options
ninja -C build.aarch64 -j $(nproc) cuphycontroller_scf \
      dapp_ring_dump dapp_ring_fakel1 dapp_ring_selftest dapp_export_test dapp_sched
```

`dapp_sched` links TensorRT (present in the container) when CMake finds
`libnvinfer`; otherwise it builds with the placeholder kernel only.

`ENABLE_DAPP_HOOK` is ON by default. The macro is attached to the `dapp_hook`
interface target rather than added globally, so enabling or disabling it
rebuilds only `nvphy` and `scf_5g_fapi` and relinks their consumers (39 ninja
steps from a warm tree), not the whole SDK.

Compiling it in changes nothing at runtime; the ring is only created when the
L2 adapter yaml carries a `dapp_hook` block.

## Enable

Add to the l2 adapter yaml (e.g. `cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml`),
at the top level next to `fapi_config_check_mask`:

```yaml
dapp_hook:
  enable: 1
  shm_name: /aerial_dapp_ring   # -> /dev/shm/aerial_dapp_ring
  ring_len: 65536               # records, power of two; 65536 * 128 B = 8 MiB
  mlock: 1                      # pin the ring so the RT thread never page-faults
  export_ul: 1                  # UL_TTI.request (the prediction input)
  export_dl: 0                  # DL_TTI.request
  export_pdus: 1                # per-PDU records in addition to the per-cell summary
  async_export: 1               # default 1: records are written by a dedicated export thread (see below)
  export_cpu: 43                # core of that thread: isolated and used by nothing else. Required for async.
  export_prio: 90               # its SCHED_FIFO priority (default 90, below msg_processing's 95)
```

### Inline or asynchronous export

Asynchronous export is the default. It needs `export_cpu`: the export thread
polls at real-time priority and must not share a core, so a block without
`export_cpu` (e.g. a yaml written before this option existed) logs a warning
and exports inline. `async_export: 0` selects inline explicitly.

Inline, the `msg_processing` thread walks every
PDU and writes the records itself: about 100 ns per record, i.e. 130..460
records = 20..45 us per slot at 16 cells, on the path whose budget is 600 us
(tick -> slot command). At 16 cells that leaves 0..16 us of margin and slots get
dropped (`sweep/results/c16_fullstack/e_hook_cost.txt`, `async_export.txt`).

Asynchronous, the `msg_processing` thread only queues one descriptor
per TTI request (and one per SLOT_END); the `dapp_export` thread walks the
message and writes the same records, in the same order, with the record time
set to the time the hook saw the message. Cost on the L2A path: about 2.5 us
per slot at 16 cells; records land in the ring 0.8 us (max 8 us) after the hook.
The request's nvIPC buffer is kept until the export thread is done with it and
is then released by the `msg_processing` thread as before
(`include/dapp_hook/dapp_async.hpp`, `PHY_module::recv_msg`). Nothing waits:
if the queue (1024 descriptors) or the held-buffer budget (256) is full the
message is not exported and the `dropped` counter of the console line
`dApp async export: ...` (one every ~10 s) goes up. The export thread polls, so
its core shows 100 % busy. At a few cells inline also fits the budget.

Sizing: one UL slot with C cells and U PUSCH PDUs per cell produces
`C * (U + 1) + 1` records. At mu=1 with 16 cells, 2 UEs per cell and 4 UL slots
per 10, that is roughly 100k records/s, so 65536 records is about 0.6 s of
history. Consumers only need to keep up on average; overrun is detected and
reported, never silently ignored.

## Read it

```bash
build/cuPHY-CP/dapp_hook/dapp_ring_dump --stats            # header and counters
build/cuPHY-CP/dapp_hook/dapp_ring_dump --follow --pdus    # live, like tail -f
build/cuPHY-CP/dapp_hook/dapp_ring_dump --type UL_TTI,SLOT_END --max 50
python3 cuPHY-CP/dapp_hook/python/dapp_ring.py --follow
python3 cuPHY-CP/dapp_hook/python/dapp_ring.py --verify    # cross-check summaries vs PDU records
```

Test a consumer without running L1:

```bash
build/cuPHY-CP/dapp_hook/dapp_ring_fakel1 --slots 4000 --cells 4 --ue-per-cell 3 &
python3 cuPHY-CP/dapp_hook/python/dapp_ring.py --follow
```

Integrity test of the ring itself (producer/consumer, torn reads, overrun):

```bash
build/cuPHY-CP/dapp_hook/dapp_ring_selftest --records 2000000 --ring-len 4096
build/cuPHY-CP/dapp_hook/dapp_ring_selftest --records 200000 --ring-len 1024 --slow-reader
```

## dapp_sched: rule-based SM scheduler prototype

`sched/` holds a first consumer that does something with the records: it keeps
a windowed estimate of how loaded cuPHY is and, when an inference request
arrives, decides how many SMs that inference may use.

There is no model. The policy is a linear rule in `sched/dapp_sched.hpp`:

1. Each UL slot gets a load index in [0,1] from the UL_TTI summaries of every
   cell, weighting PRB x layers (front end) against transport block bytes
   (LDPC decode).
2. A sliding window keeps the last N slots and the **peak** is used, because an
   inference spans several slots and the binding constraint is the worst slot
   it overlaps.
3. The peak maps linearly onto an SM reservation for cuPHY; what is left, minus
   headroom, is offered to the inference and quantised to the 8-SM granularity
   of compute capability 9.0.

If the ring goes stale or the producer dies, the tracker reports full load and
the inference gets its minimum. Protecting the real-time RAN is the safe
direction.

```bash
# no L1 needed: synthetic records in, decisions out
build/cuPHY-CP/dapp_hook/dapp_ring_fakel1 --slots 4000 --cells 4 --ue-per-cell 3 &
build/cuPHY-CP/dapp_hook/dapp_sched --self-test 5 --cells 4

# as a service: one request per connection
build/cuPHY-CP/dapp_hook/dapp_sched -s /tmp/dapp_sched.sock
echo 'RUN 10' | nc -U /tmp/dapp_sched.sock
# OK sm=48 granted=48 pct=36.4 load=0.31 ms=12.4 capped=1 reason=moderate uplink
```

Enforcement needs MPS. A context is capped by creating it with
`CU_EXEC_AFFINITY_TYPE_SM_COUNT`, which the driver only honours under MPS, and
the count is fixed at creation - so the pool creates one context per SM class
up front and picks one per request (`sched/dapp_sm_pool.hpp`). Without MPS the
component still runs and logs its decisions, but reports
`SM capping=INACTIVE`; the phase4 run scripts start MPS as part of bringing the
RAN up.

Replacing the linear rule with a trained predictor means replacing `decide()`
alone.

### Real inference: YOLO through TensorRT

With `--engine`, dapp_sched runs a real YOLO detector (ultralytics v8/11
export) instead of the placeholder kernel. `sched/dapp_yolo.hpp` holds the
TensorRT runner plus the letterbox/NMS pre- and post-processing; images are
decoded with the vendored `third_party/stb_image.h`.

TensorRT allocates in whichever CUDA context is current when an engine is
deserialised, so dapp_sched keeps **one engine instance per SM class** (5 x
~9 MB for yolov8n). Picking a budget is then just `cuCtxSetCurrent`; nothing
is rebuilt or re-uploaded per request.

Build the model once (weights, ONNX and engines are git-ignored):

```bash
# host: CPU-only ultralytics in a venv, exports ONNX + reference detections
python3 -m venv ~/.venvs/dapp_yolo && ~/.venvs/dapp_yolo/bin/pip install ultralytics onnx onnxslim onnxruntime
~/.venvs/dapp_yolo/bin/python cuPHY-CP/dapp_hook/models/export_yolo.py --model yolov8n.pt --imgsz 640
~/.venvs/dapp_yolo/bin/python cuPHY-CP/dapp_hook/models/export_1280.py --batches 8   # YOLO26 n/s/x at 1280, batch 8 (tiled 4K)

# container, as root with the MPS pipe the RAN scripts use: ONE ENGINE PER (model, batch, SM class),
# each built under the MPS percentage cap that the driver maps to exactly that SM count (see below)
docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var -e SMS="56 80 132" -e BATCHES=8 \
  -e MODELS="yolo26n_1280 yolo26s_1280 yolo26x_1280" c_aerial_troy \
  bash /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/build_engines.sh
# -> models/yolo26x_1280_b8_fp16_sm56.engine, ..._sm80.engine, ..._sm132.engine (9 files, ~6-30 min each)
```

### Engines and SM classes

**A TensorRT engine only runs near the SM count the builder saw, and the
boundary differs per model and per build.** TensorRT picks kernels by timing
them on the device it builds on; under MPS that device is the partition, and
kernels chosen for one SM count are refused by the driver at another
(`IExecutionContext::enqueueV3: Error Code 1: Cask (Cask convolution execution)`
or `Cask (Cask Gemm execution)`, then `Cuda Runtime (invalid argument)`), which
dapp_sched reports as `warm-up inference failed` for that (model, SM, file).
Measured 2026-10-14 with the bench's 1280 batch-1 engines (built under
percentage caps 30 / 50 / 100 %, i.e. 38 / 66 / 132 SMs seen by the builder)
loaded into SM-count contexts; GPU ms = H2D + network + D2H, median of 30:

| engine (SMs at build) | 16 | 32 | 48 | 56 | 64 | 80 | 132 |
|---|---|---|---|---|---|---|---|
| 26n atp100 (132) | fail G | fail G | 1.91 | 1.69 | 1.63 | 1.52 | 1.30 |
| 26n atp50 (66) | fail G | fail G | 1.82 | 1.65 | 1.58 | 1.50 | 1.34 |
| 26n atp30 (38) | fail G | 2.19 | 1.81 | 1.65 | 1.61 | 1.50 | 1.37 |
| 26s atp100 (132) | fail C | fail C | fail C | fail C | fail C | fail C | 1.65 |
| 26s atp50 (66) | fail C | fail C | fail C | 2.28 | 2.16 | 1.97 | fail G |
| 26s atp30 (38) | fail G | fail G | 2.56 | 2.29 | 2.20 | 2.01 | 1.72 |
| 26x atp100 (132) | fail C | fail C | fail C | fail C | fail C | fail C | 4.68 |
| 26x atp50 (66) | fail C | fail C | 8.41 | 7.37 | 6.85 | fail G | fail G |
| 26x atp30 (38) | fail C | fail C | 8.37 | 7.71 | 7.19 | 5.94 | 4.93 |

(G = Cask Gemm, C = Cask convolution; the percentage cap is not the point -
the SM count is, so the same holds for the bench's percentage-cap engines.)
Rules:

- **Build one engine per SM class under the cap that maps to that class and
  name it by the SM count** (`build_engines.sh` with `SMS=...`). On the GH200 a
  percentage cap grants `floor(pct/100 * 66 TPC) * 2` SMs: 37 % = 48, 43 % =
  56, 49 % = 64, 61 % = 80, 100 % = 132; `models/mps_sm_probe.py` prints what
  the builder sees and the script puts that count into the file name.
- **Give dapp_sched the engine of every class it uses** with
  `--engine NAME:SM=file` (repeatable; several models side by side). A class
  with no exact key loads the file of the largest smaller key with a warning;
  the uncapped fallback context (no MPS) loads the largest key; nothing else
  is loaded. The warm-up inference at load is the check: a failure names the
  (model, SM, file) and the process exits.
- The earlier rule here, "build under a small cap (12 % = 16 SM) and the
  engine runs in every class", was an observation on yolov8n at 640 only
  (commit b967edd, 2026-09-10: ran at 64 SMs and above, failed at 48 and
  below when built on the full GPU). It does not hold for the YOLO26 models
  (26s / 26x built at 38 SMs fail at 16 and 32; built at 66 SMs they fail
  above as well). The bench's rule "an engine runs only under the cap it was
  built with" (`sweep/matrix.yaml`) is the same fact seen from percentage
  caps and stays.
- TensorRT warns `Using an engine plan file across different models of
  devices` whenever the builder saw a different SM count; that is expected.

Check the C++ pre/post-processing against ultralytics and see the effect of
the cap with a sweep (MPS must be up, i.e. the RAN scripts have run or
`nvidia-cuda-mps-control -d` was started as root):

```bash
docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var c_aerial_troy \
  /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/dapp_hook/dapp_sched \
  --engine v8n:16=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/yolov8n_fp16_sm16.engine \
  --image  /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/bus.jpg --sweep 30 \
  | python3 cuPHY-CP/dapp_hook/tools/yolo_check.py cuPHY-CP/dapp_hook/models/reference_detections.json bus.jpg
# (one key, 16: every class 16..96 loads that file - the yolov8n case where that works)
# [sweep] SM= 16  median=1.13 ms  min=.. max=..  dets=5  mean=..  p99=..  net=mean/p99  h2d=..  d2h=..  dec=0.00  n=28  model=v8n
```

Measured on the GH200 (yolov8n, 640x640, FP16, GPU time incl. H2D/D2H):

| SM class | 16 | 32 | 48 | 64 | 96 | 132 |
|---|---|---|---|---|---|---|
| median ms | 1.13 | 0.97 | 0.90 | 0.86 | 0.84 | 0.83 |

The detections match ultralytics' ONNX output on the sample images
(`YOLO CHECK PASS`). yolov8n is launch-bound at this size, which is why the
curve is flat; a larger model or batch scales more steeply, and the same
export script produces it (`--model yolov8m.pt`).

Serving on the live ring, with a socket on `/dev/shm` so host-side clients
can reach a process inside the container:

```bash
docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var c_aerial_troy \
  /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/dapp_hook/dapp_sched \
  --engine v8n:16=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/yolov8n_fp16_sm16.engine \
  --image  /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/bus.jpg -s /dev/shm/dapp_sched.sock --cells 1
python3 cuPHY-CP/dapp_hook/tools/dapp_sched_client.py -s /dev/shm/dapp_sched.sock RUN            # default image, default model
python3 cuPHY-CP/dapp_hook/tools/dapp_sched_client.py -s /dev/shm/dapp_sched.sock -m v8n RUN /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/zidane.jpg
# OK sm=64 granted=64 pct=48.5 load=0.117 ms=0.87 pre_ms=0.0 dets=5 top=person:0.89,person:0.88,person:0.88 capped=1 reason=light uplink model=v8n net=0.80 dec=0.00
```

The request line is `RUN [model=NAME] [image]` (`-m` in the client); without
`model=` the first `--engine` model is used. `ms` is the GPU time of that
request (h2d + net + d2h on the engine stream), `net` the network alone,
`dec` the nvjpeg decode (0 on the CPU path), `pre_ms` the CPU decode/letterbox
or file read (0 when the image is cached), `dets`/`top` the detections. Image
paths are resolved by the dapp_sched process, i.e. inside the container.

### Tiled 4K inference (batch-N engines)

A batch-N engine (`yolo26x_1280_b8_fp16_sm132.engine`) is used TILED, the way
the E2E server (`e2e/infer_server.py`) does it: the JPEG is decoded on the
GH200's NVJPG hardware engines straight into the engine's device input
(`e2e/nvjpeg_dec.cu`, compiled into dapp_sched; no H2D copy): N-1 native
1280x1280 crops on a grid (3840x2160 -> 3 x 2 = 6 tiles) plus the whole image
letterboxed in the remaining slots, then the boxes of all slots are mapped
back through their crop and merged by one class-aware NMS. One nvjpeg decoder
lives in every (model, class) context. Needs nvjpeg at build time (CMake
prints `nvjpeg tiled decode enabled`); baseline JPEGs only - the hardware
decoder refuses progressive ones. `--decode gpu` puts batch-1 engines on the
same path (whole image letterboxed on the GPU).

```bash
docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var c_aerial_troy \
  /opt/nvidia/cuBB/build.aarch64/cuPHY-CP/dapp_hook/dapp_sched --classes 56,80,132 \
  --engine 26x:56=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/yolo26x_1280_b8_fp16_sm56.engine \
  --engine 26x:80=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/yolo26x_1280_b8_fp16_sm80.engine \
  --engine 26x:132=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/yolo26x_1280_b8_fp16_sm132.engine \
  --image /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/img4k/uhd_0_q85.jpg --sweep 102 --sweep-skip 2
```

`--sweep` prints, per (model, class), `median / min / max / mean / p99` of the
GPU time, `net=mean/p99`, `h2d`, `d2h` and `dec` means and the detections of
the first class; the first `--sweep-skip` runs (default 2) are left out of the
statistics.

**Output heads.** Two engine outputs are accepted (`dapp_yolo.hpp`): the raw
one-to-many head `[N,4+nc,A]` (class max, threshold and class-aware NMS on the
CPU, ~8 ms per 4K tiled request after the row-wise scan) and the YOLO26
**end-to-end (NMS-free) head `[N,K,6]`** = `x1,y1,x2,y2,score,class` per image
(`export_1280.py --nms-free`, ultralytics `nms=False`, `max_det` 300; engines
`*_b8_e2e_fp16_sm<N>.engine` via `ONNX_SUFFIX=_e2e build_engines.sh`). With the
end-to-end head the CPU only thresholds, maps each slot's boxes through its
ROI and merges the duplicates that different tiles see of the same object
(class-aware NMS between slots only): ~0.05 ms per request. The log line of
every engine says `raw head` or `end2end head`.

Measured 2026-10-15, batch-8 SM-class engines, 4K street frame (uhd_0_q85.jpg,
3840x2160 -> 6 tiles + 2 global views), MPS on, no L1, 100 runs per cell
(first 2 excluded), each class with the engine built for it:

| model | SM | gpu ms mean / p99 | net ms mean / p99 | d2h | dec | dets |
|---|---|---|---|---|---|---|
| 26n | 56 | 6.55 / 6.57 | 6.24 / 6.25 | 0.31 | 5.33 | 68 |
| 26n | 80 | 5.19 / 5.21 | 4.88 / 4.90 | 0.31 | 5.26 | 68 |
| 26n | 132 | 4.09 / 4.11 | 3.78 / 3.80 | 0.31 | 5.21 | 68 |
| 26s | 56 | 10.83 / 10.86 | 10.52 / 10.55 | 0.31 | 5.33 | 90 |
| 26s | 80 | 8.45 / 8.48 | 8.14 / 8.17 | 0.31 | 5.26 | 90 |
| 26s | 132 | 6.39 / 6.40 | 6.07 / 6.09 | 0.31 | 5.21 | 89 |
| 26x | 56 | 42.03 / 42.15 | 41.72 / 41.84 | 0.31 | 5.34 | 91 |
| 26x | 80 | 31.71 / 31.77 | 31.39 / 31.46 | 0.31 | 5.26 | 91 |
| 26x | 132 | 23.78 / 24.31 | 23.47 / 23.99 | 0.31 | 5.22 | 91 |

`dec` (NVJPG decode + tiles, ~5 ms) is outside `gpu ms` and independent of the
SM class; the C++ post-processing is 1-3 ms on the CPU. The detections equal
those of `e2e/infer_server.py` with the same engine and image (count and top
boxes). Cross-class compatibility of these engines: the sm56 engines also run
at 48 and 132 but not at 64 / 80, the sm80 engines of 26s / 26x only at 80 and
132, the sm132 engines only at 132 - which is why every class gets its own.

GPU memory (nvidia-smi, MPS): the SM-class contexts alone cost 284 / 370 /
562 MiB for 56 / 80 / 132 SMs (1,218 MiB for the three); one (model, class)
engine instance adds ~0.68 GB (26n), ~1.0 GB (26s), ~2.14 GB (26x) on top of
its context; all nine instances in one process = 12.2 GB. The 16-cell 59c
cuphycontroller holds 37.9 GB of the 97.9 GB, so cuPHY + the nine engines +
contexts use ~51 %.

## Consuming it from your own process

C/C++: include `dapp_hook/dapp_ring.hpp` and use `nv::dapp::Consumer`.
Python: `dapp_ring.DappRing` (numpy + mmap, no Aerial dependency).
Any language: the layout is plain C in `include/dapp_hook/dapp_ring_abi.h`.

```cpp
auto c = nv::dapp::Consumer::open("/aerial_dapp_ring", err);
dapp_rec_t r;
while (running) {
    uint64_t lost = 0;
    int st = c->next(r, &lost);
    if (st == nv::dapp::Consumer::NONE)      { /* poll again */ continue; }
    if (st == nv::dapp::Consumer::RESTARTED) { /* L1 restarted */ continue; }
    if (r.type == DAPP_REC_UL_TTI) { predict(r.sfn, r.slot, r.cell_id, r.u.ul_tti); }
}
```

Rules for a consumer:

- Never write to the ring. It is mapped read-only and there is exactly one writer.
- Poll; there is no wakeup mechanism. A 100–200 us poll on a spare core keeps up.
- Handle `RESTARTED` (L1 restarted, generation bumped) and `lost` (you fell behind).
- Records are a broadcast log: many independent consumers can read the same ring.

## Lifetime

The ring is a file in `/dev/shm`, so its contents survive an L1 crash and can be
inspected afterwards. On restart L1 re-initialises the same object, bumps
`generation`, and consumers see `RESTARTED`. Remove it with
`rm /dev/shm/aerial_dapp_ring` when you no longer need it.

## Files

| Path | Role |
|---|---|
| `include/dapp_hook/dapp_ring_abi.h` | shared-memory layout (plain C, static-asserted) |
| `include/dapp_hook/dapp_ring.hpp` | header-only producer/consumer |
| `include/dapp_hook/dapp_async.hpp` | asynchronous export: SPSC descriptor queue, export thread, deferred buffer release |
| `python/dapp_ring.py` | Python reader and verifier |
| `tools/dapp_ring_dump.cpp` | CLI viewer |
| `tools/dapp_ring_fakel1.cpp` | record generator for offline consumer bring-up |
| `tools/dapp_ring_selftest.cpp` | ring integrity test |
| `tools/dapp_export_test.cpp` | FAPI -> record conversion test on synthetic messages, incl. the asynchronous path |
| `sched/dapp_sched.hpp` | load tracker and SM budget policy (no CUDA) |
| `sched/dapp_sm_pool.hpp` | SM-capped CUDA context pool |
| `sched/dapp_yolo.hpp` | TensorRT YOLO runner, letterbox and NMS |
| `sched/dapp_sched_main.cpp` | scheduler process |
| `models/export_yolo.py` | YOLO -> ONNX export plus reference detections (outputs git-ignored) |
| `tools/yolo_check.py` | compares dapp_sched detections with the ultralytics reference |
| `tools/dapp_sched_client.py` | socket client for dapp_sched (no `nc` needed) |
| `third_party/stb_image.h`, `stb_image_impl.cpp` | vendored image decoder (public domain) |
| `../scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_dapp_export.cpp` | FAPI → record conversion |
