# seg_yolo: YOLO inference in sub-0.5 ms segments, each on the green-context SM partition chosen for the current slot (2026-10-31)

Goal: a YOLO tenant whose SM share can change every slot (0.5 ms) in the same process as cuphycontroller, Beaver-style
(arXiv 2610.02522), without touching TensorRT internals. TensorRT enqueues a whole engine at once and links the CUDA runtime
statically, so the only switch points are engine boundaries: the network is cut into segments, every segment is built once per
SM class (engines refuse other SM counts), and a runner enqueues segment k just in time on the stream of the class the slot clock
selects.

## Files
- `seg_split.py` (host venv `/home/troy/.venvs/dapp_yolo/bin/python`): per-layer times from `trtexec --exportProfile` of the full
  engine (attributed to ONNX nodes; Myelin/reformat layers spread between neighbours) -> greedy bins of `--bin-ms` at cuts whose
  frontier has <= `--max-frontier` tensors -> `onnx.utils.extract_model` per segment, `manifest.json` (true producer/consumer
  inputs/outputs only: a pass-through tensor declared as an output makes the TensorRT parser insert an identity with a renamed
  `__<name>` input and breaks name-keyed buffers).
- engine build: trtexec per (segment, class) under the MPS active-thread cap of the class (31 % = 40 SMs, 43 % = 56, 67 % = 88,
  none = 132), `--fp16`, I/O fp16 in `hwc8` for 4-D tensors with C % 8 == 0 (TensorRT's internal fp16 layout: no reformat kernels at
  the boundary; `chw` for the 3-channel image and the head tensors). Scripts used: session scratchpad seg_build2.sh / seg_build_fine.sh.
- `seg_runner.cu` (nvcc inside c_aerial_troy: `-I/usr/include/aarch64-linux-gnu -l:libnvinfer.so.10 -lcuda -lcudart`): green
  context + stream per class, one device buffer per tensor name, just-in-time enqueue (`--lookahead` segments in flight) on the
  class of `--schedule`[slot % 10] with the slot clock `--slot-us`, cross-stream events at switches, per-segment GPU times, and a
  detection-level comparison with the unsegmented engine (`--full`, `--input` real letterboxed image, fp16 CHW batch 8).

## Results, yolo26n 1280 batch 8 e2e, GH200 (models/seg26n: 18 segments, 0.35 ms bins at the 56-SM profile)
| run | latency p50 | note |
|---|---|---|
| full engine 132 / 88 / 40 SMs | 3.78 / 4.64 / 7.98 ms | reference |
| segmented, fixed 132 / 88 / 40 | 4.45 / 5.37 / 9.02 ms | +18 / +16 / +13 %; detections 192 = 192 matched (IoU >= 0.993) |
| slot schedule pos 7-9 -> 40, else 88 | 6.21 ms (p90 7.1) | 2.4 class switches per inference |
| slot schedule pos 7-9 -> 40, else 132 | 5.44 ms (p90 5.9) | 2.0 switches |
Segment GPU time: 132 SMs max 0.33 ms, 88 SMs max 0.39 ms (all < 0.5), 40 SMs 0.34-0.79 ms (8 of 18 above 0.5 -> finer split
`seg26n_fine`, 0.22 ms bins). First build with chw boundaries cost +37-40 % (reformat kernels); hwc8 boundaries +13-18 %.
Just-in-time enqueue adds ~45 us host time per segment (sum of segment GPU medians 5.37 ms vs 6.21 ms latency).

### Final iteration (models/seg26n_fine: 30 segments, 0.25 ms bins on the 40-SM layer profile, --max-frontier 12 --allow-int)
| run | latency p50 | segments > 0.5 ms |
|---|---|---|
| fixed 40 SMs | 9.58 ms (full 7.99, +20 %) | 2 of 30: seg03 0.68, seg04 0.54 (model.2 block on 320x320 maps: a standalone conv there is ~2x the full-engine layer time) |
| fixed 88 SMs | 5.69 ms (full 4.67, +22 %) | 0 (max 0.32) |
| slot schedule pos 7-9 -> 40, else 88 | 6.35 ms (p90 7.2), 2.7 switches | detections 192 = 192 matched |
Floor at 40 SMs: longest single layer of the full engine 0.40 ms (Myelin head group); the two early-backbone segments need per-node
cuts (or a >= 48-SM UL-window class). Int64 boundaries (TopK indices) work with int64:chw I/O formats and dtype-aware buffers.
Per-segment host cost of just-in-time enqueue ~25-45 us (sum of GPU medians 5.63 vs latency 6.35 ms).
