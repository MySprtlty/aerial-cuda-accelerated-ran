# gc_bench: CUDA green-context facts measured on the GH200 (2026-10-31, driver 575.57.08, CUDA 12.9)

Build (host or inside c_aerial_troy):
    nvcc -O2 -arch=sm_90 -o gc_bench gc_bench.cu -lcuda
    nvcc -O2 -arch=sm_90 -rdc=true -o gc_devgraph gc_devgraph.cu -lcuda -lcudadevrt
Run direct: ./gc_bench ; as an MPS client of the container daemon: CUDA_MPS_PIPE_DIRECTORY=/var ./gc_bench

Results (both direct and as MPS client unless noted):
- cuGreenCtxCreate ~2.2 ms, cuGreenCtxDestroy ~55 us (16-SM partition) -> contexts must be pre-created, never per slot.
- cuCtxSetCurrent + launch alternating between two green contexts: ~1.4-1.5 us per launch (host side).
- Plain kernel: the STREAM decides the SM set (stream of A under ctx B current, or under the primary ctx: still A's SMs).
- Overlapping partitions are allowed (two contexts split from the full set share SMs).
- cudaEventRecord in one green ctx + cudaStreamWaitEvent in another: works.
- CUDA graphs: the SM set is fixed when the kernel NODES are created (capture stream's ctx, or the ctx current at
  cudaGraphAddKernelNode); cudaGraphInstantiate / cudaGraphLaunch stream do NOT change it. A graph whose nodes were
  created under the primary ctx runs on all 132 SMs even when launched into a green-ctx stream.
- A remainder CUdevResource of cuDevSmResourceSplitByCount cannot be re-split directly (error 915): create a temporary
  green ctx from it and re-split cuGreenCtxGetDevResource() of that ctx (what cuphydriver's tmpGreenContextsForResplit does).
- Under CUDA_MPS_ACTIVE_THREAD_PERCENTAGE (49/50/73 tried) cuDeviceGetDevResource reports the capped SM count and every
  cuDevSmResourceSplitByCount fails with 915: green contexts and MPS active-thread caps do not combine on this driver.
- Under MPS (no cap) %smid is renumbered per context (A -> 0..31, B -> 0..23), so disjointness cannot be read from smid.
- gc_devgraph: device-side cudaGraphLaunch from a plain kernel returned cudaErrorNotSupported even under the primary ctx,
  so this harness does not reproduce cuPHY's PUSCH device graph launch (pusch_deviceGraphLaunchEn); that case is untested.

## gc_nest.cu (2026-11-01): SM layout of nested channel partitions and the tenant gap

Built in the container (`nvcc -O2 -std=c++17 -arch=sm_90 gc_nest.cu -o /tmp/gc_nest -lcuda`), run without
`CUDA_MPS_PIPE_DIRECTORY` (physical `%smid`). Reproduces cuphydriver's splits for PUSCH profiles 40/56/72/96 (remainders of
`split(all, 132 - Y)`, actual 44/60/76/100), PDSCH 40/56 (results of `split(all, N)`), DL ctrl 24, comms 16, PUCCH/PRACH (top 20
re-split 8:12), order (re-split of the largest PUSCH complement) and three constructions of the YOLO gap:
- results of `split(all, N)` are nested prefixes, remainders nested suffixes; DL ctrl / comms / order inside PDSCH 40, PUCCH /
  PRACH inside PUSCH 44, order disjoint from every PUSCH profile (all `[ok]`);
- SM ids are GPC-interleaved (PDSCH 40 = 0-9,16-25,32-41,48-57; PUSCH 44 = 70-77,84-91,98-105,112-131), so "first c of the
  remainder of `split(all, N)`" (V1) and "first |R1| - Y' of that remainder" (V4) overlap PUSCH (`[FAIL]`);
- V2 = remainder of `split(prefix(P), N)` with `prefix(P)` = the PUSCH split's result is disjoint from PDSCH_N, PUSCH_Y and
  ORDER, and its first-N result equals the global PDSCH_N (all `[ok]`). Gap sizes (P, N): (88,40) 48, (88,56) 32, (72,40) 32,
  (72,56) 16, (56,40) 16. This is what cuPHY-CP/dapp_hook/tenant uses.
