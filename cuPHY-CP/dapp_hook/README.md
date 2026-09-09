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
      dapp_ring_dump dapp_ring_fakel1 dapp_ring_selftest
```

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
```

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

The inference itself is a placeholder GPU workload. Replace `run_inference()`
in `sched/dapp_sched_main.cpp` with the TensorRT execution call; nothing around
it changes. Replacing the linear rule with a trained predictor means replacing
`decide()` alone.

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
| `python/dapp_ring.py` | Python reader and verifier |
| `tools/dapp_ring_dump.cpp` | CLI viewer |
| `tools/dapp_ring_fakel1.cpp` | record generator for offline consumer bring-up |
| `tools/dapp_ring_selftest.cpp` | ring integrity test |
| `sched/dapp_sched.hpp` | load tracker and SM budget policy (no CUDA) |
| `sched/dapp_sm_pool.hpp` | SM-capped CUDA context pool |
| `sched/dapp_sched_main.cpp` | scheduler process |
| `../scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_dapp_export.cpp` | FAPI → record conversion |
