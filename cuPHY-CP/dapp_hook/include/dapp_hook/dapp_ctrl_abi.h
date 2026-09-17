// Control block written by the estimator (estimator/dapp_est/control_block.py)
// and read by the tenant runner. Shared memory /dev/shm/aerial_dapp_ctrl, 64 bytes.
// Reader: spin until seq is even, copy, re-read seq, retry if it changed.
#pragma once
#include <stdint.h>

#define DAPP_CTRL_MAGIC "DAPPCTL1"
#define DAPP_CTRL_ABI 1u
#define DAPP_CTRL_SIZE 64u
#define DAPP_CTRL_SHM_NAME "/aerial_dapp_ctrl"

struct dapp_ctrl_block {
    char     magic[8];      // "DAPPCTL1"
    uint32_t abi;           // DAPP_CTRL_ABI
    uint32_t size;          // DAPP_CTRL_SIZE
    uint64_t seq;           // odd while the writer is inside an update
    uint32_t gate_slots;    // slots from slot_id in which the tenant may start; 0 = hold
    uint32_t cap_pct;       // SM cap (% of the GPU) the tenant must run under
    uint32_t slot_id;       // sfn * slots_per_frame + slot of the "now" slot
    uint16_t sfn;
    uint16_t slot;
    uint64_t ts_ns;         // ring clock timestamp of the decision
    uint64_t n_decisions;
    uint32_t batch;
    uint32_t reserved;
};
