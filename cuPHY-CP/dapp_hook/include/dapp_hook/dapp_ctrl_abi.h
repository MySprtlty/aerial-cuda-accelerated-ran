// Control block written by the estimator (estimator/dapp_est/control_block.py, one entry per tenant
// profile) or by the dispatcher (e2e/dispatcher.py, manual / timetable policy source) and read by the
// tenant runner (dapp_sched --ctrl). Shared memory /dev/shm/aerial_dapp_ctrl, 128 bytes, ABI 2:
// the common part (slot of the decision, clock, counter) plus up to DAPP_CTRL_MAX_MODELS entries
// {name, cap_pct, gate_slots}, one per model. The estimator answers "what is safe per model";
// which model runs is the dispatcher's choice (RUN model=NAME picks the entry).
// Reader: spin until seq is even, copy, re-read seq, retry if it changed.
#pragma once
#include <stdint.h>

#define DAPP_CTRL_MAGIC "DAPPCTL1"
#define DAPP_CTRL_ABI 2u
#define DAPP_CTRL_SIZE 128u
#define DAPP_CTRL_MAX_MODELS 4u
#define DAPP_CTRL_NAME_LEN 8u
#define DAPP_CTRL_SHM_NAME "/aerial_dapp_ctrl"

struct dapp_ctrl_model {
    char     name[DAPP_CTRL_NAME_LEN];   // model name as used in RUN model=NAME, NUL padded ("26x", "26s", "26n")
    uint32_t cap_pct;                    // SM cap (% of the GPU) this model must run under; 0 = do not run
    uint32_t gate_slots;                 // slots from slot_id in which this model may start; 0 = hold
};                                       // 16 bytes

struct dapp_ctrl_block {
    char     magic[8];      // "DAPPCTL1"
    uint32_t abi;           // DAPP_CTRL_ABI (2)
    uint32_t size;          // DAPP_CTRL_SIZE (128)
    uint64_t seq;           // odd while the writer is inside an update
    uint32_t slot_id;       // sfn * slots_per_frame + slot of the "now" slot
    uint16_t sfn;
    uint16_t slot;
    uint64_t ts_ns;         // CLOCK_REALTIME ns of the decision (ring clock in live mode)
    uint64_t n_decisions;
    uint32_t n_models;      // valid entries in models[]
    uint32_t reserved;
    struct dapp_ctrl_model models[DAPP_CTRL_MAX_MODELS];   // 56 .. 120
    uint32_t reserved2[2];
};

#ifdef __cplusplus
static_assert(sizeof(struct dapp_ctrl_model) == 16, "dapp_ctrl_model size");
static_assert(sizeof(struct dapp_ctrl_block) == DAPP_CTRL_SIZE, "dapp_ctrl_block size");
#endif
