/*
 * dapp_sm_pool: a pool of CUDA contexts, each capped to a different SM count.
 *
 * Under MPS, a context created with CU_EXEC_AFFINITY_TYPE_SM_COUNT is limited
 * to that many SMs. The count is fixed at context creation, so the only way to
 * change an SM budget per request is to create several contexts up front and
 * pick one. Switching is just cuCtxSetCurrent, which costs nothing.
 *
 * This mirrors what Aerial does for its own channels in
 * cuPHY-CP/cuphydriver/src/common/mps.cpp.
 *
 * Without MPS running, cuCtxCreate_v3 rejects the affinity parameter
 * (CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY, 224). The pool then falls back to one
 * uncapped context and reports capping as inactive, so the component still
 * runs and the decision is still logged - it simply is not enforced.
 */
#ifndef DAPP_SM_POOL_HPP
#define DAPP_SM_POOL_HPP

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace nv {
namespace dapp {
namespace sched {

class SmPool {
public:
    struct Slot {
        uint32_t sm = 0;      // 0 means uncapped
        CUcontext ctx = nullptr;
#if CUDA_VERSION >= 12040
        CUgreenCtx green = nullptr;   // green mode: the partition this slot's ctx was converted from
#endif
    };

    // classes: requested SM counts, e.g. {16, 32, 48, 64, 96}
    // green (2026-10-31): CUDA green contexts instead of MPS execution-affinity contexts: every class is an SM partition
    // split from the full device set (8-SM granularity on Hopper, the count may round up), converted with cuCtxFromGreenCtx so
    // streams created while it is current are confined to the partition. No MPS needed; all classes share the primary context
    // (one address space), which is what a single cuphycontroller + YOLO process would use.
    bool init(int device_ordinal, const std::vector<uint32_t>& classes, std::string& err, bool green = false)
    {
        CUresult rc = cuInit(0);
        if (rc != CUDA_SUCCESS) { err = cu_err("cuInit", rc); return false; }
        rc = cuDeviceGet(&dev_, device_ordinal);
        if (rc != CUDA_SUCCESS) { err = cu_err("cuDeviceGet", rc); return false; }

        int sm_total = 0;
        cuDeviceGetAttribute(&sm_total, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_);
        device_sm_ = static_cast<uint32_t>(sm_total);

        int mps = 0;
#if defined(CU_DEVICE_ATTRIBUTE_MPS_ENABLED)
        cuDeviceGetAttribute(&mps, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, dev_);
#endif
        mps_enabled_ = (mps == 1);

        green_ = green;
#if CUDA_VERSION >= 12040
        if (green) {
            CUdevResource all = {};
            rc = cuDeviceGetDevResource(dev_, &all, CU_DEV_RESOURCE_TYPE_SM);
            if (rc != CUDA_SUCCESS) { err = cu_err("cuDeviceGetDevResource", rc); return false; }
            for (uint32_t sm : classes) {
                if (sm == 0 || sm > device_sm_) { continue; }
                CUdevResource part = {}, rem = {}; unsigned ng = 1;
                rc = cuDevSmResourceSplitByCount(&part, &ng, &all, &rem, 0, sm);
                if (rc != CUDA_SUCCESS) { last_affinity_error_ = static_cast<int>(rc); continue; }
                CUdevResourceDesc desc; rc = cuDevResourceGenerateDesc(&desc, &part, 1);
                if (rc != CUDA_SUCCESS) { last_affinity_error_ = static_cast<int>(rc); continue; }
                CUgreenCtx g = nullptr; rc = cuGreenCtxCreate(&g, desc, dev_, CU_GREEN_CTX_DEFAULT_STREAM);
                if (rc != CUDA_SUCCESS) { last_affinity_error_ = static_cast<int>(rc); continue; }
                CUcontext ctx = nullptr; rc = cuCtxFromGreenCtx(&ctx, g);
                if (rc != CUDA_SUCCESS) { last_affinity_error_ = static_cast<int>(rc); cuGreenCtxDestroy(g); continue; }
                Slot sl; sl.sm = part.sm.smCount; sl.ctx = ctx; sl.green = g; slots_.push_back(sl);
            }
            if (slots_.empty()) { err = "no green context could be created (error " + std::to_string(last_affinity_error_) + ")"; return false; }
            std::sort(slots_.begin(), slots_.end(), [](const Slot& a, const Slot& b) { return a.sm < b.sm; });
            capping_active_ = true;
            return true;
        }
#endif
        for (uint32_t sm : classes) {
            if (sm == 0 || sm > device_sm_) { continue; }
            CUcontext ctx = nullptr;
            CUexecAffinityParam aff;
            aff.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
            aff.param.smCount.val = sm;
            rc = cuCtxCreate_v3(&ctx, &aff, 1, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, dev_);
            if (rc != CUDA_SUCCESS) {
                last_affinity_error_ = static_cast<int>(rc);
                continue; // most likely MPS is not running
            }
            // Confirm what the driver actually granted.
            CUexecAffinityParam applied;
            uint32_t granted = sm;
            if (cuCtxGetExecAffinity(&applied, CU_EXEC_AFFINITY_TYPE_SM_COUNT) == CUDA_SUCCESS) {
                granted = applied.param.smCount.val;
            }
            cuCtxPopCurrent(nullptr);
            slots_.push_back(Slot{granted, ctx});
        }

        if (slots_.empty()) {
            // Fallback: one plain context, no capping.
            rc = cuCtxCreate(&fallback_, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, dev_);
            if (rc != CUDA_SUCCESS) { err = cu_err("cuCtxCreate", rc); return false; }
            cuCtxPopCurrent(nullptr);
            capping_active_ = false;
        } else {
            std::sort(slots_.begin(), slots_.end(),
                      [](const Slot& a, const Slot& b) { return a.sm < b.sm; });
            capping_active_ = true;
        }
        return true;
    }

    ~SmPool()
    {
#if CUDA_VERSION >= 12040
        if (green_) { for (Slot& s : slots_) { if (s.green) { cuGreenCtxDestroy(s.green); } } return; }
#endif
        for (Slot& s : slots_) { if (s.ctx) { cuCtxDestroy(s.ctx); } }
        if (fallback_) { cuCtxDestroy(fallback_); }
    }

    // Largest class whose SM count is <= budget; the smallest class if none fit.
    const Slot* pick(uint32_t budget_sm) const
    {
        if (!capping_active_) { return nullptr; }
        const Slot* best = &slots_.front();
        for (const Slot& s : slots_) { if (s.sm <= budget_sm) { best = &s; } }
        return best;
    }

    // Makes the chosen context current on the calling thread.
    bool bind(const Slot* s) const
    {
        CUcontext c = s ? s->ctx : fallback_;
        return c && cuCtxSetCurrent(c) == CUDA_SUCCESS;
    }

    bool capping_active() const { return capping_active_; }
    bool green() const { return green_; }
    bool mps_enabled() const { return mps_enabled_; }
    uint32_t device_sm() const { return device_sm_; }
    const std::vector<Slot>& slots() const { return slots_; }
    int last_affinity_error() const { return last_affinity_error_; }

private:
    static std::string cu_err(const char* what, CUresult rc)
    {
        const char* n = nullptr;
        cuGetErrorName(rc, &n);
        return std::string(what) + " failed: " + (n ? n : "?") + " (" + std::to_string((int)rc) + ")";
    }

    CUdevice           dev_ = 0;
    std::vector<Slot>  slots_;
    CUcontext          fallback_ = nullptr;
    bool               capping_active_ = false;
    bool               green_ = false;
    bool               mps_enabled_ = false;
    uint32_t           device_sm_ = 0;
    int                last_affinity_error_ = 0;
};

} // namespace sched
} // namespace dapp
} // namespace nv

#endif // DAPP_SM_POOL_HPP
