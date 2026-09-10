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
    };

    // classes: requested SM counts, e.g. {16, 32, 48, 64, 96}
    bool init(int device_ordinal, const std::vector<uint32_t>& classes, std::string& err)
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
    bool               mps_enabled_ = false;
    uint32_t           device_sm_ = 0;
    int                last_affinity_error_ = 0;
};

} // namespace sched
} // namespace dapp
} // namespace nv

#endif // DAPP_SM_POOL_HPP
