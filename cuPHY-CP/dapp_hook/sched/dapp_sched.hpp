/*
 * dapp_sched: rule-based SM budget decision from FAPI ring records.
 *
 * No model. The policy is a small, explainable formula so it can be argued
 * about and tuned by hand, and later swapped for a trained predictor by
 * replacing LoadTracker::load_index() with the model output.
 *
 * Idea:
 *   1. Each UL slot gets a load index in [0,1] derived from the UL_TTI
 *      summaries of every cell in that slot.
 *   2. A sliding window keeps the last N slots. The PEAK over the window is
 *      what matters: an inference spans several slots, so the binding
 *      constraint is the worst slot it will overlap, not the average.
 *   3. The peak maps linearly to an SM reservation for cuPHY; whatever is
 *      left, minus headroom, is offered to the inference workload.
 *
 * Fail-safe: if the ring is stale or the producer is gone, the tracker
 * reports full load, so the inference side gets its minimum. Protecting the
 * real-time RAN is always the safe direction.
 *
 * Header-only, no CUDA dependency, so the policy can be unit-tested and
 * replayed offline against a recorded ring.
 */
#ifndef DAPP_SCHED_HPP
#define DAPP_SCHED_HPP

#include "dapp_hook/dapp_ring.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

namespace nv {
namespace dapp {
namespace sched {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Config {
    // --- what "fully loaded" means, per cell ---
    uint32_t max_prb        = 273;  // PRBs in the widest carrier
    uint32_t max_layers     = 4;    // UL layers per UE group at full rank
    uint32_t max_tb_kbytes  = 200;  // TB bytes per cell per slot at full load
    uint32_t num_cells      = 4;    // cells that can carry UL in one slot

    // --- how the two cost drivers are weighted ---
    // front end: channel estimate + equalisation, scales with PRB x layers
    // back end : LDPC decode, scales with transport block bytes
    double w_front_end      = 0.5;
    double w_back_end       = 0.5;

    // --- sliding window ---
    uint32_t window_slots   = 20;      // 20 slots = 10 ms at mu=1
    int64_t  stale_after_ns = 20000000; // 20 ms with no record -> treat as stale

    // --- SM budget ---
    uint32_t total_sm       = 132;  // GH200
    uint32_t sm_granularity = 8;    // CC 9.0 splits SMs in groups of 8
    uint32_t cuphy_sm_idle  = 24;   // reserved for cuPHY even with no UL traffic
    uint32_t cuphy_sm_peak  = 108;  // reserved for cuPHY at load index 1.0
    uint32_t headroom_sm    = 8;    // never handed to either side
    uint32_t infer_sm_min   = 8;    // an inference always gets at least this
    uint32_t infer_sm_max   = 96;   // and never more than this
};

// ---------------------------------------------------------------------------
// One slot's worth of aggregated UL demand
// ---------------------------------------------------------------------------
struct SlotLoad {
    uint32_t sfn = 0, slot = 0;
    uint32_t cells = 0;          // cells that reported an UL_TTI
    uint64_t prb_layers = 0;     // sum of PRB x layers over all cells
    uint64_t tb_bytes = 0;       // sum of transport block bytes
    double   index = 0.0;        // [0,1]
    bool     enqueued = true;    // false when L1 dropped the slot command
};

// ---------------------------------------------------------------------------
// Load tracker: folds ring records into a windowed peak load index
// ---------------------------------------------------------------------------
class LoadTracker {
public:
    explicit LoadTracker(const Config& cfg) : cfg_(cfg) {}

    // Feed one record. Returns true when a slot was completed by this record.
    bool on_record(const dapp_rec_t& r)
    {
        last_rec_ns_ = static_cast<int64_t>(r.ts_ns);
        switch (r.type) {
        case DAPP_REC_UL_TTI: {
            const dapp_ul_tti_t& u = r.u.ul_tti;
            if (cur_.sfn != r.sfn || cur_.slot != r.slot) {
                cur_ = SlotLoad{};
                cur_.sfn = r.sfn;
                cur_.slot = r.slot;
            }
            cur_.cells += 1;
            cur_.prb_layers += u.tot_pusch_prb_layers;
            cur_.tb_bytes += u.tot_pusch_tb_bytes;
            return false;
        }
        case DAPP_REC_SLOT_END: {
            // The slot is complete: score it and slide the window.
            if (cur_.sfn == r.sfn && cur_.slot == r.slot) {
                cur_.enqueued = (r.u.slot_end.enqueued != 0);
                cur_.index = score(cur_);
                push(cur_);
                cur_ = SlotLoad{};
                return true;
            }
            // UL-free slot: still a data point, and a low one.
            SlotLoad empty{};
            empty.sfn = r.sfn;
            empty.slot = r.slot;
            empty.enqueued = (r.u.slot_end.enqueued != 0);
            push(empty);
            return true;
        }
        default:
            return false;
        }
    }

    // Peak load index over the window, or 1.0 (fail-safe) when data is stale.
    double peak() const
    {
        if (stale()) { return 1.0; }
        double m = 0.0;
        for (const SlotLoad& s : win_) { m = std::max(m, s.index); }
        return m;
    }

    double mean() const
    {
        if (win_.empty()) { return stale() ? 1.0 : 0.0; }
        double sum = 0.0;
        for (const SlotLoad& s : win_) { sum += s.index; }
        return sum / static_cast<double>(win_.size());
    }

    bool stale() const
    {
        if (last_rec_ns_ == 0) { return true; }
        return (now_ns() - last_rec_ns_) > cfg_.stale_after_ns;
    }

    size_t window_size() const { return win_.size(); }
    const SlotLoad* newest() const { return win_.empty() ? nullptr : &win_.back(); }
    uint64_t slots_seen() const { return slots_seen_; }
    uint64_t slots_dropped() const { return slots_dropped_; }

private:
    double score(const SlotLoad& s) const
    {
        const double fe_max = static_cast<double>(cfg_.max_prb) * cfg_.max_layers * cfg_.num_cells;
        const double be_max = static_cast<double>(cfg_.max_tb_kbytes) * 1024.0 * cfg_.num_cells;
        const double fe = fe_max > 0 ? static_cast<double>(s.prb_layers) / fe_max : 0.0;
        const double be = be_max > 0 ? static_cast<double>(s.tb_bytes) / be_max : 0.0;
        const double v = cfg_.w_front_end * fe + cfg_.w_back_end * be;
        return std::min(1.0, std::max(0.0, v));
    }

    void push(const SlotLoad& s)
    {
        win_.push_back(s);
        while (win_.size() > cfg_.window_slots) { win_.pop_front(); }
        ++slots_seen_;
        if (!s.enqueued) { ++slots_dropped_; }
    }

    Config              cfg_;
    std::deque<SlotLoad> win_;
    SlotLoad            cur_{};
    int64_t             last_rec_ns_ = 0;
    uint64_t            slots_seen_ = 0;
    uint64_t            slots_dropped_ = 0;
};

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------
struct Decision {
    double   load = 1.0;        // windowed peak load index used
    uint32_t cuphy_sm = 0;      // SMs reserved for cuPHY
    uint32_t infer_sm = 0;      // SMs offered to the inference workload
    double   infer_pct = 0.0;   // as a percentage of the device
    bool     stale = false;     // decision taken without fresh ring data
    const char* reason = "";
};

inline uint32_t quantize_down(uint32_t v, uint32_t g)
{
    return g == 0 ? v : (v / g) * g;
}

// Maps a load index to an SM budget. Deliberately linear and boring.
inline Decision decide(const Config& cfg, double load, bool stale)
{
    Decision d;
    d.load  = std::min(1.0, std::max(0.0, load));
    d.stale = stale;

    const double span = static_cast<double>(cfg.cuphy_sm_peak) - static_cast<double>(cfg.cuphy_sm_idle);
    d.cuphy_sm = static_cast<uint32_t>(cfg.cuphy_sm_idle + d.load * span + 0.5);

    uint32_t left = 0;
    const uint32_t taken = d.cuphy_sm + cfg.headroom_sm;
    if (cfg.total_sm > taken) { left = cfg.total_sm - taken; }

    left = quantize_down(left, cfg.sm_granularity);
    left = std::min(left, cfg.infer_sm_max);
    left = std::max(left, cfg.infer_sm_min);

    d.infer_sm  = left;
    d.infer_pct = 100.0 * static_cast<double>(left) / static_cast<double>(cfg.total_sm);
    d.reason    = stale ? "ring stale, assuming full cuPHY load"
                        : (d.load > 0.66 ? "heavy uplink slot in window"
                        : (d.load > 0.33 ? "moderate uplink" : "light uplink"));
    return d;
}

} // namespace sched
} // namespace dapp
} // namespace nv

#endif // DAPP_SCHED_HPP
