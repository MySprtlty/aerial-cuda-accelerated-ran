/*
 * dapp_sizer.hpp (2026-11-01): the per-slot SM sizer inside the L1 process.
 *
 * The FAPI hook (scf_5g_fapi_dapp_export.cpp) already decodes every UL_TTI / DL_TTI PDU into a dapp_ul_pdu_t /
 * dapp_dl_pdu_t record; sizer_add_ul() / sizer_add_dl() turn those records into the per-kernel work units of the
 * calibrated SM model (estimator/dapp_est/work_formulas.py, ported 1:1) and accumulate them per slot. When the L2 adapter
 * closes a slot (nv_phy_module.cpp, just before l1_enqueue_phy_work) sizer_plan() picks, for PUSCH and PDSCH, the
 * smallest configured green-context profile (cuphydriver DAPP_PUSCH_PROFILES / DAPP_PDSCH_PROFILES) whose predicted
 * completion (PUSCH) or duration (PDSCH) meets the channel budget with the fitted margin (sm_model.txt, exported from
 * estimator/sm_model_fit.yaml by estimator/export_cmodel.py), hands the choice to cuphydriver
 * (cuphydriver_dapp_set_profile) and records it in g_slot_plan for the in-process YOLO tenant (dapp_tenant.hpp).
 *
 * Threads: sizer_expect() runs on the L2A msg_processing thread when a TTI request is handed to the hook; sizer_add_*()
 * runs wherever the PDU walk runs (the async export thread, or inline); sizer_plan() runs on the msg_processing thread
 * and waits (bounded) until the export thread has accounted for every submitted request of the slot.
 * Header-only, no CUDA; the ring record format is untouched.
 */
#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>

#include "dapp_ring_abi.h"

// cuphydriver_api.hpp (same signatures; declared here so the hook does not depend on cuphydriver headers)
void cuphydriver_dapp_set_profile(uint16_t sfn, uint16_t slot, uint8_t pusch_idx, uint8_t pdsch_idx);
int  cuphydriver_dapp_profile_sms(int channel, int* sm_counts, int max_count);
uint64_t cuphydriver_dapp_profile_count(int channel, int profile);
uint64_t cuphydriver_dapp_fallback_count(int channel);

namespace nv {
namespace dapp {

// ---------------------------------------------------------------------------------------------------------------------
// work units (work_formulas.py)
// ---------------------------------------------------------------------------------------------------------------------
enum { SZ_CHEST = 0, SZ_EQ, SZ_LLR, SZ_LDPC, SZ_UCI, SZ_PUSCH_K };
enum { SZ_ENC = 0, SZ_MOD, SZ_PREC, SZ_PDSCH_K };
static constexpr int    kSzSlotsPerFrame = 20;     // numerology 1 (0.5 ms slots)
static constexpr double kSzSlotUs        = 500.0;

inline void sizer_ldpc_segmentation(uint32_t tb_bytes, uint32_t rate_x10240, int& C, int& Zc, int& bg) noexcept
{
    static const int lifting[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384};
    const long A = static_cast<long>(tb_bytes) * 8;
    C = 0; Zc = 0; bg = 0;
    if (A <= 0) { return; }
    const double Rr = static_cast<double>(rate_x10240) / 10240.0;
    bg = (A <= 292 || (A <= 3824 && Rr <= 0.67) || Rr <= 0.25) ? 2 : 1;
    const long Kcb = (bg == 1) ? 8448 : 3840;
    const long B   = A + ((bg == 2 && A <= 3824) ? 16 : 24);
    long Bp;
    if (B <= Kcb) { C = 1; Bp = B; }
    else { const long L = 24; C = static_cast<int>((B + (Kcb - L) - 1) / (Kcb - L)); Bp = B + static_cast<long>(C) * L; }
    const long Kp = (Bp + C - 1) / C;
    int Kb;
    if (bg == 1) { Kb = 22; } else { Kb = (B > 640) ? 10 : (B > 560) ? 9 : (B > 192) ? 8 : 6; }
    Zc = lifting[sizeof(lifting) / sizeof(lifting[0]) - 1];
    for (int z : lifting) { if (static_cast<long>(Kb) * z >= Kp) { Zc = z; break; } }
}

// ---------------------------------------------------------------------------------------------------------------------
// per-slot accumulation table
// ---------------------------------------------------------------------------------------------------------------------
struct SizerSlot {
    std::atomic<uint32_t> key{0xFFFFFFFFu};
    std::atomic<uint32_t> expect_ul{0}, done_ul{0}, expect_dl{0}, done_dl{0};
    double   pusch[SZ_PUSCH_K] = {0, 0, 0, 0, 0};
    double   pdsch[SZ_PDSCH_K] = {0, 0, 0};
    uint32_t n_pusch = 0, n_pdsch = 0;
    uint8_t  ul_ctrl = 0, dl_ctrl = 0;     // PUCCH/PRACH/SRS resp. PDCCH/SSB/CSI-RS present
};
static constexpr unsigned kSizerSlots = 64;
inline SizerSlot g_sizer_slots[kSizerSlots];

inline uint32_t   sizer_key(uint16_t sfn, uint16_t slot) noexcept { return (static_cast<uint32_t>(sfn) << 16) | slot; }
inline SizerSlot& sizer_entry(uint16_t sfn, uint16_t slot) noexcept { return g_sizer_slots[((static_cast<uint32_t>(sfn) << 5) | (slot & 31u)) & (kSizerSlots - 1)]; }

// L2A thread: a TTI request of this slot has been handed to the hook (dir 0 = UL, 1 = DL)
inline void sizer_expect(uint16_t sfn, uint16_t slot, int dir) noexcept
{
    SizerSlot& e = sizer_entry(sfn, slot);
    const uint32_t key = sizer_key(sfn, slot);
    if (e.key.load(std::memory_order_acquire) != key) {
        e.key.store(0xFFFFFFFFu, std::memory_order_release);
        for (double& w : e.pusch) { w = 0; }
        for (double& w : e.pdsch) { w = 0; }
        e.n_pusch = 0; e.n_pdsch = 0; e.ul_ctrl = 0; e.dl_ctrl = 0;
        e.expect_ul.store(0, std::memory_order_relaxed); e.done_ul.store(0, std::memory_order_relaxed);
        e.expect_dl.store(0, std::memory_order_relaxed); e.done_dl.store(0, std::memory_order_relaxed);
        e.key.store(key, std::memory_order_release);
    }
    (dir == 0 ? e.expect_ul : e.expect_dl).fetch_add(1, std::memory_order_release);
}

// export / inline thread: one decoded PDU of the slot
inline void sizer_add_ul(uint16_t sfn, uint16_t slot, const dapp_ul_pdu_t& p) noexcept
{
    SizerSlot& e = sizer_entry(sfn, slot);
    if (e.key.load(std::memory_order_acquire) != sizer_key(sfn, slot)) { return; }
    if (p.pdu_type == DAPP_UL_PUSCH) {
        const int dmrs     = __builtin_popcount(static_cast<uint32_t>(p.ul_dmrs_sym_pos));
        const int data_sym = (p.num_sym > dmrs) ? p.num_sym - dmrs : 0;
        const double chest = static_cast<double>(p.rb_size) * 12.0 * dmrs * p.num_layers;
        const double eq    = static_cast<double>(p.rb_size) * 12.0 * data_sym * p.num_layers;
        const double llr   = eq * p.qam_mod_order;
        double ldpc = 0.0;
        if ((p.pdu_bitmap & 0x1) && p.tb_size > 0) {
            int C, Zc, bg; sizer_ldpc_segmentation(p.tb_size, p.target_code_rate, C, Zc, bg);
            int ncb = p.num_cb; if (ncb == 0) { ncb = C; }
            ldpc = (Zc > 0) ? static_cast<double>(ncb) * Zc : static_cast<double>(ncb);
        }
        const double uci = (p.pdu_bitmap & 0x2) ? static_cast<double>(p.harq_ack_bit_len) + p.csi_part1_bit_len : 0.0;
        e.pusch[SZ_CHEST] += chest; e.pusch[SZ_EQ] += eq; e.pusch[SZ_LLR] += llr; e.pusch[SZ_LDPC] += ldpc; e.pusch[SZ_UCI] += uci;
        e.n_pusch++;
    } else {
        e.ul_ctrl = 1;
    }
}
inline void sizer_add_dl(uint16_t sfn, uint16_t slot, const dapp_dl_pdu_t& p) noexcept
{
    SizerSlot& e = sizer_entry(sfn, slot);
    if (e.key.load(std::memory_order_acquire) != sizer_key(sfn, slot)) { return; }
    if (p.pdu_type == DAPP_DL_PDSCH) {
        e.pdsch[SZ_ENC]  += static_cast<double>(p.tb_size) + p.tb_size_cw1;
        e.pdsch[SZ_MOD]  += static_cast<double>(p.rb_size) * 12.0 * p.num_sym * p.num_layers;
        e.pdsch[SZ_PREC] += static_cast<double>(p.prg_bf_sum) * p.num_layers;
        e.n_pdsch++;
    } else {
        e.dl_ctrl = 1;
    }
}
// export / inline thread: the PDU walk of one TTI request is complete
inline void sizer_done(uint16_t sfn, uint16_t slot, int dir) noexcept
{
    SizerSlot& e = sizer_entry(sfn, slot);
    if (e.key.load(std::memory_order_acquire) != sizer_key(sfn, slot)) { return; }
    (dir == 0 ? e.done_ul : e.done_dl).fetch_add(1, std::memory_order_release);
}

// ---------------------------------------------------------------------------------------------------------------------
// model (sm_model.txt: "key value" lines, see estimator/export_cmodel.py)
// ---------------------------------------------------------------------------------------------------------------------
struct SizerChannel {
    bool   completion = false;               // budget kind: completion (end after t0 <= complete_by) or duration
    double s_ref = 100, fixed_us = 0, budget_us = 1e9, margin_us = 0, complete_by_us = 0;
    double start_us[kSzSlotsPerFrame];        // per slot-in-frame; NAN = use slot % 10
    double start10_us[10];
    double w_ins[5] = {0, 0, 0, 0, 0}, w_par[5] = {0, 0, 0, 0, 0};
    SizerChannel() { for (double& v : start_us) { v = NAN; } for (double& v : start10_us) { v = NAN; } }
    double start_for(int slot) const noexcept {
        const double a = start_us[slot % kSzSlotsPerFrame];
        if (!std::isnan(a)) { return a; }
        const double b = start10_us[slot % 10];
        return std::isnan(b) ? 0.0 : b;
    }
    double work_us(const double* w, int nk, double s) const noexcept {
        double t = 0.0; if (s < 1) { s = 1; }
        for (int k = 0; k < nk; k++) { if (w[k] != 0.0) { t += w[k] * (w_ins[k] + w_par[k] * s_ref / s); } }
        return t;
    }
};
struct SizerModel {
    bool         loaded = false;
    std::string  path;
    int          sm_total = 132, granularity = 8;
    SizerChannel pusch, pdsch;
    static int kernel_index(const std::string& ch, const std::string& k) noexcept {
        static const char* pk[] = {"chest", "eq", "llr", "ldpc", "uci"}; static const char* dk[] = {"enc", "mod", "prec"};
        if (ch == "PUSCH") { for (int i = 0; i < 5; i++) { if (k == pk[i]) return i; } }
        else { for (int i = 0; i < 3; i++) { if (k == dk[i]) return i; } }
        return -1;
    }
    bool load(const std::string& file, std::string& err) {
        std::ifstream f(file);
        if (!f) { err = "cannot open " + file; return false; }
        std::string line; int n = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') { continue; }
            std::istringstream is(line); std::string key, val; is >> key >> val;
            if (key.empty() || val.empty()) { continue; }
            n++;
            if (key == "sm_total") { sm_total = std::atoi(val.c_str()); continue; }
            if (key == "granularity") { granularity = std::atoi(val.c_str()); continue; }
            const size_t dot = key.find('.');
            if (dot == std::string::npos) { continue; }
            const std::string ch = key.substr(0, dot), rest = key.substr(dot + 1);
            SizerChannel* c = (ch == "PUSCH") ? &pusch : (ch == "PDSCH") ? &pdsch : nullptr;
            if (c == nullptr) { continue; }
            if (rest == "kind") { c->completion = (val == "completion"); }
            else if (rest == "s_ref") { c->s_ref = std::atof(val.c_str()); }
            else if (rest == "fixed_us") { c->fixed_us = std::atof(val.c_str()); }
            else if (rest == "budget_us") { c->budget_us = std::atof(val.c_str()); }
            else if (rest == "margin_us") { c->margin_us = std::atof(val.c_str()); }
            else if (rest == "complete_by_us") { c->complete_by_us = std::atof(val.c_str()); }
            else if (rest.rfind("start_us.", 0) == 0) { const int s = std::atoi(rest.c_str() + 9); if (s >= 0 && s < kSzSlotsPerFrame) { c->start_us[s] = std::atof(val.c_str()); } }
            else if (rest.rfind("start10_us.", 0) == 0) { const int s = std::atoi(rest.c_str() + 11); if (s >= 0 && s < 10) { c->start10_us[s] = std::atof(val.c_str()); } }
            else if (rest.rfind("w_ins.", 0) == 0) { const int k = kernel_index(ch, rest.substr(6)); if (k >= 0) { c->w_ins[k] = std::atof(val.c_str()); } }
            else if (rest.rfind("w_par.", 0) == 0) { const int k = kernel_index(ch, rest.substr(6)); if (k >= 0) { c->w_par[k] = std::atof(val.c_str()); } }
        }
        if (n == 0) { err = "no entries in " + file; return false; }
        path = file; loaded = true; return true;
    }
};
inline SizerModel g_sizer_model;

// ---------------------------------------------------------------------------------------------------------------------
// plan table for the tenant (written by sizer_plan on the L2A thread, read by the tenant thread)
// ---------------------------------------------------------------------------------------------------------------------
struct SlotPlanEntry {
    std::atomic<uint32_t> key{0xFFFFFFFFu};
    int64_t t0_ns = 0;
    int8_t  ul_prof = -1, dl_prof = -1;      // chosen profile index (-1: no PUSCH / PDSCH work)
    int16_t ul_sm = 0, dl_sm = 0;            // partition SM counts of the chosen profiles
    uint8_t ul_any = 0, dl_any = 0;          // any UL / DL channel scheduled (incl. control-only slots)
    float   pusch_t = 0, pdsch_t = 0;        // predicted completion after t0 (PUSCH) / duration (PDSCH) at the chosen profile, us
    uint8_t pusch_ok = 1, pdsch_ok = 1;
};
static constexpr unsigned kSlotPlanEntries = 128;
inline SlotPlanEntry          g_slot_plan[kSlotPlanEntries];
inline std::atomic<int64_t>   g_plan_ref_t0_ns{0};      // t0 of the newest planned slot
inline std::atomic<int64_t>   g_plan_ref_id{-1};        // its absolute slot id (sfn * 20 + slot)
inline int64_t g_plan_wrap_base = 0;                 // L2A thread only: + 1024 frames at every SFN wrap, so plan ids are monotonic
inline int64_t g_plan_last_raw = -1;
inline int64_t slot_abs_id(uint16_t sfn, uint16_t slot) noexcept { return static_cast<int64_t>(sfn) * kSzSlotsPerFrame + slot; }
// monotonic id of a planned slot (L2A thread): the raw id plus the accumulated SFN wraps
inline int64_t slot_mono_id(uint16_t sfn, uint16_t slot) noexcept
{
    const int64_t raw = slot_abs_id(sfn, slot);
    if (g_plan_last_raw >= 0 && raw + 10000 < g_plan_last_raw) { g_plan_wrap_base += 1024LL * kSzSlotsPerFrame; }
    g_plan_last_raw = raw;
    return g_plan_wrap_base + raw;
}
inline SlotPlanEntry& slot_plan_entry(int64_t abs_id) noexcept { return g_slot_plan[static_cast<uint64_t>(abs_id) & (kSlotPlanEntries - 1)]; }

// statistics (read by the tenant for its periodic log line)
struct SizerStats {
    std::atomic<uint64_t> plans{0}, incomplete{0}, pusch_slots{0}, pdsch_slots{0}, pusch_infeasible{0}, pdsch_infeasible{0};
    std::atomic<uint64_t> pusch_prof[16], pdsch_prof[16];
    std::atomic<int64_t>  wait_max_ns{0}, wait_sum_ns{0}, plan_max_ns{0}, plan_sum_ns{0};
    SizerStats() { for (auto& a : pusch_prof) { a.store(0); } for (auto& a : pdsch_prof) { a.store(0); } }
};
inline SizerStats g_sizer_stats;

struct SizerProfiles { int n_pusch = 0, n_pdsch = 0; int pusch_sm[16], pdsch_sm[16]; bool ready = false; };
inline SizerProfiles g_sizer_profiles;
inline bool    g_sizer_enabled = false;
inline int64_t g_sizer_wait_max_ns = 40000;
inline int64_t g_prev_ul_end_abs_ns = 0;     // burst check: completion of the previous U slot's PUSCH (absolute ns)

inline int64_t sizer_now_ns() noexcept { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec; }

// Reads DAPP_SIZER_MODEL (file) and DAPP_SIZER_WAIT_US; returns false (and leaves the sizer disabled) without a model.
inline bool sizer_init(std::string& err)
{
    const char* m = std::getenv("DAPP_SIZER_MODEL");
    if (m == nullptr || *m == '\0') { err = "DAPP_SIZER_MODEL not set"; return false; }
    if (!g_sizer_model.load(m, err)) { return false; }
    if (const char* w = std::getenv("DAPP_SIZER_WAIT_US")) { g_sizer_wait_max_ns = static_cast<int64_t>(std::atof(w) * 1000.0); }
    g_sizer_enabled = true;
    return true;
}
inline void sizer_refresh_profiles() noexcept
{
    g_sizer_profiles.n_pusch = cuphydriver_dapp_profile_sms(0, g_sizer_profiles.pusch_sm, 16);
    g_sizer_profiles.n_pdsch = cuphydriver_dapp_profile_sms(1, g_sizer_profiles.pdsch_sm, 16);
    g_sizer_profiles.ready   = true;
}

// smallest profile whose timed value meets the limit; the largest when none does (ok = false)
inline int sizer_pick(const SizerChannel& c, const double* work, int nk, const int* sm, int n, double extra_us, int slot, double& t_out, bool& ok) noexcept
{
    const double limit = c.completion ? (c.complete_by_us - c.margin_us) : (c.budget_us - c.margin_us);
    for (int i = 0; i < n; i++) {
        const double t = c.completion ? (c.start_for(slot) + extra_us + c.work_us(work, nk, sm[i]))
                                      : (c.fixed_us + extra_us + c.work_us(work, nk, sm[i]));
        if (t <= limit) { t_out = t; ok = true; return i; }
    }
    t_out = c.completion ? (c.start_for(slot) + extra_us + c.work_us(work, nk, sm[n - 1])) : (c.fixed_us + extra_us + c.work_us(work, nk, sm[n - 1]));
    ok = false;
    return n - 1;
}

// L2A thread, just before l1_enqueue_phy_work() of this slot. t0_ns: expected T0 of the slot (system clock ns).
inline void sizer_plan(uint16_t sfn, uint16_t slot, int64_t t0_ns) noexcept
{
    if (!g_sizer_enabled) { return; }
    if (!g_sizer_profiles.ready) { sizer_refresh_profiles(); }
    const int64_t t_start = sizer_now_ns();
    SizerSlot& e = sizer_entry(sfn, slot);
    const bool have = (e.key.load(std::memory_order_acquire) == sizer_key(sfn, slot));
    int64_t waited = 0;
    if (have) {
        // wait (bounded) for the export thread to account for every submitted request of this slot
        while (e.done_ul.load(std::memory_order_acquire) < e.expect_ul.load(std::memory_order_acquire) ||
               e.done_dl.load(std::memory_order_acquire) < e.expect_dl.load(std::memory_order_acquire)) {
            waited = sizer_now_ns() - t_start;
            if (waited > g_sizer_wait_max_ns) { g_sizer_stats.incomplete.fetch_add(1, std::memory_order_relaxed); break; }
        }
    }
    const int64_t mono_id = slot_mono_id(sfn, slot);
    SlotPlanEntry& pe = slot_plan_entry(mono_id);
    pe.key.store(0xFFFFFFFFu, std::memory_order_release);
    pe.t0_ns = t0_ns; pe.ul_prof = -1; pe.dl_prof = -1; pe.ul_sm = 0; pe.dl_sm = 0; pe.ul_any = 0; pe.dl_any = 0; pe.pusch_t = 0; pe.pdsch_t = 0; pe.pusch_ok = 1; pe.pdsch_ok = 1;
    uint8_t want_pusch = 0xFF, want_pdsch = 0xFF;
    if (have) {
        bool any_pusch = false; for (double w : e.pusch) { any_pusch = any_pusch || (w != 0.0); }
        bool any_pdsch = false; for (double w : e.pdsch) { any_pdsch = any_pdsch || (w != 0.0); }
        pe.ul_any = (any_pusch || e.ul_ctrl) ? 1 : 0;
        pe.dl_any = (any_pdsch || e.dl_ctrl) ? 1 : 0;
        if (any_pusch && g_sizer_profiles.n_pusch > 0) {
            // burst check: the previous U slot's PUSCH still running at this slot's start delays this one
            double queue_us = 0.0;
            const double start = g_sizer_model.pusch.start_for(slot);
            if (g_prev_ul_end_abs_ns != 0 && t0_ns != 0) {
                const int64_t start_abs = t0_ns + static_cast<int64_t>(start * 1000.0);
                if (g_prev_ul_end_abs_ns > start_abs) { queue_us = static_cast<double>(g_prev_ul_end_abs_ns - start_abs) / 1000.0; }
            }
            double t; bool ok;
            const int p = sizer_pick(g_sizer_model.pusch, e.pusch, SZ_PUSCH_K, g_sizer_profiles.pusch_sm, g_sizer_profiles.n_pusch, queue_us, slot, t, ok);
            pe.ul_prof = static_cast<int8_t>(p); pe.ul_sm = static_cast<int16_t>(g_sizer_profiles.pusch_sm[p]); pe.pusch_t = static_cast<float>(t); pe.pusch_ok = ok ? 1 : 0;
            want_pusch = static_cast<uint8_t>(p);
            if (t0_ns != 0) { g_prev_ul_end_abs_ns = t0_ns + static_cast<int64_t>((g_sizer_model.pusch.completion ? t : start + queue_us + t) * 1000.0); }
            g_sizer_stats.pusch_slots.fetch_add(1, std::memory_order_relaxed); g_sizer_stats.pusch_prof[p & 15].fetch_add(1, std::memory_order_relaxed);
            if (!ok) { g_sizer_stats.pusch_infeasible.fetch_add(1, std::memory_order_relaxed); }
        } else if (pe.ul_any && g_sizer_profiles.n_pusch > 0) {
            pe.ul_sm = static_cast<int16_t>(g_sizer_profiles.pusch_sm[0]);     // PUCCH / PRACH live inside the smallest PUSCH partition
        }
        if (any_pdsch && g_sizer_profiles.n_pdsch > 0) {
            double t; bool ok;
            const int p = sizer_pick(g_sizer_model.pdsch, e.pdsch, SZ_PDSCH_K, g_sizer_profiles.pdsch_sm, g_sizer_profiles.n_pdsch, 0.0, slot, t, ok);
            pe.dl_prof = static_cast<int8_t>(p); pe.dl_sm = static_cast<int16_t>(g_sizer_profiles.pdsch_sm[p]); pe.pdsch_t = static_cast<float>(t); pe.pdsch_ok = ok ? 1 : 0;
            want_pdsch = static_cast<uint8_t>(p);
            g_sizer_stats.pdsch_slots.fetch_add(1, std::memory_order_relaxed); g_sizer_stats.pdsch_prof[p & 15].fetch_add(1, std::memory_order_relaxed);
            if (!ok) { g_sizer_stats.pdsch_infeasible.fetch_add(1, std::memory_order_relaxed); }
        } else if (pe.dl_any && g_sizer_profiles.n_pdsch > 0) {
            pe.dl_sm = static_cast<int16_t>(g_sizer_profiles.pdsch_sm[0]);     // DL control channels live inside the smallest PDSCH partition
        }
    }
    cuphydriver_dapp_set_profile(sfn, slot, want_pusch, want_pdsch);
    pe.key.store(sizer_key(sfn, slot), std::memory_order_release);
    if (mono_id >= g_plan_ref_id.load(std::memory_order_relaxed) && t0_ns != 0) {   // the clock reference never moves backwards
        g_plan_ref_t0_ns.store(t0_ns, std::memory_order_relaxed);
        g_plan_ref_id.store(mono_id, std::memory_order_release);
    }
    const int64_t dt = sizer_now_ns() - t_start;
    g_sizer_stats.plans.fetch_add(1, std::memory_order_relaxed);
    g_sizer_stats.wait_sum_ns.fetch_add(waited, std::memory_order_relaxed);
    if (waited > g_sizer_stats.wait_max_ns.load(std::memory_order_relaxed)) { g_sizer_stats.wait_max_ns.store(waited, std::memory_order_relaxed); }
    g_sizer_stats.plan_sum_ns.fetch_add(dt, std::memory_order_relaxed);
    if (dt > g_sizer_stats.plan_max_ns.load(std::memory_order_relaxed)) { g_sizer_stats.plan_max_ns.store(dt, std::memory_order_relaxed); }
}

} // namespace dapp
} // namespace nv
