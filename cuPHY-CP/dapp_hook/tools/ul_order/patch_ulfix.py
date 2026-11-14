#!/usr/bin/env python3
# patch_ulfix.py <tree root> [--check]   (2026-11-14)
# PUSCH in-order gate (F1) + counters (D0) + forced CRC failure for failed-setup slots (F3b-min), cuphydriver only.
# Exact-string replacements; every anchor must occur exactly once (inside its scope). --check only verifies the anchors.
import sys, os
root = sys.argv[1]; check = '--check' in sys.argv
D = os.path.join(root, 'cuPHY-CP/cuphydriver')
P = []   # (file, scope_start or None, scope_end or None, old, new)

# ---------------------------------------------------------------- include/context.hpp
P.append(('include/context.hpp', None, None,
'''    void                    setPuschAggrFactor(uint8_t pusch_aggr_factor_);

private:
''',
'''    void                    setPuschAggrFactor(uint8_t pusch_aggr_factor_);

    // PUSCH in-order gate (2026-11-14). Every UL slot with PUSCH gets a sequence number in l1_enqueue_phy_work (slot order).
    // task_work_function_ul_aggr_3 hands the PUSCH results (CRC / RX_DATA) of sequence s to L2 only after s-1 was handed over or
    // given up, so two consecutive UL slots can no longer report in the wrong order when their GPU work finishes out of order.
    // Env: DAPP_PUSCH_INORDER=0/1 (default 1 with dApp PUSCH profiles, else 0), DAPP_PUSCH_INORDER_US (longest wait, default 3000).
    static constexpr uint64_t UL_INORDER_RING = 64;
    int                   ul_inorder_gate = 0;               ///< 1 = hold a PUSCH callback until the previous PUSCH slot reported
    int64_t               ul_inorder_gate_ns = 3000000;      ///< longest hold; after it the slot reports anyway (counted as timeout)
    std::atomic<uint64_t> ul_inorder_next{0};                ///< last sequence number handed out (0 = none)
    std::atomic<uint64_t> ul_inorder_wm{0};                  ///< every sequence <= wm has reported or was given up
    std::atomic<uint64_t> ul_inorder_done[UL_INORDER_RING];  ///< entry s % RING holds s once s is done
    std::atomic<uint64_t> ul_inorder_last_cb{0};             ///< highest sequence whose callback started (out-of-order detection)
    std::atomic<uint64_t> ul_inorder_n_cb{0}, ul_inorder_n_wait{0}, ul_inorder_n_timeout{0}, ul_inorder_n_ooo{0}, ul_inorder_n_setup_failed{0};
    std::atomic<int64_t>  ul_inorder_wait_ns_sum{0}, ul_inorder_wait_ns_max{0}, ul_inorder_log_ns{0};
    std::atomic<uint64_t> ul_inorder_logged_hard{0}, ul_inorder_logged_soft{0};
    uint64_t ulInorderNextSeq() { return ul_inorder_next.fetch_add(1) + 1; }
    bool     ulInorderPredDone(uint64_t seq) const { return seq <= ul_inorder_wm.load() + 1; }
    void     ulInorderAdvance() {
        uint64_t w = ul_inorder_wm.load();
        for(;;) {
            const uint64_t nx = w + 1;
            if(ul_inorder_done[nx % UL_INORDER_RING].load() != nx) break;
            if(ul_inorder_wm.compare_exchange_weak(w, nx)) w = nx;
        }
    }
    void     ulInorderMarkDone(uint64_t seq) {               ///< idempotent; seq 0 = slot without PUSCH
        if(seq == 0) return;
        ul_inorder_done[seq % UL_INORDER_RING].store(seq);
        ulInorderAdvance();
    }
    void     ulInorderGiveUpPred(uint64_t seq) {             ///< stop waiting for everything before seq
        uint64_t w = ul_inorder_wm.load();
        while(w + 1 < seq && !ul_inorder_wm.compare_exchange_weak(w, seq - 1)) {}
        ulInorderAdvance();
    }
    /// Bookkeeping right before the PUSCH callback of sequence seq.
    void     ulInorderOnCallback(uint64_t seq, bool waited, int64_t wait_ns, bool expired, bool setup_failed) {
        if(seq == 0) return;
        if(expired) ul_inorder_n_timeout.fetch_add(1);
        if(!ulInorderPredDone(seq)) ulInorderGiveUpPred(seq);
        uint64_t prev = ul_inorder_last_cb.load();
        for(;;) {
            if(seq < prev) { ul_inorder_n_ooo.fetch_add(1); break; }
            if(ul_inorder_last_cb.compare_exchange_weak(prev, seq)) break;
        }
        ul_inorder_n_cb.fetch_add(1);
        if(waited) {
            ul_inorder_n_wait.fetch_add(1); ul_inorder_wait_ns_sum.fetch_add(wait_ns);
            int64_t m = ul_inorder_wait_ns_max.load();
            while(wait_ns > m && !ul_inorder_wait_ns_max.compare_exchange_weak(m, wait_ns)) {}
        }
        if(setup_failed) ul_inorder_n_setup_failed.fetch_add(1);
    }
    /// true when a summary line is due: 1 s after a timeout / out-of-order report / failed setup, 10 s after a plain wait.
    bool     ulInorderLogDue(int64_t now_ns) {
        const uint64_t hard = ul_inorder_n_timeout.load() + ul_inorder_n_ooo.load() + ul_inorder_n_setup_failed.load();
        const uint64_t soft = ul_inorder_n_wait.load();
        int64_t last = ul_inorder_log_ns.load();
        const bool due = (hard != ul_inorder_logged_hard.load() && now_ns - last >= 1000000000LL) ||
                         (soft != ul_inorder_logged_soft.load() && now_ns - last >= 10000000000LL);
        if(!due || !ul_inorder_log_ns.compare_exchange_strong(last, now_ns)) return false;
        ul_inorder_logged_hard.store(hard); ul_inorder_logged_soft.store(soft);
        return true;
    }

private:
'''))

# ---------------------------------------------------------------- src/common/context.cpp
P.append(('src/common/context.cpp', None, None,
'''        if(!dapp_pdsch_profiles.empty()) { NVLOGC_FMT(TAG, "dApp PDSCH profiles [{}]: yaml mps_sm_pdsch {} replaced by the largest", list_str(dapp_pdsch_profiles), mps_sm_pdsch); mps_sm_pdsch = dapp_pdsch_profiles.back(); }
    }
''',
'''        if(!dapp_pdsch_profiles.empty()) { NVLOGC_FMT(TAG, "dApp PDSCH profiles [{}]: yaml mps_sm_pdsch {} replaced by the largest", list_str(dapp_pdsch_profiles), mps_sm_pdsch); mps_sm_pdsch = dapp_pdsch_profiles.back(); }
        // PUSCH in-order gate (2026-11-14, see context.hpp): on by default only when dApp PUSCH profiles are in use.
        const char* g_env = std::getenv("DAPP_PUSCH_INORDER");
        ul_inorder_gate = (g_env != nullptr) ? (std::atoi(g_env) != 0 ? 1 : 0) : (dapp_pusch_profiles.empty() ? 0 : 1);
        const char* g_us = std::getenv("DAPP_PUSCH_INORDER_US");
        if(g_us != nullptr && std::atoi(g_us) > 0) ul_inorder_gate_ns = static_cast<int64_t>(std::atoi(g_us)) * 1000;
        for(auto& d : ul_inorder_done) d.store(0);
        NVLOGC_FMT(TAG, "PUSCH in-order gate {} (wait limit {} us)", ul_inorder_gate ? "ON" : "OFF", ul_inorder_gate_ns / 1000);
    }
'''))

# ---------------------------------------------------------------- include/slot_map_ul.hpp
P.append(('include/slot_map_ul.hpp', None, None,
'''    PhyPuschAggr * aggr_pusch;                                  ///< PUSCH aggregator pointer (nullptr if not scheduled)
''',
'''    PhyPuschAggr * aggr_pusch;                                  ///< PUSCH aggregator pointer (nullptr if not scheduled)
    uint64_t       ul_pusch_seq = 0;                            ///< PUSCH in-order gate: sequence number of this slot (0 = no PUSCH)
'''))

# ---------------------------------------------------------------- src/uplink/slot_map_ul.cpp
P.append(('src/uplink/slot_map_ul.cpp', None, None,
'''    if((num_active_cells > 0) && (prev_cells + num_cells < num_active_cells))
        return 0;

    bool isSrs = false;
''',
'''    if((num_active_cells > 0) && (prev_cells + num_cells < num_active_cells))
        return 0;

    // PUSCH in-order gate: whatever happened to this slot, later slots must not wait for it any more
    pdctx->ulInorderMarkDone(ul_pusch_seq);
    ul_pusch_seq = 0;

    bool isSrs = false;
'''))

# ---------------------------------------------------------------- src/common/cuphydriver_api.cpp
P.append(('src/common/cuphydriver_api.cpp', None, None,
'''                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SlotMapUL aggrSetPhy");
                goto cleanup_err;
            }
            //Check for Exit handler in flight and skip enqueing of UL tasks accordingly
''',
'''                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SlotMapUL aggrSetPhy");
                goto cleanup_err;
            }
            // PUSCH in-order gate: slot commands arrive in slot order, so this numbers the UL slots that carry PUSCH
            slot_map_ul->ul_pusch_seq = (aggr_pusch_ptr != nullptr) ? pdctx->ulInorderNextSeq() : 0;
            //Check for Exit handler in flight and skip enqueing of UL tasks accordingly
'''))

# ---------------------------------------------------------------- src/uplink/task_function_ul_aggr.cpp  (task_work_function_ul_aggr_3 only)
A3 = 'int task_work_function_ul_aggr_3(Worker* worker, void* param, int first_cell, int num_cells,int num_ul_tasks)\n{'
A3E = 'int task_work_function_ul_aggr_3_srs(Worker* worker, void* param, int first_cell, int num_cells,int num_ul_tasks)'
P.append(('src/uplink/task_function_ul_aggr.cpp', A3, A3E,
'''    bool pusch_reorder_kernel_timeout_state = false;
    bool pusch_terminating_state = false;
''',
'''    bool pusch_reorder_kernel_timeout_state = false;
    bool pusch_terminating_state = false;
    // PUSCH in-order gate (2026-11-14, see context.hpp)
    const uint64_t ul_seq = (pusch != nullptr) ? slot_map->ul_pusch_seq : 0;
    bool pusch_cb_pending = false, pusch_cb_waited = false;
    t_ns pusch_cb_pending_t(0);
'''))
P.append(('src/uplink/task_function_ul_aggr.cpp', A3, A3E,
'''                    case WAIT_ACTION_COMPLETED:
                        slot_map->timings.end_t_ul_pusch_compl[0] = Time::nowNs();

                        TI_ADD("Validate PUSCH");
                        pusch->validate(cell_timeout_list,gpu_early_harq_timeout,slot_map->aggr_ulbuf_pcap_capture, slot_map->aggr_ulbuf_pcap_capture_ts);

                        TI_ADD("Callback PUSCH");
                        slot_map->timings.start_t_ul_pusch_cb[0] = Time::nowNs();
                        pusch->callback(cell_timeout_list,gpu_early_harq_timeout);
                        slot_map->timings.end_t_ul_pusch_cb[0] = Time::nowNs();

                        TI_ADD("Done PUSCH");
                        break;
                }
            }

            //PUCCH Wait Processing
''',
'''                    case WAIT_ACTION_COMPLETED:
                        slot_map->timings.end_t_ul_pusch_compl[0] = Time::nowNs();
                        // validate + callback run below, once the previous PUSCH slot has reported (PUSCH in-order gate)
                        pusch_cb_pending = true;
                        pusch_cb_pending_t = slot_map->timings.end_t_ul_pusch_compl[0];
                        break;
                }
            }

            // PUSCH in-order gate (2026-11-14): the PUSCH results of consecutive UL slots go to L2 in slot order. A slot whose GPU
            // work finished before the previous PUSCH slot reported is held here (PUCCH / PRACH / SRS below are not held), at most
            // ul_inorder_gate_ns. With the gate off the callback runs at once, as before; the counters still record the order.
            if(pusch_cb_pending)
            {
                const int64_t held_ns = (Time::nowNs() - pusch_cb_pending_t).count();
                const bool pred_done = (ul_seq == 0) || pdctx->ulInorderPredDone(ul_seq);
                const bool expired = (!pred_done) && pdctx->ul_inorder_gate && (held_ns > pdctx->ul_inorder_gate_ns);
                if(pred_done || !pdctx->ul_inorder_gate || expired)
                {
                    TI_ADD("Validate PUSCH");
                    pusch->validate(cell_timeout_list,gpu_early_harq_timeout,slot_map->aggr_ulbuf_pcap_capture, slot_map->aggr_ulbuf_pcap_capture_ts);

                    TI_ADD("Callback PUSCH");
                    pdctx->ulInorderOnCallback(ul_seq, pusch_cb_waited, held_ns, expired, pusch->getSetupStatus() != CH_SETUP_DONE_NO_ERROR);
                    slot_map->timings.start_t_ul_pusch_cb[0] = Time::nowNs();
                    pusch->callback(cell_timeout_list,gpu_early_harq_timeout);
                    slot_map->timings.end_t_ul_pusch_cb[0] = Time::nowNs();

                    TI_ADD("Done PUSCH");
                    pdctx->ulInorderMarkDone(ul_seq);
                    pusch_cb_pending = false;
                    if(ul_seq != 0 && pdctx->ulInorderLogDue(slot_map->timings.end_t_ul_pusch_cb[0].count()))
                    {
                        NVLOGC_FMT(TAG, "PUSCH in-order gate: slots {} held {} (sum {} us, max {} us) timeout {} out-of-order {} setup-failed {}",
                                   pdctx->ul_inorder_n_cb.load(), pdctx->ul_inorder_n_wait.load(), pdctx->ul_inorder_wait_ns_sum.load() / 1000,
                                   pdctx->ul_inorder_wait_ns_max.load() / 1000, pdctx->ul_inorder_n_timeout.load(), pdctx->ul_inorder_n_ooo.load(),
                                   pdctx->ul_inorder_n_setup_failed.load());
                    }
                }
                else
                {
                    pusch_cb_waited = true;
                }
            }

            //PUCCH Wait Processing
'''))
P.append(('src/uplink/task_function_ul_aggr.cpp', A3, A3E,
'''    } while(still_waiting);

    cleanup:

    slot_map->addSlotEndTask();
''',
'''    } while(still_waiting || pusch_cb_pending);   // pusch_cb_pending: PUSCH finished on the GPU, its callback is held by the in-order gate

    cleanup:

    pdctx->ulInorderMarkDone(ul_seq);   // also covers the early exits above (no callback for this slot): later slots must not wait for it
    slot_map->addSlotEndTask();
'''))

# ---------------------------------------------------------------- src/uplink/phypusch_aggr.cpp  (PhyPuschAggr::validate)
P.append(('src/uplink/phypusch_aggr.cpp', None, None,
'''    uint32_t hb_slot_index=0;
    totNumTbCrc = dyn_params.pCellGrpDynPrm->nUes;
    for(uint32_t i = 0; i < totNumTbCrc; ++i)
    {
        if(dyn_params.pCellGrpDynPrm->pUePrms[i].pduBitmap & 0x01)
        {
            uint32_t offsetsTbCrc = pStartOffsetsTbCrc[i];
            if(hb_slot_index < hq_buffer_counter && hb_slot[hb_slot_index]->getCellDynIdx()==-1){
''',
'''    uint32_t hb_slot_index=0;
    totNumTbCrc = dyn_params.pCellGrpDynPrm->nUes;
    // Failed setup (2026-11-14; e.g. bucketAllocateBuffer refused a HARQ buffer that is still in flight): cuphyRunPuschRx did not
    // run for this slot, so the output buffers hold values of earlier slots. Report every TB as CRC failure, with the TB CRC offsets
    // rebuilt here (running TB count, the layout cuPHY setup phase 1 writes) because they may be stale as well. okTimeout makes the
    // callback free the NDI=1 HARQ buffers that were allocated before the failure (their contents were not reset).
    const bool setup_failed = (getSetupStatus() != CH_SETUP_DONE_NO_ERROR);
    uint32_t tb_run = 0;
    for(uint32_t i = 0; i < totNumTbCrc; ++i)
    {
        if(dyn_params.pCellGrpDynPrm->pUePrms[i].pduBitmap & 0x01)
        {
            if(setup_failed)
            {
                pStartOffsetsTbCrc[i] = tb_run;
                pTbCrcs[tb_run] = 1;
                okTimeout = true;
            }
            tb_run++;
            uint32_t offsetsTbCrc = pStartOffsetsTbCrc[i];
            if(hb_slot_index < hq_buffer_counter && hb_slot[hb_slot_index]->getCellDynIdx()==-1){
'''))

rc = 0
for f, s0, s1, old, new in P:
    p = os.path.join(D, f); t = open(p).read()
    a = 0; b = len(t)
    if s0 is not None:
        if t.count(s0) != 1 or t.count(s1) != 1: print(f"!! {f}: scope markers {t.count(s0)}/{t.count(s1)}"); rc = 1; continue
        a = t.index(s0); b = t.index(s1)
    n = t[a:b].count(old)
    if n != 1:
        print(f"!! {f}: anchor occurs {n} times: {old.strip().splitlines()[0][:70]}" + ("  (already patched)" if t[a:b].count(new) == 1 else "")); rc = 1; continue
    if not check:
        i = a + t[a:b].index(old); t = t[:i] + new + t[i + len(old):]; open(p, 'w').write(t)
    print(f"ok {f}: {old.strip().splitlines()[0][:70]}")
sys.exit(rc)
