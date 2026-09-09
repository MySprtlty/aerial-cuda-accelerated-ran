/*
 * dapp_ring_fakel1: emit a realistic record stream into the dApp ring WITHOUT
 * running cuphycontroller. Use it to bring up and test a consumer (the Python
 * reader, a GPU scheduler, dapp_ring_dump) before touching the real L1.
 *
 *   dapp_ring_fakel1 [-n /aerial_dapp_ring] [--slots N] [--cells C]
 *                    [--rate-us 500] [--ue-per-cell U] [--drop-every K]
 *
 * The record pattern matches the real hook: per UL slot, for each cell,
 * UL_PDU records followed by one UL_TTI summary, then one SLOT_END for the
 * slot. A TDD-like pattern marks 4 of every 10 slots as uplink.
 */
#include "dapp_hook/dapp_ring.hpp"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

using namespace nv::dapp;

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int) { g_stop = 1; }

static bool is_ul_slot(uint16_t slot) { return (slot % 10) >= 6; } // slots 6..9 uplink

int main(int argc, char** argv)
{
    std::string name = DAPP_RING_DEFAULT_NAME;
    long slots = 2000, cells = 4, rate_us = 500, ue_per_cell = 2, drop_every = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-n" || a == "--name") && i + 1 < argc) name = argv[++i];
        else if (a == "--slots" && i + 1 < argc) slots = std::atol(argv[++i]);
        else if (a == "--cells" && i + 1 < argc) cells = std::atol(argv[++i]);
        else if (a == "--rate-us" && i + 1 < argc) rate_us = std::atol(argv[++i]);
        else if (a == "--ue-per-cell" && i + 1 < argc) ue_per_cell = std::atol(argv[++i]);
        else if (a == "--drop-every" && i + 1 < argc) drop_every = std::atol(argv[++i]);
        else {
            std::fprintf(stderr, "usage: %s [-n /name] [--slots N] [--cells C] [--rate-us U] [--ue-per-cell U] [--drop-every K]\n", argv[0]);
            return 2;
        }
    }

    Producer::Config cfg;
    cfg.name          = name;
    cfg.ring_len      = DAPP_RING_DEFAULT_LEN;
    cfg.do_mlock      = false;
    cfg.slot_advance  = 3;
    cfg.mu            = 1;
    cfg.num_cells     = static_cast<uint32_t>(cells);
    cfg.producer_name = "fake_l1";
    std::string err;
    auto ring = Producer::create(cfg, err);
    if (!ring) { std::fprintf(stderr, "producer create failed: %s\n", err.c_str()); return 1; }
    std::printf("fake L1 producing on %s: slots=%ld cells=%ld rate=%ldus ue/cell=%ld\n", name.c_str(), slots, cells, rate_us, ue_per_cell);

    std::signal(SIGINT, on_sigint);
    uint16_t sfn = 0, slot = 0;
    long emitted_slots = 0;
    uint64_t rng = 0x243F6A8885A308D3ULL;
    auto next_rand = [&rng]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };

    while (!g_stop && (slots <= 0 || emitted_slots < slots)) {
        const bool ul = is_ul_slot(slot);
        const int64_t tick = now_ns();

        uint32_t cells_with_ul = 0;
        if (ul) {
            for (long c = 0; c < cells; ++c) {
                // Load varies per slot and per cell so a predictor has something to learn.
                const long n_ue = 1 + static_cast<long>(next_rand() % static_cast<uint64_t>(ue_per_cell));
                uint16_t n_pusch = 0;
                uint32_t tot_prb = 0, tot_layers = 0, tot_tb = 0, tot_cb = 0, tot_prb_layers = 0;

                for (long u = 0; u < n_ue; ++u) {
                    const uint16_t rb_size = static_cast<uint16_t>(4 + next_rand() % 270);
                    const uint8_t  layers  = static_cast<uint8_t>(1 + next_rand() % 4);
                    const uint8_t  mcs     = static_cast<uint8_t>(next_rand() % 28);
                    const uint32_t tb      = static_cast<uint32_t>(rb_size) * layers * (mcs + 2) * 3;
                    dapp_rec_t*    r = ring->begin(DAPP_REC_UL_PDU, sfn, slot, static_cast<uint16_t>(c));
                    dapp_ul_pdu_t& p = r->u.ul_pdu;
                    p.pdu_index      = static_cast<uint8_t>(u);
                    p.pdu_type       = DAPP_UL_PUSCH;
                    p.rnti           = static_cast<uint16_t>(0x1000 + c * 16 + u);
                    p.handle         = static_cast<uint32_t>(c * 100 + u);
                    p.rb_start       = 0;
                    p.rb_size        = rb_size;
                    p.start_sym      = 0;
                    p.num_sym        = 14;
                    p.num_layers     = layers;
                    p.mcs_index      = mcs;
                    p.mcs_table      = 0;
                    p.qam_mod_order  = static_cast<uint8_t>(mcs < 10 ? 2 : (mcs < 18 ? 4 : 6));
                    p.target_code_rate = static_cast<uint16_t>(1000 + mcs * 200);
                    p.pdu_bitmap     = 0x1;
                    p.tb_size        = tb;
                    p.num_cb         = static_cast<uint16_t>(1 + tb / 1056);
                    p.ul_dmrs_sym_pos = 0x4;
                    p.dmrs_config_type = 0;
                    p.num_dmrs_cdm_grps_no_data = 2;
                    p.rv_index       = 0;
                    p.harq_process_id = static_cast<uint8_t>(u & 0xf);
                    p.ndi            = 1;
                    p.bwp_size       = 273;
                    p.bwp_start      = 0;
                    p.pdu_size       = 96;
                    ring->commit(r);

                    ++n_pusch;
                    tot_prb += rb_size;
                    tot_layers += layers;
                    tot_prb_layers += static_cast<uint32_t>(rb_size) * layers;
                    tot_tb += tb;
                    tot_cb += p.num_cb;
                }

                dapp_rec_t*    r = ring->begin(DAPP_REC_UL_TTI, sfn, slot, static_cast<uint16_t>(c));
                dapp_ul_tti_t& s = r->u.ul_tti;
                s.num_pdus       = static_cast<uint8_t>(n_pusch);
                s.num_ulsch      = static_cast<uint8_t>(n_pusch);
                s.n_pusch        = n_pusch;
                s.msg_len        = 128 + n_pusch * 96;
                s.body_len       = s.msg_len - 2;
                s.ts_l2_send_ns  = now_ns() - 20000;
                s.tot_pusch_prb  = tot_prb;
                s.tot_pusch_layers = tot_layers;
                s.tot_pusch_tb_bytes = tot_tb;
                s.tot_pusch_cb   = tot_cb;
                s.tot_pusch_prb_layers = tot_prb_layers;
                ring->commit(r);
                ring->count_ul_tti();
                ++cells_with_ul;
            }
        }

        const bool dropped = (drop_every > 0) && (emitted_slots % drop_every == drop_every - 1);
        dapp_rec_t*      r = ring->begin(DAPP_REC_SLOT_END, sfn, slot, 0xFFFF);
        dapp_slot_end_t& d = r->u.slot_end;
        d.enqueued            = dropped ? 0u : 1u;
        d.slot_end_rcvd       = 1;
        d.is_ul               = ul ? 1u : 0u;
        d.is_dl               = ul ? 0u : 1u;
        d.enqueue_ret         = dropped ? -1 : 0;
        d.num_cells           = cells_with_ul;
        d.cmd_size            = static_cast<uint32_t>(cells);
        d.tick_original_ns    = tick;
        d.t0_ns               = tick + 3 * 500000;
        d.l1_slot_ind_tick_ns = tick;
        d.l2a_latency_ns      = 150000 + static_cast<int64_t>(next_rand() % 100000);
        d.l2a_start_ns        = tick + 50000;
        d.l2a_end_ns          = tick + 120000;
        ring->commit(r);
        ring->count_slot_end();
        if (dropped) { ring->count_slot_dropped(); }
        ring->heartbeat();

        if (++slot >= 20) { slot = 0; sfn = static_cast<uint16_t>((sfn + 1) % 1024); }
        ++emitted_slots;
        if (rate_us > 0) { std::this_thread::sleep_for(std::chrono::microseconds(rate_us)); }
    }
    std::printf("done: %ld slots, head=%llu\n", emitted_slots, (unsigned long long)ring->head());
    return 0;
}
