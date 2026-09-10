/*
 * dapp_ring_dump: print the contents of the dApp FAPI hook ring.
 *
 *   dapp_ring_dump [-n /aerial_dapp_ring] [--follow] [--from-start] [--pdus]
 *                  [--type UL_TTI,SLOT_END,...] [--cell N] [--max N] [--stats]
 *
 * Default: prints the header, then every record from the oldest available
 * one and exits when it catches up. --follow keeps polling (like tail -f).
 */
#include "dapp_hook/dapp_ring.hpp"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <time.h>

using namespace nv::dapp;

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int) { g_stop = 1; }

static void print_hdr(const dapp_ring_hdr_t& h)
{
    std::printf("ring: name=%s abi=%u rec_size=%u ring_len=%u generation=%u producer_pid=%u mlock=%u\n",
                h.producer_name, h.abi_version, h.rec_size, h.ring_len, h.generation, h.producer_pid, h.flags & 1u);
    std::printf("      slot_advance=%u mu=%u num_cells=%u head=%llu heartbeat_age_ms=%.1f\n",
                h.slot_advance, h.mu, h.num_cells, (unsigned long long)h.head,
                (double)(now_ns() - (int64_t)h.heartbeat_ns) / 1e6);
    std::printf("      cnt_ul_tti=%llu cnt_dl_tti=%llu cnt_slot_end=%llu cnt_slot_dropped=%llu cnt_pdu_truncated=%llu\n",
                (unsigned long long)h.cnt_ul_tti, (unsigned long long)h.cnt_dl_tti, (unsigned long long)h.cnt_slot_end,
                (unsigned long long)h.cnt_slot_dropped, (unsigned long long)h.cnt_pdu_truncated);
}

static void fmt_ts(uint64_t ns, char* buf, size_t len)
{
    time_t s = (time_t)(ns / 1000000000ULL);
    struct tm tmv;
    localtime_r(&s, &tmv);
    size_t n = strftime(buf, len, "%H:%M:%S", &tmv);
    std::snprintf(buf + n, len - n, ".%06llu", (unsigned long long)((ns % 1000000000ULL) / 1000ULL));
}

static void print_rec(const dapp_rec_t& r, bool pdus)
{
    char ts[32];
    fmt_ts(r.ts_ns, ts, sizeof(ts));
    switch (r.type) {
    case DAPP_REC_PRODUCER_START:
        std::printf("#%-8llu %s PRODUCER_START pid=%u gen=%u slot_advance=%u mu=%u cells=%u ring_len=%u\n",
                    (unsigned long long)r.seq, ts, r.u.start.pid, r.u.start.generation, r.u.start.slot_advance,
                    r.u.start.mu, r.u.start.num_cells, r.u.start.ring_len);
        break;
    case DAPP_REC_UL_TTI: {
        const dapp_ul_tti_t& u = r.u.ul_tti;
        std::printf("#%-8llu %s UL_TTI   %4u.%-2u cell=%-2u pdus=%u (pusch=%u pucch=%u prach=%u srs=%u) ulsch=%u ulcch=%u rach=%u"
                    " prb=%u layers=%u prb*layers=%u tb_bytes=%u cb=%u pucch_prb=%u msg_len=%u l2_send_age_us=%.1f%s\n",
                    (unsigned long long)r.seq, ts, r.sfn, r.slot, r.cell_id, u.num_pdus, u.n_pusch, u.n_pucch, u.n_prach, u.n_srs,
                    u.num_ulsch, u.num_ulcch, u.rach_present, u.tot_pusch_prb, u.tot_pusch_layers, u.tot_pusch_prb_layers,
                    u.tot_pusch_tb_bytes, u.tot_pusch_cb, u.tot_pucch_prb, u.msg_len,
                    u.ts_l2_send_ns > 0 ? (double)((int64_t)r.ts_ns - u.ts_l2_send_ns) / 1e3 : -1.0,
                    u.pdu_truncated ? " TRUNCATED" : "");
        break;
    }
    case DAPP_REC_UL_PDU: {
        if (!pdus) { return; }
        const dapp_ul_pdu_t& p = r.u.ul_pdu;
        std::printf("#%-8llu %s   UL_PDU %4u.%-2u cell=%-2u [%u] %-5s rnti=0x%04x", (unsigned long long)r.seq, ts, r.sfn, r.slot,
                    r.cell_id, p.pdu_index, ul_pdu_type_name(p.pdu_type), p.rnti);
        switch (p.pdu_type) {
        case DAPP_UL_PUSCH:
            std::printf(" rb=%u+%u sym=%u+%u layers=%u mcs=%u/%u qam=%u cr=%u tb=%uB cb=%u rv=%u harq=%u ndi=%u tp=%u dmrs_pos=0x%x bitmap=0x%x",
                        p.rb_start, p.rb_size, p.start_sym, p.num_sym, p.num_layers, p.mcs_index, p.mcs_table, p.qam_mod_order,
                        p.target_code_rate, p.tb_size, p.num_cb, p.rv_index, p.harq_process_id, p.ndi, p.transform_precoding,
                        p.ul_dmrs_sym_pos, p.pdu_bitmap);
            if (p.pdu_bitmap & 2u) { std::printf(" uci(harq=%u csi1=%u)", p.harq_ack_bit_len, p.csi_part1_bit_len); }
            break;
        case DAPP_UL_PUCCH:
            std::printf(" fmt=%u prb=%u+%u sym=%u+%u harq=%u csi1=%u csi2=%u sr=%u", p.format_type, p.rb_start, p.rb_size,
                        p.start_sym, p.num_sym, p.bit_len_harq, p.bit_len_csi1, p.bit_len_csi2, p.sr_flag);
            break;
        case DAPP_UL_PRACH:
            std::printf(" format=%u ocas=%u num_ra=%u start_sym=%u", p.prach_format, p.num_prach_ocas, p.num_ra, p.start_sym);
            break;
        case DAPP_UL_SRS:
            std::printf(" ports=%u sym=%u rep=%u comb=%u bw_idx=%u cfg_idx=%u bwp=%u+%u", p.num_layers, p.srs_num_symbols,
                        p.srs_num_repetitions, p.srs_comb_size, p.srs_bandwidth_index, p.srs_config_index, p.bwp_start, p.bwp_size);
            break;
        default: break;
        }
        std::printf("\n");
        break;
    }
    case DAPP_REC_DL_TTI: {
        const dapp_dl_tti_t& d = r.u.dl_tti;
        std::printf("#%-8llu %s DL_TTI   %4u.%-2u cell=%-2u pdus=%u (pdcch=%u pdsch=%u csirs=%u ssb=%u) dci=%u prb=%u layers=%u prb*layers=%u tb_bytes=%u msg_len=%u%s\n",
                    (unsigned long long)r.seq, ts, r.sfn, r.slot, r.cell_id, d.num_pdus, d.n_pdcch, d.n_pdsch, d.n_csirs, d.n_ssb,
                    d.n_dci, d.tot_pdsch_prb, d.tot_pdsch_layers, d.tot_pdsch_prb_layers, d.tot_pdsch_tb_bytes, d.msg_len,
                    d.pdu_truncated ? " TRUNCATED" : "");
        break;
    }
    case DAPP_REC_DL_PDU: {
        if (!pdus) { return; }
        const dapp_dl_pdu_t& p = r.u.dl_pdu;
        std::printf("#%-8llu %s   DL_PDU %4u.%-2u cell=%-2u [%u] %-6s", (unsigned long long)r.seq, ts, r.sfn, r.slot, r.cell_id,
                    p.pdu_index, dl_pdu_type_name(p.pdu_type));
        switch (p.pdu_type) {
        case DAPP_DL_PDSCH:
            std::printf(" rnti=0x%04x rb=%u+%u sym=%u+%u layers=%u cw=%u mcs=%u/%u qam=%u cr=%u tb=%uB tb1=%uB fapi_idx=%u", p.rnti,
                        p.rb_start, p.rb_size, p.start_sym, p.num_sym, p.num_layers, p.num_codewords, p.mcs_index, p.mcs_table,
                        p.qam_mod_order, p.target_code_rate, p.tb_size, p.tb_size_cw1, p.fapi_pdu_index);
            break;
        case DAPP_DL_PDCCH:
            std::printf(" dci=%u sym=%u+%u bwp=%u+%u", p.num_dl_dci, p.start_sym, p.num_sym, p.bwp_start, p.bwp_size);
            break;
        case DAPP_DL_CSI_RS:
            std::printf(" rb=%u+%u row=%u", p.rb_start, p.rb_size, p.csirs_row);
            break;
        case DAPP_DL_SSB:
            std::printf(" block=%u", p.ssb_block_index);
            break;
        default: break;
        }
        std::printf("\n");
        break;
    }
    case DAPP_REC_SLOT_END: {
        const dapp_slot_end_t& s = r.u.slot_end;
        std::printf("#%-8llu %s SLOT_END %4u.%-2u %s ret=%d trig=%s cells=%u cmd_size=%u ul=%u dl=%u csirs=%u l2a_latency_us=%.1f t0_in_us=%.1f\n",
                    (unsigned long long)r.seq, ts, r.sfn, r.slot, s.enqueued ? "ENQUEUED" : "DROPPED ", s.enqueue_ret,
                    s.slot_end_rcvd ? "slot_rsp" : "tick    ", s.num_cells, s.cmd_size, s.is_ul, s.is_dl, s.is_csirs,
                    (double)s.l2a_latency_ns / 1e3, s.t0_ns ? (double)(s.t0_ns - (int64_t)r.ts_ns) / 1e3 : 0.0);
        break;
    }
    default:
        std::printf("#%-8llu %s type=%u sfn=%u slot=%u cell=%u\n", (unsigned long long)r.seq, ts, r.type, r.sfn, r.slot, r.cell_id);
        break;
    }
}

int main(int argc, char** argv)
{
    std::string name = DAPP_RING_DEFAULT_NAME;
    bool follow = false, from_start = true, pdus = false, stats_only = false;
    long max_records = -1;
    int cell_filter = -1;
    std::vector<uint16_t> type_filter;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-n" || a == "--name") && i + 1 < argc) { name = argv[++i]; }
        else if (a == "--follow" || a == "-f") { follow = true; }
        else if (a == "--from-start") { from_start = true; }
        else if (a == "--new") { from_start = false; }
        else if (a == "--pdus" || a == "-p") { pdus = true; }
        else if (a == "--stats") { stats_only = true; }
        else if (a == "--max" && i + 1 < argc) { max_records = std::atol(argv[++i]); }
        else if (a == "--cell" && i + 1 < argc) { cell_filter = std::atoi(argv[++i]); }
        else if (a == "--type" && i + 1 < argc) {
            std::string t = argv[++i];
            size_t pos = 0;
            while (pos <= t.size()) {
                size_t c = t.find(',', pos);
                std::string tok = t.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
                if (tok == "UL_TTI") type_filter.push_back(DAPP_REC_UL_TTI);
                else if (tok == "UL_PDU") { type_filter.push_back(DAPP_REC_UL_PDU); pdus = true; }
                else if (tok == "DL_TTI") type_filter.push_back(DAPP_REC_DL_TTI);
                else if (tok == "DL_PDU") { type_filter.push_back(DAPP_REC_DL_PDU); pdus = true; }
                else if (tok == "SLOT_END") type_filter.push_back(DAPP_REC_SLOT_END);
                else if (tok == "PRODUCER_START") type_filter.push_back(DAPP_REC_PRODUCER_START);
                if (c == std::string::npos) break;
                pos = c + 1;
            }
        }
        else {
            std::fprintf(stderr, "usage: %s [-n /shm_name] [--follow] [--new] [--pdus] [--stats] [--type A,B] [--cell N] [--max N]\n", argv[0]);
            return 2;
        }
    }

    std::string err;
    auto c = Consumer::open(name, err);
    if (!c) {
        std::fprintf(stderr, "cannot open ring %s: %s\n", name.c_str(), err.c_str());
        return 1;
    }
    print_hdr(c->hdr());
    if (stats_only) { return 0; }
    if (!from_start) { c->seek_to_head(); }

    std::signal(SIGINT, on_sigint);
    dapp_rec_t r;
    long printed = 0;
    uint64_t lost = 0;
    while (!g_stop) {
        int st = c->next(r, &lost);
        if (st == Consumer::RESTARTED) {
            std::printf("--- producer restarted (generation %u) ---\n", c->hdr().generation);
            continue;
        }
        if (st == Consumer::NONE) {
            if (!follow) { break; }
            struct timespec ts = {0, 200000}; // 200 us
            nanosleep(&ts, nullptr);
            continue;
        }
        if (lost) { std::printf("--- lost %llu records (reader too slow) ---\n", (unsigned long long)lost); }
        if (cell_filter >= 0 && static_cast<int>(r.cell_id) != cell_filter && r.type != DAPP_REC_SLOT_END && r.type != DAPP_REC_PRODUCER_START) { continue; }
        if (!type_filter.empty()) {
            bool ok = false;
            for (uint16_t t : type_filter) { if (t == r.type) { ok = true; break; } }
            if (!ok) { continue; }
        }
        print_rec(r, pdus);
        if (max_records > 0 && ++printed >= max_records) { break; }
    }
    std::fflush(stdout);
    return 0;
}
