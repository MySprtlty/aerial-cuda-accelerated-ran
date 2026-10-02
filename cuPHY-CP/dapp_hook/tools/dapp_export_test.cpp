/*
 * dapp_export_test: unit test for the FAPI -> ring record conversion.
 *
 * Builds SCF FAPI messages byte by byte in memory (the same packed structs the
 * L1 parses), runs dapp_export_dl_tti / dapp_export_ul_tti on them exactly as
 * the hook does, reads the ring back with the consumer, and checks every
 * field that a load predictor depends on. Exercises the variable-length tails
 * that are easy to get wrong:
 *   - PDSCH: codewords -> end block -> optional PTRS -> precoding/BF block
 *   - PDCCH: a list of DCIs whose stride depends on each DCI's own BF block
 *           and payload size
 *   - truncation: a message cut short must be reported, never over-read
 *   - asynchronous export (dapp_async.hpp): same records in the same order as
 *     the inline path, record time = hook time, buffer claim / deferred
 *     release, and no blocking when the queue or the held budget is full
 *
 * No L1, no GPU, no MPS. Exit code 0 on success.
 */
#include "scf_5g_fapi_dapp_export.hpp"
#include "dapp_hook/dapp_ring.hpp"
#include "dapp_hook/dapp_async.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace nv::dapp;

static int g_fail = 0;
#define CHECK(cond, ...)                                                            \
    do {                                                                            \
        if (!(cond)) {                                                              \
            ++g_fail;                                                               \
            std::printf("  FAIL %s:%d  " #cond "  ", __FILE__, __LINE__);            \
            std::printf(__VA_ARGS__);                                               \
            std::printf("\n");                                                      \
        }                                                                           \
    } while (0)

// Little byte builder mirroring how testMAC lays a message out.
struct Buf {
    std::vector<uint8_t> b;
    template <class T> T* put(const T& v) { size_t o = b.size(); b.resize(o + sizeof(T)); std::memcpy(&b[o], &v, sizeof(T)); return reinterpret_cast<T*>(&b[o]); }
    size_t pos() const { return b.size(); }
    void   pad(size_t n) { b.resize(b.size() + n, 0); }
    template <class T> T* at(size_t o) { return reinterpret_cast<T*>(&b[o]); }
};

// ---- builders -----------------------------------------------------------
static void put_bf(Buf& m, uint16_t num_prgs, uint16_t prg_size, uint8_t dig_bf)
{
    scf_fapi_tx_precoding_beamforming_t bf{};
    bf.num_prgs = num_prgs; bf.prg_size = prg_size; bf.dig_bf_interfaces = dig_bf;
    m.put(bf);
    m.pad(static_cast<size_t>(num_prgs) * 2 + static_cast<size_t>(num_prgs) * dig_bf * 2); // pm_idx + beam_idx
}

static void put_dci(Buf& m, uint16_t rnti, uint8_t al, uint16_t bits, uint16_t prgs, uint8_t dig_bf)
{
    scf_fapi_dl_dci_t d{};
    d.rnti = rnti; d.aggregation_level = al; d.cce_index = 0;
    m.put(d);
    put_bf(m, prgs, 2, dig_bf);
    scf_fapi_pdcch_tx_power_info_t pw{};
    m.put(pw);
    scf_fapi_pdcch_dci_payload_t pl{};
    pl.payload_size_bits = bits;
    m.put(pl);
    m.pad((bits + 7u) / 8u);
}

// Returns the generic PDU header offset so pdu_size can be patched afterwards.
static size_t begin_pdu(Buf& m, uint16_t type)
{
    scf_fapi_generic_pdu_info_t g{};
    g.pdu_type = type;
    size_t off = m.pos();
    m.put(g);
    return off;
}
static void end_pdu(Buf& m, size_t off) { m.at<scf_fapi_generic_pdu_info_t>(off)->pdu_size = static_cast<uint16_t>(m.pos() - off); }

static void put_pdcch(Buf& m, const std::vector<std::tuple<uint8_t, uint16_t, uint16_t, uint8_t>>& dcis)
{
    size_t off = begin_pdu(m, DL_TTI_PDU_TYPE_PDCCH);
    scf_fapi_pdcch_pdu_t p{};
    p.bwp.bwp_size = 273; p.start_sym_index = 0; p.duration_sym = 2;
    p.num_dl_dci = static_cast<uint16_t>(dcis.size());
    m.put(p);
    uint16_t rnti = 0x100;
    for (auto& [al, bits, prgs, bf] : dcis) { put_dci(m, rnti++, al, bits, prgs, bf); }
    end_pdu(m, off);
}

static void put_pdsch(Buf& m, uint16_t rnti, uint8_t mcs, uint32_t tb, uint16_t rb, uint8_t layers, bool ptrs,
                      uint16_t prgs, uint16_t prg_size, uint8_t dig_bf)
{
    size_t off = begin_pdu(m, DL_TTI_PDU_TYPE_PDSCH);
    scf_fapi_pdsch_pdu_t p{};
    p.pdu_bitmap = ptrs ? 0x1 : 0x0; p.rnti = rnti; p.pdu_index = 7; p.bwp.bwp_size = 273; p.num_codewords = 1;
    m.put(p);
    scf_fapi_pdsch_codeword_t cw{};
    cw.mcs_index = mcs; cw.qam_mod_order = 6; cw.target_code_rate = 6000; cw.tb_size = tb;
    m.put(cw);
    scf_fapi_pdsch_pdu_end_t e{};
    e.num_of_layers = layers; e.rb_start = 10; e.rb_size = rb; e.start_sym_index = 2; e.num_symbols = 12; e.dl_dmrs_sym_pos = 0x4;
    m.put(e);
    if (ptrs) { scf_fapi_pdsch_ptrs_t t{}; m.put(t); }
    put_bf(m, prgs, prg_size, dig_bf);
    end_pdu(m, off);
}

static size_t begin_dl_tti(Buf& m, uint16_t sfn, uint16_t slot, uint8_t num_pdus)
{
    scf_fapi_dl_tti_req_t h{};
    h.msg_hdr.type_id = SCF_FAPI_DL_TTI_REQUEST; h.sfn = sfn; h.slot = slot; h.num_pdus = num_pdus;
    size_t off = m.pos();
    m.put(h);
    return off;
}

// ---- drain helper -------------------------------------------------------
static std::vector<dapp_rec_t> drain(Consumer& c)
{
    std::vector<dapp_rec_t> out;
    dapp_rec_t r;
    while (c.next(r, nullptr) == Consumer::RECORD) { out.push_back(r); }
    return out;
}

int main()
{
    const std::string name = "/dapp_export_test";
    Producer::Config cfg;
    cfg.name = name; cfg.ring_len = 1024; cfg.do_mlock = false;
    std::string err;
    auto ring = Producer::create(cfg, err);
    if (!ring) { std::printf("ring create failed: %s\n", err.c_str()); return 1; }
    auto cons = Consumer::open(name, err);
    if (!cons) { std::printf("consumer open failed: %s\n", err.c_str()); return 1; }
    cons->seek_to_head();

    // ---------------------------------------------------------------------
    // Case 1: DL_TTI with one PDCCH (3 DCIs of AL 1,4,16) and two PDSCH
    //         (one without PTRS, one with), realistic sizes.
    // ---------------------------------------------------------------------
    std::printf("case 1: PDCCH with 3 DCIs + PDSCH x2 (PTRS off/on)\n");
    {
        Buf m;
        size_t hoff = begin_dl_tti(m, 421, 15, 3);
        put_pdcch(m, {{1, 39, 1, 4}, {4, 41, 1, 4}, {16, 60, 2, 4}});
        put_pdsch(m, 0x4601, 20, 12345, 100, 2, false, 25, 4, 4);   // 100 PRB / prg 4 -> 25 PRGs
        put_pdsch(m, 0x4602, 9, 3000, 24, 1, true, 6, 4, 4);        // PTRS present
        m.at<scf_fapi_dl_tti_req_t>(hoff)->msg_hdr.length = static_cast<uint32_t>(m.pos() - hoff);

        nv_ipc_msg_t ipc{};
        ipc.msg_buf = m.b.data(); ipc.msg_len = static_cast<int32_t>(m.pos()); ipc.cell_id = 0;
        scf_5g_fapi::dapp_export_dl_tti(*ring, 0, *m.at<scf_fapi_dl_tti_req_t>(hoff), ipc, 0);

        auto recs = drain(*cons);
        CHECK(recs.size() == 4, "got %zu records, want 3 PDU + 1 summary", recs.size());
        if (recs.size() == 4) {
            const dapp_dl_pdu_t& pdcch = recs[0].u.dl_pdu;
            CHECK(recs[0].type == DAPP_REC_DL_PDU && pdcch.pdu_type == DAPP_DL_PDCCH, "rec0 type=%u pdu_type=%u", recs[0].type, pdcch.pdu_type);
            CHECK(pdcch.num_dl_dci == 3, "num_dl_dci=%u", pdcch.num_dl_dci);
            CHECK(pdcch.agg_level_sum == 21, "agg_level_sum=%u want 21", pdcch.agg_level_sum);
            CHECK(pdcch.agg_level_max == 16, "agg_level_max=%u want 16", pdcch.agg_level_max);
            CHECK(pdcch.dci_payload_bits_sum == 140, "dci_payload_bits_sum=%u want 140", pdcch.dci_payload_bits_sum);
            CHECK(pdcch.prg_bf_sum == 16, "pdcch prg_bf_sum=%u want 1*4+1*4+2*4=16", pdcch.prg_bf_sum);
            CHECK(pdcch.dci_truncated == 0, "dci_truncated=%u", pdcch.dci_truncated);

            const dapp_dl_pdu_t& a = recs[1].u.dl_pdu;
            CHECK(a.pdu_type == DAPP_DL_PDSCH && a.rnti == 0x4601, "rec1 pdu_type=%u rnti=0x%x", a.pdu_type, a.rnti);
            CHECK(a.mcs_index == 20 && a.tb_size == 12345 && a.rb_size == 100 && a.num_layers == 2, "pdsch A fields mcs=%u tb=%u rb=%u l=%u", a.mcs_index, a.tb_size, a.rb_size, a.num_layers);
            CHECK(a.num_prgs == 25 && a.prg_size == 4 && a.dig_bf_interfaces == 4, "pdsch A bf prg=%u size=%u bf=%u", a.num_prgs, a.prg_size, a.dig_bf_interfaces);
            CHECK(a.prg_bf_sum == 100, "pdsch A prg_bf_sum=%u want 100", a.prg_bf_sum);
            CHECK(a.fapi_pdu_index == 7, "fapi_pdu_index=%u", a.fapi_pdu_index);

            const dapp_dl_pdu_t& b = recs[2].u.dl_pdu;
            CHECK(b.rnti == 0x4602 && b.tb_size == 3000, "pdsch B rnti=0x%x tb=%u", b.rnti, b.tb_size);
            CHECK(b.num_prgs == 6 && b.dig_bf_interfaces == 4, "pdsch B (PTRS path) bf prg=%u bf=%u", b.num_prgs, b.dig_bf_interfaces);

            const dapp_dl_tti_t& s = recs[3].u.dl_tti;
            CHECK(recs[3].type == DAPP_REC_DL_TTI, "rec3 type=%u", recs[3].type);
            CHECK(s.n_pdcch == 1 && s.n_pdsch == 2 && s.n_dci == 3, "counts pdcch=%u pdsch=%u dci=%u", s.n_pdcch, s.n_pdsch, s.n_dci);
            CHECK(s.tot_pdsch_prb == 124 && s.tot_pdsch_layers == 3, "tot prb=%u layers=%u", s.tot_pdsch_prb, s.tot_pdsch_layers);
            CHECK(s.tot_pdsch_prb_layers == 224, "tot prb*layers=%u want 100*2+24*1", s.tot_pdsch_prb_layers);
            CHECK(s.tot_pdsch_tb_bytes == 15345, "tot tb=%u", s.tot_pdsch_tb_bytes);
            CHECK(s.tot_pdsch_prg_bf == 124, "tot_pdsch_prg_bf=%u want 100+24", s.tot_pdsch_prg_bf);
            CHECK(s.tot_dci_agg_level == 21 && s.max_dci_agg_level == 16, "dci al sum=%u max=%u", s.tot_dci_agg_level, s.max_dci_agg_level);
            CHECK(s.tot_dci_payload_bits == 140, "tot_dci_payload_bits=%u", s.tot_dci_payload_bits);
            CHECK(s.tot_pdcch_prg_bf == 16, "tot_pdcch_prg_bf=%u", s.tot_pdcch_prg_bf);
            CHECK(s.max_pdsch_mcs == 20, "max_pdsch_mcs=%u", s.max_pdsch_mcs);
            CHECK(s.pdu_truncated == 0, "pdu_truncated=%u", s.pdu_truncated);
        }
    }

    // ---------------------------------------------------------------------
    // Case 2: the same PDCCH, but msg_len cut inside the second DCI. The walk
    //         must stop, flag truncation, and never read past msg_len.
    // ---------------------------------------------------------------------
    std::printf("case 2: DCI list truncated by msg_len\n");
    {
        Buf m;
        size_t hoff = begin_dl_tti(m, 421, 16, 1);
        put_pdcch(m, {{2, 39, 1, 4}, {8, 41, 1, 4}, {8, 41, 1, 4}});
        size_t full = m.pos();
        m.at<scf_fapi_dl_tti_req_t>(hoff)->msg_hdr.length = static_cast<uint32_t>(full - hoff);

        nv_ipc_msg_t ipc{};
        ipc.msg_buf = m.b.data();
        ipc.msg_len = static_cast<int32_t>(full - 30); // chop into the DCI list
        scf_5g_fapi::dapp_export_dl_tti(*ring, 0, *m.at<scf_fapi_dl_tti_req_t>(hoff), ipc, 0);

        auto recs = drain(*cons);
        // The PDU header claims a size beyond msg_len, so the walker refuses the whole PDU.
        CHECK(recs.size() == 1, "got %zu records, want summary only", recs.size());
        if (!recs.empty()) {
            const dapp_dl_tti_t& s = recs.back().u.dl_tti;
            CHECK(s.pdu_truncated == 1, "pdu_truncated=%u want 1", s.pdu_truncated);
            CHECK(s.n_pdcch == 0, "n_pdcch=%u want 0 (PDU rejected as a whole)", s.n_pdcch);
        }
    }

    // ---------------------------------------------------------------------
    // Case 3: PDU size is honest but the DCI list inside overruns it (a DCI
    //         claims a huge payload). The per-DCI walk must stop inside the PDU.
    // ---------------------------------------------------------------------
    std::printf("case 3: DCI overruns its own PDU\n");
    {
        Buf m;
        size_t hoff = begin_dl_tti(m, 421, 17, 1);
        size_t off = begin_pdu(m, DL_TTI_PDU_TYPE_PDCCH);
        scf_fapi_pdcch_pdu_t p{};
        p.num_dl_dci = 2;
        m.put(p);
        put_dci(m, 0x200, 4, 40, 1, 4);
        // second DCI header + BF + power + payload header claiming 4000 bits, but no payload bytes follow
        scf_fapi_dl_dci_t d{}; d.aggregation_level = 8; m.put(d);
        put_bf(m, 1, 2, 4);
        scf_fapi_pdcch_tx_power_info_t pw{}; m.put(pw);
        scf_fapi_pdcch_dci_payload_t pl{}; pl.payload_size_bits = 4000; m.put(pl);
        end_pdu(m, off);
        // Add a second, valid PDU after it so we can prove the outer walk continued.
        put_pdsch(m, 0x4603, 5, 500, 8, 1, false, 2, 4, 4);
        m.at<scf_fapi_dl_tti_req_t>(hoff)->num_pdus = 2;
        m.at<scf_fapi_dl_tti_req_t>(hoff)->msg_hdr.length = static_cast<uint32_t>(m.pos() - hoff);

        nv_ipc_msg_t ipc{};
        ipc.msg_buf = m.b.data(); ipc.msg_len = static_cast<int32_t>(m.pos());
        scf_5g_fapi::dapp_export_dl_tti(*ring, 0, *m.at<scf_fapi_dl_tti_req_t>(hoff), ipc, 0);

        auto recs = drain(*cons);
        CHECK(recs.size() == 3, "got %zu records, want PDCCH + PDSCH + summary", recs.size());
        if (recs.size() == 3) {
            const dapp_dl_pdu_t& pdcch = recs[0].u.dl_pdu;
            // Both DCI headers were readable, so both ALs are counted (4+8); only the
            // second payload could not be validated, and that is flagged.
            CHECK(pdcch.agg_level_sum == 12, "agg_level_sum=%u want 12", pdcch.agg_level_sum);
            CHECK(pdcch.dci_truncated == 1, "dci_truncated=%u want 1", pdcch.dci_truncated);
            CHECK(recs[1].u.dl_pdu.rnti == 0x4603, "outer walk did not reach the PDSCH after the bad PDCCH");
            CHECK(recs[2].u.dl_tti.pdu_truncated == 1, "summary pdu_truncated=%u want 1", recs[2].u.dl_tti.pdu_truncated);
        }
    }

    // ---------------------------------------------------------------------
    // Case 4: UL_TTI regression - one PUSCH with data + UCI sections.
    // ---------------------------------------------------------------------
    std::printf("case 4: UL_TTI PUSCH with data and UCI sections\n");
    {
        Buf m;
        scf_fapi_ul_tti_req_t h{};
        h.msg_hdr.type_id = SCF_FAPI_UL_TTI_REQUEST; h.sfn = 421; h.slot = 18; h.num_pdus = 1; h.num_ulsch = 1;
        size_t hoff = m.pos(); m.put(h);
        size_t off = begin_pdu(m, UL_TTI_PDU_TYPE_PUSCH);
        scf_fapi_pusch_pdu_t p{};
        p.pdu_bitmap = 0x3; p.rnti = 0x0002; p.rb_start = 100; p.rb_size = 30; p.start_symbol_index = 0; p.num_of_symbols = 14;
        p.num_of_layers = 2; p.mcs_index = 10; p.mcs_table = 1; p.qam_mod_order = 4; p.target_code_rate = 6580; p.transform_precoding = 1;
        m.put(p);
        scf_fapi_pusch_data_t d{}; d.tb_size = 3009; d.num_cb = 3; d.rv_index = 0; d.harq_process_id = 5; d.new_data_indicator = 1; m.put(d);
        scf_fapi_pusch_uci_t u{}; u.harq_ack_bit_length = 2; u.csi_part_1_bit_length = 11; m.put(u);
        end_pdu(m, off);
        m.at<scf_fapi_ul_tti_req_t>(hoff)->msg_hdr.length = static_cast<uint32_t>(m.pos() - hoff);

        nv_ipc_msg_t ipc{};
        ipc.msg_buf = m.b.data(); ipc.msg_len = static_cast<int32_t>(m.pos());
        scf_5g_fapi::dapp_export_ul_tti(*ring, 0, *m.at<scf_fapi_ul_tti_req_t>(hoff), ipc, 0);

        auto recs = drain(*cons);
        CHECK(recs.size() == 2, "got %zu records", recs.size());
        if (recs.size() == 2) {
            const dapp_ul_pdu_t& q = recs[0].u.ul_pdu;
            CHECK(q.rnti == 2 && q.rb_size == 30 && q.num_layers == 2 && q.mcs_index == 10 && q.tb_size == 3009 && q.num_cb == 3,
                  "pusch rnti=%u rb=%u l=%u mcs=%u tb=%u cb=%u", q.rnti, q.rb_size, q.num_layers, q.mcs_index, q.tb_size, q.num_cb);
            CHECK(q.harq_process_id == 5 && q.ndi == 1 && q.transform_precoding == 1, "harq=%u ndi=%u tp=%u", q.harq_process_id, q.ndi, q.transform_precoding);
            CHECK(q.harq_ack_bit_len == 2 && q.csi_part1_bit_len == 11, "uci harq=%u csi1=%u", q.harq_ack_bit_len, q.csi_part1_bit_len);
            const dapp_ul_tti_t& s = recs[1].u.ul_tti;
            CHECK(s.n_pusch == 1 && s.tot_pusch_prb == 30 && s.tot_pusch_prb_layers == 60 && s.tot_pusch_tb_bytes == 3009 && s.tot_pusch_cb == 3,
                  "ul summary pusch=%u prb=%u prb*l=%u tb=%u cb=%u", s.n_pusch, s.tot_pusch_prb, s.tot_pusch_prb_layers, s.tot_pusch_tb_bytes, s.tot_pusch_cb);
        }
    }

    // ---------------------------------------------------------------------
    // Case 5: asynchronous export. One PUSCH UL_TTI per "cell message".
    // ---------------------------------------------------------------------
    std::printf("case 5: asynchronous export (queue, order, timestamps, deferred release)\n");
    {
        auto build_ul = [](Buf& m, uint16_t sfn, uint16_t slot, uint16_t rnti) -> size_t {
            scf_fapi_ul_tti_req_t h{};
            h.msg_hdr.type_id = SCF_FAPI_UL_TTI_REQUEST; h.sfn = sfn; h.slot = slot; h.num_pdus = 1; h.num_ulsch = 1;
            size_t hoff = m.pos(); m.put(h);
            size_t off = begin_pdu(m, UL_TTI_PDU_TYPE_PUSCH);
            scf_fapi_pusch_pdu_t p{};
            p.pdu_bitmap = 0x1; p.rnti = rnti; p.rb_size = 34; p.num_of_symbols = 14; p.num_of_layers = 4; p.mcs_index = 27; p.qam_mod_order = 8;
            m.put(p);
            scf_fapi_pusch_data_t d{}; d.tb_size = 15117; d.num_cb = 15; m.put(d);
            end_pdu(m, off);
            m.at<scf_fapi_ul_tti_req_t>(hoff)->msg_hdr.length = static_cast<uint32_t>(m.pos() - hoff);
            return hoff;
        };
        auto same_payload = [](const dapp_rec_t& a, const dapp_rec_t& b) {
            // everything except seq and ts_ns
            return std::memcmp(reinterpret_cast<const uint8_t*>(&a) + 16, reinterpret_cast<const uint8_t*>(&b) + 16, sizeof(dapp_rec_t) - 16) == 0;
        };

        // reference: the inline path on one message
        Buf ref; size_t ref_off = build_ul(ref, 500, 4, 0x1000);
        nv_ipc_msg_t ref_ipc{}; ref_ipc.msg_buf = ref.b.data(); ref_ipc.msg_len = static_cast<int32_t>(ref.pos());
        scf_5g_fapi::dapp_export_ul_tti(*ring, 0, *ref.at<scf_fapi_ul_tti_req_t>(ref_off), ref_ipc, 777);
        auto ref_recs = drain(*cons);
        CHECK(ref_recs.size() == 2, "inline reference gave %zu records", ref_recs.size());

        auto ax = AsyncExporter::create(ring.get());

        // (a) export thread not running: the hand-off never blocks, it refuses
        const unsigned N = AsyncExporter::kMaxHeld;
        std::vector<Buf> msgs(N + 2);
        std::vector<size_t> offs(N + 2);
        std::vector<nv_ipc_msg_t> ipcs(N + 2);
        unsigned accepted = 0;
        for (unsigned i = 0; i < N + 1; ++i) {
            offs[i] = build_ul(msgs[i], 500, 4, 0x1000);       // same content as the reference
            ipcs[i] = nv_ipc_msg_t{}; ipcs[i].msg_buf = msgs[i].b.data(); ipcs[i].msg_len = static_cast<int32_t>(msgs[i].pos()); ipcs[i].cell_id = static_cast<int32_t>(i % 16);
            const bool ok = scf_5g_fapi::dapp_submit_ul_tti(*ax, static_cast<uint16_t>(i % 16), *msgs[i].at<scf_fapi_ul_tti_req_t>(offs[i]), ipcs[i], 777);
            if (ok) {
                ++accepted;
                CHECK(ax->take_claim(ipcs[i].msg_buf, &ipcs[i], sizeof(nv_ipc_msg_t)), "claim %u not taken", i);
                CHECK(!ax->take_claim(ipcs[i].msg_buf, &ipcs[i], sizeof(nv_ipc_msg_t)), "claim %u taken twice", i);
            }
        }
        CHECK(accepted == N, "accepted %u hand-offs with the export thread stopped, want %u (held budget)", accepted, N);
        nv_ipc_msg_t rel{};
        CHECK(!ax->pop_releasable(&rel, sizeof(rel)), "a buffer became releasable before the export thread ran");
        {
            auto st = ax->stats();
            CHECK(st.dropped == 1 && st.orphans == 0 && st.done == 0, "stats dropped=%llu orphans=%llu done=%llu",
                  static_cast<unsigned long long>(st.dropped), static_cast<unsigned long long>(st.orphans), static_cast<unsigned long long>(st.done));
        }
        CHECK(drain(*cons).empty(), "records appeared without the export thread");

        // SLOT_END goes through the same queue (no buffer attached)
        {
            AsyncExporter::Item* it = ax->reserve(false);
            CHECK(it != nullptr, "no room for SLOT_END");
            if (it != nullptr) {
                it->fn = async_slot_end_handler; it->ts_rx_ns = now_ns(); it->ts_l2_send_ns = 0; it->fapi_msg = nullptr;
                it->kind = AsyncExporter::KIND_SLOT_END; it->cell_id = 0xFFFF; it->sfn = 500; it->slot = 4; it->slot_dropped = 0;
                it->slot_end = dapp_slot_end_t{}; it->slot_end.enqueued = 1; it->slot_end.cmd_size = 16;
                ax->publish(it, nullptr);
            }
        }

        // (b) start the thread: everything is written, in hand-off order
        std::string aerr;
        CHECK(ax->start(-1, 0, aerr), "export thread start: %s", aerr.c_str());
        for (int i = 0; i < 2000; ++i) {
            // not stats(): that would reset the delay figures checked below
            if (ring->head() >= ref_recs.back().seq + 2ull * N + 1) { break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        auto recs = drain(*cons);
        CHECK(recs.size() == 2 * N + 1, "async wrote %zu records, want %u", recs.size(), 2 * N + 1);
        if (recs.size() == 2 * N + 1 && ref_recs.size() == 2) {
            unsigned bad_payload = 0, bad_cell = 0, bad_ts = 0, bad_seq = 0;
            for (unsigned i = 0; i < N; ++i) {
                dapp_rec_t a = recs[2 * i], b = recs[2 * i + 1];
                if (a.cell_id != i % 16 || b.cell_id != i % 16) { ++bad_cell; }
                if (a.ts_ns != b.ts_ns || (i > 0 && a.ts_ns < recs[2 * i - 1].ts_ns)) { ++bad_ts; }   // one hook time per message
                if (b.seq != a.seq + 1) { ++bad_seq; }
                a.cell_id = ref_recs[0].cell_id; b.cell_id = ref_recs[1].cell_id;
                if (!same_payload(a, ref_recs[0]) || !same_payload(b, ref_recs[1])) { ++bad_payload; }
            }
            CHECK(bad_payload == 0, "%u messages differ from the inline records", bad_payload);
            CHECK(bad_cell == 0 && bad_seq == 0 && bad_ts == 0, "order/time: cell=%u seq=%u ts=%u", bad_cell, bad_seq, bad_ts);
            const dapp_rec_t& e = recs[2 * N];
            CHECK(e.type == DAPP_REC_SLOT_END && e.sfn == 500 && e.slot == 4, "last record type=%u %u.%u", e.type, e.sfn, e.slot);
            CHECK(e.u.slot_end.num_cells == 16 && e.u.slot_end.enqueued == 1 && e.u.slot_end.cmd_size == 16,
                  "slot_end num_cells=%u enqueued=%u cmd_size=%u", e.u.slot_end.num_cells, e.u.slot_end.enqueued, e.u.slot_end.cmd_size);
        }

        // (c) buffers come back for release in hand-off order, exactly once
        unsigned released = 0, wrong = 0;
        while (ax->pop_releasable(&rel, sizeof(rel))) {
            if (rel.msg_buf != ipcs[released].msg_buf || rel.cell_id != ipcs[released].cell_id) { ++wrong; }
            ++released;
        }
        CHECK(released == N && wrong == 0, "released %u (want %u), %u out of order", released, N, wrong);

        // (d) room again after the release; an unclaimed hand-off is counted as an orphan
        offs[N] = build_ul(msgs[N], 501, 5, 0x2000);
        ipcs[N] = nv_ipc_msg_t{}; ipcs[N].msg_buf = msgs[N].b.data(); ipcs[N].msg_len = static_cast<int32_t>(msgs[N].pos());
        CHECK(scf_5g_fapi::dapp_submit_ul_tti(*ax, 3, *msgs[N].at<scf_fapi_ul_tti_req_t>(offs[N]), ipcs[N], 0), "hand-off refused after release");
        offs[N + 1] = build_ul(msgs[N + 1], 501, 5, 0x2001);
        ipcs[N + 1] = nv_ipc_msg_t{}; ipcs[N + 1].msg_buf = msgs[N + 1].b.data(); ipcs[N + 1].msg_len = static_cast<int32_t>(msgs[N + 1].pos());
        CHECK(scf_5g_fapi::dapp_submit_ul_tti(*ax, 4, *msgs[N + 1].at<scf_fapi_ul_tti_req_t>(offs[N + 1]), ipcs[N + 1], 0), "second hand-off refused");
        CHECK(ax->take_claim(ipcs[N + 1].msg_buf, &ipcs[N + 1], sizeof(nv_ipc_msg_t)), "claim not taken");
        for (int i = 0; i < 2000 && ring->head() < recs.back().seq + 4; ++i) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        auto st = ax->stats();
        CHECK(st.orphans == 1 && st.dropped == 1 && st.done == N + 3, "stats orphans=%llu dropped=%llu done=%llu",
              static_cast<unsigned long long>(st.orphans), static_cast<unsigned long long>(st.dropped), static_cast<unsigned long long>(st.done));
        CHECK(st.delay_n == N + 3 && st.delay_max_ns >= 0, "delay samples %llu", static_cast<unsigned long long>(st.delay_n));
        CHECK(drain(*cons).size() == 4, "last two messages not exported");
        ax->stop_and_join();
    }

    Producer::unlink(name);
    std::printf("%s (%d failures)\n", g_fail == 0 ? "EXPORT TEST PASS" : "EXPORT TEST FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
