/*
 * dApp hook: export UL_TTI.request / DL_TTI.request contents to the
 * shared-memory ring. See scf_5g_fapi_dapp_export.hpp.
 *
 * Runs on the L2 adapter msg_processing thread: no allocation, no locking,
 * no syscalls, and every read of the nvIPC buffer is bounded by msg_len.
 */
#ifdef ENABLE_DAPP_HOOK

#include "scf_5g_fapi_dapp_export.hpp"

#include <cstdint>
#include <cstring>

namespace scf_5g_fapi
{

namespace
{

// Bounded iterator over a FAPI PDU list inside an nvIPC message buffer.
struct pdu_walker
{
    const uint8_t* cur;
    const uint8_t* end; // one past the last valid byte of the nvIPC message buffer
    bool           truncated = false;

    pdu_walker(const uint8_t* payload, const nv_ipc_msg_t& ipc) :
        cur(payload),
        end(static_cast<const uint8_t*>(ipc.msg_buf) + (ipc.msg_len > 0 ? ipc.msg_len : 0))
    {
        if (ipc.msg_buf == nullptr || cur == nullptr || cur > end) { end = cur; }
    }

    // Next generic PDU header, or nullptr when the buffer is exhausted.
    const scf_fapi_generic_pdu_info_t* next()
    {
        if (cur + sizeof(scf_fapi_generic_pdu_info_t) > end)
        {
            truncated = (cur != end);
            return nullptr;
        }
        auto*          pdu  = reinterpret_cast<const scf_fapi_generic_pdu_info_t*>(cur);
        const uint16_t size = pdu->pdu_size;
        if (size < sizeof(scf_fapi_generic_pdu_info_t) || cur + size > end)
        {
            truncated = true;
            return nullptr;
        }
        cur += size;
        return pdu;
    }

    static bool fits(const uint8_t* p, size_t n, const uint8_t* limit) { return p + n <= limit; }
};

inline uint16_t clamp_u16(uint32_t v) { return v > 0xFFFFu ? 0xFFFFu : static_cast<uint16_t>(v); }

} // namespace

void dapp_export_ul_tti(nv::dapp::Producer& ring, uint16_t cell_id, const scf_fapi_ul_tti_req_t& msg,
                        const nv_ipc_msg_t& ipc, int64_t ts_l2_send_ns)
{
    const uint16_t sfn        = msg.sfn;
    const uint16_t slot       = msg.slot;
    const bool     write_pdus = nv::dapp::export_pdus();

    uint16_t n_pusch = 0, n_pucch = 0, n_prach = 0, n_srs = 0;
    uint32_t tot_prb = 0, tot_layers = 0, tot_tb = 0, tot_cb = 0, tot_prb_layers = 0;
    uint32_t tot_pucch_prb = 0, tot_srs_ports = 0;

    pdu_walker     w(msg.payload, ipc);
    const unsigned num_pdus = msg.num_pdus;
    for (unsigned i = 0; i < num_pdus; ++i)
    {
        const scf_fapi_generic_pdu_info_t* pdu = w.next();
        if (pdu == nullptr) { break; }
        const uint8_t* cfg   = pdu->pdu_config;
        const uint8_t* limit = reinterpret_cast<const uint8_t*>(pdu) + pdu->pdu_size;

        dapp_ul_pdu_t o{};
        o.pdu_index = static_cast<uint8_t>(i);
        o.pdu_type  = static_cast<uint8_t>(pdu->pdu_type);
        o.pdu_size  = pdu->pdu_size;

        switch (pdu->pdu_type)
        {
        case UL_TTI_PDU_TYPE_PUSCH:
        {
            if (!pdu_walker::fits(cfg, sizeof(scf_fapi_pusch_pdu_t), limit)) { break; }
            const auto& p               = *reinterpret_cast<const scf_fapi_pusch_pdu_t*>(cfg);
            o.rnti                      = p.rnti;
            o.handle                    = p.handle;
            o.bwp_size                  = p.bwp.bwp_size;
            o.bwp_start                 = p.bwp.bwp_start;
            o.target_code_rate          = p.target_code_rate;
            o.qam_mod_order             = p.qam_mod_order;
            o.mcs_index                 = p.mcs_index;
            o.mcs_table                 = p.mcs_table;
            o.transform_precoding       = p.transform_precoding;
            o.num_layers                = p.num_of_layers;
            o.ul_dmrs_sym_pos           = p.ul_dmrs_sym_pos;
            o.dmrs_config_type          = p.dmrs_config_type;
            o.num_dmrs_cdm_grps_no_data = p.num_dmrs_cdm_groups_no_data;
            o.rb_start                  = p.rb_start;
            o.rb_size                   = p.rb_size;
            o.start_sym                 = p.start_symbol_index;
            o.num_sym                   = p.num_of_symbols;
            o.pdu_bitmap                = p.pdu_bitmap;

            const uint8_t* next = p.payload;
            if (p.pdu_bitmap & 0x1)
            {
                if (pdu_walker::fits(next, sizeof(scf_fapi_pusch_data_t), limit))
                {
                    const auto& d     = *reinterpret_cast<const scf_fapi_pusch_data_t*>(next);
                    o.rv_index        = d.rv_index;
                    o.harq_process_id = d.harq_process_id;
                    o.ndi             = d.new_data_indicator;
                    o.tb_size         = d.tb_size;
                    o.num_cb          = d.num_cb;
                }
                next += sizeof(scf_fapi_pusch_data_t);
            }
            if (p.pdu_bitmap & 0x2)
            {
                if (pdu_walker::fits(next, sizeof(scf_fapi_pusch_uci_t), limit))
                {
                    const auto& u       = *reinterpret_cast<const scf_fapi_pusch_uci_t*>(next);
                    o.harq_ack_bit_len  = u.harq_ack_bit_length;
                    o.csi_part1_bit_len = u.csi_part_1_bit_length;
                }
            }
            ++n_pusch;
            tot_prb += o.rb_size;
            tot_layers += o.num_layers;
            tot_prb_layers += static_cast<uint32_t>(o.rb_size) * o.num_layers;
            tot_tb += o.tb_size;
            tot_cb += o.num_cb;
            break;
        }
        case UL_TTI_PDU_TYPE_PUCCH:
        {
            if (!pdu_walker::fits(cfg, sizeof(scf_fapi_pucch_pdu_t), limit)) { break; }
            const auto& p  = *reinterpret_cast<const scf_fapi_pucch_pdu_t*>(cfg);
            o.rnti         = p.rnti;
            o.handle       = p.handle;
            o.bwp_size     = p.bwp.bwp_size;
            o.bwp_start    = p.bwp.bwp_start;
            o.format_type  = p.format_type;
            o.rb_start     = p.prb_start;
            o.rb_size      = p.prb_size;
            o.start_sym    = p.start_symbol_index;
            o.num_sym      = p.num_of_symbols;
            o.sr_flag      = p.sr_flag;
            o.bit_len_harq = p.bit_len_harq;
            o.bit_len_csi1 = p.bit_len_csi_part_1;
            o.bit_len_csi2 = p.bit_len_csi_part_2;
            o.num_layers   = 1;
            ++n_pucch;
            tot_pucch_prb += o.rb_size;
            break;
        }
        case UL_TTI_PDU_TYPE_PRACH:
        {
            if (!pdu_walker::fits(cfg, sizeof(scf_fapi_prach_pdu_t), limit)) { break; }
            const auto& p    = *reinterpret_cast<const scf_fapi_prach_pdu_t*>(cfg);
            o.prach_format   = p.prach_format;
            o.num_prach_ocas = p.num_prach_ocas;
            o.num_ra         = p.num_ra;
            o.start_sym      = p.prach_start_symbol;
            ++n_prach;
            break;
        }
        case UL_TTI_PDU_TYPE_SRS:
        {
            if (!pdu_walker::fits(cfg, sizeof(scf_fapi_srs_pdu_t), limit)) { break; }
            const auto& p         = *reinterpret_cast<const scf_fapi_srs_pdu_t*>(cfg);
            o.rnti                = p.rnti;
            o.handle              = p.handle;
            o.bwp_size            = p.bwp.bwp_size;
            o.bwp_start           = p.bwp.bwp_start;
            o.num_layers          = p.num_ant_ports;
            o.srs_num_symbols     = p.num_symbols;
            o.num_sym             = p.num_symbols;
            o.srs_num_repetitions = p.num_repetitions;
            o.srs_comb_size       = p.comb_size;
            o.srs_bandwidth_index = p.bandwidth_index;
            o.srs_config_index    = p.config_index;
            o.start_sym           = p.time_start_position;
            ++n_srs;
            tot_srs_ports += o.num_layers;
            break;
        }
        default:
            break;
        }

        if (write_pdus)
        {
            dapp_rec_t* r = ring.begin(DAPP_REC_UL_PDU, sfn, slot, cell_id);
            r->u.ul_pdu   = o;
            ring.commit(r);
        }
    }

    dapp_rec_t*    r     = ring.begin(DAPP_REC_UL_TTI, sfn, slot, cell_id);
    dapp_ul_tti_t& s     = r->u.ul_tti;
    s.num_pdus           = msg.num_pdus;
    s.rach_present       = msg.rach_present;
    s.num_ulsch          = msg.num_ulsch;
    s.num_ulcch          = msg.num_ulcch;
    s.ngroup             = msg.ngroup;
    s.pdu_truncated      = w.truncated ? 1u : 0u;
    s.msg_len            = ipc.msg_len;
    s.body_len           = msg.msg_hdr.length;
    s.ts_l2_send_ns      = ts_l2_send_ns;
    s.n_pusch            = n_pusch;
    s.n_pucch            = n_pucch;
    s.n_prach            = n_prach;
    s.n_srs              = n_srs;
    s.tot_pusch_prb      = tot_prb;
    s.tot_pusch_layers   = tot_layers;
    s.tot_pusch_tb_bytes = tot_tb;
    s.tot_pusch_cb       = tot_cb;
    s.tot_pusch_prb_layers = tot_prb_layers;
    s.tot_pucch_prb      = tot_pucch_prb;
    s.tot_srs_ports      = tot_srs_ports;
    ring.commit(r);
    ring.count_ul_tti();
    if (w.truncated) { ring.count_pdu_truncated(); }
}

void dapp_export_dl_tti(nv::dapp::Producer& ring, uint16_t cell_id, const scf_fapi_dl_tti_req_t& msg,
                        const nv_ipc_msg_t& ipc, int64_t ts_l2_send_ns)
{
    const uint16_t sfn        = msg.sfn;
    const uint16_t slot       = msg.slot;
    const bool     write_pdus = nv::dapp::export_pdus();

    uint16_t n_pdcch = 0, n_pdsch = 0, n_csirs = 0, n_ssb = 0;
    uint32_t n_dci = 0, tot_prb = 0, tot_layers = 0, tot_tb = 0, tot_prb_layers = 0;
    uint32_t tot_pdsch_prg_bf = 0, tot_dci_al = 0, tot_dci_bits = 0, tot_pdcch_prg_bf = 0;
    uint8_t  max_dci_al = 0, max_pdsch_mcs = 0;

    pdu_walker     w(msg.payload, ipc);
    const unsigned num_pdus = msg.num_pdus;
    for (unsigned i = 0; i < num_pdus; ++i)
    {
        const scf_fapi_generic_pdu_info_t* pdu = w.next();
        if (pdu == nullptr) { break; }
        const uint8_t* cfg   = pdu->pdu_config;
        const uint8_t* limit = reinterpret_cast<const uint8_t*>(pdu) + pdu->pdu_size;

        dapp_dl_pdu_t o{};
        o.pdu_index = static_cast<uint8_t>(i);
        o.pdu_type  = static_cast<uint8_t>(pdu->pdu_type);
        o.pdu_size  = pdu->pdu_size;

        switch (pdu->pdu_type)
        {
        case DL_TTI_PDU_TYPE_PDSCH:
        {
            if (!pdu_walker::fits(cfg, sizeof(scf_fapi_pdsch_pdu_t), limit)) { break; }
            const auto& p    = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(cfg);
            o.pdu_bitmap     = p.pdu_bitmap;
            o.rnti           = p.rnti;
            o.fapi_pdu_index = p.pdu_index;
            o.bwp_size       = p.bwp.bwp_size;
            o.bwp_start      = p.bwp.bwp_start;
            o.num_codewords  = p.num_codewords;

            const uint8_t* cw_ptr = reinterpret_cast<const uint8_t*>(p.codewords);
            if (p.num_codewords >= 1 && pdu_walker::fits(cw_ptr, sizeof(scf_fapi_pdsch_codeword_t), limit))
            {
                const auto& cw     = *reinterpret_cast<const scf_fapi_pdsch_codeword_t*>(cw_ptr);
                o.target_code_rate = cw.target_code_rate;
                o.qam_mod_order    = cw.qam_mod_order;
                o.mcs_index        = cw.mcs_index;
                o.mcs_table        = cw.mcs_table;
                o.rv_index         = cw.rv_index;
                o.tb_size          = cw.tb_size;
            }
            if (p.num_codewords >= 2 &&
                pdu_walker::fits(cw_ptr + sizeof(scf_fapi_pdsch_codeword_t), sizeof(scf_fapi_pdsch_codeword_t), limit))
            {
                const auto& cw1 =
                    *reinterpret_cast<const scf_fapi_pdsch_codeword_t*>(cw_ptr + sizeof(scf_fapi_pdsch_codeword_t));
                o.tb_size_cw1 = cw1.tb_size;
            }
            const uint8_t* end_ptr = cw_ptr + static_cast<size_t>(p.num_codewords) * sizeof(scf_fapi_pdsch_codeword_t);
            if (pdu_walker::fits(end_ptr, sizeof(scf_fapi_pdsch_pdu_end_t), limit))
            {
                const auto& e               = *reinterpret_cast<const scf_fapi_pdsch_pdu_end_t*>(end_ptr);
                o.num_layers                = e.num_of_layers;
                o.dl_dmrs_sym_pos           = e.dl_dmrs_sym_pos;
                o.dmrs_config_type          = e.dmrs_config_type;
                o.num_dmrs_cdm_grps_no_data = e.num_dmrs_cdm_grps_no_data;
                o.rb_start                  = e.rb_start;
                o.rb_size                   = e.rb_size;
                o.start_sym                 = e.start_sym_index;
                o.num_sym                   = e.num_symbols;

                // The precoding/beamforming section follows the fixed tail,
                // after the optional PTRS block when pdu_bitmap bit 0 is set
                // (same walk as scf_5g_slot_commands_pdsch_csirs.cpp).
                const uint8_t* bf_ptr = end_ptr + sizeof(scf_fapi_pdsch_pdu_end_t);
                if (p.pdu_bitmap & 0x1)
                {
                    bf_ptr = pdu_walker::fits(bf_ptr, sizeof(scf_fapi_pdsch_ptrs_t), limit)
                                 ? bf_ptr + sizeof(scf_fapi_pdsch_ptrs_t)
                                 : nullptr;
                }
                if (bf_ptr != nullptr && pdu_walker::fits(bf_ptr, sizeof(scf_fapi_tx_precoding_beamforming_t), limit))
                {
                    const auto& bf      = *reinterpret_cast<const scf_fapi_tx_precoding_beamforming_t*>(bf_ptr);
                    o.num_prgs          = bf.num_prgs;
                    o.prg_size          = bf.prg_size;
                    o.dig_bf_interfaces = bf.dig_bf_interfaces;
                    o.prg_bf_sum        = static_cast<uint32_t>(bf.num_prgs) * bf.dig_bf_interfaces;
                }
            }
            ++n_pdsch;
            tot_prb += o.rb_size;
            tot_layers += o.num_layers;
            tot_prb_layers += static_cast<uint32_t>(o.rb_size) * o.num_layers;
            tot_tb += o.tb_size + o.tb_size_cw1;
            tot_pdsch_prg_bf += o.prg_bf_sum;
            if (o.mcs_index > max_pdsch_mcs) { max_pdsch_mcs = o.mcs_index; }
            break;
        }
        case DL_TTI_PDU_TYPE_PDCCH:
        {
            if (!pdu_walker::fits(cfg, sizeof(scf_fapi_pdcch_pdu_t), limit)) { break; }
            const auto& p = *reinterpret_cast<const scf_fapi_pdcch_pdu_t*>(cfg);
            o.bwp_size    = p.bwp.bwp_size;
            o.bwp_start   = p.bwp.bwp_start;
            o.start_sym   = p.start_sym_index;
            o.num_sym     = p.duration_sym;
            o.num_dl_dci  = p.num_dl_dci;

            // Each DCI is variable length: header, precoding/beamforming block
            // sized by its own num_prgs/dig_bf_interfaces, tx power info, then
            // the payload. Stride formula mirrors scf_5g_slot_commands_pdcch.cpp.
            uint32_t al_sum = 0, bits_sum = 0, prg_bf = 0;
            uint8_t  al_max = 0;
            bool     truncated = false;
            const uint8_t* dci_ptr = reinterpret_cast<const uint8_t*>(p.dl_dci);
            for (uint16_t k = 0; k < p.num_dl_dci; ++k)
            {
                if (!pdu_walker::fits(dci_ptr, sizeof(scf_fapi_dl_dci_t), limit)) { truncated = true; break; }
                const auto&    dci = *reinterpret_cast<const scf_fapi_dl_dci_t*>(dci_ptr);
                const uint8_t* q   = dci_ptr + sizeof(scf_fapi_dl_dci_t);
                if (!pdu_walker::fits(q, sizeof(scf_fapi_tx_precoding_beamforming_t), limit)) { truncated = true; break; }
                const auto&  bf      = *reinterpret_cast<const scf_fapi_tx_precoding_beamforming_t*>(q);
                const size_t bf_size = sizeof(scf_fapi_tx_precoding_beamforming_t) +
                                       static_cast<size_t>(bf.num_prgs) * sizeof(uint16_t) +
                                       static_cast<size_t>(bf.num_prgs) * bf.dig_bf_interfaces * sizeof(uint16_t);
                q += bf_size + sizeof(scf_fapi_pdcch_tx_power_info_t);
                if (!pdu_walker::fits(q, sizeof(scf_fapi_pdcch_dci_payload_t), limit)) { truncated = true; break; }
                const auto& pl = *reinterpret_cast<const scf_fapi_pdcch_dci_payload_t*>(q);
                al_sum   += dci.aggregation_level;
                bits_sum += pl.payload_size_bits;
                prg_bf   += static_cast<uint32_t>(bf.num_prgs) * bf.dig_bf_interfaces;
                if (dci.aggregation_level > al_max) { al_max = dci.aggregation_level; }
                dci_ptr = q + sizeof(scf_fapi_pdcch_dci_payload_t) + (static_cast<size_t>(pl.payload_size_bits) + 7u) / 8u;
                if (dci_ptr > limit) { truncated = true; break; } // payload claims more bytes than the PDU holds
            }
            o.agg_level_sum        = clamp_u16(al_sum);
            o.agg_level_max        = al_max;
            o.dci_payload_bits_sum = clamp_u16(bits_sum);
            o.prg_bf_sum           = prg_bf;
            o.dci_truncated        = truncated ? 1u : 0u;

            ++n_pdcch;
            n_dci += p.num_dl_dci;
            tot_dci_al += al_sum;
            tot_dci_bits += bits_sum;
            tot_pdcch_prg_bf += prg_bf;
            if (al_max > max_dci_al) { max_dci_al = al_max; }
            if (truncated) { w.truncated = true; }
            break;
        }
        case DL_TTI_PDU_TYPE_CSI_RS:
        {
            if (!pdu_walker::fits(cfg, sizeof(scf_fapi_csi_rsi_pdu_t), limit)) { break; }
            const auto& p = *reinterpret_cast<const scf_fapi_csi_rsi_pdu_t*>(cfg);
            o.bwp_size    = p.bwp.bwp_size;
            o.bwp_start   = p.bwp.bwp_start;
            o.rb_start    = p.start_rb;
            o.rb_size     = p.num_of_rbs;
            o.csirs_row   = p.row;
            o.start_sym   = p.sym_l0;
            ++n_csirs;
            break;
        }
        case DL_TTI_PDU_TYPE_SSB:
        {
            if (!pdu_walker::fits(cfg, sizeof(scf_fapi_ssb_pdu_t), limit)) { break; }
            const auto& p     = *reinterpret_cast<const scf_fapi_ssb_pdu_t*>(cfg);
            o.ssb_block_index = p.ssb_block_index;
            ++n_ssb;
            break;
        }
        default:
            break;
        }

        if (write_pdus)
        {
            dapp_rec_t* r = ring.begin(DAPP_REC_DL_PDU, sfn, slot, cell_id);
            r->u.dl_pdu   = o;
            ring.commit(r);
        }
    }

    dapp_rec_t*    r       = ring.begin(DAPP_REC_DL_TTI, sfn, slot, cell_id);
    dapp_dl_tti_t& s       = r->u.dl_tti;
    s.num_pdus             = msg.num_pdus;
    s.ngroup               = msg.ngroup;
    s.pdu_truncated        = w.truncated ? 1u : 0u;
    s.msg_len              = ipc.msg_len;
    s.body_len             = msg.msg_hdr.length;
    s.ts_l2_send_ns        = ts_l2_send_ns;
    s.n_pdcch              = n_pdcch;
    s.n_pdsch              = n_pdsch;
    s.n_csirs              = n_csirs;
    s.n_ssb                = n_ssb;
    s.n_dci                = n_dci;
    s.tot_pdsch_prb        = tot_prb;
    s.tot_pdsch_layers     = tot_layers;
    s.tot_pdsch_tb_bytes   = tot_tb;
    s.tot_pdsch_prb_layers = tot_prb_layers;
    s.tot_pdsch_prg_bf     = tot_pdsch_prg_bf;
    s.tot_dci_agg_level    = tot_dci_al;
    s.tot_dci_payload_bits = tot_dci_bits;
    s.max_dci_agg_level    = max_dci_al;
    s.max_pdsch_mcs        = max_pdsch_mcs;
    s.tot_pdcch_prg_bf     = tot_pdcch_prg_bf;
    ring.commit(r);
    ring.count_dl_tti();
    if (w.truncated) { ring.count_pdu_truncated(); }
}

} // namespace scf_5g_fapi

#endif // ENABLE_DAPP_HOOK
