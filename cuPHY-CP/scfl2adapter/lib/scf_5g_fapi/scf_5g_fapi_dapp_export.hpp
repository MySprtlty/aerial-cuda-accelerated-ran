/*
 * dApp hook: export UL_TTI.request / DL_TTI.request contents to the
 * shared-memory ring (see cuPHY-CP/dapp_hook). Called on the L2 adapter
 * msg_processing thread; must stay allocation-free and bounded.
 */
#ifndef SCF_5G_FAPI_DAPP_EXPORT_HPP_INCLUDED_
#define SCF_5G_FAPI_DAPP_EXPORT_HPP_INCLUDED_

#ifdef ENABLE_DAPP_HOOK

#include "scf_5g_fapi.h"
#include "nv_ipc.h"
#include "dapp_hook/dapp_ring.hpp"

namespace scf_5g_fapi
{

// Writes one DAPP_REC_UL_PDU record per PDU followed by one DAPP_REC_UL_TTI
// summary record. `ipc` bounds the walk (never reads past msg_buf + msg_len).
void dapp_export_ul_tti(nv::dapp::Producer& ring, uint16_t cell_id, const scf_fapi_ul_tti_req_t& msg,
                        const nv_ipc_msg_t& ipc, int64_t ts_l2_send_ns);

// Same for DL_TTI.request: DAPP_REC_DL_PDU records followed by DAPP_REC_DL_TTI.
void dapp_export_dl_tti(nv::dapp::Producer& ring, uint16_t cell_id, const scf_fapi_dl_tti_req_t& msg,
                        const nv_ipc_msg_t& ipc, int64_t ts_l2_send_ns);

} // namespace scf_5g_fapi

#endif // ENABLE_DAPP_HOOK
#endif // SCF_5G_FAPI_DAPP_EXPORT_HPP_INCLUDED_
