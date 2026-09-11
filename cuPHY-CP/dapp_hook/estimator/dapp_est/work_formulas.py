"""
Per-PDU "work" of every cuPHY kernel group, computed from FAPI fields only.

PROVISIONAL. These are proxies for the amount of GPU work each kernel group
does, chosen so that measured kernel durations should scale roughly
linearly with them. They are the only place such formulas live; when
kernel_map.md (kernel-by-kernel measurements) exists, change them here and
re-run calibrate.py to refit the weights in rules.yaml.

Channel -> kernel groups:
  PUSCH : chest, eq, llr, ldpc, uci
  PDSCH : enc, mod, prec
  PDCCH : agg, bits
  PUCCH : f01, f234
  PRACH : ocas               (time-domain occasions)
  SRS   : ports
  SSB   : present            (fixed cost group, 0/1)
  CSIRS : count              (fixed cost group, number of PDUs)
"""
from . import ringio as R

PROVISIONAL = True

CHANNELS = ("PUSCH", "PUCCH", "PRACH", "SRS", "PDSCH", "PDCCH", "SSB", "CSIRS")
KERNELS = {
    "PUSCH": ("chest", "eq", "llr", "ldpc", "uci"),
    "PDSCH": ("enc", "mod", "prec"),
    "PDCCH": ("agg", "bits"),
    "PUCCH": ("f01", "f234"),
    "PRACH": ("ocas",),
    "SRS": ("ports",),
    "SSB": ("present",),
    "CSIRS": ("count",),
}
UL_CHANNELS = ("PUSCH", "PUCCH", "PRACH", "SRS")
DL_CHANNELS = ("PDSCH", "PDCCH", "SSB", "CSIRS")

SC_PER_PRB = 12

# 38.212 5.2.2 LDPC segmentation
_LIFTING = sorted({2, 4, 8, 16, 32, 64, 128, 256, 3, 6, 12, 24, 48, 96, 192, 384, 5, 10, 20, 40, 80, 160, 320,
                   7, 14, 28, 56, 112, 224, 9, 18, 36, 72, 144, 288, 11, 22, 44, 88, 176, 352, 13, 26, 52, 104,
                   208, 15, 30, 60, 120, 240})
_KCB_BG1, _KCB_BG2 = 8448, 3840
_CRC_TB, _CRC_CB = 24, 24


def popcount(x):
    return bin(int(x) & 0xFFFFFFFF).count("1")


def ldpc_segmentation(tb_bytes, target_code_rate_x10240):
    """(num_cb, Zc, bg) per 38.212 5.2.2 / 7.2.2 from TB size in bytes and code rate * 10240 (FAPI unit)."""
    A = int(tb_bytes) * 8
    if A <= 0:
        return 0, 0, 0
    Rr = float(target_code_rate_x10240) / 10240.0
    if A <= 292 or (A <= 3824 and Rr <= 0.67) or Rr <= 0.25:
        bg = 2
    else:
        bg = 1
    Kcb = _KCB_BG1 if bg == 1 else _KCB_BG2
    B = A + (16 if (bg == 2 and A <= 3824) else _CRC_TB)
    if B <= Kcb:
        C, L = 1, 0
        Bp = B
    else:
        L = _CRC_CB
        C = -(-B // (Kcb - L))
        Bp = B + C * L
    Kp = -(-Bp // C)
    if bg == 1:
        Kb = 22
    else:
        Kb = 10 if B > 640 else 9 if B > 560 else 8 if B > 192 else 6
    Zc = next((z for z in _LIFTING if Kb * z >= Kp), _LIFTING[-1])
    return C, Zc, bg


def work_ul_pdu(p):
    """p: ringio UL_PDU record. Returns (channel, {kernel: work})."""
    t = p.pdu_type
    if t == R.UL_PUSCH:
        dmrs = popcount(p.ul_dmrs_sym_pos)
        data_sym = max(0, p.num_sym - dmrs)
        chest = p.rb_size * SC_PER_PRB * dmrs * p.num_layers
        eq = p.rb_size * SC_PER_PRB * data_sym * p.num_layers
        llr = eq * p.qam_mod_order
        ncb = p.num_cb
        if p.pdu_bitmap & 0x1 and p.tb_size > 0:
            C, Zc, _ = ldpc_segmentation(p.tb_size, p.target_code_rate)
            if ncb == 0:
                ncb = C          # testMAC leaves num_cb = 0; derive it
            ldpc = ncb * Zc if Zc > 0 else ncb
        else:
            ldpc = 0
        uci = (p.harq_ack_bit_len + p.csi_part1_bit_len) if (p.pdu_bitmap & 0x2) else 0
        return "PUSCH", {"chest": chest, "eq": eq, "llr": llr, "ldpc": ldpc, "uci": uci}
    if t == R.UL_PUCCH:
        bits = p.bit_len_harq + p.bit_len_csi1 + p.bit_len_csi2
        if p.format_type in (0, 1):
            return "PUCCH", {"f01": 1, "f234": 0}
        return "PUCCH", {"f01": 0, "f234": bits * p.rb_size}
    if t == R.UL_PRACH:
        # num_ra is the frequency-domain occasion *index* (0..7) in SCF FAPI, not a count
        return "PRACH", {"ocas": max(1, p.num_prach_ocas)}
    if t == R.UL_SRS:
        return "SRS", {"ports": p.num_layers * max(1, p.srs_num_symbols)}
    return None, {}


def work_dl_pdu(p):
    """p: ringio DL_PDU record. Returns (channel, {kernel: work})."""
    t = p.pdu_type
    if t == R.DL_PDSCH:
        enc = p.tb_size + p.tb_size_cw1
        mod = p.rb_size * SC_PER_PRB * p.num_sym * p.num_layers
        prec = p.prg_bf_sum * p.num_layers
        return "PDSCH", {"enc": enc, "mod": mod, "prec": prec}
    if t == R.DL_PDCCH:
        return "PDCCH", {"agg": p.agg_level_sum, "bits": p.dci_payload_bits_sum}
    if t == R.DL_SSB:
        return "SSB", {"present": 1}
    if t == R.DL_CSI_RS:
        return "CSIRS", {"count": 1}
    return None, {}
