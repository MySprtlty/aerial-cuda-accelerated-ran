"""
Pure-Python reader for dApp ring records (ABI v2, cuPHY-CP/dapp_hook/include/
dapp_hook/dapp_ring_abi.h). Works on a live ring (/dev/shm/aerial_dapp_ring),
on a raw copy of the shm file, and on a record stream written by
tools/dapp_ring_record.py. No numpy: the live loop must not depend on it.

The struct layouts below are checked against python/dapp_ring.py (which is
checked against the C header) in tests/test_ringio.py.
"""
import mmap
import os
import struct
from collections import namedtuple

ABI_VERSION = 2
HDR_SIZE = 4096
REC_SIZE = 128
MAGIC = b"DAPPRNG1"

REC_UL_PDU, REC_UL_TTI, REC_DL_PDU, REC_DL_TTI, REC_SLOT_END, REC_PRODUCER_START = 1, 2, 3, 4, 5, 6
UL_PRACH, UL_PUSCH, UL_PUCCH, UL_SRS = 0, 1, 2, 3
DL_PDCCH, DL_PDSCH, DL_CSI_RS, DL_SSB = 0, 1, 2, 3

# ---- layouts (little endian, natural alignment, no implicit padding) ----
COMMON = struct.Struct("<QQHHHH")          # seq, ts_ns, type, sfn, slot, cell_id  (24 B)
COMMON_FIELDS = ("seq", "ts_ns", "type", "sfn", "slot", "cell_id")

UL_PDU_FMT = "<BBBBHHIHHBBBBHHIHHBBBBBBHHHHBBHHHBBBBBBBBH"
UL_PDU_FIELDS = ("pdu_index", "pdu_type", "start_sym", "num_sym", "pdu_size", "rnti", "handle",
                 "rb_start", "rb_size", "num_layers", "mcs_index", "mcs_table", "qam_mod_order",
                 "target_code_rate", "pdu_bitmap", "tb_size", "num_cb", "ul_dmrs_sym_pos",
                 "transform_precoding", "dmrs_config_type", "num_dmrs_cdm_grps_no_data", "rv_index",
                 "harq_process_id", "ndi", "bwp_size", "bwp_start", "harq_ack_bit_len",
                 "csi_part1_bit_len", "format_type", "sr_flag", "bit_len_harq", "bit_len_csi1",
                 "bit_len_csi2", "prach_format", "num_prach_ocas", "num_ra", "srs_num_symbols",
                 "srs_num_repetitions", "srs_comb_size", "srs_bandwidth_index", "srs_config_index",
                 "reserved")

UL_TTI_FMT = "<BBBBBBHIIqHHHHIIIIIIII"
UL_TTI_FIELDS = ("num_pdus", "rach_present", "num_ulsch", "num_ulcch", "ngroup", "pdu_truncated",
                 "reserved0", "msg_len", "body_len", "ts_l2_send_ns", "n_pusch", "n_pucch", "n_prach",
                 "n_srs", "tot_pusch_prb", "tot_pusch_layers", "tot_pusch_tb_bytes", "tot_pusch_cb",
                 "tot_pusch_prb_layers", "tot_pucch_prb", "tot_srs_ports", "reserved1")

DL_PDU_FMT = "<BBBBHHHHBBBBBBHIIHHHHBBHHBBHHBBHHBBII"
DL_PDU_FIELDS = ("pdu_index", "pdu_type", "start_sym", "num_sym", "pdu_size", "rnti", "rb_start",
                 "rb_size", "num_layers", "num_codewords", "mcs_index", "mcs_table", "qam_mod_order",
                 "rv_index", "target_code_rate", "tb_size", "tb_size_cw1", "num_dl_dci", "pdu_bitmap",
                 "bwp_size", "bwp_start", "dmrs_config_type", "num_dmrs_cdm_grps_no_data",
                 "dl_dmrs_sym_pos", "fapi_pdu_index", "csirs_row", "ssb_block_index", "num_prgs",
                 "prg_size", "dig_bf_interfaces", "agg_level_max", "agg_level_sum",
                 "dci_payload_bits_sum", "dci_truncated", "reserved8", "prg_bf_sum", "reserved")

DL_TTI_FMT = "<BBBBIIIqHHHHIIIIIIIIBBHI"
DL_TTI_FIELDS = ("num_pdus", "ngroup", "pdu_truncated", "reserved0", "msg_len", "body_len", "reserved1",
                 "ts_l2_send_ns", "n_pdcch", "n_pdsch", "n_csirs", "n_ssb", "n_dci", "tot_pdsch_prb",
                 "tot_pdsch_layers", "tot_pdsch_tb_bytes", "tot_pdsch_prb_layers", "tot_pdsch_prg_bf",
                 "tot_dci_agg_level", "tot_dci_payload_bits", "max_dci_agg_level", "max_pdsch_mcs",
                 "reserved2", "tot_pdcch_prg_bf")

SLOT_END_FMT = "<BBBBB3xiIIIqqqqqq"
SLOT_END_FIELDS = ("enqueued", "slot_end_rcvd", "is_ul", "is_dl", "is_csirs", "enqueue_ret", "num_cells",
                   "cmd_size", "reserved1", "tick_original_ns", "t0_ns", "l2a_latency_ns",
                   "l1_slot_ind_tick_ns", "l2a_start_ns", "l2a_end_ns")

START_FMT = "<IIIIIIII"
START_FIELDS = ("pid", "generation", "slot_advance", "mu", "num_cells", "ring_len", "abi_version", "rec_size")

_LAYOUT = {
    REC_UL_PDU: (struct.Struct(UL_PDU_FMT), UL_PDU_FIELDS, 64),
    REC_UL_TTI: (struct.Struct(UL_TTI_FMT), UL_TTI_FIELDS, 64),
    REC_DL_PDU: (struct.Struct(DL_PDU_FMT), DL_PDU_FIELDS, 64),
    REC_DL_TTI: (struct.Struct(DL_TTI_FMT), DL_TTI_FIELDS, 72),
    REC_SLOT_END: (struct.Struct(SLOT_END_FMT), SLOT_END_FIELDS, 72),
    REC_PRODUCER_START: (struct.Struct(START_FMT), START_FIELDS, 32),
}
for _t, (_s, _f, _sz) in _LAYOUT.items():
    assert _s.size == _sz, (_t, _s.size, _sz)
    assert len(_f) == len(_s.unpack(b"\0" * _s.size)), _t

_TUPLES = {t: namedtuple("Rec%d" % t, COMMON_FIELDS + f) for t, (_, f, _) in _LAYOUT.items()}
Header = namedtuple("Header", "abi_version rec_size ring_len hdr_size generation producer_pid start_ts_ns "
                              "head heartbeat_ns slot_advance mu num_cells flags")
_HDR = struct.Struct("<8sIIIIIIQQQIIII")


def decode(buf, off=0):
    """Decode one 128-byte record at buf[off:]. Returns a namedtuple or None for an empty/unknown slot."""
    seq, ts, typ, sfn, slot, cell = COMMON.unpack_from(buf, off)
    lay = _LAYOUT.get(typ)
    if seq == 0 or lay is None:
        return None
    s, _, _ = lay
    return _TUPLES[typ]._make((seq, ts, typ, sfn, slot, cell) + s.unpack_from(buf, off + COMMON.size))


def decode_header(buf):
    magic, abi, rec, rl, hs, gen, pid, sts, head, hb, sa, mu, nc, fl = _HDR.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError("bad ring magic %r" % magic)
    if abi != ABI_VERSION or rec != REC_SIZE or hs != HDR_SIZE:
        raise ValueError("ABI mismatch: abi=%d rec=%d hdr=%d" % (abi, rec, hs))
    return Header(abi, rec, rl, hs, gen, pid, sts, head, hb, sa, mu, nc, fl)


class RecordFile:
    """Iterate records from (a) a raw stream file from dapp_ring_record.py or (b) a copy of the shm image."""

    def __init__(self, path, hdr_path=None):
        self.path = path
        self.header = None
        with open(path, "rb") as f:
            self.data = f.read()
        if self.data[:8] == MAGIC:
            self.header = decode_header(self.data)
            self.body_off = HDR_SIZE
            self.image = True
        else:
            self.body_off = 0
            self.image = False
            hp = hdr_path or (path + ".hdr")
            if os.path.exists(hp):
                with open(hp, "rb") as f:
                    self.header = decode_header(f.read())
        self.n = (len(self.data) - self.body_off) // REC_SIZE

    def __iter__(self):
        d, o = self.data, self.body_off
        recs = []
        for i in range(self.n):
            r = decode(d, o + i * REC_SIZE)
            if r is not None:
                recs.append(r)
        if self.image:
            recs.sort(key=lambda r: r.seq)
        return iter(recs)


class LiveRing:
    """Follow the live shm ring. next() returns one record, or None if nothing new. Detects lag."""

    def __init__(self, name="/aerial_dapp_ring", margin_div=8):
        path = name if name.startswith("/dev/shm/") else "/dev/shm/" + name.lstrip("/")
        self.fd = os.open(path, os.O_RDONLY)
        size = os.fstat(self.fd).st_size
        self.mm = mmap.mmap(self.fd, size, prot=mmap.PROT_READ)
        self.header = decode_header(self.mm)
        self.ring_len = self.header.ring_len
        self.mask = self.ring_len - 1
        self.margin = self.ring_len // margin_div
        self.generation = self.header.generation
        self.cursor = max(1, self.head() - self.ring_len + self.margin + 1)
        self.lost = 0

    def head(self):
        return struct.unpack_from("<Q", self.mm, 40)[0]

    def heartbeat_ns(self):
        return struct.unpack_from("<Q", self.mm, 48)[0]

    def seek_to_head(self):
        self.cursor = self.head() + 1

    def next(self):
        head = self.head()
        gen = struct.unpack_from("<I", self.mm, 24)[0]
        if gen != self.generation:
            self.generation = gen
            self.cursor = 1
            return "RESTARTED"
        if self.cursor > head:
            return None
        if head - self.cursor + 1 + self.margin > self.ring_len:
            lost = head - self.ring_len + self.margin + 1 - self.cursor
            self.lost += lost
            self.cursor += lost
        off = HDR_SIZE + (self.cursor & self.mask) * REC_SIZE
        seq = struct.unpack_from("<Q", self.mm, off)[0]
        if seq != self.cursor:
            return None   # being written, or already overwritten: try again
        r = decode(self.mm, off)
        if r is None:
            self.cursor += 1
            return None
        # re-check seq after the payload read (seqlock); a torn read shows a changed seq
        if struct.unpack_from("<Q", self.mm, off)[0] != self.cursor:
            return None
        self.cursor += 1
        return r
