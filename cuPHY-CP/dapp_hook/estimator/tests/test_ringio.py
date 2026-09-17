import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

from dapp_est import ringio as R           # noqa: E402
from dapp_est import work_formulas as W    # noqa: E402
from dapp_est import slot_context as S     # noqa: E402


class LayoutAgainstNumpyReader(unittest.TestCase):
    """The pure-Python struct layouts must match python/dapp_ring.py (itself checked against the C header)."""

    def _check(self, fmt, fields, np_dtype):
        import numpy as np  # test-only dependency
        st = struct.Struct(fmt)
        # offsets of each struct field: walk the format
        offs = []
        pos = 0
        i = 1
        while i < len(fmt):
            ch = fmt[i]
            n = ""
            while ch.isdigit():
                n += ch
                i += 1
                ch = fmt[i]
            cnt = int(n) if n else 1
            if ch == "x":
                pos += cnt
                i += 1
                continue
            size = struct.calcsize("<" + ch)
            offs.append((pos, size))
            pos += size * cnt
            i += 1
        self.assertEqual(len(offs), len(fields))
        P = 24
        for (o, sz), name in zip(offs, fields):
            if name.startswith("reserved"):
                continue
            self.assertIn(name, np_dtype.names, name)
            self.assertEqual(np_dtype.fields[name][1], P + o, "offset of %s" % name)
            self.assertEqual(np_dtype[name].itemsize, sz, "size of %s" % name)

    def test_layouts(self):
        import dapp_ring as dr
        self._check(R.UL_PDU_FMT, R.UL_PDU_FIELDS, dr.UL_PDU_DTYPE)
        self._check(R.UL_TTI_FMT, R.UL_TTI_FIELDS, dr.UL_TTI_DTYPE)
        self._check(R.DL_PDU_FMT, R.DL_PDU_FIELDS, dr.DL_PDU_DTYPE)
        self._check(R.DL_TTI_FMT, R.DL_TTI_FIELDS, dr.DL_TTI_DTYPE)
        self._check(R.SLOT_END_FMT, R.SLOT_END_FIELDS, dr.SLOT_END_DTYPE)

    def test_decode_roundtrip(self):
        buf = bytearray(128)
        R.COMMON.pack_into(buf, 0, 42, 123456789, R.REC_UL_PDU, 951, 17, 3)
        s = struct.Struct(R.UL_PDU_FMT)
        vals = [0] * len(R.UL_PDU_FIELDS)
        f = list(R.UL_PDU_FIELDS)
        vals[f.index("pdu_type")] = R.UL_PUSCH
        vals[f.index("rb_size")] = 100
        vals[f.index("num_sym")] = 14
        vals[f.index("num_layers")] = 2
        vals[f.index("ul_dmrs_sym_pos")] = 0x4
        vals[f.index("qam_mod_order")] = 8
        vals[f.index("tb_size")] = 10434
        vals[f.index("target_code_rate")] = 6825
        vals[f.index("pdu_bitmap")] = 1
        s.pack_into(buf, 24, *vals)
        r = R.decode(bytes(buf))
        self.assertEqual((r.seq, r.sfn, r.slot, r.cell_id, r.rb_size, r.tb_size), (42, 951, 17, 3, 100, 10434))
        ch, w = W.work_ul_pdu(r)
        self.assertEqual(ch, "PUSCH")
        self.assertEqual(w["chest"], 100 * 12 * 1 * 2)
        self.assertEqual(w["eq"], 100 * 12 * 13 * 2)
        self.assertEqual(w["llr"], w["eq"] * 8)
        self.assertGreater(w["ldpc"], 0)


class LdpcSegmentation(unittest.TestCase):
    def test_small_tb_bg2_single_cb(self):
        C, Zc, bg = W.ldpc_segmentation(100, 3000)   # 800 bits, R=0.29 -> BG2, one code block
        self.assertEqual((C, bg), (1, 2))
        self.assertEqual(Zc, 88)                        # Kb=10: 10*Zc >= 800+16 -> 88

    def test_large_tb_bg1(self):
        C, Zc, bg = W.ldpc_segmentation(73000, 9000)  # 584 kbit, R=0.88 -> BG1, Kcb 8448
        self.assertEqual(bg, 1)
        B = 73000 * 8 + 24
        self.assertEqual(C, -(-B // (8448 - 24)))
        self.assertEqual(Zc, 384)

    def test_mcs20_example_from_ring(self):
        C, Zc, bg = W.ldpc_segmentation(2049, 6825)   # 16392 bits, R=0.67 -> BG1 (A>3824), C=2
        self.assertEqual((bg, C), (1, 2))
        Kp = -(-(2049 * 8 + 24 + 2 * 24) // 2)
        self.assertGreaterEqual(22 * Zc, Kp)


class ContextFromRecords(unittest.TestCase):
    def _rec(self, typ, sfn, slot, cell, **kw):
        buf = bytearray(128)
        R.COMMON.pack_into(buf, 0, self.seq, self.seq * 1000, typ, sfn, slot, cell)
        self.seq += 1
        s, fields, _ = R._LAYOUT[typ]
        vals = [0] * len(fields)
        for k, v in kw.items():
            vals[list(fields).index(k)] = v
        s.pack_into(buf, 24, *vals)
        return R.decode(bytes(buf))

    def test_two_cells_one_slot(self):
        self.seq = 1
        b = S.ContextBuilder()
        recs = [
            self._rec(R.REC_DL_PDU, 1, 2, 0, pdu_type=R.DL_PDSCH, rnti=1, rb_size=273, num_sym=12, num_layers=4, tb_size=100000, prg_bf_sum=4),
            self._rec(R.REC_DL_PDU, 1, 2, 0, pdu_type=R.DL_PDCCH, agg_level_sum=8, dci_payload_bits_sum=80),
            self._rec(R.REC_DL_TTI, 1, 2, 0, num_pdus=2, n_pdsch=1, n_pdcch=1),
            self._rec(R.REC_DL_PDU, 1, 2, 1, pdu_type=R.DL_PDSCH, rnti=2, rb_size=100, num_sym=12, num_layers=2, tb_size=30000, prg_bf_sum=2),
            self._rec(R.REC_DL_TTI, 1, 2, 1, num_pdus=1, n_pdsch=1),
            self._rec(R.REC_SLOT_END, 1, 2, 0xFFFF, enqueued=1, is_dl=1, num_cells=2, t0_ns=5000, tick_original_ns=3500),
        ]
        closed = []
        for r in recs:
            closed += b.feed(r)
        self.assertEqual(len(closed), 1)
        c = closed[0]
        self.assertTrue(c.complete)
        self.assertEqual(c.n_cells(), 2)
        self.assertEqual(c.ch["PDSCH"].count, 2)
        self.assertEqual(c.ch["PDSCH"].total["enc"], 130000)
        self.assertEqual(c.ch["PDSCH"].critical["enc"], 100000)
        self.assertEqual(c.ch["PDSCH"].total["mod"], 273 * 12 * 12 * 4 + 100 * 12 * 12 * 2)
        self.assertEqual(c.ch["PDCCH"].total["agg"], 8)
        self.assertEqual(len(c.ch["PDSCH"].rntis), 2)
        self.assertEqual(c.t0_ns, 5000)
        self.assertEqual(sorted(c.cell_work.keys()), [0, 1])


if __name__ == "__main__":
    unittest.main()
