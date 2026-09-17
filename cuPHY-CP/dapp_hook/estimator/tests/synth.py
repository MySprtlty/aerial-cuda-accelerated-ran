"""Synthetic ring records/images for tests (never the production ring name)."""
import os
import struct

from dapp_est import ringio as R

COMMON_FIELDS = ("seq", "ts_ns", "type", "sfn", "slot", "cell_id")


def pack_record(seq, ts_ns, typ, sfn, slot, cell, **payload):
    s, fields, _ = R._LAYOUT[typ]
    vals = []
    for f in fields:
        vals.append(payload.pop(f, 0))
    if payload:
        raise KeyError("unknown fields for type %d: %s" % (typ, sorted(payload)))
    body = R.COMMON.pack(seq, ts_ns, typ, sfn, slot, cell) + s.pack(*vals)
    return body + b"\0" * (R.REC_SIZE - len(body))


def make_rec(typ, sfn=0, slot=0, cell=0, seq=1, ts_ns=0, **payload):
    return R.decode(pack_record(seq, ts_ns, typ, sfn, slot, cell, **payload))


def pusch(sfn, slot, cell, rnti, rb=273, sym=14, layers=2, qam=6, tb=50000, cr=8000, dmrs=0x804, **kw):
    return dict(pdu_type=R.UL_PUSCH, rnti=rnti, rb_size=rb, num_sym=sym, num_layers=layers, qam_mod_order=qam,
                tb_size=tb, target_code_rate=cr, ul_dmrs_sym_pos=dmrs, pdu_bitmap=1, **kw)


def pdsch(rnti, rb=273, sym=12, layers=4, tb=100000, prg=1):
    return dict(pdu_type=R.DL_PDSCH, rnti=rnti, rb_size=rb, num_sym=sym, num_layers=layers, tb_size=tb, tb_size_cw1=0,
                prg_bf_sum=prg)


def slot_records(seq, ts_ns, sfn, slot, cells, ul, n_ue=3, heavy=1.0):
    """Records of one slot: PDUs for every cell, then SLOT_END. Returns (records, next_seq)."""
    recs = []
    for cell in range(cells):
        for u in range(n_ue):
            if ul:
                recs.append(pack_record(seq, ts_ns, R.REC_UL_PDU, sfn, slot, cell,
                                        **pusch(sfn, slot, cell, 100 + u, tb=int(50000 * heavy))))
            else:
                recs.append(pack_record(seq, ts_ns, R.REC_DL_PDU, sfn, slot, cell,
                                        **pdsch(100 + u, tb=int(100000 * heavy))))
            seq += 1
            ts_ns += 1000
    recs.append(pack_record(seq, ts_ns, R.REC_SLOT_END, sfn, slot, 0, t0_ns=ts_ns + 1500000, tick_original_ns=ts_ns,
                            enqueued=1, is_ul=int(ul), is_dl=int(not ul), num_cells=cells))
    seq += 1
    return recs, seq


def make_stream(n_slots=60, cells=2, n_ue=3, t_start=1000000000000, heavy_slots=()):
    """A DDDSUUDDDD-like stream: slots 4,5 of every 10 are UL. Returns (bytes, n_records)."""
    seq = 1
    ts = t_start
    out = []
    for i in range(n_slots):
        sfn, slot = i // 20, i % 20
        ul = (slot % 10) in (4, 5)
        heavy = 3.0 if i in heavy_slots else 1.0
        recs, seq = slot_records(seq, ts, sfn, slot, cells, ul, n_ue, heavy)
        out.extend(recs)
        ts += 500000
    return b"".join(out), seq - 1


def make_image(stream, n_records, ring_len=None, mu=1, slot_advance=3, num_cells=2):
    """Wrap a record stream into an shm-image (4096 B header + ring body)."""
    if ring_len is None:
        ring_len = 1
        while ring_len < n_records + 1:
            ring_len *= 2
    hdr = R._HDR.pack(R.MAGIC, R.ABI_VERSION, R.REC_SIZE, ring_len, R.HDR_SIZE, 1, os.getpid(), 0,
                      n_records, 0, slot_advance, mu, num_cells, 0)
    hdr += b"\0" * (R.HDR_SIZE - len(hdr))
    body = bytearray(ring_len * R.REC_SIZE)
    for i in range(n_records):
        seq = i + 1
        off = (seq & (ring_len - 1)) * R.REC_SIZE
        body[off:off + R.REC_SIZE] = stream[i * R.REC_SIZE:(i + 1) * R.REC_SIZE]
    return hdr + bytes(body)
