#!/usr/bin/env python3
"""
Reader for the dApp FAPI hook ring exported by cuphycontroller
(cuPHY-CP/dapp_hook). Pure Python + numpy, no libnvipc needed.

    from dapp_ring import DappRing
    ring = DappRing("/aerial_dapp_ring")      # /dev/shm/aerial_dapp_ring
    for rec in ring.follow():                 # generator of numpy structured records
        if rec["type"] == REC_UL_TTI:
            ...

CLI:
    python3 dapp_ring.py [--name /aerial_dapp_ring] [--follow] [--pdus] [--slots N]
    python3 dapp_ring.py --verify              # per-slot aggregation consistency check

The layout mirrors include/dapp_hook/dapp_ring_abi.h exactly (offsets are
explicit and checked against the header fields at open time).
"""
import argparse
import mmap
import os
import sys
import time
from collections import defaultdict

import numpy as np

MAGIC = 0x31474E5250504144
ABI_VERSION = 2
HDR_SIZE = 4096
REC_SIZE = 128

REC_NONE, REC_UL_PDU, REC_UL_TTI, REC_DL_PDU, REC_DL_TTI, REC_SLOT_END, REC_PRODUCER_START = range(7)
REC_NAMES = {REC_UL_PDU: "UL_PDU", REC_UL_TTI: "UL_TTI", REC_DL_PDU: "DL_PDU", REC_DL_TTI: "DL_TTI",
             REC_SLOT_END: "SLOT_END", REC_PRODUCER_START: "PRODUCER_START"}
UL_PDU_NAMES = {0: "PRACH", 1: "PUSCH", 2: "PUCCH", 3: "SRS"}
DL_PDU_NAMES = {0: "PDCCH", 1: "PDSCH", 2: "CSI_RS", 3: "SSB"}

HDR_DTYPE = np.dtype({
    "names": ["magic", "abi_version", "rec_size", "ring_len", "hdr_size", "generation", "producer_pid",
              "start_ts_ns", "head", "heartbeat_ns", "slot_advance", "mu", "num_cells", "flags",
              "cnt_ul_tti", "cnt_dl_tti", "cnt_slot_end", "cnt_slot_dropped", "cnt_pdu_truncated", "cnt_records",
              "producer_name"],
    "formats": ["<u8", "<u4", "<u4", "<u4", "<u4", "<u4", "<u4",
                "<u8", "<u8", "<u8", "<u4", "<u4", "<u4", "<u4",
                "<u8", "<u8", "<u8", "<u8", "<u8", "<u8",
                "S64"],
    "offsets": [0, 8, 12, 16, 20, 24, 28,
                32, 40, 48, 56, 60, 64, 68,
                72, 80, 88, 96, 104, 112,
                120],
    "itemsize": HDR_SIZE,
})

# Common prefix (24 bytes) + every payload variant as overlapping views.
COMMON = [("seq", "<u8", 0), ("ts_ns", "<u8", 8), ("type", "<u2", 16), ("sfn", "<u2", 18), ("slot", "<u2", 20), ("cell_id", "<u2", 22)]
P = 24  # payload offset


def _dtype(fields):
    names, formats, offsets = zip(*fields)
    return np.dtype({"names": list(names), "formats": list(formats), "offsets": list(offsets), "itemsize": REC_SIZE})


REC_DTYPE = _dtype(COMMON)

UL_PDU_DTYPE = _dtype(COMMON + [
    ("pdu_index", "u1", P + 0), ("pdu_type", "u1", P + 1), ("start_sym", "u1", P + 2), ("num_sym", "u1", P + 3),
    ("pdu_size", "<u2", P + 4), ("rnti", "<u2", P + 6), ("handle", "<u4", P + 8),
    ("rb_start", "<u2", P + 12), ("rb_size", "<u2", P + 14),
    ("num_layers", "u1", P + 16), ("mcs_index", "u1", P + 17), ("mcs_table", "u1", P + 18), ("qam_mod_order", "u1", P + 19),
    ("target_code_rate", "<u2", P + 20), ("pdu_bitmap", "<u2", P + 22), ("tb_size", "<u4", P + 24),
    ("num_cb", "<u2", P + 28), ("ul_dmrs_sym_pos", "<u2", P + 30),
    ("transform_precoding", "u1", P + 32), ("dmrs_config_type", "u1", P + 33), ("num_dmrs_cdm_grps_no_data", "u1", P + 34),
    ("rv_index", "u1", P + 35), ("harq_process_id", "u1", P + 36), ("ndi", "u1", P + 37),
    ("bwp_size", "<u2", P + 38), ("bwp_start", "<u2", P + 40), ("harq_ack_bit_len", "<u2", P + 42), ("csi_part1_bit_len", "<u2", P + 44),
    ("format_type", "u1", P + 46), ("sr_flag", "u1", P + 47), ("bit_len_harq", "<u2", P + 48), ("bit_len_csi1", "<u2", P + 50),
    ("bit_len_csi2", "<u2", P + 52), ("prach_format", "u1", P + 54), ("num_prach_ocas", "u1", P + 55), ("num_ra", "u1", P + 56),
    ("srs_num_symbols", "u1", P + 57), ("srs_num_repetitions", "u1", P + 58), ("srs_comb_size", "u1", P + 59),
    ("srs_bandwidth_index", "u1", P + 60), ("srs_config_index", "u1", P + 61),
])

UL_TTI_DTYPE = _dtype(COMMON + [
    ("num_pdus", "u1", P + 0), ("rach_present", "u1", P + 1), ("num_ulsch", "u1", P + 2), ("num_ulcch", "u1", P + 3),
    ("ngroup", "u1", P + 4), ("pdu_truncated", "u1", P + 5), ("msg_len", "<u4", P + 8), ("body_len", "<u4", P + 12),
    ("ts_l2_send_ns", "<i8", P + 16), ("n_pusch", "<u2", P + 24), ("n_pucch", "<u2", P + 26), ("n_prach", "<u2", P + 28),
    ("n_srs", "<u2", P + 30), ("tot_pusch_prb", "<u4", P + 32), ("tot_pusch_layers", "<u4", P + 36),
    ("tot_pusch_tb_bytes", "<u4", P + 40), ("tot_pusch_cb", "<u4", P + 44), ("tot_pusch_prb_layers", "<u4", P + 48),
    ("tot_pucch_prb", "<u4", P + 52), ("tot_srs_ports", "<u4", P + 56),
])

DL_PDU_DTYPE = _dtype(COMMON + [
    ("pdu_index", "u1", P + 0), ("pdu_type", "u1", P + 1), ("start_sym", "u1", P + 2), ("num_sym", "u1", P + 3),
    ("pdu_size", "<u2", P + 4), ("rnti", "<u2", P + 6), ("rb_start", "<u2", P + 8), ("rb_size", "<u2", P + 10),
    ("num_layers", "u1", P + 12), ("num_codewords", "u1", P + 13), ("mcs_index", "u1", P + 14), ("mcs_table", "u1", P + 15),
    ("qam_mod_order", "u1", P + 16), ("rv_index", "u1", P + 17), ("target_code_rate", "<u2", P + 18),
    ("tb_size", "<u4", P + 20), ("tb_size_cw1", "<u4", P + 24), ("num_dl_dci", "<u2", P + 28), ("pdu_bitmap", "<u2", P + 30),
    ("bwp_size", "<u2", P + 32), ("bwp_start", "<u2", P + 34), ("dmrs_config_type", "u1", P + 36),
    ("num_dmrs_cdm_grps_no_data", "u1", P + 37), ("dl_dmrs_sym_pos", "<u2", P + 38), ("fapi_pdu_index", "<u2", P + 40),
    ("csirs_row", "u1", P + 42), ("ssb_block_index", "u1", P + 43),
    ("num_prgs", "<u2", P + 44), ("prg_size", "<u2", P + 46), ("dig_bf_interfaces", "u1", P + 48),
    ("agg_level_max", "u1", P + 49), ("agg_level_sum", "<u2", P + 50), ("dci_payload_bits_sum", "<u2", P + 52),
    ("dci_truncated", "u1", P + 54), ("prg_bf_sum", "<u4", P + 56),
])

DL_TTI_DTYPE = _dtype(COMMON + [
    ("num_pdus", "u1", P + 0), ("ngroup", "u1", P + 1), ("pdu_truncated", "u1", P + 2), ("msg_len", "<u4", P + 4),
    ("body_len", "<u4", P + 8), ("ts_l2_send_ns", "<i8", P + 16), ("n_pdcch", "<u2", P + 24), ("n_pdsch", "<u2", P + 26),
    ("n_csirs", "<u2", P + 28), ("n_ssb", "<u2", P + 30), ("n_dci", "<u4", P + 32), ("tot_pdsch_prb", "<u4", P + 36),
    ("tot_pdsch_layers", "<u4", P + 40), ("tot_pdsch_tb_bytes", "<u4", P + 44), ("tot_pdsch_prb_layers", "<u4", P + 48),
    ("tot_pdsch_prg_bf", "<u4", P + 52), ("tot_dci_agg_level", "<u4", P + 56), ("tot_dci_payload_bits", "<u4", P + 60),
    ("max_dci_agg_level", "u1", P + 64), ("max_pdsch_mcs", "u1", P + 65), ("tot_pdcch_prg_bf", "<u4", P + 68),
])

SLOT_END_DTYPE = _dtype(COMMON + [
    ("enqueued", "u1", P + 0), ("slot_end_rcvd", "u1", P + 1), ("is_ul", "u1", P + 2), ("is_dl", "u1", P + 3),
    ("is_csirs", "u1", P + 4), ("enqueue_ret", "<i4", P + 8), ("num_cells", "<u4", P + 12), ("cmd_size", "<u4", P + 16),
    ("tick_original_ns", "<i8", P + 24), ("t0_ns", "<i8", P + 32), ("l2a_latency_ns", "<i8", P + 40),
    ("l1_slot_ind_tick_ns", "<i8", P + 48), ("l2a_start_ns", "<i8", P + 56), ("l2a_end_ns", "<i8", P + 64),
])

START_DTYPE = _dtype(COMMON + [
    ("pid", "<u4", P + 0), ("generation", "<u4", P + 4), ("slot_advance", "<u4", P + 8), ("mu", "<u4", P + 12),
    ("num_cells", "<u4", P + 16), ("ring_len", "<u4", P + 20), ("abi_version", "<u4", P + 24), ("rec_size", "<u4", P + 28),
])

TYPE_DTYPES = {REC_UL_PDU: UL_PDU_DTYPE, REC_UL_TTI: UL_TTI_DTYPE, REC_DL_PDU: DL_PDU_DTYPE, REC_DL_TTI: DL_TTI_DTYPE,
               REC_SLOT_END: SLOT_END_DTYPE, REC_PRODUCER_START: START_DTYPE}


class DappRing:
    """Wait-free reader with its own cursor. Safe to run while L1 is writing."""

    def __init__(self, name="/aerial_dapp_ring", from_start=True):
        path = "/dev/shm/" + name.lstrip("/")
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        size = os.fstat(self.fd).st_size
        if size < HDR_SIZE:
            raise RuntimeError("ring too small; producer not started?")
        self.mm = mmap.mmap(self.fd, size, prot=mmap.PROT_READ)
        self.buf = memoryview(self.mm)
        hdr = self.header()
        if int(hdr["magic"]) != MAGIC:
            raise RuntimeError("bad magic 0x%x (ring being initialised or not a dapp ring)" % int(hdr["magic"]))
        if int(hdr["abi_version"]) != ABI_VERSION or int(hdr["rec_size"]) != REC_SIZE or int(hdr["hdr_size"]) != HDR_SIZE:
            raise RuntimeError("ABI mismatch: %s" % dict((k, int(hdr[k])) for k in ("abi_version", "rec_size", "hdr_size")))
        self.ring_len = int(hdr["ring_len"])
        self.mask = self.ring_len - 1
        self.generation = int(hdr["generation"])
        self.cursor = 1 if from_start else self.head() + 1
        self.lost = 0
        self._recs = np.frombuffer(self.buf, dtype=REC_DTYPE, count=self.ring_len, offset=HDR_SIZE)

    def header(self):
        return np.frombuffer(self.buf, dtype=HDR_DTYPE, count=1, offset=0)[0]

    def head(self):
        return int(np.frombuffer(self.buf, dtype="<u8", count=1, offset=40)[0])

    def producer_alive(self, max_age_s=2.0):
        hdr = self.header()
        pid = int(hdr["producer_pid"])
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return (time.time() * 1e9 - int(hdr["heartbeat_ns"])) < max_age_s * 1e9

    def _decode(self, raw):
        t = int(np.frombuffer(raw, dtype="<u2", count=1, offset=16)[0])
        dt = TYPE_DTYPES.get(t, REC_DTYPE)
        return np.frombuffer(raw, dtype=dt, count=1)[0]

    def next(self):
        """Returns a decoded record, None if nothing new, or 'RESTARTED'."""
        gen = int(self.header()["generation"])
        if gen != self.generation:
            self.generation = gen
            self.cursor = 1
            return "RESTARTED"
        for _ in range(8):
            head = self.head()
            if self.cursor > head:
                return None
            margin = self.ring_len // 8
            if head - self.cursor + 1 + margin > self.ring_len:
                new_cursor = head + 1 + margin - self.ring_len
                self.lost += new_cursor - self.cursor
                self.cursor = new_cursor
            off = HDR_SIZE + (self.cursor & self.mask) * REC_SIZE
            s1 = int(np.frombuffer(self.buf, dtype="<u8", count=1, offset=off)[0])
            if s1 != self.cursor:
                if s1 > self.cursor:
                    self.lost += s1 - self.cursor
                    self.cursor = s1
                    continue
                return None
            raw = bytes(self.buf[off:off + REC_SIZE])
            s2 = int(np.frombuffer(self.buf, dtype="<u8", count=1, offset=off)[0])
            if s1 != s2:
                continue
            self.cursor += 1
            return self._decode(raw)
        return None

    def follow(self, poll_s=0.0002, stop_when_idle=False):
        while True:
            r = self.next()
            if r is None:
                if stop_when_idle:
                    return
                time.sleep(poll_s)
                continue
            if isinstance(r, str):
                continue
            yield r

    def close(self):
        self.buf.release()
        self.mm.close()
        os.close(self.fd)


def fmt_rec(r, pdus=False):
    t = int(r["type"])
    ts = time.strftime("%H:%M:%S", time.localtime(int(r["ts_ns"]) // 10**9)) + ".%06d" % ((int(r["ts_ns"]) % 10**9) // 1000)
    head = "#%-8d %s %-8s %4d.%-2d cell=%-2d" % (int(r["seq"]), ts, REC_NAMES.get(t, "?"), int(r["sfn"]), int(r["slot"]), int(r["cell_id"]))
    if t == REC_UL_TTI:
        return head + " pdus=%d pusch=%d pucch=%d prach=%d srs=%d prb=%d layers=%d prb*layers=%d tb_bytes=%d cb=%d%s" % (
            r["num_pdus"], r["n_pusch"], r["n_pucch"], r["n_prach"], r["n_srs"], r["tot_pusch_prb"], r["tot_pusch_layers"],
            r["tot_pusch_prb_layers"], r["tot_pusch_tb_bytes"], r["tot_pusch_cb"], " TRUNCATED" if r["pdu_truncated"] else "")
    if t == REC_UL_PDU:
        if not pdus:
            return None
        kind = UL_PDU_NAMES.get(int(r["pdu_type"]), "?")
        if kind == "PUSCH":
            return head + "   [%d] PUSCH rnti=0x%04x rb=%d+%d sym=%d+%d layers=%d mcs=%d/%d qam=%d tb=%dB cb=%d rv=%d" % (
                r["pdu_index"], r["rnti"], r["rb_start"], r["rb_size"], r["start_sym"], r["num_sym"], r["num_layers"],
                r["mcs_index"], r["mcs_table"], r["qam_mod_order"], r["tb_size"], r["num_cb"], r["rv_index"])
        if kind == "PUCCH":
            return head + "   [%d] PUCCH rnti=0x%04x fmt=%d prb=%d+%d sym=%d+%d harq=%d csi1=%d" % (
                r["pdu_index"], r["rnti"], r["format_type"], r["rb_start"], r["rb_size"], r["start_sym"], r["num_sym"],
                r["bit_len_harq"], r["bit_len_csi1"])
        if kind == "PRACH":
            return head + "   [%d] PRACH format=%d ocas=%d num_ra=%d" % (r["pdu_index"], r["prach_format"], r["num_prach_ocas"], r["num_ra"])
        return head + "   [%d] SRS rnti=0x%04x ports=%d sym=%d rep=%d" % (r["pdu_index"], r["rnti"], r["num_layers"], r["srs_num_symbols"], r["srs_num_repetitions"])
    if t == REC_DL_TTI:
        return head + " pdus=%d pdcch=%d pdsch=%d csirs=%d ssb=%d dci=%d prb=%d layers=%d tb_bytes=%d prg_bf=%d dci_al=%d al_max=%d dci_bits=%d mcs_max=%d%s" % (
            r["num_pdus"], r["n_pdcch"], r["n_pdsch"], r["n_csirs"], r["n_ssb"], r["n_dci"], r["tot_pdsch_prb"],
            r["tot_pdsch_layers"], r["tot_pdsch_tb_bytes"], r["tot_pdsch_prg_bf"], r["tot_dci_agg_level"],
            r["max_dci_agg_level"], r["tot_dci_payload_bits"], r["max_pdsch_mcs"], " TRUNCATED" if r["pdu_truncated"] else "")
    if t == REC_DL_PDU:
        if not pdus:
            return None
        kind = DL_PDU_NAMES.get(int(r["pdu_type"]), "?")
        if kind == "PDSCH":
            return head + "   [%d] PDSCH rnti=0x%04x rb=%d+%d sym=%d+%d layers=%d mcs=%d qam=%d tb=%dB prg=%dx%d bf=%d" % (
                r["pdu_index"], r["rnti"], r["rb_start"], r["rb_size"], r["start_sym"], r["num_sym"], r["num_layers"],
                r["mcs_index"], r["qam_mod_order"], r["tb_size"], r["num_prgs"], r["prg_size"], r["dig_bf_interfaces"])
        if kind == "PDCCH":
            return head + "   [%d] PDCCH dci=%d al_sum=%d al_max=%d bits=%d prg_bf=%d%s" % (
                r["pdu_index"], r["num_dl_dci"], r["agg_level_sum"], r["agg_level_max"], r["dci_payload_bits_sum"],
                r["prg_bf_sum"], " DCI_TRUNC" if r["dci_truncated"] else "")
        return head + "   [%d] %s" % (r["pdu_index"], kind)
    if t == REC_SLOT_END:
        return head + " %s ret=%d trig=%s cells=%d ul=%d dl=%d l2a_latency_us=%.1f" % (
            "ENQUEUED" if r["enqueued"] else "DROPPED", r["enqueue_ret"], "slot_rsp" if r["slot_end_rcvd"] else "tick",
            r["num_cells"], r["is_ul"], r["is_dl"], int(r["l2a_latency_ns"]) / 1e3)
    if t == REC_PRODUCER_START:
        return head + " pid=%d gen=%d slot_advance=%d mu=%d cells=%d ring_len=%d" % (
            r["pid"], r["generation"], r["slot_advance"], r["mu"], r["num_cells"], r["ring_len"])
    return head


def verify(ring, max_slots):
    """Groups records per (sfn, slot, cell) and checks the summary against the PDU records."""
    groups = defaultdict(lambda: {"pdus": [], "sum": None})
    slots = {}
    checked = 0
    bad = 0
    for r in ring.follow(stop_when_idle=True):
        t = int(r["type"])
        key = (int(r["sfn"]), int(r["slot"]), int(r["cell_id"]))
        if t == REC_UL_PDU:
            groups[key]["pdus"].append(r)
        elif t == REC_UL_TTI:
            g = groups.pop(key, {"pdus": [], "sum": None})
            pusch = [p for p in g["pdus"] if int(p["pdu_type"]) == 1]
            ok = (len(g["pdus"]) == int(r["num_pdus"]) or bool(r["pdu_truncated"])) and \
                 len(pusch) == int(r["n_pusch"]) and \
                 sum(int(p["rb_size"]) for p in pusch) == int(r["tot_pusch_prb"]) and \
                 sum(int(p["tb_size"]) for p in pusch) == int(r["tot_pusch_tb_bytes"])
            checked += 1
            if not ok:
                bad += 1
                print("MISMATCH", key, "pdus=%d/%d pusch=%d/%d" % (len(g["pdus"]), int(r["num_pdus"]), len(pusch), int(r["n_pusch"])))
            slots.setdefault(key[:2], []).append(r)
        elif t == REC_SLOT_END:
            k = (int(r["sfn"]), int(r["slot"]))
            cells = slots.pop(k, [])
            tot_prb = sum(int(c["tot_pusch_prb"]) for c in cells)
            tot_tb = sum(int(c["tot_pusch_tb_bytes"]) for c in cells)
            print("slot %4d.%-2d %-8s cells_with_ul_tti=%-2d tot_pusch_prb=%-5d tot_tb_bytes=%-8d l2a_latency_us=%.1f" % (
                k[0], k[1], "ENQUEUED" if r["enqueued"] else "DROPPED", len(cells), tot_prb, tot_tb, int(r["l2a_latency_ns"]) / 1e3))
            if max_slots and checked >= max_slots:
                break
    print("checked %d UL_TTI summaries, %d mismatches, %d records lost by reader" % (checked, bad, ring.lost))
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="/aerial_dapp_ring")
    ap.add_argument("--follow", "-f", action="store_true")
    ap.add_argument("--new", action="store_true", help="skip records already in the ring")
    ap.add_argument("--pdus", "-p", action="store_true")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    ring = DappRing(args.name, from_start=not args.new)
    h = ring.header()
    print("ring: name=%s gen=%d pid=%d ring_len=%d head=%d slot_advance=%d mu=%d cells=%d alive=%s" % (
        h["producer_name"].decode(errors="ignore"), h["generation"], h["producer_pid"], h["ring_len"], h["head"],
        h["slot_advance"], h["mu"], h["num_cells"], ring.producer_alive()))
    print("      cnt_ul_tti=%d cnt_dl_tti=%d cnt_slot_end=%d cnt_slot_dropped=%d cnt_pdu_truncated=%d" % (
        h["cnt_ul_tti"], h["cnt_dl_tti"], h["cnt_slot_end"], h["cnt_slot_dropped"], h["cnt_pdu_truncated"]))
    if args.verify:
        sys.exit(0 if verify(ring, args.max) else 1)
    n = 0
    try:
        for r in ring.follow(stop_when_idle=not args.follow):
            line = fmt_rec(r, args.pdus)
            if line:
                print(line)
                n += 1
                if args.max and n >= args.max:
                    break
    except KeyboardInterrupt:
        pass
    if ring.lost:
        print("reader lost %d records" % ring.lost, file=sys.stderr)


if __name__ == "__main__":
    main()
