/*
 * dApp FAPI hook ring: shared-memory ABI.
 *
 * This header is plain C so the layout can be consumed from C, C++ and
 * Python (see python/dapp_ring.py, which mirrors these offsets).
 *
 * Layout of the POSIX shared memory object (/dev/shm/<name>):
 *
 *   [ dapp_ring_hdr_t : DAPP_RING_HDR_SIZE bytes ]
 *   [ dapp_rec_t[ring_len] : ring_len * DAPP_REC_SIZE bytes ]
 *
 * Single producer (the L1 msg_processing thread), any number of readers.
 * Records are a broadcast log: record with sequence number `seq` lives at
 * index (seq & (ring_len-1)); seq starts at 1 and 0 means "empty". Readers
 * keep their own cursor and detect overrun by comparing with hdr.head.
 *
 * Torn-read protection (seqlock per record): the producer stores seq=0,
 * release-fences, writes the payload, then release-stores the new seq.
 * A reader loads seq (acquire), copies the record, acquire-fences, loads
 * seq again and accepts the copy only if both loads equal its cursor.
 */
#ifndef DAPP_RING_ABI_H
#define DAPP_RING_ABI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DAPP_RING_MAGIC        0x31474E5250504144ULL /* "DAPPRNG1" as little-endian u64 */
#define DAPP_RING_ABI_VERSION  2u   /* 2: DL precoding + DCI fields appended */
#define DAPP_RING_HDR_SIZE     4096u
#define DAPP_REC_SIZE          128u
#define DAPP_REC_PAYLOAD_SIZE  (DAPP_REC_SIZE - 24u)
#define DAPP_RING_DEFAULT_NAME "/aerial_dapp_ring"
#define DAPP_RING_DEFAULT_LEN  65536u   /* records; 65536 * 128 B = 8 MiB */
#define DAPP_RING_NAME_MAX     64u

/* Record types (dapp_rec_t.type) */
enum dapp_rec_type {
    DAPP_REC_NONE           = 0,
    DAPP_REC_UL_PDU         = 1, /* one per UL_TTI.request PDU (PRACH/PUSCH/PUCCH/SRS)          */
    DAPP_REC_UL_TTI         = 2, /* one per cell per slot, written AFTER its UL_PDU records      */
    DAPP_REC_DL_PDU         = 3, /* one per DL_TTI.request PDU (PDCCH/PDSCH/CSI-RS/SSB)          */
    DAPP_REC_DL_TTI         = 4, /* one per cell per slot, written AFTER its DL_PDU records      */
    DAPP_REC_SLOT_END       = 5, /* one per slot: slot command enqueued to cuphydriver or dropped */
    DAPP_REC_PRODUCER_START = 6, /* first record after the producer (re)started                  */
};

/* pdu_type values, identical to SCF FAPI */
enum dapp_ul_pdu_type { DAPP_UL_PRACH = 0, DAPP_UL_PUSCH = 1, DAPP_UL_PUCCH = 2, DAPP_UL_SRS = 3 };
enum dapp_dl_pdu_type { DAPP_DL_PDCCH = 0, DAPP_DL_PDSCH = 1, DAPP_DL_CSI_RS = 2, DAPP_DL_SSB = 3 };

/* ---- payloads (all naturally aligned, no implicit padding) -------------- */

typedef struct dapp_ul_pdu_s {
    uint8_t  pdu_index;           /* 0  index inside the UL_TTI.request           */
    uint8_t  pdu_type;            /* 1  dapp_ul_pdu_type                          */
    uint8_t  start_sym;           /* 2  PUSCH/PUCCH start symbol, PRACH start sym */
    uint8_t  num_sym;             /* 3  PUSCH/PUCCH symbols, SRS num_symbols      */
    uint16_t pdu_size;            /* 4  FAPI pdu_size                             */
    uint16_t rnti;                /* 6                                            */
    uint32_t handle;              /* 8                                            */
    uint16_t rb_start;            /* 12 PUSCH rb_start / PUCCH prb_start          */
    uint16_t rb_size;             /* 14 PUSCH rb_size  / PUCCH prb_size           */
    uint8_t  num_layers;          /* 16 PUSCH layers, SRS num_ant_ports           */
    uint8_t  mcs_index;           /* 17 */
    uint8_t  mcs_table;           /* 18 */
    uint8_t  qam_mod_order;       /* 19 */
    uint16_t target_code_rate;    /* 20 */
    uint16_t pdu_bitmap;          /* 22 PUSCH pdu_bitmap                          */
    uint32_t tb_size;             /* 24 PUSCH tb_size (bytes) when pdu_bitmap&1   */
    uint16_t num_cb;              /* 28 PUSCH num_cb                              */
    uint16_t ul_dmrs_sym_pos;     /* 30 */
    uint8_t  transform_precoding; /* 32 */
    uint8_t  dmrs_config_type;    /* 33 */
    uint8_t  num_dmrs_cdm_grps_no_data; /* 34 */
    uint8_t  rv_index;            /* 35 */
    uint8_t  harq_process_id;     /* 36 */
    uint8_t  ndi;                 /* 37 */
    uint16_t bwp_size;            /* 38 */
    uint16_t bwp_start;           /* 40 */
    uint16_t harq_ack_bit_len;    /* 42 PUSCH UCI (pdu_bitmap&2)                  */
    uint16_t csi_part1_bit_len;   /* 44 PUSCH UCI                                  */
    uint8_t  format_type;         /* 46 PUCCH                                      */
    uint8_t  sr_flag;             /* 47 PUCCH                                      */
    uint16_t bit_len_harq;        /* 48 PUCCH                                      */
    uint16_t bit_len_csi1;        /* 50 PUCCH                                      */
    uint16_t bit_len_csi2;        /* 52 PUCCH                                      */
    uint8_t  prach_format;        /* 54 PRACH                                      */
    uint8_t  num_prach_ocas;      /* 55 PRACH                                      */
    uint8_t  num_ra;              /* 56 PRACH                                      */
    uint8_t  srs_num_symbols;     /* 57 SRS                                        */
    uint8_t  srs_num_repetitions; /* 58 SRS                                        */
    uint8_t  srs_comb_size;       /* 59 SRS                                        */
    uint8_t  srs_bandwidth_index; /* 60 SRS                                        */
    uint8_t  srs_config_index;    /* 61 SRS                                        */
    uint16_t reserved;            /* 62 */
} dapp_ul_pdu_t;                  /* 64 bytes */

typedef struct dapp_ul_tti_s {
    uint8_t  num_pdus;            /* 0  */
    uint8_t  rach_present;        /* 1  */
    uint8_t  num_ulsch;           /* 2  */
    uint8_t  num_ulcch;           /* 3  */
    uint8_t  ngroup;              /* 4  */
    uint8_t  pdu_truncated;       /* 5  1 if the PDU walk hit the buffer end       */
    uint16_t reserved0;           /* 6  */
    uint32_t msg_len;             /* 8  nvIPC msg_len                              */
    uint32_t body_len;            /* 12 FAPI body header length                    */
    int64_t  ts_l2_send_ns;       /* 16 nvIPC send timestamp (0 if unavailable)    */
    uint16_t n_pusch;             /* 24 */
    uint16_t n_pucch;             /* 26 */
    uint16_t n_prach;             /* 28 */
    uint16_t n_srs;               /* 30 */
    uint32_t tot_pusch_prb;       /* 32 sum(rb_size)                               */
    uint32_t tot_pusch_layers;    /* 36 sum(num_layers)                            */
    uint32_t tot_pusch_tb_bytes;  /* 40 sum(tb_size)                               */
    uint32_t tot_pusch_cb;        /* 44 sum(num_cb)                                */
    uint32_t tot_pusch_prb_layers;/* 48 sum(rb_size*num_layers)                    */
    uint32_t tot_pucch_prb;       /* 52 sum(prb_size)                              */
    uint32_t tot_srs_ports;       /* 56 sum(num_ant_ports)                         */
    uint32_t reserved1;           /* 60 */
} dapp_ul_tti_t;                  /* 64 bytes */

typedef struct dapp_dl_pdu_s {
    uint8_t  pdu_index;           /* 0  */
    uint8_t  pdu_type;            /* 1  dapp_dl_pdu_type                          */
    uint8_t  start_sym;           /* 2  PDSCH start_sym_index / PDCCH start_sym    */
    uint8_t  num_sym;             /* 3  PDSCH num_symbols / PDCCH duration_sym     */
    uint16_t pdu_size;            /* 4  */
    uint16_t rnti;                /* 6  */
    uint16_t rb_start;            /* 8  PDSCH rb_start / CSI-RS start_rb           */
    uint16_t rb_size;             /* 10 PDSCH rb_size  / CSI-RS num_of_rbs         */
    uint8_t  num_layers;          /* 12 PDSCH */
    uint8_t  num_codewords;       /* 13 PDSCH */
    uint8_t  mcs_index;           /* 14 PDSCH cw0 */
    uint8_t  mcs_table;           /* 15 PDSCH cw0 */
    uint8_t  qam_mod_order;       /* 16 PDSCH cw0 */
    uint8_t  rv_index;            /* 17 PDSCH cw0 */
    uint16_t target_code_rate;    /* 18 PDSCH cw0 */
    uint32_t tb_size;             /* 20 PDSCH cw0 tb_size (bytes)                  */
    uint32_t tb_size_cw1;         /* 24 PDSCH cw1 tb_size                          */
    uint16_t num_dl_dci;          /* 28 PDCCH */
    uint16_t pdu_bitmap;          /* 30 PDSCH */
    uint16_t bwp_size;            /* 32 */
    uint16_t bwp_start;           /* 34 */
    uint8_t  dmrs_config_type;    /* 36 PDSCH */
    uint8_t  num_dmrs_cdm_grps_no_data; /* 37 PDSCH */
    uint16_t dl_dmrs_sym_pos;     /* 38 PDSCH */
    uint16_t fapi_pdu_index;      /* 40 PDSCH pdu_index (links TX_DATA.request)    */
    uint8_t  csirs_row;           /* 42 CSI-RS */
    uint8_t  ssb_block_index;     /* 43 SSB */
    uint16_t num_prgs;            /* 44 PDSCH precoding: PRGs in this PDU          */
    uint16_t prg_size;            /* 46 PDSCH precoding: PRBs per PRG              */
    uint8_t  dig_bf_interfaces;   /* 48 PDSCH precoding: digital BF ports          */
    uint8_t  agg_level_max;       /* 49 PDCCH: max aggregation level over DCIs     */
    uint16_t agg_level_sum;       /* 50 PDCCH: sum of aggregation levels           */
    uint16_t dci_payload_bits_sum;/* 52 PDCCH: sum of payload_size_bits            */
    uint8_t  dci_truncated;       /* 54 PDCCH: 1 if the DCI walk hit the PDU end   */
    uint8_t  reserved8;           /* 55 */
    uint32_t prg_bf_sum;          /* 56 sum(num_prgs x dig_bf_interfaces): one term for PDSCH, over DCIs for PDCCH */
    uint32_t reserved;            /* 60 */
} dapp_dl_pdu_t;                  /* 64 bytes */

typedef struct dapp_dl_tti_s {
    uint8_t  num_pdus;            /* 0  */
    uint8_t  ngroup;              /* 1  */
    uint8_t  pdu_truncated;       /* 2  */
    uint8_t  reserved0;           /* 3  */
    uint32_t msg_len;             /* 4  */
    uint32_t body_len;            /* 8  */
    uint32_t reserved1;           /* 12 */
    int64_t  ts_l2_send_ns;       /* 16 */
    uint16_t n_pdcch;             /* 24 */
    uint16_t n_pdsch;             /* 26 */
    uint16_t n_csirs;             /* 28 */
    uint16_t n_ssb;               /* 30 */
    uint32_t n_dci;               /* 32 sum(num_dl_dci)                            */
    uint32_t tot_pdsch_prb;       /* 36 */
    uint32_t tot_pdsch_layers;    /* 40 */
    uint32_t tot_pdsch_tb_bytes;  /* 44 */
    uint32_t tot_pdsch_prb_layers;/* 48 */
    uint32_t tot_pdsch_prg_bf;    /* 52 sum over PDSCH of num_prgs x dig_bf_interfaces (precoding work) */
    uint32_t tot_dci_agg_level;   /* 56 sum of aggregation levels over every DCI            */
    uint32_t tot_dci_payload_bits;/* 60 sum of DCI payload bits                              */
    uint8_t  max_dci_agg_level;   /* 64 */
    uint8_t  max_pdsch_mcs;       /* 65 max mcs_index over PDSCH codeword 0                 */
    uint16_t reserved2;           /* 66 */
    uint32_t tot_pdcch_prg_bf;    /* 68 sum over DCIs of num_prgs x dig_bf_interfaces       */
} dapp_dl_tti_t;                  /* 72 bytes */

typedef struct dapp_slot_end_s {
    uint8_t  enqueued;            /* 0  1: l1_enqueue_phy_work() was called        */
    uint8_t  slot_end_rcvd;       /* 1  1: triggered by SLOT.response, 0: by tick  */
    uint8_t  is_ul;               /* 2  */
    uint8_t  is_dl;               /* 3  */
    uint8_t  is_csirs;            /* 4  */
    uint8_t  reserved0[3];        /* 5  */
    int32_t  enqueue_ret;         /* 8  return of l1_enqueue_phy_work (-1: not called) */
    uint32_t num_cells;           /* 12 cells with commands in this slot           */
    uint32_t cmd_size;            /* 16 slot_cmd.cells.size()                      */
    uint32_t reserved1;           /* 20 */
    int64_t  tick_original_ns;    /* 24 slot_cmd.tick_original                     */
    int64_t  t0_ns;               /* 32 expected T0 of the slot (0 if unknown)     */
    int64_t  l2a_latency_ns;      /* 40 L2+L2A latency measured by L1              */
    int64_t  l1_slot_ind_tick_ns; /* 48 time the tick fired for this slot          */
    int64_t  l2a_start_ns;        /* 56 first FAPI message of the slot             */
    int64_t  l2a_end_ns;          /* 64 process_phy_commands entry                 */
} dapp_slot_end_t;                /* 72 bytes */

typedef struct dapp_producer_start_s {
    uint32_t pid;                 /* 0  */
    uint32_t generation;          /* 4  */
    uint32_t slot_advance;        /* 8  */
    uint32_t mu;                  /* 12 */
    uint32_t num_cells;           /* 16 */
    uint32_t ring_len;            /* 20 */
    uint32_t abi_version;         /* 24 */
    uint32_t rec_size;            /* 28 */
} dapp_producer_start_t;          /* 32 bytes */

/* ---- record ------------------------------------------------------------- */

typedef struct dapp_rec_s {
    uint64_t seq;                 /* 0  sequence number, 0 = empty (written last) */
    uint64_t ts_ns;               /* 8  CLOCK_REALTIME ns when the hook ran        */
    uint16_t type;                /* 16 dapp_rec_type                              */
    uint16_t sfn;                 /* 18 */
    uint16_t slot;                /* 20 */
    uint16_t cell_id;             /* 22 L1 cell index (0xFFFF for slot-level)      */
    union {
        dapp_ul_pdu_t         ul_pdu;
        dapp_ul_tti_t         ul_tti;
        dapp_dl_pdu_t         dl_pdu;
        dapp_dl_tti_t         dl_tti;
        dapp_slot_end_t       slot_end;
        dapp_producer_start_t start;
        uint8_t               raw[DAPP_REC_PAYLOAD_SIZE];
    } u;                          /* 24 .. 127 */
} dapp_rec_t;

/* ---- header (one page) -------------------------------------------------- */

typedef struct dapp_ring_hdr_s {
    uint64_t magic;               /* 0   DAPP_RING_MAGIC                          */
    uint32_t abi_version;         /* 8   DAPP_RING_ABI_VERSION                    */
    uint32_t rec_size;            /* 12  DAPP_REC_SIZE                            */
    uint32_t ring_len;            /* 16  number of records (power of two)         */
    uint32_t hdr_size;            /* 20  DAPP_RING_HDR_SIZE                       */
    uint32_t generation;          /* 24  incremented on every producer start      */
    uint32_t producer_pid;        /* 28  */
    uint64_t start_ts_ns;         /* 32  */
    uint64_t head;                /* 40  seq of the newest committed record       */
    uint64_t heartbeat_ns;        /* 48  last producer write time                 */
    uint32_t slot_advance;        /* 56  */
    uint32_t mu;                  /* 60  */
    uint32_t num_cells;           /* 64  */
    uint32_t flags;               /* 68  bit0: mlock succeeded                    */
    uint64_t cnt_ul_tti;          /* 72  UL_TTI.request messages exported         */
    uint64_t cnt_dl_tti;          /* 80  */
    uint64_t cnt_slot_end;        /* 88  */
    uint64_t cnt_slot_dropped;    /* 96  */
    uint64_t cnt_pdu_truncated;   /* 104 */
    uint64_t cnt_records;         /* 112 == head                                  */
    char     producer_name[DAPP_RING_NAME_MAX]; /* 120 */
    uint8_t  reserved[DAPP_RING_HDR_SIZE - 184];
} dapp_ring_hdr_t;

/* ---- compile-time layout checks ----------------------------------------- */
#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
#ifdef __cplusplus
#define DAPP_STATIC_ASSERT(c, m) static_assert(c, m)
#else
#define DAPP_STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif
DAPP_STATIC_ASSERT(sizeof(dapp_ul_pdu_t) == 64, "dapp_ul_pdu_t size");
DAPP_STATIC_ASSERT(sizeof(dapp_ul_tti_t) == 64, "dapp_ul_tti_t size");
DAPP_STATIC_ASSERT(sizeof(dapp_dl_pdu_t) == 64, "dapp_dl_pdu_t size");
DAPP_STATIC_ASSERT(sizeof(dapp_dl_tti_t) == 72, "dapp_dl_tti_t size");
DAPP_STATIC_ASSERT(sizeof(dapp_slot_end_t) == 72, "dapp_slot_end_t size");
DAPP_STATIC_ASSERT(sizeof(dapp_producer_start_t) == 32, "dapp_producer_start_t size");
DAPP_STATIC_ASSERT(sizeof(dapp_rec_t) == DAPP_REC_SIZE, "dapp_rec_t size");
DAPP_STATIC_ASSERT(offsetof(dapp_rec_t, u) == 24, "dapp_rec_t payload offset");
DAPP_STATIC_ASSERT(offsetof(dapp_ul_pdu_t, tb_size) == 24, "dapp_ul_pdu_t.tb_size offset");
DAPP_STATIC_ASSERT(offsetof(dapp_ul_tti_t, ts_l2_send_ns) == 16, "dapp_ul_tti_t.ts_l2_send_ns offset");
DAPP_STATIC_ASSERT(offsetof(dapp_slot_end_t, tick_original_ns) == 24, "dapp_slot_end_t.tick_original_ns offset");
DAPP_STATIC_ASSERT(offsetof(dapp_dl_pdu_t, prg_bf_sum) == 56, "dapp_dl_pdu_t.prg_bf_sum offset");
DAPP_STATIC_ASSERT(offsetof(dapp_dl_tti_t, tot_pdcch_prg_bf) == 68, "dapp_dl_tti_t.tot_pdcch_prg_bf offset");
DAPP_STATIC_ASSERT(sizeof(dapp_ring_hdr_t) == DAPP_RING_HDR_SIZE, "dapp_ring_hdr_t size");
DAPP_STATIC_ASSERT(offsetof(dapp_ring_hdr_t, head) == 40, "dapp_ring_hdr_t.head offset");
DAPP_STATIC_ASSERT(offsetof(dapp_ring_hdr_t, producer_name) == 120, "dapp_ring_hdr_t.producer_name offset");
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DAPP_RING_ABI_H */
