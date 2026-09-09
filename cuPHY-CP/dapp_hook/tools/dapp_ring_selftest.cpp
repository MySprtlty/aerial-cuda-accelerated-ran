/*
 * dapp_ring_selftest: exercises the ring without the L1.
 *
 * One producer thread writes a deterministic stream of records at high rate
 * (optionally faster than the consumer so overrun handling is exercised),
 * one consumer thread reads them back and checks:
 *   - no torn record is ever accepted (payload checksum matches seq)
 *   - sequence numbers are strictly increasing
 *   - lost counts reported by the consumer match the gaps observed
 * Exit code 0 on success.
 *
 *   dapp_ring_selftest [-n /dapp_selftest] [--records N] [--ring-len L] [--slow-reader]
 */
#include "dapp_hook/dapp_ring.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

using namespace nv::dapp;

static uint32_t checksum(const dapp_rec_t& r)
{
    // FNV-1a over everything except seq and ts (ts is set by begin()).
    uint32_t h = 2166136261u;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&r);
    for (size_t i = 16; i < sizeof(r); ++i) {
        if (i >= offsetof(dapp_rec_t, u) + 60 && i < offsetof(dapp_rec_t, u) + 64) { continue; } // checksum slot
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

int main(int argc, char** argv)
{
    std::string name = "/dapp_selftest";
    uint64_t records = 2000000;
    uint32_t ring_len = 4096;
    bool slow_reader = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-n" && i + 1 < argc) name = argv[++i];
        else if (a == "--records" && i + 1 < argc) records = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--ring-len" && i + 1 < argc) ring_len = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--slow-reader") slow_reader = true;
        else { std::fprintf(stderr, "usage: %s [-n /name] [--records N] [--ring-len L] [--slow-reader]\n", argv[0]); return 2; }
    }

    Producer::Config cfg;
    cfg.name = name;
    cfg.ring_len = ring_len;
    cfg.do_mlock = false;
    cfg.slot_advance = 3;
    cfg.mu = 1;
    cfg.num_cells = 20;
    cfg.producer_name = "selftest";
    std::string err;
    auto prod = Producer::create(cfg, err);
    if (!prod) { std::fprintf(stderr, "producer create failed: %s\n", err.c_str()); return 1; }

    auto cons = Consumer::open(name, err);
    if (!cons) { std::fprintf(stderr, "consumer open failed: %s\n", err.c_str()); return 1; }

    std::atomic<bool> done{false};
    std::atomic<uint64_t> produced{0};

    std::thread writer([&] {
        for (uint64_t i = 0; i < records; ++i) {
            const uint16_t sfn = static_cast<uint16_t>((i / 20) % 1024);
            const uint16_t slot = static_cast<uint16_t>(i % 20);
            dapp_rec_t* r = prod->begin(DAPP_REC_UL_PDU, sfn, slot, static_cast<uint16_t>(i % 20));
            r->u.ul_pdu.pdu_index = static_cast<uint8_t>(i & 0x3f);
            r->u.ul_pdu.pdu_type = DAPP_UL_PUSCH;
            r->u.ul_pdu.rnti = static_cast<uint16_t>(0x1000 + (i & 0xfff));
            r->u.ul_pdu.rb_size = static_cast<uint16_t>(1 + (i % 273));
            r->u.ul_pdu.num_layers = static_cast<uint8_t>(1 + (i % 4));
            r->u.ul_pdu.tb_size = static_cast<uint32_t>(i * 7);
            r->u.ul_pdu.handle = static_cast<uint32_t>(i);
            uint32_t cs = checksum(*r);
            std::memcpy(&r->u.raw[60], &cs, sizeof(cs));
            prod->commit(r);
            produced.store(i + 1, std::memory_order_relaxed);
        }
        prod->heartbeat();
        done.store(true, std::memory_order_release);
    });

    uint64_t got = 0, torn = 0, gaps = 0, lost_reported = 0, last_seq = 0, restarts = 0;
    dapp_rec_t r;
    // The PRODUCER_START record is seq 1 and precedes the writer's records.
    while (true) {
        uint64_t lost = 0;
        int st = cons->next(r, &lost);
        if (st == Consumer::RESTARTED) { restarts++; continue; }
        if (st == Consumer::NONE) {
            if (done.load(std::memory_order_acquire) && cons->cursor() > cons->head()) break;
            if (slow_reader) { std::this_thread::sleep_for(std::chrono::microseconds(50)); }
            continue;
        }
        lost_reported += lost;
        if (r.seq <= last_seq) { std::fprintf(stderr, "seq not increasing: %llu after %llu\n", (unsigned long long)r.seq, (unsigned long long)last_seq); return 1; }
        if (last_seq != 0 && r.seq != last_seq + 1) { gaps += r.seq - last_seq - 1; }
        last_seq = r.seq;
        if (r.type == DAPP_REC_UL_PDU) {
            uint32_t cs; std::memcpy(&cs, &r.u.raw[60], sizeof(cs));
            if (cs != checksum(r) || r.u.ul_pdu.handle + 2 != r.seq) { torn++; }
        }
        got++;
        if (slow_reader && (got % 3 == 0)) { std::this_thread::sleep_for(std::chrono::microseconds(2)); }
    }
    writer.join();

    const uint64_t total = records + 1; // + PRODUCER_START
    std::printf("produced=%llu consumed=%llu lost_reported=%llu gaps_observed=%llu torn=%llu restarts=%llu head=%llu\n",
                (unsigned long long)total, (unsigned long long)got, (unsigned long long)lost_reported,
                (unsigned long long)gaps, (unsigned long long)torn, (unsigned long long)restarts,
                (unsigned long long)cons->head());

    bool ok = (torn == 0) && (got + gaps == total) && (gaps == lost_reported) && (cons->head() == total);
    std::printf("%s\n", ok ? "SELFTEST PASS" : "SELFTEST FAIL");
    Producer::unlink(name);
    return ok ? 0 : 1;
}
