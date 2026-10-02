/*
 * dApp FAPI hook: asynchronous export.
 *
 * With the inline hook the L2 adapter msg_processing thread walks every PDU of
 * a TTI request and writes one ring record per PDU (~100 ns each, 130..460
 * records per slot at 16 cells), which costs 20..45 us of the 600 us L2+L2A
 * budget. In asynchronous mode that thread only hands a small descriptor to a
 * single-producer/single-consumer queue; a dedicated export thread walks the
 * message and writes the very same records, in the same order, to the ring.
 *
 *   L2A thread  : submit (reserve + publish), take_claim, pop_releasable
 *   export thread: run()  - the ONLY writer of the ring once started
 *
 * Buffer lifetime: a TTI descriptor points into the nvIPC message buffer, so
 * the L2 adapter must not release that buffer until the export thread is done
 * with it. publish() records a claim on the buffer; the caller that would
 * release it asks take_claim() and, when it matches, lets the exporter hold the
 * descriptor instead. pop_releasable() gives the descriptors back once the
 * export thread has passed them (the release itself stays on the L2A thread).
 *
 * Nothing here blocks the L2A thread: when the queue or the held-buffer budget
 * is full the message is simply not exported and a counter is incremented.
 * The ring ABI (dapp_ring_abi.h) is unchanged.
 */
#ifndef DAPP_ASYNC_HPP
#define DAPP_ASYNC_HPP

#include "dapp_ring.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <thread>

#include <pthread.h>
#include <sched.h>

namespace nv {
namespace dapp {

class AsyncExporter {
public:
    static constexpr uint32_t kQueueLen = 1024; // descriptors, power of two (one slot at 16 cells uses <= 33)
    static constexpr uint32_t kHeldLen  = 512;  // held-buffer FIFO, power of two
    static constexpr uint32_t kMaxHeld  = 256;  // nvIPC buffers whose release may be deferred at any time
    static constexpr size_t   kIpcBytes = 48;   // >= sizeof(nv_ipc_msg_t); opaque here (no nvIPC dependency)

    enum Kind : uint16_t { KIND_UL_TTI = 1, KIND_DL_TTI = 2, KIND_SLOT_END = 3 };

    struct Item;
    using Handler = void (*)(Producer& ring, const Item& it);

    struct Item {
        Handler         fn;            // runs on the export thread
        uint64_t        seq;           // hand-off sequence of a TTI item (0: no buffer attached)
        int64_t         ts_rx_ns;      // L2A time of the hook; becomes the record timestamp
        int64_t         ts_l2_send_ns;
        const void*     fapi_msg;      // FAPI body inside the nvIPC message buffer
        uint16_t        kind;
        uint16_t        cell_id;
        uint16_t        sfn;
        uint16_t        slot;
        uint8_t         slot_dropped;  // SLOT_END: slot command was not enqueued
        uint8_t         reserved[7];
        uint8_t         ipc[kIpcBytes]; // nv_ipc_msg_t of the message
        dapp_slot_end_t slot_end;       // SLOT_END payload (num_cells filled by the export thread)
    };

    struct Stats {
        uint64_t submitted;     // descriptors handed to the export thread
        uint64_t dropped;       // not exported: queue or held-buffer budget full
        uint64_t orphans;       // claims that nobody took (buffer released while possibly still in use)
        uint64_t done;          // descriptors written by the export thread
        uint32_t max_depth;     // queue depth high-water mark
        uint32_t max_held;      // deferred-release high-water mark
        int64_t  delay_sum_ns;  // sum of (record written - hook time) since the last read
        int64_t  delay_max_ns;  // max of the same since the last read
        uint64_t delay_n;
    };

    static std::unique_ptr<AsyncExporter> create(Producer* ring) { return std::unique_ptr<AsyncExporter>(new AsyncExporter(ring)); }

    // Starts the export thread. cpu < 0: no pinning; fifo_prio <= 0: scheduling policy left alone.
    bool start(int cpu, int fifo_prio, std::string& err)
    {
        try {
            thread_ = std::thread([this] { run(); });
        } catch (const std::exception& e) {
            err = std::string("thread start failed: ") + e.what();
            return false;
        }
        (void)pthread_setname_np(thread_.native_handle(), "dapp_export");
        if (cpu >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            const int rc = pthread_setaffinity_np(thread_.native_handle(), sizeof(set), &set);
            if (rc != 0) { err = std::string("setaffinity failed: ") + std::strerror(rc); }
        }
        if (fifo_prio > 0) {
            sched_param sp{};
            sp.sched_priority = fifo_prio;
            const int rc = pthread_setschedparam(thread_.native_handle(), SCHED_FIFO, &sp);
            if (rc != 0) { err = std::string("setschedparam failed: ") + std::strerror(rc); }
        }
        return true;
    }

    void stop_and_join() noexcept
    {
        stop_.store(true, std::memory_order_release);
        if (thread_.joinable()) { thread_.join(); }
    }

    ~AsyncExporter() { stop_and_join(); }

    // ---- L2A thread ---------------------------------------------------------
    // reserve(): the descriptor to fill, or nullptr when the message cannot be
    // exported right now (never waits). holds_buffer: the item will point into
    // an nvIPC buffer whose release has to be deferred.
    inline Item* reserve(bool holds_buffer) noexcept
    {
        const uint64_t t     = tail_.load(std::memory_order_relaxed);
        const uint64_t depth = t - head_.load(std::memory_order_acquire);
        if (depth >= kQueueLen || (holds_buffer && (handed_ - released_) >= kMaxHeld)) {
            ++dropped_;
            return nullptr;
        }
        if (depth + 1 > max_depth_) { max_depth_ = static_cast<uint32_t>(depth + 1); }
        return &q_[t & (kQueueLen - 1)];
    }

    // publish(): makes a filled descriptor visible to the export thread. msg_buf
    // is the nvIPC message buffer the item points into (nullptr: none); the
    // buffer is claimed until take_claim() or the next publish().
    inline void publish(Item* it, const void* msg_buf) noexcept
    {
        if (msg_buf != nullptr) {
            it->seq = ++handed_;
            // previous claim never taken: its buffer was released the normal way
            if (claim_buf_ != nullptr) { ++orphans_; ++released_; }
            claim_buf_ = msg_buf;
            claim_seq_ = it->seq;
        } else {
            it->seq = 0;
        }
        ++submitted_;
        tail_.store(tail_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    }

    // take_claim(): called where the L2 adapter is about to release `msg_buf`.
    // `ipc` is the descriptor it would have released (nbytes <= kIpcBytes).
    // True: the exporter now holds that descriptor, do not release it.
    inline bool take_claim(const void* msg_buf, const void* ipc, size_t nbytes) noexcept
    {
        if (claim_buf_ == nullptr || claim_buf_ != msg_buf) { return false; }
        Held& h = held_[held_tail_ & (kHeldLen - 1)];
        h.seq = claim_seq_;
        std::memcpy(h.ipc, ipc, nbytes <= kIpcBytes ? nbytes : kIpcBytes);
        ++held_tail_;
        claim_buf_ = nullptr;
        const uint32_t n = held_tail_ - held_head_;
        if (n > max_held_) { max_held_ = n; }
        return true;
    }

    // pop_releasable(): copies the next descriptor the export thread is done
    // with into `ipc_out` (nbytes <= kIpcBytes). False when there is none.
    inline bool pop_releasable(void* ipc_out, size_t nbytes) noexcept
    {
        if (held_head_ == held_tail_) { return false; }
        const Held& h = held_[held_head_ & (kHeldLen - 1)];
        if (h.seq > done_seq_.load(std::memory_order_acquire)) { return false; }
        std::memcpy(ipc_out, h.ipc, nbytes <= kIpcBytes ? nbytes : kIpcBytes);
        ++held_head_;
        ++released_;
        return true;
    }

    // Snapshot for logging (L2A thread). The delay figures are reset by the read.
    inline Stats stats() noexcept
    {
        Stats s{};
        s.submitted    = submitted_;
        s.dropped      = dropped_;
        s.orphans      = orphans_;
        s.done         = done_.load(std::memory_order_relaxed);
        s.max_depth    = max_depth_;
        s.max_held     = max_held_;
        s.delay_sum_ns = delay_sum_ns_.exchange(0, std::memory_order_relaxed);
        s.delay_max_ns = delay_max_ns_.exchange(0, std::memory_order_relaxed);
        s.delay_n      = delay_n_.exchange(0, std::memory_order_relaxed);
        return s;
    }

    AsyncExporter(const AsyncExporter&) = delete;
    AsyncExporter& operator=(const AsyncExporter&) = delete;

private:
    explicit AsyncExporter(Producer* ring) : ring_(ring)
    {
        std::memset(static_cast<void*>(q_), 0, sizeof(q_)); // pre-fault
        std::memset(static_cast<void*>(held_), 0, sizeof(held_));
    }

    static inline void cpu_relax() noexcept
    {
#if defined(__aarch64__)
        __asm__ __volatile__("isb");
#elif defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
    }

    // Export thread: the only writer of the ring.
    void run() noexcept
    {
        while (!stop_.load(std::memory_order_acquire)) {
            const uint64_t h = head_.load(std::memory_order_relaxed);
            if (h == tail_.load(std::memory_order_acquire)) {
                cpu_relax();
                continue;
            }
            const Item& it = q_[h & (kQueueLen - 1)];
            ring_->set_ts_override(static_cast<uint64_t>(it.ts_rx_ns));
            it.fn(*ring_, it);
            ring_->set_ts_override(0);
            const int64_t d = now_ns() - it.ts_rx_ns;
            delay_sum_ns_.fetch_add(d, std::memory_order_relaxed);
            delay_n_.fetch_add(1, std::memory_order_relaxed);
            if (d > delay_max_ns_.load(std::memory_order_relaxed)) { delay_max_ns_.store(d, std::memory_order_relaxed); }
            // the buffer behind a TTI item may be released from here on
            if (it.seq != 0) { done_seq_.store(it.seq, std::memory_order_release); }
            done_.fetch_add(1, std::memory_order_relaxed);
            head_.store(h + 1, std::memory_order_release);
        }
    }

    struct Held {
        uint64_t seq;
        uint8_t  ipc[kIpcBytes];
    };

    Producer* ring_;

    // L2A thread only
    alignas(64) std::atomic<uint64_t> tail_{0};
    uint64_t    handed_    = 0; // TTI items published
    uint64_t    released_  = 0; // TTI buffers given back through pop_releasable()
    uint64_t    submitted_ = 0;
    uint64_t    dropped_   = 0;
    uint64_t    orphans_   = 0;
    uint32_t    max_depth_ = 0;
    uint32_t    max_held_  = 0;
    uint32_t    held_head_ = 0;
    uint32_t    held_tail_ = 0;
    const void* claim_buf_ = nullptr;
    uint64_t    claim_seq_ = 0;

    // export thread only (read by the L2A thread)
    alignas(64) std::atomic<uint64_t> head_{0};
    std::atomic<uint64_t> done_seq_{0};
    std::atomic<uint64_t> done_{0};
    std::atomic<int64_t>  delay_sum_ns_{0};
    std::atomic<int64_t>  delay_max_ns_{0};
    std::atomic<uint64_t> delay_n_{0};

    alignas(64) std::atomic<bool> stop_{false};
    std::thread thread_;

    alignas(64) Held held_[kHeldLen];
    alignas(64) Item q_[kQueueLen];
};

// SLOT_END on the export thread: same record as the inline hook writes.
inline void async_slot_end_handler(Producer& ring, const AsyncExporter::Item& it)
{
    dapp_rec_t* r           = ring.begin(DAPP_REC_SLOT_END, it.sfn, it.slot, 0xFFFF);
    r->u.slot_end           = it.slot_end;
    r->u.slot_end.num_cells = ring.take_cells(it.sfn, it.slot);
    ring.commit(r);
    ring.count_slot_end();
    if (it.slot_dropped) { ring.count_slot_dropped(); }
    ring.heartbeat();
}

//----------------------------------------------------------------------------
// Process-wide exporter. Null unless the yaml asks for async_export: 1.
// The object is intentionally never freed: the L2A thread may still be inside
// submit when the process exits. Only the export thread is stopped, before the
// producer (defined earlier, so destroyed later) unmaps the ring.
//----------------------------------------------------------------------------
inline AsyncExporter* g_async = nullptr;

inline AsyncExporter* async_exporter() noexcept { return g_async; }

struct AsyncStopper {
    ~AsyncStopper()
    {
        if (g_async != nullptr) { g_async->stop_and_join(); }
    }
};
inline AsyncStopper g_async_stopper;

inline bool install_async_exporter(int cpu, int fifo_prio, std::string& err)
{
    if (g_async != nullptr) {
        err = "dApp async exporter already installed";
        return false;
    }
    if (g_producer == nullptr) {
        err = "dApp ring producer not installed";
        return false;
    }
    auto ax = AsyncExporter::create(g_producer);
    if (!ax->start(cpu, fifo_prio, err)) { return false; }
    g_async = ax.release();
    return true;
}

} // namespace dapp
} // namespace nv

#endif // DAPP_ASYNC_HPP
