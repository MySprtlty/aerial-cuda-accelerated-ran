/*
 * dApp FAPI hook ring: header-only C++ producer / consumer.
 *
 * Producer  - owned by the L1 (cuphycontroller) process. Created once at
 *             startup (not real-time), written from exactly ONE thread
 *             (the L2 adapter msg_processing thread). The real-time path
 *             (begin()/commit()) does no allocation, no locking and no
 *             system calls except clock_gettime (vDSO).
 * Consumer  - any process (dApp, dump tool). Read-only mapping, own cursor,
 *             wait-free, detects torn records, overruns and producer restarts.
 *
 * See dapp_ring_abi.h for the shared-memory layout.
 */
#ifndef DAPP_RING_HPP
#define DAPP_RING_HPP

#include "dapp_ring_abi.h"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nv {
namespace dapp {

inline int64_t now_ns() noexcept
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

inline size_t ring_bytes(uint32_t ring_len) noexcept
{
    return static_cast<size_t>(DAPP_RING_HDR_SIZE) + static_cast<size_t>(ring_len) * DAPP_REC_SIZE;
}

inline bool is_pow2(uint32_t v) noexcept { return v != 0 && (v & (v - 1)) == 0; }

//----------------------------------------------------------------------------
// Producer
//----------------------------------------------------------------------------
class Producer {
public:
    struct Config {
        std::string name         = DAPP_RING_DEFAULT_NAME; // "/name" -> /dev/shm/name
        uint32_t    ring_len     = DAPP_RING_DEFAULT_LEN;  // power of two
        bool        do_mlock     = true;
        uint32_t    slot_advance = 0;
        uint32_t    mu           = 0;
        uint32_t    num_cells    = 0;
        std::string producer_name = "cuphycontroller";
    };

    // Creates (or re-initialises) the shared memory object. Not real-time safe.
    // Returns nullptr and fills `err` on failure.
    static std::unique_ptr<Producer> create(const Config& cfg, std::string& err)
    {
        if (!is_pow2(cfg.ring_len) || cfg.ring_len < 16) {
            err = "ring_len must be a power of two >= 16";
            return nullptr;
        }
        if (cfg.name.empty() || cfg.name[0] != '/' || cfg.name.size() >= DAPP_RING_NAME_MAX) {
            err = "shm name must start with '/' and be shorter than 64 characters";
            return nullptr;
        }

        const size_t bytes = ring_bytes(cfg.ring_len);
        int fd = ::shm_open(cfg.name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0) {
            err = std::string("shm_open failed: ") + std::strerror(errno);
            return nullptr;
        }
        // Readers may run as a different user: make the object world-readable.
        (void)::fchmod(fd, 0666);

        // Preserve the generation counter of a previous producer instance.
        uint32_t prev_generation = 0;
        {
            struct stat st{};
            if (::fstat(fd, &st) == 0 && static_cast<size_t>(st.st_size) >= DAPP_RING_HDR_SIZE) {
                dapp_ring_hdr_t old{};
                if (::pread(fd, &old, sizeof(old), 0) == static_cast<ssize_t>(sizeof(old)) &&
                    old.magic == DAPP_RING_MAGIC) {
                    prev_generation = old.generation;
                }
            }
        }

        if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
            err = std::string("ftruncate failed: ") + std::strerror(errno);
            ::close(fd);
            return nullptr;
        }
        void* base = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            err = std::string("mmap failed: ") + std::strerror(errno);
            ::close(fd);
            return nullptr;
        }

        std::unique_ptr<Producer> p(new Producer());
        p->fd_    = fd;
        p->base_  = base;
        p->bytes_ = bytes;
        p->hdr_   = static_cast<dapp_ring_hdr_t*>(base);
        p->recs_  = reinterpret_cast<dapp_rec_t*>(static_cast<uint8_t*>(base) + DAPP_RING_HDR_SIZE);
        p->mask_  = cfg.ring_len - 1;
        p->name_  = cfg.name;

        // Invalidate the header first so a concurrent reader never trusts a
        // half-initialised ring, then zero everything (this also pre-faults
        // every page so the real-time thread never takes a page fault).
        __atomic_store_n(&p->hdr_->magic, 0ULL, __ATOMIC_RELEASE);
        std::memset(base, 0, bytes);

        bool locked = false;
        if (cfg.do_mlock) {
            locked = (::mlock(base, bytes) == 0);
        }

        dapp_ring_hdr_t* h = p->hdr_;
        h->abi_version  = DAPP_RING_ABI_VERSION;
        h->rec_size     = DAPP_REC_SIZE;
        h->ring_len     = cfg.ring_len;
        h->hdr_size     = DAPP_RING_HDR_SIZE;
        h->generation   = prev_generation + 1;
        h->producer_pid = static_cast<uint32_t>(::getpid());
        h->start_ts_ns  = static_cast<uint64_t>(now_ns());
        h->head         = 0;
        h->heartbeat_ns = h->start_ts_ns;
        h->slot_advance = cfg.slot_advance;
        h->mu           = cfg.mu;
        h->num_cells    = cfg.num_cells;
        h->flags        = locked ? 1u : 0u;
        std::strncpy(h->producer_name, cfg.producer_name.c_str(), DAPP_RING_NAME_MAX - 1);
        __atomic_store_n(&h->magic, DAPP_RING_MAGIC, __ATOMIC_RELEASE);

        p->next_seq_ = 1;
        p->mlocked_  = locked;

        dapp_rec_t* r = p->begin(DAPP_REC_PRODUCER_START, 0, 0, 0xFFFF);
        r->u.start.pid          = h->producer_pid;
        r->u.start.generation   = h->generation;
        r->u.start.slot_advance = cfg.slot_advance;
        r->u.start.mu           = cfg.mu;
        r->u.start.num_cells    = cfg.num_cells;
        r->u.start.ring_len     = cfg.ring_len;
        r->u.start.abi_version  = DAPP_RING_ABI_VERSION;
        r->u.start.rec_size     = DAPP_REC_SIZE;
        p->commit(r);
        return p;
    }

    // The shared memory object is deliberately NOT unlinked: it stays in
    // /dev/shm for post-mortem inspection and is re-initialised by the next
    // producer. Call unlink() explicitly if you want it removed.
    ~Producer()
    {
        if (base_ != nullptr && base_ != MAP_FAILED) {
            if (mlocked_) { (void)::munlock(base_, bytes_); }
            ::munmap(base_, bytes_);
        }
        if (fd_ >= 0) { ::close(fd_); }
    }

    static int unlink(const std::string& name) { return ::shm_unlink(name.c_str()); }

    // ---- real-time path -------------------------------------------------
    // begin(): returns the record slot for the next sequence number with the
    // common prefix filled and the payload zeroed. Fill r->u.* then commit().
    inline dapp_rec_t* begin(uint16_t type, uint16_t sfn, uint16_t slot, uint16_t cell_id) noexcept
    {
        dapp_rec_t* r = &recs_[next_seq_ & mask_];
        __atomic_store_n(&r->seq, 0ULL, __ATOMIC_RELAXED); // invalidate for readers
        __atomic_thread_fence(__ATOMIC_RELEASE);            // ... before the payload changes
        r->ts_ns   = static_cast<uint64_t>(now_ns());
        r->type    = type;
        r->sfn     = sfn;
        r->slot    = slot;
        r->cell_id = cell_id;
        std::memset(&r->u, 0, sizeof(r->u));
        return r;
    }

    inline void commit(dapp_rec_t* r) noexcept
    {
        const uint64_t seq = next_seq_;
        __atomic_store_n(&r->seq, seq, __ATOMIC_RELEASE);
        __atomic_store_n(&hdr_->head, seq, __ATOMIC_RELEASE);
        hdr_->cnt_records = seq;
        next_seq_ = seq + 1;
    }

    inline void heartbeat() noexcept { hdr_->heartbeat_ns = static_cast<uint64_t>(now_ns()); }

    // Plain (non-atomic) counters: single writer, readers only need approximate values.
    inline void count_ul_tti() noexcept        { hdr_->cnt_ul_tti++; }
    inline void count_dl_tti() noexcept        { hdr_->cnt_dl_tti++; }
    inline void count_slot_end() noexcept      { hdr_->cnt_slot_end++; }
    inline void count_slot_dropped() noexcept  { hdr_->cnt_slot_dropped++; }
    inline void count_pdu_truncated() noexcept { hdr_->cnt_pdu_truncated++; }

    // ---- info -----------------------------------------------------------
    const std::string& name() const noexcept { return name_; }
    uint64_t head() const noexcept { return __atomic_load_n(&hdr_->head, __ATOMIC_ACQUIRE); }
    uint32_t ring_len() const noexcept { return mask_ + 1; }
    uint32_t generation() const noexcept { return hdr_->generation; }
    bool mlocked() const noexcept { return mlocked_; }
    size_t bytes() const noexcept { return bytes_; }

    Producer(const Producer&) = delete;
    Producer& operator=(const Producer&) = delete;

private:
    Producer() = default;

    int              fd_       = -1;
    void*            base_     = nullptr;
    size_t           bytes_    = 0;
    dapp_ring_hdr_t* hdr_      = nullptr;
    dapp_rec_t*      recs_     = nullptr;
    uint64_t         mask_     = 0;
    uint64_t         next_seq_ = 1;
    bool             mlocked_  = false;
    std::string      name_;
};

//----------------------------------------------------------------------------
// Consumer
//----------------------------------------------------------------------------
class Consumer {
public:
    enum Status : int {
        NONE      = 0,  // no new record
        RECORD    = 1,  // `out` holds a valid record
        RESTARTED = -1, // producer restarted (generation changed); cursor was reset
    };

    // Opens an existing ring read-only. Fails if it does not exist yet.
    static std::unique_ptr<Consumer> open(const std::string& name, std::string& err)
    {
        int fd = ::shm_open(name.c_str(), O_RDONLY, 0);
        if (fd < 0) {
            err = std::string("shm_open failed: ") + std::strerror(errno);
            return nullptr;
        }
        dapp_ring_hdr_t hdr{};
        if (::pread(fd, &hdr, sizeof(hdr), 0) != static_cast<ssize_t>(sizeof(hdr))) {
            err = "ring header not readable (producer not started yet?)";
            ::close(fd);
            return nullptr;
        }
        if (hdr.magic != DAPP_RING_MAGIC) {
            err = "bad magic (ring being initialised or not a dapp ring)";
            ::close(fd);
            return nullptr;
        }
        if (hdr.abi_version != DAPP_RING_ABI_VERSION || hdr.rec_size != DAPP_REC_SIZE ||
            hdr.hdr_size != DAPP_RING_HDR_SIZE || !is_pow2(hdr.ring_len)) {
            err = "ABI mismatch: abi_version=" + std::to_string(hdr.abi_version) +
                  " rec_size=" + std::to_string(hdr.rec_size) +
                  " ring_len=" + std::to_string(hdr.ring_len);
            ::close(fd);
            return nullptr;
        }
        const size_t bytes = ring_bytes(hdr.ring_len);
        void* base = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            err = std::string("mmap failed: ") + std::strerror(errno);
            ::close(fd);
            return nullptr;
        }
        std::unique_ptr<Consumer> c(new Consumer());
        c->fd_    = fd;
        c->base_  = base;
        c->bytes_ = bytes;
        c->hdr_   = static_cast<const dapp_ring_hdr_t*>(base);
        c->recs_  = reinterpret_cast<const dapp_rec_t*>(static_cast<const uint8_t*>(base) + DAPP_RING_HDR_SIZE);
        c->mask_  = hdr.ring_len - 1;
        c->generation_ = hdr.generation;
        c->cursor_ = 1; // start from the oldest available record
        return c;
    }

    ~Consumer()
    {
        if (base_ != nullptr && base_ != MAP_FAILED) { ::munmap(base_, bytes_); }
        if (fd_ >= 0) { ::close(fd_); }
    }

    const dapp_ring_hdr_t& hdr() const noexcept { return *hdr_; }
    uint64_t head() const noexcept { return __atomic_load_n(&hdr_->head, __ATOMIC_ACQUIRE); }
    uint64_t cursor() const noexcept { return cursor_; }
    uint64_t lost() const noexcept { return lost_total_; }
    uint32_t ring_len() const noexcept { return static_cast<uint32_t>(mask_ + 1); }

    // Skip everything already in the ring; only records written after this call are returned.
    void seek_to_head() noexcept { cursor_ = head() + 1; }
    void seek(uint64_t seq) noexcept { cursor_ = seq == 0 ? 1 : seq; }

    // Producer liveness: true if the producer pid exists and the heartbeat is recent.
    bool producer_alive(int64_t max_age_ns = 2000000000LL) const noexcept
    {
        const uint32_t pid = hdr_->producer_pid;
        if (pid == 0 || ::kill(static_cast<pid_t>(pid), 0) != 0) { return false; }
        const int64_t age = now_ns() - static_cast<int64_t>(__atomic_load_n(&hdr_->heartbeat_ns, __ATOMIC_ACQUIRE));
        return age <= max_age_ns;
    }

    // Copies the next record into `out`. Returns RECORD, NONE or RESTARTED.
    // `lost` (optional) receives the number of records skipped because of overrun.
    int next(dapp_rec_t& out, uint64_t* lost = nullptr) noexcept
    {
        if (lost) { *lost = 0; }

        const uint32_t gen = __atomic_load_n(&hdr_->generation, __ATOMIC_ACQUIRE);
        if (gen != generation_) {
            generation_ = gen;
            cursor_     = 1;
            return RESTARTED;
        }

        for (int attempt = 0; attempt < 8; ++attempt) {
            const uint64_t head = this->head();
            if (cursor_ > head) { return NONE; }

            // Overrun: the producer has already wrapped past our cursor.
            // Keep a safety margin of 1/8 ring so we do not race the writer.
            const uint64_t margin = (mask_ + 1) / 8;
            if (head - cursor_ + 1 + margin > mask_ + 1) {
                const uint64_t new_cursor = head + 1 + margin - (mask_ + 1);
                const uint64_t skipped = new_cursor - cursor_;
                lost_total_ += skipped;
                if (lost) { *lost += skipped; }
                cursor_ = new_cursor;
            }

            const dapp_rec_t* r = &recs_[cursor_ & mask_];
            const uint64_t s1 = __atomic_load_n(&r->seq, __ATOMIC_ACQUIRE);
            if (s1 != cursor_) {
                if (s1 > cursor_) {
                    // Overwritten between the head check and here: skip forward.
                    const uint64_t skipped = s1 - cursor_;
                    lost_total_ += skipped;
                    if (lost) { *lost += skipped; }
                    cursor_ = s1;
                    continue;
                }
                return NONE; // being written (seq==0) or not yet visible
            }
            std::memcpy(&out, r, sizeof(out));
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            const uint64_t s2 = __atomic_load_n(&r->seq, __ATOMIC_RELAXED);
            if (s1 != s2) { continue; } // torn read: retry
            ++cursor_;
            return RECORD;
        }
        return NONE;
    }

    Consumer(const Consumer&) = delete;
    Consumer& operator=(const Consumer&) = delete;

private:
    Consumer() = default;

    int                    fd_    = -1;
    void*                  base_  = nullptr;
    size_t                 bytes_ = 0;
    const dapp_ring_hdr_t* hdr_   = nullptr;
    const dapp_rec_t*      recs_  = nullptr;
    uint64_t               mask_  = 0;
    uint64_t               cursor_ = 1;
    uint64_t               lost_total_ = 0;
    uint32_t               generation_ = 0;
};

//----------------------------------------------------------------------------
// Process-wide producer
//
// Installed once from the L2 adapter start-up path, before any real-time
// thread runs. The hot path reads a plain pointer (null when the hook is
// disabled), so the cost of a disabled hook is one predictable branch.
//----------------------------------------------------------------------------
inline std::unique_ptr<Producer> g_producer_owner;
inline Producer*                 g_producer     = nullptr;
inline bool                      g_export_ul    = true;
inline bool                      g_export_dl    = false;
inline bool                      g_export_pdus  = true;

inline Producer* producer() noexcept { return g_producer; }
inline bool export_ul() noexcept { return g_export_ul; }
inline bool export_dl() noexcept { return g_export_dl; }
inline bool export_pdus() noexcept { return g_export_pdus; }

inline bool install_producer(const Producer::Config& cfg, std::string& err)
{
    if (g_producer != nullptr) {
        err = "dApp ring producer already installed";
        return false;
    }
    auto p = Producer::create(cfg, err);
    if (!p) { return false; }
    g_producer_owner = std::move(p);
    g_producer       = g_producer_owner.get();
    return true;
}

inline void shutdown_producer() noexcept
{
    g_producer = nullptr;
    g_producer_owner.reset();
}

inline const char* rec_type_name(uint16_t t) noexcept
{
    switch (t) {
        case DAPP_REC_UL_PDU:         return "UL_PDU";
        case DAPP_REC_UL_TTI:         return "UL_TTI";
        case DAPP_REC_DL_PDU:         return "DL_PDU";
        case DAPP_REC_DL_TTI:         return "DL_TTI";
        case DAPP_REC_SLOT_END:       return "SLOT_END";
        case DAPP_REC_PRODUCER_START: return "PRODUCER_START";
        default:                      return "?";
    }
}

inline const char* ul_pdu_type_name(uint8_t t) noexcept
{
    switch (t) {
        case DAPP_UL_PRACH: return "PRACH";
        case DAPP_UL_PUSCH: return "PUSCH";
        case DAPP_UL_PUCCH: return "PUCCH";
        case DAPP_UL_SRS:   return "SRS";
        default:            return "?";
    }
}

inline const char* dl_pdu_type_name(uint8_t t) noexcept
{
    switch (t) {
        case DAPP_DL_PDCCH:  return "PDCCH";
        case DAPP_DL_PDSCH:  return "PDSCH";
        case DAPP_DL_CSI_RS: return "CSI_RS";
        case DAPP_DL_SSB:    return "SSB";
        default:             return "?";
    }
}

} // namespace dapp
} // namespace nv

#endif // DAPP_RING_HPP
