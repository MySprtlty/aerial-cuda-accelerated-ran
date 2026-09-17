/*
 * CtrlReader: read-only view of the estimator's control block
 * (/dev/shm/aerial_dapp_ctrl, layout in dapp_hook/dapp_ctrl_abi.h).
 *
 * The estimator (estimator/live.py) rewrites the block at every slot under a
 * seqlock (seq odd while writing). A reader copies the block when seq is even
 * and the same before and after the copy. No locks, no syscalls after open().
 */
#pragma once
#include "dapp_hook/dapp_ctrl_abi.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace nv::dapp::sched {

class CtrlReader {
public:
    CtrlReader() = default;
    ~CtrlReader() { close(); }
    CtrlReader(const CtrlReader&) = delete;
    CtrlReader& operator=(const CtrlReader&) = delete;

    static std::string shm_path(const std::string& name)
    {
        if (name.rfind("/dev/shm/", 0) == 0) { return name; }
        return std::string("/dev/shm/") + (name.empty() || name[0] != '/' ? name : name.substr(1));
    }

    // Maps the block. Fails (without side effects) while the estimator has not created it yet.
    bool open(const std::string& name, std::string& err)
    {
        close();
        const std::string path = shm_path(name);
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = path + ": " + std::strerror(errno); return false; }
        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size < (off_t)DAPP_CTRL_SIZE) {
            ::close(fd);
            err = path + ": too small";
            return false;
        }
        void* m = ::mmap(nullptr, DAPP_CTRL_SIZE, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) { err = path + ": mmap: " + std::strerror(errno); return false; }
        const dapp_ctrl_block* b = static_cast<const dapp_ctrl_block*>(m);
        if (std::memcmp(b->magic, DAPP_CTRL_MAGIC, 8) != 0 || b->abi != DAPP_CTRL_ABI) {
            ::munmap(m, DAPP_CTRL_SIZE);
            err = path + ": bad magic/abi";
            return false;
        }
        map_  = m;
        path_ = path;
        return true;
    }

    bool is_open() const { return map_ != nullptr; }
    const std::string& path() const { return path_; }

    void close()
    {
        if (map_) { ::munmap(map_, DAPP_CTRL_SIZE); map_ = nullptr; }
    }

    // Seqlock read. false if the writer was mid-update on every attempt (never seen in practice:
    // the writer holds the odd state for a few hundred nanoseconds).
    bool read(dapp_ctrl_block& out) const
    {
        if (!map_) { return false; }
        const dapp_ctrl_block* b = static_cast<const dapp_ctrl_block*>(map_);
        for (int i = 0; i < 16; ++i) {
            const uint64_t s1 = __atomic_load_n(&b->seq, __ATOMIC_ACQUIRE);
            if (s1 & 1u) { continue; }
            std::memcpy(&out, b, sizeof(out));
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            const uint64_t s2 = __atomic_load_n(&b->seq, __ATOMIC_ACQUIRE);
            if (s1 == s2) { return true; }
        }
        return false;
    }

    static int64_t now_ns()
    {
        timespec ts{};
        clock_gettime(CLOCK_REALTIME, &ts);
        return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    }

private:
    void*       map_ = nullptr;
    std::string path_;
};

} // namespace nv::dapp::sched
