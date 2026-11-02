// dapp_agent.cpp (2026-11-02): see dapp_agent.hpp and docs/XAPP_INTERFACE.md
#include "dapp_agent.hpp"
#include "dapp_json.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace nv { namespace dapp { namespace agent {

namespace {
int64_t now_real_ns() { timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec; }
void set_nonblock(int fd) { const int f = fcntl(fd, F_GETFL, 0); fcntl(fd, F_SETFL, f | O_NONBLOCK); }
constexpr size_t kMaxOut = 8u << 20;     // a client that cannot drain 8 MB is dropped (slow consumer)
constexpr size_t kMaxLine = 64u << 10;
constexpr size_t kMaxJpeg = 64u << 20;
}

bool Server::start(const Config& c, std::string& err) {
    cfg_ = c;
    lfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd_ < 0) { err = std::string("socket: ") + std::strerror(errno); return false; }
    int one = 1; setsockopt(lfd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_ANY); a.sin_port = htons(static_cast<uint16_t>(cfg_.port));
    if (bind(lfd_, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) { err = "bind port " + std::to_string(cfg_.port) + ": " + std::strerror(errno); close(lfd_); lfd_ = -1; return false; }
    if (listen(lfd_, 8) < 0) { err = std::string("listen: ") + std::strerror(errno); close(lfd_); lfd_ = -1; return false; }
    set_nonblock(lfd_);
    if (!cfg_.udp.empty()) {
        const size_t k = cfg_.udp.rfind(':');
        if (k == std::string::npos) { err = "DAPP_AGENT_UDP wants host:port"; return false; }
        addrinfo hints{}; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_DGRAM; addrinfo* res = nullptr;
        if (getaddrinfo(cfg_.udp.substr(0, k).c_str(), cfg_.udp.substr(k + 1).c_str(), &hints, &res) != 0 || !res) { err = "DAPP_AGENT_UDP: cannot resolve " + cfg_.udp; return false; }
        sockaddr_in* ua = new sockaddr_in; std::memcpy(ua, res->ai_addr, sizeof(sockaddr_in)); uaddr_ = ua; freeaddrinfo(res);
        ufd_ = socket(AF_INET, SOCK_DGRAM, 0); set_nonblock(ufd_);
    }
    if (pipe(wake_fd_) != 0) { err = "pipe"; return false; }
    set_nonblock(wake_fd_[0]); set_nonblock(wake_fd_[1]);
    stop_.store(false);
    th_ = std::thread([this] { run(); });
    return true;
}

void Server::stop() {
    if (!th_.joinable()) return;
    stop_.store(true); const char x = 1; (void)!write(wake_fd_[1], &x, 1);
    th_.join();
    for (Client& c : clients_) { if (c.fd >= 0) close(c.fd); }
    clients_.clear();
    if (lfd_ >= 0) { close(lfd_); lfd_ = -1; }
    if (ufd_ >= 0) { close(ufd_); ufd_ = -1; }
    delete static_cast<sockaddr_in*>(uaddr_); uaddr_ = nullptr;
    close(wake_fd_[0]); close(wake_fd_[1]);
}

bool Server::pop_request(Request& r) { std::lock_guard<std::mutex> g(qm_); if (reqs_.empty()) return false; r = std::move(reqs_.front()); reqs_.pop_front(); return true; }
size_t Server::queue_depth() { std::lock_guard<std::mutex> g(qm_); return reqs_.size(); }
void Server::push_result(const Result& r) { { std::lock_guard<std::mutex> g(rm_); results_.push_back(r); } const char x = 1; (void)!write(wake_fd_[1], &x, 1); }
void Server::publish_estimate(const std::string& j) { { std::lock_guard<std::mutex> g(em_); latest_estimate_ = j; est_seq_++; } const char x = 1; (void)!write(wake_fd_[1], &x, 1); }
void Server::publish_sm_report(const std::string& j) { { std::lock_guard<std::mutex> g(em_); pending_report_ = j; } const char x = 1; (void)!write(wake_fd_[1], &x, 1); }
void Server::set_report_fn(std::function<std::string(int)> f) { report_fn_ = std::move(f); }
void Server::set_estimate_fn(std::function<std::string()> f) { estimate_fn_ = std::move(f); }

void Server::send(Client& c, const std::string& line) { if (c.fd < 0) return; c.out += line; c.out += '\n'; if (c.out.size() > kMaxOut) c.want_close = true; }
void Server::flush(Client& c) {
    while (c.fd >= 0 && !c.out.empty()) {
        const ssize_t n = ::send(c.fd, c.out.data(), c.out.size(), MSG_NOSIGNAL);
        if (n > 0) { c.out.erase(0, static_cast<size_t>(n)); continue; }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
        c.want_close = true; return;
    }
}
void Server::drop(int idx) { Client& c = clients_[idx]; if (c.fd >= 0) { close(c.fd); c.fd = -1; } c.sub_estimate = c.sub_report = false; c.in.clear(); c.out.clear(); c.in_payload = false; }

std::string Server::result_json(const Result& r) const {
    json::Obj o; o.kv("type", "result").kv("req_id", static_cast<unsigned long long>(r.id)).kv("status", r.status);
    if (!r.reason.empty()) o.kv("reason", r.reason);
    o.kv("model", r.model).kv("sm", r.sm).kv("sm_mean", r.sm_mean)
     .kv("t_recv_ns", static_cast<long long>(r.t_recv_ns)).kv("t_start_ns", static_cast<long long>(r.t_start_ns)).kv("t_end_ns", static_cast<long long>(r.t_end_ns))
     .kv("wait_ms", r.wait_ms).kv("decode_ms", r.decode_ms).kv("infer_ms", r.infer_ms).kv("latency_ms", r.latency_ms)
     .kv("detections", r.detections).kv("tiles", r.tiles);
    if (!r.dets_json.empty()) o.raw("dets", r.dets_json);
    return o.str();
}

void Server::handle_line(Client& c, int idx, const std::string& line) {
    json::Value v; std::string err;
    if (!json::parse(line, v, err) || v.type != json::Value::Object) { send(c, json::Obj().kv("type", "error").kv("error", "bad json: " + err).str()); return; }
    const std::string type = v.s("type");
    if (type == "hello") {
        c.role = v.s("role", "unknown"); c.name = v.s("name", c.role);
        c.sub_estimate = v.has_in_array("subscribe", "estimate"); c.sub_report = v.has_in_array("subscribe", "sm_report");
        c.every = std::max(1, static_cast<int>(v.i("every", 1))); c.period_ms = std::max(10, static_cast<int>(v.i("period_ms", 1000)));
        c.next_report_ns = now_real_ns() + static_cast<int64_t>(c.period_ms) * 1000000LL;
        // a returning role replaces the older connection of the same role
        for (size_t j = 0; j < clients_.size(); j++) { if (static_cast<int>(j) != idx && clients_[j].fd >= 0 && clients_[j].role == c.role && clients_[j].role != "unknown") clients_[j].want_close = true; }
        json::Obj o; o.kv("type", "welcome").kv("version", 1).raw("models", cfg_.models_json).kv("slot_us", 500).kv("cells", cfg_.cells).kv("tdd", cfg_.tdd).kv("gpu_sm", cfg_.gpu_sm);
        send(c, o.str()); return;
    }
    if (type == "run") {
        Request r; r.id = static_cast<uint64_t>(v.n("req_id", 0)); r.client = idx; r.model = v.s("model", "26x");
        if (const json::Value* q = v.get("qos")) { r.deadline_ms = q->n("deadline_ms", 0.0); r.priority = static_cast<int>(q->i("priority", 0)); }
        r.want_dets = v.t("want_dets", false);
        const long long bytes = v.i("bytes", 0);
        if (bytes <= 0 || static_cast<size_t>(bytes) > kMaxJpeg) { send(c, json::Obj().kv("type", "result").kv("req_id", static_cast<unsigned long long>(r.id)).kv("status", "fail").kv("reason", "bad bytes").str()); return; }
        c.pending = std::move(r); c.pending.jpeg.reserve(static_cast<size_t>(bytes)); c.in_payload = true; c.payload_left = static_cast<size_t>(bytes); return;
    }
    if (type == "get") {
        const std::string what = v.s("what");
        if (what == "estimate") { std::string e; { std::lock_guard<std::mutex> g(em_); e = latest_estimate_; } if (e.empty() && estimate_fn_) e = estimate_fn_(); send(c, e.empty() ? json::Obj().kv("type", "estimate").kv("status", "none").str() : e); return; }
        if (what == "sm_report") { send(c, report_fn_ ? report_fn_(c.period_ms) : json::Obj().kv("type", "sm_report").kv("status", "none").str()); return; }
        if (what == "models") { json::Obj o; o.kv("type", "welcome").kv("version", 1).raw("models", cfg_.models_json).kv("slot_us", 500).kv("cells", cfg_.cells).kv("tdd", cfg_.tdd).kv("gpu_sm", cfg_.gpu_sm); send(c, o.str()); return; }
        send(c, json::Obj().kv("type", "error").kv("error", "unknown get: " + what).str()); return;
    }
    if (type == "ping") { send(c, json::Obj().kv("type", "pong").kv("t_ns", static_cast<long long>(now_real_ns())).str()); return; }
    send(c, json::Obj().kv("type", "error").kv("error", "unknown type: " + type).str());
}

void Server::run() {
    pthread_setname_np(pthread_self(), "dapp_agent");
    if (cfg_.cpu >= 0) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cfg_.cpu, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs); }
    std::vector<pollfd> pfds; char buf[65536];
    while (!stop_.load(std::memory_order_acquire)) {
        pfds.clear();
        pfds.push_back({lfd_, POLLIN, 0}); pfds.push_back({wake_fd_[0], POLLIN, 0});
        for (Client& c : clients_) { if (c.fd >= 0) pfds.push_back({c.fd, static_cast<short>(POLLIN | (c.out.empty() ? 0 : POLLOUT)), 0}); }
        poll(pfds.data(), pfds.size(), 2);
        // accept
        if (pfds[0].revents & POLLIN) {
            for (;;) {
                sockaddr_in pa{}; socklen_t pl = sizeof pa; const int fd = accept(lfd_, reinterpret_cast<sockaddr*>(&pa), &pl);
                if (fd < 0) break;
                set_nonblock(fd); int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
                Client nc; nc.fd = fd; size_t slot = clients_.size();
                for (size_t j = 0; j < clients_.size(); j++) { if (clients_[j].fd < 0) { slot = j; break; } }
                if (slot == clients_.size()) clients_.push_back(nc); else clients_[slot] = nc;
            }
        }
        if (pfds[1].revents & POLLIN) { while (read(wake_fd_[0], buf, sizeof buf) > 0) { } }
        // read
        for (size_t i = 0; i < clients_.size(); i++) {
            Client& c = clients_[i]; if (c.fd < 0) continue;
            for (;;) {
                const ssize_t n = recv(c.fd, buf, sizeof buf, 0);
                if (n == 0) { c.want_close = true; break; }
                if (n < 0) { if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) c.want_close = true; break; }
                size_t off = 0;
                while (off < static_cast<size_t>(n)) {
                    if (c.in_payload) {
                        const size_t take = std::min(c.payload_left, static_cast<size_t>(n) - off);
                        c.pending.jpeg.insert(c.pending.jpeg.end(), buf + off, buf + off + take); off += take; c.payload_left -= take;
                        if (c.payload_left == 0) {
                            c.in_payload = false; c.pending.t_recv_ns = now_real_ns(); n_req_++;
                            bool accepted = false;
                            { std::lock_guard<std::mutex> g(qm_); if (static_cast<int>(reqs_.size()) < cfg_.max_queue) { reqs_.push_back(std::move(c.pending)); accepted = true; } }
                            if (!accepted) { n_rej_++; send(c, json::Obj().kv("type", "result").kv("req_id", static_cast<unsigned long long>(c.pending.id)).kv("status", "fail").kv("reason", "overload").kv("model", c.pending.model).str()); }
                            c.pending = Request();
                        }
                        continue;
                    }
                    const char* nl = static_cast<const char*>(std::memchr(buf + off, '\n', static_cast<size_t>(n) - off));
                    if (!nl) { c.in.append(buf + off, static_cast<size_t>(n) - off); if (c.in.size() > kMaxLine) c.want_close = true; break; }
                    c.in.append(buf + off, static_cast<size_t>(nl - (buf + off))); off = static_cast<size_t>(nl - buf) + 1;
                    std::string line; line.swap(c.in);
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (!line.empty()) handle_line(c, static_cast<int>(i), line);
                }
                if (static_cast<size_t>(n) < sizeof buf) break;
            }
        }
        // results from the tenant
        { std::deque<Result> rs; { std::lock_guard<std::mutex> g(rm_); rs.swap(results_); }
          for (const Result& r : rs) { if (r.client >= 0 && r.client < static_cast<int>(clients_.size()) && clients_[r.client].fd >= 0) send(clients_[r.client], result_json(r)); } }
        // estimates (latest wins) and reports
        { std::string est, rep; uint64_t seq = 0; { std::lock_guard<std::mutex> g(em_); seq = est_seq_; if (seq != est_sent_) { est = latest_estimate_; est_sent_ = seq; } rep.swap(pending_report_); }
          if (!est.empty()) {
              for (Client& c : clients_) { if (c.fd >= 0 && c.sub_estimate) { if ((c.est_count++ % static_cast<uint64_t>(c.every)) == 0) send(c, est); } }
              if (ufd_ >= 0 && uaddr_) { std::string d = est + "\n"; sendto(ufd_, d.data(), d.size(), MSG_DONTWAIT, static_cast<sockaddr*>(uaddr_), sizeof(sockaddr_in)); }
          }
          if (!rep.empty()) { for (Client& c : clients_) { if (c.fd >= 0 && c.sub_report) send(c, rep); } }
          const int64_t now = now_real_ns();
          for (Client& c : clients_) {
              if (c.fd >= 0 && c.sub_report && report_fn_ && now >= c.next_report_ns) { send(c, report_fn_(c.period_ms)); c.next_report_ns = now + static_cast<int64_t>(c.period_ms) * 1000000LL; }
          }
        }
        // write + close
        for (size_t i = 0; i < clients_.size(); i++) { Client& c = clients_[i]; if (c.fd < 0) continue; flush(c); if (c.want_close) drop(static_cast<int>(i)); }
    }
}

}}} // namespace nv::dapp::agent
