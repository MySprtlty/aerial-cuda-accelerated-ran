/*
 * dapp_agent.hpp (2026-11-02): the xApp-facing server of the in-process dApp (docs/XAPP_INTERFACE.md).
 *
 * One thread ("dapp_agent", core DAPP_AGENT_CPU) owns a TCP listening socket (DAPP_AGENT_PORT, default 9810) and the
 * client connections: it parses the JSON-lines protocol, hands RUN requests (JPEG bytes attached) to the tenant thread
 * through a queue, sends the tenant's results back to the requesting client, and fans the tenant's per-slot estimates and
 * periodic SM reports out to the subscribed clients (optionally also as UDP datagrams, DAPP_AGENT_UDP=host:port).
 * Nothing here touches CUDA; the tenant thread (dapp_tenant.cu) is the only GPU user.
 */
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nv { namespace dapp { namespace agent {

struct Request {
    uint64_t id = 0; int client = -1;
    std::string model;                 // "26n" .. "26x"
    double deadline_ms = 0.0; int priority = 0; bool want_dets = false;
    std::vector<unsigned char> jpeg;
    int64_t t_recv_ns = 0;             // CLOCK_REALTIME when the last byte arrived
};
struct Result {
    uint64_t id = 0; int client = -1;
    std::string status = "fail", reason, model;
    int sm = 0; double sm_mean = 0.0;
    int64_t t_recv_ns = 0, t_start_ns = 0, t_end_ns = 0;
    double wait_ms = 0.0, decode_ms = 0.0, infer_ms = 0.0, latency_ms = 0.0;
    int detections = 0, tiles = 0;
    std::string dets_json;             // "[...]" when want_dets, else empty
};

struct Config {
    int port = 9810; std::string udp; int cpu = -1; int max_queue = 8;
    std::string models_json = "[]";    // welcome: ["26n",...]
    int cells = 0; std::string tdd = "DDDDDDSUUU"; int gpu_sm = 132;
};

class Server {
public:
    ~Server() { stop(); }                                  // a joinable thread at static destruction would std::terminate (L1 exit handler)
    bool start(const Config& c, std::string& err);
    void stop();
    // tenant side
    bool pop_request(Request& r);                           // non-blocking
    size_t queue_depth();
    void push_result(const Result& r);
    void publish_estimate(const std::string& json_line);    // every slot; fanned out per subscriber's "every"
    void publish_sm_report(const std::string& json_line);   // whoever subscribed to sm_report
    // sm_report producer: the tenant installs a function that renders the current report (called on the agent thread)
    void set_report_fn(std::function<std::string(int window_ms)> f);
    void set_estimate_fn(std::function<std::string()> f);   // one-shot "get estimate"
    // stats
    uint64_t requests_received() const { return n_req_.load(); }
    uint64_t rejected() const { return n_rej_.load(); }
private:
    struct Client {
        int fd = -1; std::string role, name; bool sub_estimate = false, sub_report = false; int every = 1; int period_ms = 1000;
        int64_t next_report_ns = 0; uint64_t est_count = 0; bool want_close = false;
        std::string in; std::string out;                   // buffers
        // a RUN header seen, waiting for `bytes` of payload
        bool in_payload = false; size_t payload_left = 0; Request pending;
    };
    void run();
    void handle_line(Client& c, int idx, const std::string& line);
    void send(Client& c, const std::string& line);
    void flush(Client& c);
    void drop(int idx);
    std::string result_json(const Result& r) const;

    Config cfg_;
    std::thread th_; std::atomic<bool> stop_{false};
    int lfd_ = -1, ufd_ = -1; void* uaddr_ = nullptr;   // sockaddr_in of DAPP_AGENT_UDP
    std::vector<Client> clients_; std::mutex cm_;          // clients_ is only touched by the agent thread except through the queues
    std::mutex qm_; std::deque<Request> reqs_;             // tenant pops
    std::mutex rm_; std::deque<Result> results_;           // agent thread drains
    std::mutex em_; std::string latest_estimate_; uint64_t est_seq_ = 0, est_sent_ = 0; std::string pending_report_;
    std::function<std::string(int)> report_fn_; std::function<std::string()> estimate_fn_;
    std::atomic<uint64_t> n_req_{0}, n_rej_{0};
    int wake_fd_[2] = {-1, -1};
};

}}} // namespace nv::dapp::agent
