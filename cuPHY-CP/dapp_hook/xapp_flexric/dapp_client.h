// dapp_client.h (2026-11-05) -- header-only C client for the DGIST dApp agent
// (cuphycontroller process, TCP 9810, JSON lines; protocol: cuPHY-CP/dapp_hook/docs/XAPP_INTERFACE.md).
//
// Drop this file into any xApp (FlexRIC or not) and:
//
//   #include "dapp_client.h"
//   dapp_client_t c;
//   if (dapp_connect(&c, "127.0.0.1", 9810) != 0) ...;
//   dapp_hello(&c, "ra", "my-xapp", 1, 1, 5000);                   // subscribe estimates, every slot, wait 5 s for welcome
//   dapp_estimate_t e;  dapp_latest_estimate(&c, &e);              // est_ms[m], yolo_sm_now, queue ...
//   int m = dapp_pick_model(&e, 40.0);                               // largest model whose est_ms <= 40 ms
//   dapp_result_t r;    dapp_run(&c, jpeg, len, DAPP_MODELS[m], 40, 0, 5000, &r);   // deadline 40 ms, no dets, 5 s timeout
//   printf("%s sm %d infer %.1f latency %.1f\n", r.status, r.sm, r.infer_ms, r.latency_ms);
//   dapp_close(&c);
//
// Thread model: one receive thread per connection parses every line the agent pushes (welcome, estimate,
// sm_report, result). dapp_run() blocks until the result with the same req_id arrives (or the timeout).
// All functions are safe to call from one application thread; sends are serialised with a mutex.
// No external dependencies (POSIX sockets + pthreads). The JSON handling is key-based extraction, enough
// for the flat messages of this protocol; it is not a general JSON parser.
#ifndef DAPP_CLIENT_H
#define DAPP_CLIENT_H

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DAPP_NMODELS 5
static const char* const DAPP_MODELS[DAPP_NMODELS] = {"26n", "26s", "26m", "26l", "26x"};   /* smallest .. largest */

typedef struct {
    long long t_ns;                 /* agent CLOCK_REALTIME ns when the estimate was made */
    int sfn, slot;                  /* air slot */
    int ul_sm, dl_sm;               /* cuPHY PUSCH / PDSCH SMs in that slot */
    int yolo_sm_now, yolo_sm_min;   /* SMs YOLO can use now / minimum over the planned horizon */
    int queue, inflight;            /* agent request queue state */
    double est_ms[DAPP_NMODELS];    /* "if RUN is sent now": queue wait + decode + inference, per model (<0 = unknown) */
    int est_sm[DAPP_NMODELS];       /* SM count behind each estimate */
    unsigned long seq;              /* number of estimates received so far */
    long long rx_ns;                /* local CLOCK_REALTIME ns of reception */
} dapp_estimate_t;

typedef struct {
    unsigned long long req_id;
    int ok;                         /* 1 = status "success" */
    char status[16];                /* success | fail | timeout */
    char reason[32];                /* on fail: no_engine, decode_error, overload, paused, internal, bad bytes */
    char model[8];
    int sm;                         /* SMs the inference started with */
    double sm_mean;                 /* mean SMs over the inference (chunks may move between classes) */
    double wait_ms, decode_ms, infer_ms, latency_ms;   /* server side: queue wait, JPEG decode, inference, receive->reply */
    double e2e_ms;                  /* client side: send -> result */
    int detections, tiles;
    long long t_recv_ns, t_start_ns, t_end_ns;        /* agent timestamps (CLOCK_REALTIME ns) */
    char json[8192];                /* the raw result line (truncated to this size; dets included when want_dets) */
} dapp_result_t;

typedef int (*dapp_on_line_fn)(const char* line, size_t len, void* user);   /* optional raw hook, called from the rx thread */

typedef struct {
    int fd;
    pthread_t rx;
    int rx_started;
    pthread_mutex_t mu;             /* protects everything below + serialises sends */
    pthread_cond_t cv;
    int closed;
    /* welcome */
    int have_welcome;
    char welcome[1024];
    char models[64], tdd[32];
    int gpu_sm, cells, slot_us, version;
    /* latest pushes */
    dapp_estimate_t est;
    int have_est;
    char report[2048];
    int have_report;
    /* results waiting to be collected by dapp_wait_result() */
    struct { unsigned long long id; char* json; } pending[32];
    unsigned long long next_id;
    dapp_on_line_fn on_line;
    void* user;
} dapp_client_t;

/* ------------------------------------------------------------------ small JSON helpers (key-based) */
static const char* dapp__jkey(const char* s, const char* end, const char* key) {
    char pat[64]; const size_t n = (size_t)snprintf(pat, sizeof pat, "\"%s\":", key);
    for (const char* p = s; p && p + n <= end; ) { p = (const char*)memmem(p, (size_t)(end - p), pat, n); if (!p) return NULL; return p + n; }
    return NULL;
}
static double dapp__jnum(const char* s, const char* end, const char* key, double def) {
    const char* v = dapp__jkey(s, end, key); if (!v) return def;
    if (!strncmp(v, "null", 4)) return def;
    return strtod(v, NULL);
}
static int dapp__jstr(const char* s, const char* end, const char* key, char* out, size_t n) {
    const char* v = dapp__jkey(s, end, key); if (!v || *v != '"') { if (n) out[0] = 0; return 0; }
    v++; const char* q = (const char*)memchr(v, '"', (size_t)(end - v)); if (!q) { if (n) out[0] = 0; return 0; }
    size_t len = (size_t)(q - v); if (len >= n) len = n - 1; memcpy(out, v, len); out[len] = 0; return 1;
}
static int dapp__jbool(const char* s, const char* end, const char* key) {
    const char* v = dapp__jkey(s, end, key); return v && !strncmp(v, "true", 4);
}
/* returns the span of the object/array value of key (from '{'/'[' to the matching close), or NULL */
static const char* dapp__jobj(const char* s, const char* end, const char* key, const char** oend) {
    const char* v = dapp__jkey(s, end, key); if (!v || (*v != '{' && *v != '[')) return NULL;
    const char open = *v, close = open == '{' ? '}' : ']'; int depth = 0; int instr = 0;
    for (const char* p = v; p < end; p++) {
        if (instr) { if (*p == '\\') p++; else if (*p == '"') instr = 0; continue; }
        if (*p == '"') instr = 1; else if (*p == open) depth++; else if (*p == close) { if (--depth == 0) { *oend = p + 1; return v; } }
    }
    return NULL;
}
static long long dapp__now_ns(void) { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec; }

/* ------------------------------------------------------------------ internals */
static int dapp__send_all(dapp_client_t* c, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    while (len) { ssize_t n = send(c->fd, p, len, MSG_NOSIGNAL); if (n < 0) { if (errno == EINTR) continue; return -1; } p += n; len -= (size_t)n; }
    return 0;
}
static void dapp__parse_estimate(dapp_client_t* c, const char* s, const char* e) {
    dapp_estimate_t x; memset(&x, 0, sizeof x);
    x.t_ns = (long long)dapp__jnum(s, e, "t_ns", 0); x.sfn = (int)dapp__jnum(s, e, "sfn", -1); x.slot = (int)dapp__jnum(s, e, "slot", -1);
    x.ul_sm = (int)dapp__jnum(s, e, "ul_sm", 0); x.dl_sm = (int)dapp__jnum(s, e, "dl_sm", 0);
    x.yolo_sm_now = (int)dapp__jnum(s, e, "yolo_sm_now", 0); x.yolo_sm_min = (int)dapp__jnum(s, e, "yolo_sm_min", 0);
    x.queue = (int)dapp__jnum(s, e, "queue", 0); x.inflight = (int)dapp__jnum(s, e, "inflight", 0);
    const char *oe = NULL, *o = dapp__jobj(s, e, "est_ms", &oe);
    for (int m = 0; m < DAPP_NMODELS; m++) x.est_ms[m] = o ? dapp__jnum(o, oe, DAPP_MODELS[m], -1.0) : -1.0;
    o = dapp__jobj(s, e, "est_sm", &oe);
    for (int m = 0; m < DAPP_NMODELS; m++) x.est_sm[m] = o ? (int)dapp__jnum(o, oe, DAPP_MODELS[m], 0) : 0;
    x.rx_ns = dapp__now_ns();
    x.seq = c->est.seq + 1; c->est = x; c->have_est = 1;
}
static void dapp__handle_line(dapp_client_t* c, char* line, size_t len) {
    const char* e = line + len; char type[24];
    if (!dapp__jstr(line, e, "type", type, sizeof type)) return;
    pthread_mutex_lock(&c->mu);
    if (!strcmp(type, "welcome")) {
        snprintf(c->welcome, sizeof c->welcome, "%.*s", (int)len, line);
        const char *oe = NULL, *o = dapp__jobj(line, e, "models", &oe);
        if (o) snprintf(c->models, sizeof c->models, "%.*s", (int)(oe - o), o);
        dapp__jstr(line, e, "tdd", c->tdd, sizeof c->tdd);
        c->gpu_sm = (int)dapp__jnum(line, e, "gpu_sm", 0); c->cells = (int)dapp__jnum(line, e, "cells", 0);
        c->slot_us = (int)dapp__jnum(line, e, "slot_us", 500); c->version = (int)dapp__jnum(line, e, "version", 0);
        c->have_welcome = 1;
    } else if (!strcmp(type, "estimate")) {
        dapp__parse_estimate(c, line, e);
    } else if (!strcmp(type, "sm_report")) {
        snprintf(c->report, sizeof c->report, "%.*s", (int)len, line); c->have_report = 1;
    } else if (!strcmp(type, "result")) {
        const unsigned long long id = (unsigned long long)dapp__jnum(line, e, "req_id", 0);
        int slot = -1;
        for (int i = 0; i < 32; i++) if (!c->pending[i].json) { slot = i; break; }
        if (slot < 0) { free(c->pending[0].json); c->pending[0].json = NULL; slot = 0; }   /* nobody collected: drop the oldest */
        c->pending[slot].id = id; c->pending[slot].json = strndup(line, len);
    }
    pthread_cond_broadcast(&c->cv);
    pthread_mutex_unlock(&c->mu);
}
static void* dapp__rx_thread(void* arg) {
    dapp_client_t* c = (dapp_client_t*)arg;
    const size_t cap = 1 << 20; char* buf = (char*)malloc(cap); size_t have = 0;
    for (;;) {
        ssize_t n = recv(c->fd, buf + have, cap - 1 - have, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        have += (size_t)n;
        size_t start = 0;
        for (;;) {
            char* nl = (char*)memchr(buf + start, '\n', have - start); if (!nl) break;
            const size_t len = (size_t)(nl - (buf + start));
            if (len) { if (c->on_line) c->on_line(buf + start, len, c->user); dapp__handle_line(c, buf + start, len); }
            start += len + 1;
        }
        if (start) { memmove(buf, buf + start, have - start); have -= start; }
        if (have == cap - 1) have = 0;   /* a line longer than the buffer: drop it */
    }
    free(buf);
    pthread_mutex_lock(&c->mu); c->closed = 1; pthread_cond_broadcast(&c->cv); pthread_mutex_unlock(&c->mu);
    return NULL;
}
static int dapp__wait_ms(dapp_client_t* c, int ms, struct timespec* until) {   /* cond wait helper; mutex held */
    if (ms >= 0) return pthread_cond_timedwait(&c->cv, &c->mu, until);
    return pthread_cond_wait(&c->cv, &c->mu);
}
static void dapp__deadline(struct timespec* ts, int ms) {
    clock_gettime(CLOCK_REALTIME, ts); ts->tv_sec += ms / 1000; ts->tv_nsec += (long)(ms % 1000) * 1000000L;
    if (ts->tv_nsec >= 1000000000L) { ts->tv_sec++; ts->tv_nsec -= 1000000000L; }
}

/* ------------------------------------------------------------------ public API */
/* connect to the agent; returns 0 or -1 (errno set / message on stderr) */
static int dapp_connect(dapp_client_t* c, const char* host, int port) {
    memset(c, 0, sizeof *c); c->fd = -1; c->next_id = (unsigned long long)(dapp__now_ns() / 1000000LL) & 0xFFFFFFULL;
    for (int m = 0; m < DAPP_NMODELS; m++) c->est.est_ms[m] = -1.0;
    pthread_mutex_init(&c->mu, NULL); pthread_cond_init(&c->cv, NULL);
    char ports[16]; snprintf(ports, sizeof ports, "%d", port);
    struct addrinfo hints, *res = NULL; memset(&hints, 0, sizeof hints); hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    const int rc = getaddrinfo(host, ports, &hints, &res);
    if (rc != 0) { fprintf(stderr, "dapp_connect: %s:%d: %s\n", host, port, gai_strerror(rc)); return -1; }
    for (struct addrinfo* a = res; a; a = a->ai_next) {
        int fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol); if (fd < 0) continue;
        if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) { c->fd = fd; break; }
        close(fd);
    }
    freeaddrinfo(res);
    if (c->fd < 0) { fprintf(stderr, "dapp_connect: cannot connect to %s:%d (%s)\n", host, port, strerror(errno)); return -1; }
    int one = 1; setsockopt(c->fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    if (pthread_create(&c->rx, NULL, dapp__rx_thread, c) != 0) { close(c->fd); c->fd = -1; return -1; }
    c->rx_started = 1;
    return 0;
}

/* hello: role "ra" (Resource Allocator: estimates + run) or "slicing" (sm_report). every = push an estimate every N slots
   (0 = no estimate stream); period_ms = sm_report period (role slicing). Waits up to wait_ms for the welcome; returns 1 if it came. */
static int dapp_hello(dapp_client_t* c, const char* role, const char* name, int subscribe_estimate, int every, int wait_ms) {
    char msg[512];
    const int sub_report = !strcmp(role, "slicing");
    snprintf(msg, sizeof msg, "{\"type\":\"hello\",\"role\":\"%s\",\"name\":\"%s\",\"subscribe\":[%s%s%s],\"every\":%d,\"period_ms\":1000}\n",
             role, name ? name : role, subscribe_estimate ? "\"estimate\"" : "", subscribe_estimate && sub_report ? "," : "", sub_report ? "\"sm_report\"" : "", every > 0 ? every : 1);
    pthread_mutex_lock(&c->mu); const int rc = dapp__send_all(c, msg, strlen(msg)); pthread_mutex_unlock(&c->mu);
    if (rc) return 0;
    struct timespec until; dapp__deadline(&until, wait_ms);
    pthread_mutex_lock(&c->mu);
    while (!c->have_welcome && !c->closed) { if (dapp__wait_ms(c, wait_ms, &until) == ETIMEDOUT) break; }
    const int ok = c->have_welcome; pthread_mutex_unlock(&c->mu);
    return ok;
}

/* copy of the newest estimate; returns 1 if any estimate has been received */
static int dapp_latest_estimate(dapp_client_t* c, dapp_estimate_t* out) {
    pthread_mutex_lock(&c->mu); const int ok = c->have_est; if (ok) *out = c->est; pthread_mutex_unlock(&c->mu); return ok;
}
/* wait until a newer estimate than seq arrives (or timeout); returns 1 on a new estimate */
static int dapp_wait_estimate(dapp_client_t* c, unsigned long seq, int timeout_ms, dapp_estimate_t* out) {
    struct timespec until; dapp__deadline(&until, timeout_ms);
    pthread_mutex_lock(&c->mu);
    while (!c->closed && c->est.seq <= seq) { if (dapp__wait_ms(c, timeout_ms, &until) == ETIMEDOUT) break; }
    const int ok = c->est.seq > seq; if (ok) *out = c->est; pthread_mutex_unlock(&c->mu); return ok;
}
/* Resource-Allocator policy: the largest model whose estimated server time fits the deadline, else the smallest.
   Returns a DAPP_MODELS index. With no estimate yet it returns 0 (26n). */
static int dapp_pick_model(const dapp_estimate_t* e, double deadline_ms) {
    int best = 0;
    if (!e) return best;
    for (int m = 0; m < DAPP_NMODELS; m++) if (e->est_ms[m] >= 0 && e->est_ms[m] <= deadline_ms) best = m;
    return best;
}

/* send one RUN without waiting; returns the req_id (0 on error). deadline_ms <= 0 = no QoS field. */
static unsigned long long dapp_run_async(dapp_client_t* c, const void* jpeg, size_t len, const char* model, int deadline_ms, int want_dets) {
    char hdr[256];
    pthread_mutex_lock(&c->mu);
    const unsigned long long id = ++c->next_id;
    if (deadline_ms > 0)
        snprintf(hdr, sizeof hdr, "{\"type\":\"run\",\"req_id\":%llu,\"model\":\"%s\",\"qos\":{\"deadline_ms\":%d,\"priority\":0},\"image\":\"jpeg\",\"bytes\":%zu,\"want_dets\":%s}\n",
                 id, model, deadline_ms, len, want_dets ? "true" : "false");
    else
        snprintf(hdr, sizeof hdr, "{\"type\":\"run\",\"req_id\":%llu,\"model\":\"%s\",\"qos\":{},\"image\":\"jpeg\",\"bytes\":%zu,\"want_dets\":%s}\n",
                 id, model, len, want_dets ? "true" : "false");
    int rc = dapp__send_all(c, hdr, strlen(hdr)); if (!rc) rc = dapp__send_all(c, jpeg, len);
    pthread_mutex_unlock(&c->mu);
    return rc ? 0ULL : id;
}
/* wait for the result of req_id; fills *r (status "timeout" if it did not arrive). t_send_ns = value of dapp_now_ns() before the send. */
static int dapp_wait_result(dapp_client_t* c, unsigned long long req_id, int timeout_ms, long long t_send_ns, dapp_result_t* r) {
    memset(r, 0, sizeof *r); r->req_id = req_id; strcpy(r->status, "timeout");
    struct timespec until; dapp__deadline(&until, timeout_ms);
    char* json = NULL;
    pthread_mutex_lock(&c->mu);
    for (;;) {
        for (int i = 0; i < 32; i++) if (c->pending[i].json && c->pending[i].id == req_id) { json = c->pending[i].json; c->pending[i].json = NULL; break; }
        if (json || c->closed) break;
        if (dapp__wait_ms(c, timeout_ms, &until) == ETIMEDOUT) break;
    }
    pthread_mutex_unlock(&c->mu);
    r->e2e_ms = (double)(dapp__now_ns() - t_send_ns) / 1e6;
    if (!json) return 0;
    const size_t len = strlen(json); const char* e = json + len;
    snprintf(r->json, sizeof r->json, "%s", json);
    dapp__jstr(json, e, "status", r->status, sizeof r->status); dapp__jstr(json, e, "reason", r->reason, sizeof r->reason);
    dapp__jstr(json, e, "model", r->model, sizeof r->model);
    r->ok = !strcmp(r->status, "success");
    r->sm = (int)dapp__jnum(json, e, "sm", 0); r->sm_mean = dapp__jnum(json, e, "sm_mean", 0);
    r->wait_ms = dapp__jnum(json, e, "wait_ms", 0); r->decode_ms = dapp__jnum(json, e, "decode_ms", 0);
    r->infer_ms = dapp__jnum(json, e, "infer_ms", 0); r->latency_ms = dapp__jnum(json, e, "latency_ms", 0);
    r->detections = (int)dapp__jnum(json, e, "detections", 0); r->tiles = (int)dapp__jnum(json, e, "tiles", 0);
    r->t_recv_ns = (long long)dapp__jnum(json, e, "t_recv_ns", 0); r->t_start_ns = (long long)dapp__jnum(json, e, "t_start_ns", 0); r->t_end_ns = (long long)dapp__jnum(json, e, "t_end_ns", 0);
    free(json);
    return 1;
}
/* RUN and wait: returns 1 when a result line arrived (check r->ok / r->status), 0 on timeout or send error */
static int dapp_run(dapp_client_t* c, const void* jpeg, size_t len, const char* model, int deadline_ms, int want_dets, int timeout_ms, dapp_result_t* r) {
    const long long t0 = dapp__now_ns();
    const unsigned long long id = dapp_run_async(c, jpeg, len, model, deadline_ms, want_dets);
    if (!id) { memset(r, 0, sizeof *r); strcpy(r->status, "send_error"); return 0; }
    return dapp_wait_result(c, id, timeout_ms, t0, r);
}
/* one-shot query: what = "estimate" | "models" | "sm_report"; the answer arrives through the normal push path */
static int dapp_get(dapp_client_t* c, const char* what) {
    char msg[96]; snprintf(msg, sizeof msg, "{\"type\":\"get\",\"what\":\"%s\"}\n", what);
    pthread_mutex_lock(&c->mu); const int rc = dapp__send_all(c, msg, strlen(msg)); pthread_mutex_unlock(&c->mu); return rc == 0;
}
static long long dapp_now_ns(void) { return dapp__now_ns(); }
static void dapp_close(dapp_client_t* c) {
    if (c->fd >= 0) { shutdown(c->fd, SHUT_RDWR); close(c->fd); c->fd = -1; }
    if (c->rx_started) { pthread_join(c->rx, NULL); c->rx_started = 0; }
    for (int i = 0; i < 32; i++) { free(c->pending[i].json); c->pending[i].json = NULL; }
}

#ifdef __cplusplus
}
#endif
#endif /* DAPP_CLIENT_H */
