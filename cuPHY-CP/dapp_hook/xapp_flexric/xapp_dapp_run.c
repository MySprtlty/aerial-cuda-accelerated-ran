/*
 * xapp_dapp_run.c (2026-11-05) — a FlexRIC xApp that sends RUN requests to the DGIST dApp agent.
 *
 * Two independent interfaces live in one xApp:
 *   E2 (FlexRIC)   : xApp --E42--> nearRT-RIC --E2--> OAI gNB E2 agent.  Used here to read RAN state (MAC SM indications:
 *                    connected UEs, DL/UL throughput, PUSCH SNR). E2 ends at the gNB L2; nothing on E2 reaches the L1 or
 *                    the GPU, so the inference request cannot travel over E2.
 *   dApp agent     : xApp --TCP 9810 JSON--> cuphycontroller process (tenant/dapp_agent.cpp).  The agent pushes a per-slot
 *                    estimate (how long each YOLO model would take if sent now, from the FAPI-derived SM plan); the xApp
 *                    sends RUN + JPEG and gets the result. Protocol: docs/XAPP_INTERFACE.md, client: dapp_client.h.
 *
 * Usage (our options first, FlexRIC's -c/-p after them):
 *   xapp_dapp_run [--agent HOST:PORT] [--image FILE | --images DIR] [--model auto|26n|26s|26m|26l|26x]
 *                 [--deadline-ms N] [--rate R] [--count N] [--want-dets] [--mac-period 1_ms|2_ms|5_ms|10_ms]
 *                 [--no-ric] [--csv FILE] [-c flexric.conf]
 *   --no-ric skips FlexRIC entirely (dApp path only). Built without FlexRIC (xapp_dapp_run_noric) --no-ric is implied.
 *
 * Model policy "auto" = the Resource-Allocator rule of docs/XAPP_INTERFACE.md: the largest model whose est_ms fits
 * --deadline-ms, else the smallest. The deadline is also sent as qos.deadline_ms, so the agent refuses (fail/overload)
 * instead of serving late when its own estimate says the deadline cannot be met.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "dapp_client.h"

#ifndef NO_FLEXRIC
#include "xApp/e42_xapp_api.h"
#include "util/ngran_types.h"
#include "util/time_now_us.h"
#endif

/* ------------------------------------------------------------------ options */
static struct {
    char host[128]; int port;
    char image[512], images[512], csv[512];
    char model[8]; int deadline_ms; double rate; int count; int want_dets; int no_ric;
    char mac_period[16];
} opt = { "127.0.0.1", 9810, "", "", "", "auto", 40, 2.0, 20, 0, 0, "10_ms" };

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int s) { (void)s; if (g_stop) _exit(130); g_stop = 1; }   /* second signal: leave immediately */

static void usage(const char* p) {
    fprintf(stderr, "usage: %s [--agent HOST:PORT] [--image FILE | --images DIR] [--model auto|26n|26s|26m|26l|26x] [--deadline-ms N]\n"
                    "          [--rate R] [--count N] [--want-dets] [--mac-period 1_ms|2_ms|5_ms|10_ms] [--no-ric] [--csv FILE] [-c flexric.conf]\n", p);
}

/* takes our --options out of argv; what remains (argv[0], -c, -p) goes to FlexRIC's init_fr_args */
static int parse_args(int argc, char** argv, char** fargv) {
    int fargc = 0; fargv[fargc++] = argv[0];
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i]; const char* v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define NEED(x) do { if (!v) { fprintf(stderr, "%s needs a value\n", a); return -1; } x; i++; } while (0)
        if (!strcmp(a, "--agent")) NEED({ char* col = strrchr(v, ':'); if (col) { snprintf(opt.host, sizeof opt.host, "%.*s", (int)(col - v), v); opt.port = atoi(col + 1); } else snprintf(opt.host, sizeof opt.host, "%s", v); });
        else if (!strcmp(a, "--image")) NEED(snprintf(opt.image, sizeof opt.image, "%s", v));
        else if (!strcmp(a, "--images")) NEED(snprintf(opt.images, sizeof opt.images, "%s", v));
        else if (!strcmp(a, "--csv")) NEED(snprintf(opt.csv, sizeof opt.csv, "%s", v));
        else if (!strcmp(a, "--model")) NEED(snprintf(opt.model, sizeof opt.model, "%s", v));
        else if (!strcmp(a, "--deadline-ms")) NEED(opt.deadline_ms = atoi(v));
        else if (!strcmp(a, "--rate")) NEED(opt.rate = atof(v));
        else if (!strcmp(a, "--count")) NEED(opt.count = atoi(v));
        else if (!strcmp(a, "--mac-period")) NEED(snprintf(opt.mac_period, sizeof opt.mac_period, "%s", v));
        else if (!strcmp(a, "--want-dets")) opt.want_dets = 1;
        else if (!strcmp(a, "--no-ric")) opt.no_ric = 1;
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(argv[0]); return -1; }
        else if (!strcmp(a, "-c") || !strcmp(a, "-p")) { fargv[fargc++] = argv[i]; if (v) { fargv[fargc++] = argv[i + 1]; i++; } }
        else { fprintf(stderr, "unknown option %s\n", a); usage(argv[0]); return -1; }
#undef NEED
    }
    fargv[fargc] = NULL;
    return fargc;
}

/* ------------------------------------------------------------------ images */
typedef struct { char path[512]; unsigned char* data; size_t len; } image_t;
static image_t* g_img = NULL; static int g_nimg = 0;

static int load_file(image_t* im) {
    FILE* f = fopen(im->path, "rb"); if (!f) { perror(im->path); return -1; }
    struct stat st; if (fstat(fileno(f), &st) != 0) { fclose(f); return -1; }
    im->len = (size_t)st.st_size; im->data = (unsigned char*)malloc(im->len);
    const size_t n = fread(im->data, 1, im->len, f); fclose(f);
    if (n != im->len) { fprintf(stderr, "%s: short read\n", im->path); return -1; }
    if (im->len < 4 || im->data[0] != 0xFF || im->data[1] != 0xD8) { fprintf(stderr, "%s: not a JPEG (the agent decodes baseline JPEG with nvjpeg)\n", im->path); return -1; }
    return 0;
}
static int cmp_str(const void* a, const void* b) { return strcmp(((const image_t*)a)->path, ((const image_t*)b)->path); }
static int load_images(void) {
    if (opt.image[0]) { g_img = (image_t*)calloc(1, sizeof(image_t)); snprintf(g_img[0].path, sizeof g_img[0].path, "%s", opt.image); g_nimg = 1; return load_file(&g_img[0]); }
    if (!opt.images[0]) { fprintf(stderr, "--image FILE or --images DIR is required (e.g. ../models/img4k/img4k_0.jpg)\n"); return -1; }
    DIR* d = opendir(opt.images); if (!d) { perror(opt.images); return -1; }
    struct dirent* de; int cap = 16; g_img = (image_t*)calloc((size_t)cap, sizeof(image_t));
    while ((de = readdir(d)) != NULL) {
        const size_t l = strlen(de->d_name);
        if (l < 5 || strcasecmp(de->d_name + l - 4, ".jpg") || strstr(de->d_name, "prog")) continue;   /* progressive JPEGs are not decodable by the NVJPG engine */
        if (g_nimg == cap) { cap *= 2; g_img = (image_t*)realloc(g_img, (size_t)cap * sizeof(image_t)); }
        memset(&g_img[g_nimg], 0, sizeof(image_t)); snprintf(g_img[g_nimg].path, sizeof g_img[g_nimg].path, "%s/%s", opt.images, de->d_name); g_nimg++;
    }
    closedir(d);
    if (!g_nimg) { fprintf(stderr, "no baseline .jpg in %s\n", opt.images); return -1; }
    qsort(g_img, (size_t)g_nimg, sizeof(image_t), cmp_str);
    for (int i = 0; i < g_nimg; i++) if (load_file(&g_img[i])) return -1;
    return 0;
}

/* ------------------------------------------------------------------ RAN state from E2 (MAC service model) */
static pthread_mutex_t ran_mu = PTHREAD_MUTEX_INITIALIZER;
static struct {
    unsigned long long ind;          /* indications received */
    unsigned ues;                    /* UEs in the last indication */
    double dl_mbps, ul_mbps;         /* from the delta of the cumulative aggregate TBS between indications */
    double pusch_snr;                /* mean over UEs, last indication */
    int wb_cqi, dl_mcs, ul_mcs;      /* first UE, last indication */
    unsigned long long dl_tbs, ul_tbs; long long tstamp_us;
} ran;

#ifndef NO_FLEXRIC
static void sm_cb_mac(sm_ag_if_rd_t const* rd) {
    if (!rd || rd->type != INDICATION_MSG_AGENT_IF_ANS_V0 || rd->ind.type != MAC_STATS_V0) return;
    mac_ind_msg_t const* m = &rd->ind.mac.msg;
    unsigned long long dl = 0, ul = 0; double snr = 0;
    for (uint32_t i = 0; i < m->len_ue_stats; i++) { dl += m->ue_stats[i].dl_aggr_tbs; ul += m->ue_stats[i].ul_aggr_tbs; snr += m->ue_stats[i].pusch_snr; }
    pthread_mutex_lock(&ran_mu);
    ran.ind++;
    if (ran.tstamp_us && m->tstamp > ran.tstamp_us) {
        const double dt = (double)(m->tstamp - ran.tstamp_us) / 1e6;
        ran.dl_mbps = dl >= ran.dl_tbs ? (double)(dl - ran.dl_tbs) * 8.0 / dt / 1e6 : 0.0;
        ran.ul_mbps = ul >= ran.ul_tbs ? (double)(ul - ran.ul_tbs) * 8.0 / dt / 1e6 : 0.0;
    }
    ran.dl_tbs = dl; ran.ul_tbs = ul; ran.tstamp_us = m->tstamp; ran.ues = m->len_ue_stats;
    ran.pusch_snr = m->len_ue_stats ? snr / m->len_ue_stats : 0.0;
    if (m->len_ue_stats) { ran.wb_cqi = m->ue_stats[0].wb_cqi; ran.dl_mcs = m->ue_stats[0].dl_mcs1; ran.ul_mcs = m->ue_stats[0].ul_mcs1; }
    pthread_mutex_unlock(&ran_mu);
}
#endif

static const char* model_name(int m) { return (m >= 0 && m < DAPP_NMODELS) ? DAPP_MODELS[m] : "?"; }
static int model_index(const char* s) { for (int m = 0; m < DAPP_NMODELS; m++) if (!strcmp(s, DAPP_MODELS[m])) return m; return -1; }

int main(int argc, char** argv) {
    char* fargv[64];
    setvbuf(stdout, NULL, _IOLBF, 0);   /* line-buffered even when piped: progress stays visible */
    const int fargc = parse_args(argc, argv, fargv);
    if (fargc < 0) return 2;
#ifdef NO_FLEXRIC
    opt.no_ric = 1;
#endif
    const int fixed = strcmp(opt.model, "auto") ? model_index(opt.model) : -1;
    if (strcmp(opt.model, "auto") && fixed < 0) { fprintf(stderr, "unknown model %s\n", opt.model); return 2; }
    if (strcmp(opt.mac_period, "1_ms") && strcmp(opt.mac_period, "2_ms") && strcmp(opt.mac_period, "5_ms") && strcmp(opt.mac_period, "10_ms")) {
        fprintf(stderr, "--mac-period must be 1_ms, 2_ms, 5_ms or 10_ms (the MAC service model accepts only these; mac_sm_ric.c)\n"); return 2;   /* anything else asserts inside libe42_xapp */
    }
    if (opt.count > 0 && load_images()) return 2;

    /* 1. dApp agent: connect + hello (role ra, per-slot estimates) */
    dapp_client_t c;
    if (dapp_connect(&c, opt.host, opt.port)) return 1;
    if (!dapp_hello(&c, "ra", "flexric-xapp-dapp-run", 1, 1, 5000)) { fprintf(stderr, "no welcome from the dApp agent at %s:%d\n", opt.host, opt.port); dapp_close(&c); return 1; }
    printf("[dapp] connected to %s:%d: models %s, tdd %s, cells %d, gpu_sm %d, slot %d us\n", opt.host, opt.port, c.models, c.tdd, c.cells, c.gpu_sm, c.slot_us);

    /* 2. FlexRIC: register with the nearRT-RIC, list E2 nodes, subscribe to the MAC service model */
    int nnodes = 0;
#ifndef NO_FLEXRIC
    e2_node_arr_xapp_t nodes = {0};
    sm_ans_xapp_t* mac_h = NULL;
    if (!opt.no_ric) {
        fr_args_t args = init_fr_args(fargc, fargv);
        printf("[e2] registering with the nearRT-RIC (conf %s; E42 on SCTP 36422). FlexRIC keeps retrying every 5 s until the RIC has an E2 node (gNB) ...\n", args.conf_file);
        init_xapp_api(&args);
        sleep(1);
        printf("[e2] registered; asking for the connected E2 nodes ...\n");
        nodes = e2_nodes_xapp_api();
        nnodes = nodes.len;
        printf("[e2] registered with the nearRT-RIC; connected E2 nodes = %d\n", nnodes);
        if (nnodes > 0) mac_h = (sm_ans_xapp_t*)calloc((size_t)nnodes, sizeof(sm_ans_xapp_t));
        for (int i = 0; i < nnodes; i++) {
            e2_node_connected_xapp_t* n = &nodes.n[i];
            printf("[e2] node %d: nb_id %d plmn %d/%d type %s, RAN functions:", i, n->id.nb_id.nb_id, n->id.plmn.mcc, n->id.plmn.mnc, get_ngran_name(n->id.type));
            int has_mac = 0;
            for (size_t j = 0; j < n->len_rf; j++) { printf(" %d", n->rf[j].id); if (n->rf[j].id == 142) has_mac = 1; }
            printf("\n");
            if (has_mac && (n->id.type == ngran_gNB || n->id.type == ngran_gNB_DU || n->id.type == ngran_eNB)) {
                mac_h[i] = report_sm_xapp_api(&n->id, 142, (void*)opt.mac_period, sm_cb_mac);
                printf("[e2] MAC SM (142) subscription every %s: %s\n", opt.mac_period, mac_h[i].success ? "ok" : mac_h[i].u.reason);
            }
        }
        if (nnodes == 0) printf("[e2] no E2 node yet (gNB E2 agent not connected) — continuing with dApp RUNs only\n");
    }
#else
    (void)fargc;
#endif
    signal(SIGINT, on_sigint); signal(SIGTERM, on_sigint);

    /* 3. RUN loop */
    FILE* csv = NULL;
    if (opt.csv[0]) { csv = fopen(opt.csv, "w"); if (csv) fprintf(csv, "i,req_id,model,status,reason,sm,sm_mean,wait_ms,decode_ms,infer_ms,latency_ms,e2e_ms,detections,est_ms_model,yolo_sm_now,ues,dl_mbps,ul_mbps\n"); }
    int n_ok = 0, n_fail = 0, n_timeout = 0; double sum_lat = 0, sum_e2e = 0; int per_model[DAPP_NMODELS] = {0};
    const double period_s = opt.rate > 0 ? 1.0 / opt.rate : 0.0;
    dapp_estimate_t e; memset(&e, 0, sizeof e);
    if (opt.count > 0 && !dapp_wait_estimate(&c, 0, 2000, &e)) printf("[dapp] no estimate within 2 s (agent not pushing?) — policy falls back to 26n\n");
    for (int i = 0; i < opt.count && !g_stop; i++) {
        const long long t_iter = dapp_now_ns();
        const int have_e = dapp_latest_estimate(&c, &e);
        const int m = fixed >= 0 ? fixed : dapp_pick_model(have_e ? &e : NULL, (double)opt.deadline_ms);
        const image_t* im = &g_img[i % g_nimg];
        dapp_result_t r;
        dapp_run(&c, im->data, im->len, model_name(m), opt.deadline_ms, opt.want_dets, opt.deadline_ms * 4 + 3000, &r);
        if (r.ok) { n_ok++; sum_lat += r.latency_ms; sum_e2e += r.e2e_ms; per_model[m]++; } else if (!strcmp(r.status, "timeout")) n_timeout++; else n_fail++;
        pthread_mutex_lock(&ran_mu);
        printf("#%-3d %s %-7s sm %3d (mean %5.1f) wait %5.1f dec %5.1f inf %5.1f lat %6.1f e2e %6.1f ms det %3d | est %s %.1f ms yolo_sm %d q %d",
               i + 1, r.ok ? "ok  " : r.status, r.ok ? r.model : r.reason, r.sm, r.sm_mean, r.wait_ms, r.decode_ms, r.infer_ms, r.latency_ms, r.e2e_ms, r.detections,
               model_name(m), have_e ? e.est_ms[m] : -1.0, have_e ? e.yolo_sm_now : -1, have_e ? e.queue : -1);
        if (nnodes > 0) printf(" | RAN: ind %llu UEs %u DL %.2f UL %.2f Mbps SNR %.1f CQI %d", ran.ind, ran.ues, ran.dl_mbps, ran.ul_mbps, ran.pusch_snr, ran.wb_cqi);
        printf("\n");
        if (csv) fprintf(csv, "%d,%llu,%s,%s,%s,%d,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%.2f,%d,%u,%.3f,%.3f\n", i + 1, r.req_id, model_name(m), r.status, r.reason, r.sm, r.sm_mean, r.wait_ms, r.decode_ms, r.infer_ms, r.latency_ms, r.e2e_ms, r.detections,
                         have_e ? e.est_ms[m] : -1.0, have_e ? e.yolo_sm_now : -1, ran.ues, ran.dl_mbps, ran.ul_mbps);
        pthread_mutex_unlock(&ran_mu);
        fflush(stdout);
        if (period_s > 0) { const double left = period_s - (double)(dapp_now_ns() - t_iter) / 1e9; if (left > 0) usleep((useconds_t)(left * 1e6)); }
    }
    if (csv) fclose(csv);
    if (opt.count > 0) {
        printf("[dapp] %d requests: ok %d fail %d timeout %d; mean server latency %.1f ms, mean e2e %.1f ms; models", n_ok + n_fail + n_timeout, n_ok, n_fail, n_timeout,
               n_ok ? sum_lat / n_ok : 0.0, n_ok ? sum_e2e / n_ok : 0.0);
        for (int m = 0; m < DAPP_NMODELS; m++) if (per_model[m]) printf(" %s x%d", DAPP_MODELS[m], per_model[m]);
        printf("\n");
    }

    /* 4. shutdown */
#ifndef NO_FLEXRIC
    if (!opt.no_ric) {
        for (int i = 0; i < nnodes; i++) if (mac_h && mac_h[i].success && mac_h[i].u.handle != 0) rm_report_sm_xapp_api(mac_h[i].u.handle);
        free(mac_h);
        free_e2_node_arr_xapp(&nodes);
        printf("[e2] stopping the xApp ...\n");
        for (int k = 0; try_stop_xapp_api() == false; k++) { usleep(1000); if (k == 10000) { printf("[e2] stop not acknowledged after 10 s, leaving anyway\n"); break; } }
        pthread_mutex_lock(&ran_mu); printf("[e2] stopped; MAC indications received: %llu\n", ran.ind); pthread_mutex_unlock(&ran_mu);
    }
#endif
    dapp_close(&c);
    return (opt.count > 0 && n_ok == 0) ? 1 : 0;
}
