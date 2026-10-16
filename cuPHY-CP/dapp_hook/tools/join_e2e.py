#!/usr/bin/env python3
"""
join_e2e: join the UE-side csv(s) of e2e/ue_client.py (or e2e_client.py) with the dispatcher csv on req_id and decompose the
E2E latency per request (H3):

    uplink    = t_arrival (dispatcher, first header byte) - t_send (UE)          image transfer UE -> gNB -> host
    net_in    = t_enqueue - t_arrival                                           rest of the 1 MB body arriving at the dispatcher
    queue     = t_dequeue - t_enqueue                                           waiting behind other requests
    handover  = t_start - t_dequeue                                             shm write + unix socket + gate wait (dapp_sched)
    service   = t_end - t_start                                                 decode + GPU + post (dapp_sched)
    reply_out = t_reply - t_end                                                 dispatcher reply assembly
    downlink  = t_reply (UE, last reply byte) - t_reply (dispatcher)            tiny reply host -> gNB -> UE
    rtt (UE)  = t_reply (UE) - t_send (UE)                                      the only UE-local number; = sum of the above

The UE clock is normally NOT synchronised to the host (no PTP on the laptop). Two modes:
  --ue-synced          the UE clock is trusted (NTP/PTP), uplink/downlink are taken as measured
  default              the offset is estimated from the downlink leg: offset = median(t_reply_ue - t_reply_disp) - dl_ms,
                       where --dl-ms (default 8) is the assumed one-way latency of a ~300-byte reply over the 5G DL; uplink
                       is then rtt - server - dl. The split uplink/downlink is therefore an estimate, the SUM is exact.

    python3 join_e2e.py --disp disp.csv --ue ue1.csv [--ue ue2.csv ...] [--first-s 10] [--out joined.csv] [--dl-ms 8] [--ue-synced]
Prints: per UE and overall mean / p50 / p99 of each term, model shares, lost/error counts, offered and served rates.
"""
import argparse
import csv
import sys


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v); return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def fnum(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


TERMS = ["uplink", "net_in", "queue", "handover", "service", "reply_out", "downlink", "rtt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disp", required=True); ap.add_argument("--ue", action="append", required=True)
    ap.add_argument("--first-s", type=float, default=10.0); ap.add_argument("--out", default="")
    ap.add_argument("--dl-ms", type=float, default=8.0); ap.add_argument("--ue-synced", action="store_true")
    a = ap.parse_args()
    disp = {fnum(r["req_id"]): r for r in csv.DictReader(open(a.disp))}
    joined = []
    for path in a.ue:
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        synced_key = "t_send_unix_ns" in rows[0]
        t_first = min(fnum(r.get("t_send_unix_ns", 0)) for r in rows) if synced_key else 0
        # clock offset UE -> host from the downlink leg
        pairs = [(fnum(r["t_reply_unix_ns"]), fnum(disp[fnum(r["req_id"])]["t_reply"])) for r in rows
                 if fnum(r["req_id"]) in disp and r.get("status", "0") == "0" and fnum(r.get("t_reply_unix_ns", 0))]
        if not pairs:
            print("%s: no joinable requests" % path); continue
        diffs = sorted(u - d for u, d in pairs)
        offset_ns = 0 if a.ue_synced else diffs[len(diffs) // 2] - int(a.dl_ms * 1e6)
        ue_id = rows[0].get("ue_id", path)
        for r in rows:
            rid = fnum(r["req_id"]); d = disp.get(rid)
            t_send = fnum(r["t_send_unix_ns"]) - offset_ns; t_rep_ue = fnum(r["t_reply_unix_ns"]) - offset_ns
            if t_send - (t_first - offset_ns) < a.first_s * 1e9:
                continue
            j = {"ue": ue_id, "req_id": rid, "status": r.get("status", ""), "model": (d or {}).get("model", r.get("engine", "")), "bytes": r.get("bytes", "")}
            if d and r.get("status", "0") == "0":
                ta, te, tq, ts_, tend, trp = (fnum(d[k]) for k in ("t_arrival", "t_enqueue", "t_dequeue", "t_start", "t_end", "t_reply"))
                j.update(uplink=(ta - t_send) / 1e6, net_in=(te - ta) / 1e6, queue=(tq - te) / 1e6, handover=(ts_ - tq) / 1e6 if ts_ else float("nan"),
                         service=(tend - ts_) / 1e6 if ts_ and tend else float("nan"), reply_out=(trp - tend) / 1e6 if tend else float("nan"),
                         downlink=(t_rep_ue - trp) / 1e6, rtt=(t_rep_ue - t_send) / 1e6)
            joined.append(j)
        print("%s: UE %s, %d rows, joined %d, clock offset UE-host %s%.3f ms (%s)" % (path, ue_id, len(rows), sum(1 for j in joined if j["ue"] == ue_id and "rtt" in j),
              "+" if offset_ns >= 0 else "", offset_ns / 1e6, "trusted UE clock" if a.ue_synced else "estimated from the downlink leg, dl %.0f ms assumed" % a.dl_ms))
    if a.out:
        keys = ["ue", "req_id", "status", "model", "bytes"] + TERMS
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); [w.writerow(j) for j in joined]
    ues = sorted({j["ue"] for j in joined}, key=str)
    print("\n| UE | served | lost / errors | " + " | ".join("%s mean / p50 / p99" % t for t in TERMS) + " | models |")
    print("|---|---|---|" + "---|" * len(TERMS) + "---|")
    for ue in ues + ["all"]:
        js = [j for j in joined if ue == "all" or j["ue"] == ue]
        ok = [j for j in js if "rtt" in j]
        cells = []
        for t in TERMS:
            v = [j[t] for j in ok if j.get(t) == j.get(t)]
            cells.append("%.1f / %.1f / %.1f" % (sum(v) / len(v), pct(v, 50), pct(v, 99)) if v else "-")
        models = {}
        for j in ok:
            models[j["model"]] = models.get(j["model"], 0) + 1
        print("| %s | %d | %d / %d | %s | %s |" % (ue, len(ok), sum(1 for j in js if j["status"] == "9"), sum(1 for j in js if j["status"] not in ("0", "9")),
              " | ".join(cells), " ".join("%s %.0f%%" % (m, 100.0 * k / len(ok)) for m, k in sorted(models.items())) if ok else "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
