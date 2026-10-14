#!/usr/bin/env python3
"""
req_summary: aggregate the per-request csv of e2e/dispatcher.py.

    python3 req_summary.py disp.csv [disp2.csv ...] [--first-s 1] [--out-csv out.csv]

Per file: offered req/s (served / span), E2E mean / p99 (t_arrival -> t_reply), wait (t_enqueue -> t_dequeue)
mean / p99, decode / inference / post-processing means (dapp_sched stamps: decode = t_start -> inference start is
inside dapp_sched, so decode+infer = t_start -> t_end minus the C++ post part; here: gpu = infer ms is not in the
csv, so decode+infer+post = t_start -> t_end), gate wait (t_sched_recv -> t_gate_pass), dispatcher overhead
(t_arrival -> t_sent_to_sched, t_end -> t_reply), model shares, throughput (served / span), max queue depth,
HOLD count. Requests of the first --first-s seconds are excluded (warm-up). Pure Python.
"""
import argparse
import csv
import sys


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def mean(v):
    return sum(v) / len(v) if v else 0.0


def summarize(path, first_s):
    rows = [r for r in csv.DictReader(open(path, newline=""))]
    if not rows:
        return None
    t0 = min(int(r["t_arrival"]) for r in rows)
    rows = [r for r in rows if int(r["t_arrival"]) - t0 >= first_s * 1e9]
    ok = [r for r in rows if r["status"] == "0"]
    if not rows:
        return None
    span = (max(int(r["t_reply"]) for r in rows) - min(int(r["t_arrival"]) for r in rows)) / 1e9
    ms = lambda r, a, b: (int(r[b]) - int(r[a])) / 1e6   # noqa: E731
    e2e = [ms(r, "t_arrival", "t_reply") for r in rows]
    wait = [ms(r, "t_enqueue", "t_dequeue") for r in rows]
    sched = [ms(r, "t_start", "t_end") for r in ok if int(r["t_start"]) and int(r["t_end"])]
    gate = [ms(r, "t_sched_recv", "t_gate_pass") for r in ok if int(r["t_sched_recv"]) and int(r["t_gate_pass"])]
    recv_lag = [ms(r, "t_sent_to_sched", "t_sched_recv") for r in ok if int(r["t_sched_recv"])]
    pre = [ms(r, "t_arrival", "t_sent_to_sched") for r in rows]
    post = [ms(r, "t_end", "t_reply") for r in ok if int(r["t_end"])]
    shm = [ms(r, "t_enqueue", "t_shm_written") for r in rows]
    models = {}
    for r in rows:
        models[r["model"]] = models.get(r["model"], 0) + 1
    # target rule: predicted E2E of the chosen model vs the measured one
    err = []
    for r in rows:
        v = r.get("pred_e2e_" + r["model"], "")
        if v not in ("", None):
            err.append(float(v) - (int(r["t_reply"]) - int(r["t_arrival"])) / 1e6)
    return {
        "pred_err_mean": mean(err) if err else None, "pred_err_p99": pct(err, 99) if err else None,
        "pred_err_abs_mean": mean([abs(x) for x in err]) if err else None, "pred_n": len(err),
        "file": path, "n": len(rows), "ok": len(ok), "span_s": span,
        "req_s": len(rows) / span if span > 0 else 0.0,
        "e2e_mean": mean(e2e), "e2e_p99": pct(e2e, 99),
        "wait_mean": mean(wait), "wait_p99": pct(wait, 99),
        "sched_mean": mean(sched), "sched_p99": pct(sched, 99),
        "gate_mean": mean(gate), "recv_lag_mean": mean(recv_lag),
        "pre_mean": mean(pre), "shm_mean": mean(shm), "post_mean": mean(post),
        "models": models, "max_depth": max(int(r["queue_depth"]) for r in rows) + 1,
        "holds": sum(int(r["n_hold"]) for r in rows), "errors": len(rows) - len(ok),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--first-s", type=float, default=1.0)
    ap.add_argument("--out-csv", default="")
    a = ap.parse_args()
    out = []
    print("| file | n | req/s | E2E mean / p99 ms | wait mean / p99 | dapp_sched start->end mean / p99 | gate wait | "
          "dispatcher: arrival->sent, shm write, end->reply | models | max depth | HOLD | err |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in a.csv:
        s = summarize(p, a.first_s)
        if s is None:
            print("| %s | (no rows) |" % p)
            continue
        out.append(s)
        print("| %s | %d | %.2f | %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | %.2f | %.2f, %.2f, %.2f | %s | %d | %d | %d |" % (
            s["file"].split("/")[-1], s["n"], s["req_s"], s["e2e_mean"], s["e2e_p99"], s["wait_mean"], s["wait_p99"],
            s["sched_mean"], s["sched_p99"], s["gate_mean"], s["pre_mean"], s["shm_mean"], s["post_mean"],
            " ".join("%s:%.0f%%" % (m, 100.0 * k / s["n"]) for m, k in sorted(s["models"].items())),
            s["max_depth"], s["holds"], s["errors"]))
        if s["pred_n"]:
            print("   target rule: predicted - measured E2E of the chosen model (ms): mean %.1f  p99 %.1f  |abs| mean %.1f  (n=%d)" % (
                s["pred_err_mean"], s["pred_err_p99"], s["pred_err_abs_mean"], s["pred_n"]))
    if a.out_csv and out:
        keys = [k for k in out[0] if k != "models"] + ["models"]
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in out:
                r = dict(s)
                r["models"] = " ".join("%s:%d" % kv for kv in sorted(s["models"].items()))
                w.writerow(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
