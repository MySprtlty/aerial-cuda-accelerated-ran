#!/usr/bin/env python3
"""
e2e_client: the UE side of the E2E latency measurement. Sends JPEG files to infer_server.py as an
OPEN-LOOP stream (fixed request rate, arrival times fixed in advance, so a slow path queues instead
of thinning the stream) and records, per request, the round trip from the first byte sent to the last
byte of the reply received, plus the server-side timings the reply carries. Pure Python 3 + stdlib, so it
runs on the UE laptop as it is.

    python3 e2e_client.py --server 10.0.0.1:9000 --images ../models/img4k --rate 10 --seconds 60 \
        [--arrivals fixed|poisson|jitter] [--seed 1] [--engine-sel -1] [--log client.csv] [--warm 3]

Output csv: req_id, t_arrival_ns (schedule), t_send_ns, t_reply_ns, rtt_ms (send start -> reply complete),
late_ms (schedule -> send start: client-side queueing when the socket is behind), bytes, status, engine,
n_det, and the server's net_in/queue/decode/infer/server ms. A summary line at the end: offered, served,
rtt mean/p50/p99/max, server p50, implied air+transport p50 (= rtt - server).
"""
import argparse
import csv
import glob
import json
import os
import socket
import struct
import sys
import threading
import time

HDR = struct.Struct("<4sIII")
MAGIC_Q, MAGIC_R = b"DAPQ", b"DAPR"


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("server closed")
        got += k
    return bytes(buf)


def percentile(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    i = min(len(s) - 1, max(0, int(p / 100.0 * len(s))))
    return s[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="host:port")
    ap.add_argument("--images", required=True, help="a JPEG file or a directory (uhd_*_q85.jpg are used round-robin)")
    ap.add_argument("--rate", type=float, default=10.0, help="requests per second")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--arrivals", choices=["fixed", "poisson", "jitter"], default="fixed",
                    help="fixed = 1/rate apart; poisson = exponential gaps; jitter = 1/rate apart +- --jitter-ms (a periodic camera whose "
                         "frames are not phase-locked to the gNB slot pattern, and that never queues behind itself)")
    ap.add_argument("--jitter-ms", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--engine-sel", type=int, default=-1, help="-1 = server policy, else engine index on the server")
    ap.add_argument("--log", default="client.csv")
    ap.add_argument("--warm", type=int, default=3, help="requests sent before the measurement (TCP window, server caches)")
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args()

    if os.path.isdir(a.images):
        files = sorted(glob.glob(os.path.join(a.images, "uhd_*_q85.jpg"))) or sorted(glob.glob(os.path.join(a.images, "*.jpg")))
    else:
        files = [a.images]
    if not files:
        print("no images under", a.images)
        return 1
    blobs = [open(f, "rb").read() for f in files]
    print("images: %d, %.2f MB mean" % (len(blobs), sum(map(len, blobs)) / len(blobs) / 1e6), flush=True)

    host, port = a.server.rsplit(":", 1)
    s = socket.create_connection((host, int(port)), timeout=a.timeout)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sel = 0xFFFFFFFF if a.engine_sel < 0 else a.engine_sel

    pending = {}          # req_id -> (t_arrival, t_send, nbytes)
    results = []
    lock = threading.Lock()
    done = threading.Event()

    def receiver():
        try:
            while not done.is_set():
                magic, req_id, status, n = HDR.unpack(recv_exact(s, HDR.size))
                body = recv_exact(s, n)
                t_r = time.perf_counter()
                if magic != MAGIC_R:
                    print("bad reply magic", flush=True)
                    break
                j = json.loads(body)
                with lock:
                    ta, ts, nb = pending.pop(req_id)
                results.append((req_id, ta, ts, t_r, nb, status, j))
        except (ConnectionError, OSError, socket.timeout) as e:
            if not done.is_set():
                print("receiver stopped: %s" % e, flush=True)

    threading.Thread(target=receiver, daemon=True).start()

    def send(req_id, blob, t_arrival):
        t_s = time.perf_counter()
        with lock:
            pending[req_id] = (t_arrival, t_s, len(blob))
        s.sendall(HDR.pack(MAGIC_Q, req_id, sel, len(blob)) + blob)

    # warm-up requests, closed loop
    for i in range(a.warm):
        send(1000000 + i, blobs[i % len(blobs)], time.perf_counter())
        t0 = time.perf_counter()
        while any(k >= 1000000 for k in list(pending)) and time.perf_counter() - t0 < a.timeout:
            time.sleep(0.001)
    results.clear()

    import random
    rng = random.Random(a.seed)
    n = int(a.rate * a.seconds)
    if a.arrivals == "fixed":
        offs = [i / a.rate for i in range(n)]
    elif a.arrivals == "poisson":
        offs, t = [], 0.0
        for i in range(n):
            offs.append(t)
            t += rng.expovariate(a.rate)
    else:
        j = min(a.jitter_ms / 1e3, 0.45 / a.rate)
        offs = sorted(max(0.0, i / a.rate + rng.uniform(-j, j)) for i in range(n))
    base = time.perf_counter() + 0.05
    unix0 = time.time_ns() - int(time.perf_counter() * 1e9)      # unix_ns = unix0 + perf_counter_ns
    print("sending %d requests at %.1f/s (%s) ..." % (n, a.rate, a.arrivals), flush=True)
    for i in range(n):
        t_arr = base + offs[i]
        while True:
            d = t_arr - time.perf_counter()
            if d <= 0:
                break
            time.sleep(d - 0.001 if d > 0.0015 else 0)
        send(i, blobs[i % len(blobs)], t_arr)
    # drain
    t0 = time.perf_counter()
    while pending and time.perf_counter() - t0 < a.timeout:
        time.sleep(0.01)
    done.set()
    try:
        s.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    s.close()

    results.sort(key=lambda r: r[0])
    with open(a.log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req_id", "t_arrival_ns", "t_send_ns", "t_reply_ns", "rtt_ms", "late_ms", "bytes", "status", "engine", "n_det",
                    "srv_net_in_ms", "srv_queue_ms", "srv_decode_ms", "srv_infer_ms", "srv_server_ms", "srv_post_ms", "t_send_unix_ns", "t_reply_unix_ns"])
        for req_id, ta, ts, tr, nb, st, j in results:
            tm = j.get("t_ms", {})
            w.writerow([req_id, int((ta - base) * 1e9), int((ts - base) * 1e9), int((tr - base) * 1e9), round((tr - ts) * 1e3, 3), round((ts - ta) * 1e3, 3), nb, st,
                        j.get("engine", ""), j.get("n_det", 0), tm.get("net_in", ""), tm.get("queue", ""), tm.get("decode", ""), tm.get("infer", ""), tm.get("server", ""),
                        tm.get("post", ""), unix0 + int(ts * 1e9), unix0 + int(tr * 1e9)])
    rtt = [(tr - ts) * 1e3 for _, ta, ts, tr, nb, st, j in results]
    e2e = [(tr - ta) * 1e3 for _, ta, ts, tr, nb, st, j in results]      # schedule -> reply (includes client-side lateness)
    srv = [j.get("t_ms", {}).get("server", 0.0) for *_, j in results]
    print("offered %d served %d lost %d | rtt ms mean %.1f p50 %.1f p99 %.1f max %.1f | schedule->reply p50 %.1f p99 %.1f | server p50 %.1f -> air+transport p50 %.1f ms" % (
        n, len(results), len(pending), sum(rtt) / max(1, len(rtt)), percentile(rtt, 50), percentile(rtt, 99), max(rtt) if rtt else 0,
        percentile(e2e, 50), percentile(e2e, 99), percentile(srv, 50), percentile(rtt, 50) - percentile(srv, 50)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
