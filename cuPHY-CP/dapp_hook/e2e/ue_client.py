#!/usr/bin/env python3
"""
ue_client: the UE-side request generator for the December demo (H3). One process per UE (laptop behind the 5G UE / RU),
pure Python 3 stdlib, no PTP needed on the UE. Differences from e2e_client.py (the lab client):

  - one PERSISTENT TCP connection to the dispatcher (TCP_NODELAY, keepalive) that is re-opened automatically when the air
    link drops it; requests in flight on a dropped connection are logged with status 9 (lost) and the stream continues
  - request ids carry the UE id (req_id = ue_id << 24 | seq) so several UEs can be joined with the dispatcher csv (tools/join_e2e.py)
  - open-loop camera stream: --rate req/s, --arrivals fixed | poisson | jitter (default jitter), --glob default uhd_*_q60.jpg
    (~1 MB per 4K frame: 10 req/s = ~79 Mbps, just under a 97 Mbps uplink; use --rate 8 to leave headroom)
  - UE-side csv with UNIX ns stamps (t_send_unix_ns = first header byte written, t_reply_unix_ns = last reply byte read) plus
    the server's t_ms decomposition from the reply; a progress line every --progress-s seconds

    python3 ue_client.py --server 10.45.0.1:9007 --images ./img4k --ue-id 1 --rate 8 --seconds 600 --log ue1.csv
"""
import argparse
import csv
import glob
import json
import os
import random
import socket
import struct
import sys
import threading
import time

HDR = struct.Struct("<4sIII")
MAGIC_Q, MAGIC_R = b"DAPQ", b"DAPR"
SEL_POLICY = 0xFFFFFFFF
LOST = 9


def recv_exact(sock, n):
    buf = bytearray(n); view = memoryview(buf); got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("server closed")
        got += k
    return bytes(buf)


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v); return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


class Link:
    """Persistent connection with reconnect; receiver thread completes pending requests."""

    def __init__(self, host, port, timeout, on_reply, on_lost):
        self.host, self.port, self.timeout = host, int(port), timeout
        self.on_reply, self.on_lost = on_reply, on_lost
        self.sock = None
        self.lock = threading.Lock()
        self.pending = {}
        self.stop = threading.Event()
        self.reconnects = 0
        self.connect()

    def connect(self):
        while not self.stop.is_set():
            try:
                s = socket.create_connection((self.host, self.port), timeout=self.timeout)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self.sock = s
                threading.Thread(target=self.receiver, args=(s,), daemon=True).start()
                return
            except OSError as e:
                print("%s connect failed: %s, retry in 1 s" % (time.strftime("%H:%M:%S"), e), flush=True)
                time.sleep(1.0)

    def receiver(self, s):
        try:
            while not self.stop.is_set():
                magic, req_id, status, n = HDR.unpack(recv_exact(s, HDR.size))
                body = recv_exact(s, n)
                t_r = time.perf_counter()
                if magic != MAGIC_R:
                    raise ConnectionError("bad reply magic")
                with self.lock:
                    p = self.pending.pop(req_id, None)
                if p is not None:
                    self.on_reply(req_id, p, t_r, status, json.loads(body))
        except (ConnectionError, OSError, socket.timeout, ValueError) as e:
            if not self.stop.is_set() and s is self.sock:
                print("%s link lost: %s" % (time.strftime("%H:%M:%S"), e), flush=True)
                with self.lock:
                    lost = list(self.pending.items()); self.pending.clear()
                for req_id, p in lost:
                    self.on_lost(req_id, p)
                try:
                    s.close()
                except OSError:
                    pass
                self.reconnects += 1
                self.connect()

    def send(self, req_id, blob, t_arrival, sel):
        s = self.sock
        t_s = time.perf_counter()
        with self.lock:
            self.pending[req_id] = (t_arrival, t_s, len(blob))
        try:
            s.sendall(HDR.pack(MAGIC_Q, req_id, sel, len(blob)) + blob)
        except OSError as e:
            with self.lock:
                p = self.pending.pop(req_id, None)
            if p is not None:
                self.on_lost(req_id, p)
            print("%s send failed: %s" % (time.strftime("%H:%M:%S"), e), flush=True)

    def close(self):
        self.stop.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR); self.sock.close()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="host:port of e2e/dispatcher.py (via the UE's 5G data path)")
    ap.add_argument("--images", required=True); ap.add_argument("--glob", default="uhd_*_q60.jpg")
    ap.add_argument("--ue-id", type=int, default=1, help="1..255, written into the request ids (req_id = ue_id << 24 | seq)")
    ap.add_argument("--rate", type=float, default=8.0); ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--arrivals", choices=["fixed", "poisson", "jitter"], default="jitter"); ap.add_argument("--jitter-ms", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--engine-sel", type=int, default=-1, help="-1 = dispatcher policy")
    ap.add_argument("--log", default="ue.csv"); ap.add_argument("--warm", type=int, default=2); ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--progress-s", type=float, default=10.0)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.images, a.glob))) if os.path.isdir(a.images) else [a.images]
    if not files:
        print("no images under", a.images); return 1
    blobs = [open(f, "rb").read() for f in files]
    mean_mb = sum(map(len, blobs)) / len(blobs) / 1e6
    print("UE %d: %d images, %.2f MB mean -> %.1f Mbps at %.1f req/s; server %s" % (a.ue_id, len(blobs), mean_mb, mean_mb * 8 * a.rate, a.rate, a.server), flush=True)
    host, port = a.server.rsplit(":", 1)
    sel = SEL_POLICY if a.engine_sel < 0 else a.engine_sel
    results, lock = [], threading.Lock()
    unix0 = time.time_ns() - time.perf_counter_ns()          # unix_ns = unix0 + perf_counter_ns

    def on_reply(req_id, p, t_r, status, j):
        with lock:
            results.append((req_id, p[0], p[1], t_r, p[2], status, j))

    def on_lost(req_id, p):
        with lock:
            results.append((req_id, p[0], p[1], time.perf_counter(), p[2], LOST, {}))

    link = Link(host, port, a.timeout, on_reply, on_lost)
    base_id = (a.ue_id & 0xFF) << 24
    for i in range(a.warm):                                     # warm-up, closed loop, not logged
        link.send(base_id | (0xFFFFFF - i), blobs[i % len(blobs)], time.perf_counter(), sel)
        t0 = time.perf_counter()
        while link.pending and time.perf_counter() - t0 < a.timeout:
            time.sleep(0.001)
    with lock:
        results.clear()
    rng = random.Random(a.seed * 1000 + a.ue_id)
    n = int(a.rate * a.seconds)
    if a.arrivals == "fixed":
        offs = [i / a.rate for i in range(n)]
    elif a.arrivals == "poisson":
        offs, t = [], 0.0
        for i in range(n):
            offs.append(t); t += rng.expovariate(a.rate)
    else:
        j = min(a.jitter_ms / 1e3, 0.45 / a.rate)
        offs = sorted(max(0.0, i / a.rate + rng.uniform(-j, j)) for i in range(n))
    base = time.perf_counter() + 0.05
    next_prog = base + a.progress_s
    print("sending %d requests at %.1f/s (%s) for %.0f s ..." % (n, a.rate, a.arrivals, a.seconds), flush=True)
    for i in range(n):
        t_arr = base + offs[i]
        while True:
            d = t_arr - time.perf_counter()
            if d <= 0:
                break
            time.sleep(d - 0.001 if d > 0.0015 else 0)
        link.send(base_id | i, blobs[i % len(blobs)], t_arr, sel)
        if time.perf_counter() >= next_prog:
            next_prog += a.progress_s
            with lock:
                recent = [r for r in results if r[3] >= time.perf_counter() - a.progress_s]
            ok = [(r[3] - r[2]) * 1e3 for r in recent if r[5] == 0]
            print("%s +%4.0f s: sent %d, replies %d (lost %d) in the last %.0f s, rtt p50 %.0f p99 %.0f ms, in flight %d, reconnects %d" % (
                time.strftime("%H:%M:%S"), time.perf_counter() - base, i + 1, len(ok), sum(1 for r in recent if r[5] == LOST), a.progress_s,
                pct(ok, 50), pct(ok, 99), len(link.pending), link.reconnects), flush=True)
    t0 = time.perf_counter()
    while link.pending and time.perf_counter() - t0 < a.timeout:
        time.sleep(0.01)
    link.close()
    with lock:
        rows = sorted(results, key=lambda r: r[0])
    with open(a.log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req_id", "ue_id", "seq", "t_arrival_unix_ns", "t_send_unix_ns", "t_reply_unix_ns", "rtt_ms", "late_ms", "bytes", "status", "engine", "n_det",
                    "srv_net_in_ms", "srv_queue_ms", "srv_decode_ms", "srv_infer_ms", "srv_post_ms", "srv_server_ms", "reason"])
        for req_id, ta, ts, tr, nb, st, j in rows:
            tm = j.get("t_ms", {})
            w.writerow([req_id, req_id >> 24, req_id & 0xFFFFFF, unix0 + int(ta * 1e9), unix0 + int(ts * 1e9), unix0 + int(tr * 1e9), round((tr - ts) * 1e3, 3),
                        round((ts - ta) * 1e3, 3), nb, st, j.get("engine", ""), j.get("n_det", 0), tm.get("net_in", ""), tm.get("queue", ""), tm.get("decode", ""),
                        tm.get("infer", ""), tm.get("post", ""), tm.get("server", ""), j.get("reason", "")])
    ok = [r for r in rows if r[5] == 0]
    rtt = [(r[3] - r[2]) * 1e3 for r in ok]; srv = [r[6].get("t_ms", {}).get("server", 0.0) for r in ok]
    e2e = [(r[3] - r[1]) * 1e3 for r in ok]
    models = {}
    for r in ok:
        models[r[6].get("engine", "?")] = models.get(r[6].get("engine", "?"), 0) + 1
    print("UE %d: offered %d, served %d, lost %d, errors %d, reconnects %d | rtt ms mean %.1f p50 %.1f p99 %.1f max %.1f | schedule->reply p50 %.1f p99 %.1f | "
          "server p50 %.1f -> air + transport p50 %.1f ms | models %s | csv %s" % (
              a.ue_id, n, len(ok), sum(1 for r in rows if r[5] == LOST), sum(1 for r in rows if r[5] not in (0, LOST)), link.reconnects,
              sum(rtt) / max(1, len(rtt)), pct(rtt, 50), pct(rtt, 99), max(rtt) if rtt else 0, pct(e2e, 50), pct(e2e, 99), pct(srv, 50), pct(rtt, 50) - pct(srv, 50),
              " ".join("%s %.0f%%" % (m, 100.0 * k / len(ok)) for m, k in sorted(models.items())) if ok else "-", a.log), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
