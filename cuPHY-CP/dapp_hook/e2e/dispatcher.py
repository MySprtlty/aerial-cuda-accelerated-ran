#!/usr/bin/env python3
"""
dispatcher: the queue / model-selection / record stage between the clients and dapp_sched.

    [client] --DAPQ/TCP--> [dispatcher: queue, choose model, log] --RUN model=M <path> (unix socket)--> [dapp_sched]

The front end is the one of infer_server.py (same DAPQ/DAPR protocol, one thread per connection, one FIFO
queue, the same reply JSON), so e2e_client.py works unchanged. There is no TensorRT here: a request's JPEG is
written to /dev/shm/dapq_<pid>_<n>.jpg as soon as it is complete (that overlaps the inference in progress),
the worker dequeues it, picks the model, sends "RUN model=<m> <path>" to dapp_sched over its unix socket, waits
for the reply (dapp_sched runs one request at a time; --inflight N workers queue at its socket), answers the
client and removes the file. Every request leaves one csv row with CLOCK_REALTIME ns stamps of every hand-over.

Roles: the estimator (estimator/live.py) or this process (manual / timetable) says per model what is SAFE
(cap_pct, gate_slots) in the control block (ABI 2); this process decides WHICH model runs (policy.yaml,
choose_model(): rule fixed | target, replaceable).

    python3 dispatcher.py --sched-sock /dev/shm/dapp_sched.sock --port 9000 --log disp.csv \\
        --policy-source manual --manual "26x=100/40,26s=100/40,26n=100/40"
    python3 dispatcher.py ... --policy-source timetable --timetable tt.csv      # t_s,cap_26x,cap_26s,cap_26n,gate
    python3 dispatcher.py ... --policy-source live                              # estimator writes the block

manual / timetable: the dispatcher itself writes the control block every --ctrl-period-ms (5) with the wall
clock, so dapp_sched --ctrl sees the same values (one code path in dapp_sched). live: it only reads the block.

Rules (policy.yaml `rule`):
  fixed   cap_26x >= 100 -> 26x; else 26s while its share of the last window_s is below ratio_26s_max, else 26x.
  target  "the most accurate model whose predicted E2E fits target_e2e_ms": per request and model m with
          cap_m > 0, pred_e2e_m = queue_ahead x mean service time of the last service_window requests
          + profiles/yolo26<m>_b8.yaml dur_ms[8][floor key of cap_m] + overhead_ms; the first of 26x, 26s, 26n
          with pred <= target, else the candidate with the smallest pred. queue_ahead = requests ahead of this
          one when it arrived (waiting + in service); service time = t_reply - t_dequeue of finished requests.
  both    cap_26x == 0 or 26x's gate closed > gate_closed_ms -> 26n (26s once 26n's share >= ratio_26n_max);
          safety caps on the shares of the last window_s: 26s <= ratio_26s_max (else 26x), 26n <= ratio_26n_max
          (else 26s).

Per request csv (ns, CLOCK_REALTIME): req_id, t_arrival (first header byte), t_enqueue, t_dequeue, t_shm_written,
t_sent_to_sched, t_sched_recv, t_gate_pass, t_start, t_end (dapp_sched's stamps), t_reply, model, cap_26x,
cap_26s, cap_26n, gate_slots (of the chosen model), chosen_reason, queue_depth (ahead at enqueue), n_det, n_hold,
status, pred_e2e_26x/26s/26n (ms, target rule; blank when not a candidate). tools/req_summary.py aggregates it.
"""
import argparse
import csv
import json
import os
import queue
import signal
import socket
import struct
import sys
import threading
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "estimator"))
from dapp_est import control_block as CB      # noqa: E402

try:
    import yaml
except ImportError:                           # policy.yaml is optional: defaults below
    yaml = None

HDR = struct.Struct("<4sIII")
MAGIC_Q, MAGIC_R = b"DAPQ", b"DAPR"
SEL_POLICY = 0xFFFFFFFF
CSV_FIELDS = ["req_id", "t_arrival", "t_enqueue", "t_dequeue", "t_shm_written", "t_sent_to_sched", "t_sched_recv",
              "t_gate_pass", "t_start", "t_end", "t_reply", "model", "cap_26x", "cap_26s", "cap_26n", "gate_slots",
              "chosen_reason", "queue_depth", "n_det", "n_hold", "status", "pred_e2e_26x", "pred_e2e_26s", "pred_e2e_26n"]

DEFAULT_POLICY = {
    "rule": "fixed",                      # fixed | target
    "models": ["26x", "26s", "26n"],      # accuracy order = the --engine-sel index a client may force
    "window_s": 60.0,                     # W: share of a model among the requests of the last W seconds
    "ratio_26s_max": 0.6,                 # 26s is chosen while its share is below this
    "ratio_26n_max": 0.1,                 # 26n likewise (else 26s)
    "gate_closed_ms": 100.0,              # E: 26x's gate closed for longer than this = fall back to 26n
    "hold_retry_ms": 2000.0,              # keep re-sending RUN on HOLD for this long, then give up (status 2)
    "target_e2e_ms": 67.0,                # target rule: predicted E2E budget
    "overhead_ms": 8.0,                   # target rule: decode + post-processing + dispatcher/dapp_sched hand-overs
    "service_window": 20,                 # target rule: mean service time over the last N finished requests
    "profiles_dir": os.path.join(HERE, "..", "estimator", "profiles"),
    "profile_batch": 8,
}


def now_ns():
    return time.time_ns()


def load_profiles(cfg, models):
    """{model: {cap: dur_ms}} from profiles/yolo26<m>_b8.yaml (PyYAML)."""
    out = {}
    if yaml is None:
        return out
    for m in models:
        p = os.path.join(cfg["profiles_dir"], "yolo%s_b%d.yaml" % (m, cfg["profile_batch"]))
        if not os.path.exists(p):
            continue
        with open(p) as f:
            y = yaml.safe_load(f)
        d = y.get("dur_ms", {}).get(cfg["profile_batch"]) or y.get("dur_ms", {}).get(str(cfg["profile_batch"]))
        if d:
            out[m] = {int(k): float(v) for k, v in d.items()}
    return out


def dur_at_cap(table, cap):
    """dur_ms at the largest profile cap key <= cap (the smallest key when cap is below all keys)."""
    ks = sorted(table)
    below = [k for k in ks if k <= cap]
    return table[below[-1]] if below else table[ks[0]]


# ---------------------------------------------------------------------------
# policy: which model runs. Replace choose_model() to try another rule; everything else stays.
# ---------------------------------------------------------------------------
class History:
    def __init__(self, window_s, service_window):
        self.window_ns = int(window_s * 1e9)
        self.dq = deque()                  # (t_ns, model)
        self.gate_open_since = {}          # model -> t_ns of the last time its gate was seen open
        self.service = deque(maxlen=max(1, int(service_window)))   # ms, finished requests

    def add(self, t, model):
        self.dq.append((t, model))
        self.trim(t)

    def trim(self, t):
        lo = t - self.window_ns
        while self.dq and self.dq[0][0] < lo:
            self.dq.popleft()

    def ratio(self, model, t):
        self.trim(t)
        if not self.dq:
            return 0.0
        return sum(1 for _, m in self.dq if m == model) / float(len(self.dq))

    def note_gate(self, model, open_, t):
        if open_ or model not in self.gate_open_since:
            self.gate_open_since[model] = t if open_ else self.gate_open_since.get(model, t)

    def gate_closed_ms(self, model, t):
        return (t - self.gate_open_since.get(model, t)) / 1e6

    def add_service(self, ms):
        self.service.append(ms)

    def mean_service(self, default_ms):
        return sum(self.service) / len(self.service) if self.service else default_ms


def choose_model(entries, hist, t, cfg, profiles=None, queue_ahead=0):
    """entries: {model: (cap_pct, gate_slots)}; returns (model, reason, {model: pred_e2e_ms})."""
    preds = {}
    cx, gx = entries.get("26x", (0, 0))
    hist.note_gate("26x", gx > 0, t)
    closed_ms = hist.gate_closed_ms("26x", t)
    # common clause 1: no 26x at all -> the small model
    if cx == 0 or closed_ms > cfg["gate_closed_ms"]:
        why = "cap_26x=0" if cx == 0 else "gate_26x closed %.0f ms" % closed_ms
        if hist.ratio("26n", t) < cfg["ratio_26n_max"]:
            return "26n", why + " -> 26n", preds
        return "26s", why + ", 26n share >= %.2f -> 26s" % cfg["ratio_26n_max"], preds
    if cfg.get("rule", "fixed") == "target":
        # predicted E2E per candidate model
        svc = hist.mean_service(cfg["overhead_ms"] + 30.0)
        wait = queue_ahead * svc
        for m in cfg["models"]:
            cap = entries.get(m, (0, 0))[0]
            if cap <= 0 or m not in (profiles or {}):
                continue
            preds[m] = wait + dur_at_cap(profiles[m], cap) + cfg["overhead_ms"]
        if not preds:
            return "26n", "target: no candidate with cap > 0 and a profile -> 26n", preds
        target = cfg["target_e2e_ms"]
        chosen = None
        for m in cfg["models"]:                # accuracy order
            if m in preds and preds[m] <= target:
                chosen = m
                break
        if chosen is None:
            chosen = min(preds, key=preds.get)
            why = "target: none <= %.0f ms, min pred %s %.1f" % (target, chosen, preds[chosen])
        else:
            why = "target: %s pred %.1f <= %.0f ms (ahead %d x %.1f)" % (chosen, preds[chosen], target, queue_ahead, svc)
    else:
        if cx >= 100:
            return "26x", "cap_26x=100", preds
        r = hist.ratio("26s", t)
        if r < cfg["ratio_26s_max"]:
            return "26s", "cap_26x=%d, 26s share %.2f < %.2f" % (cx, r, cfg["ratio_26s_max"]), preds
        return "26x", "cap_26x=%d, 26s share %.2f >= %.2f" % (cx, r, cfg["ratio_26s_max"]), preds
    # common clause 2: share safety caps
    if chosen == "26s" and hist.ratio("26s", t) >= cfg["ratio_26s_max"]:
        chosen, why = "26x", why + "; 26s share >= %.2f -> 26x" % cfg["ratio_26s_max"]
    elif chosen == "26n" and hist.ratio("26n", t) >= cfg["ratio_26n_max"]:
        chosen, why = "26s", why + "; 26n share >= %.2f -> 26s" % cfg["ratio_26n_max"]
    return chosen, why, preds


# ---------------------------------------------------------------------------
# policy sources: where the per-model (cap, gate) come from
# ---------------------------------------------------------------------------
class ManualSource:
    """Fixed values; this process writes them to the control block."""
    def __init__(self, spec, models):
        self.values = {m: (100, 40) for m in models}
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            name, v = item.split("=")
            cap, gate = v.split("/") if "/" in v else (v, "40")
            self.values[name.strip()] = (int(cap), int(gate))

    def entries(self, t):
        return dict(self.values)


class TimetableSource:
    """Step function from a csv: t_s, cap_26x, cap_26s, cap_26n, gate  (t_s relative to the dispatcher start)."""
    def __init__(self, path, models):
        self.rows = []
        with open(path) as f:
            for r in csv.DictReader(f):
                self.rows.append((float(r["t_s"]), {"26x": int(r["cap_26x"]), "26s": int(r["cap_26s"]), "26n": int(r["cap_26n"])}, int(r["gate"])))
        self.rows.sort(key=lambda x: x[0])
        if not self.rows:
            raise ValueError("empty timetable")
        self.t0 = now_ns()
        self.models = models

    def entries(self, t):
        el = (t - self.t0) / 1e9
        cur = self.rows[0]
        for r in self.rows:
            if r[0] <= el:
                cur = r
        caps, gate = cur[1], cur[2]
        return {m: (caps.get(m, 0), gate) for m in self.models}


class LiveSource:
    """The estimator writes the block; read it."""
    def __init__(self, name, models):
        self.name = name
        self.models = models
        self.cb = None

    def entries(self, t):
        if self.cb is None:
            try:
                self.cb = CB.ControlBlock(self.name, create=False)
            except (OSError, ValueError):
                return {m: (0, 0) for m in self.models}
        d = self.cb.read()
        if not d:
            return {m: (0, 0) for m in self.models}
        got = {m["name"]: (m["cap_pct"], m["gate_slots"]) for m in d["models"]}
        return {m: got.get(m, (0, 0)) for m in self.models}


class CtrlWriter(threading.Thread):
    """manual / timetable: publish the source's values in the control block every period (wall-clock stamp)."""
    def __init__(self, name, source, models, period_ms):
        super().__init__(daemon=True)
        self.cb = CB.ControlBlock(name)
        self.source, self.models, self.period = source, models, period_ms / 1e3
        self.n = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            t = now_ns()
            e = self.source.entries(t)
            self.n += 1
            self.cb.write(0, 0, 0, t, self.n, [(m, e[m][0], e[m][1]) for m in self.models])
            time.sleep(self.period)


# ---------------------------------------------------------------------------
def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed")
        got += k
    return bytes(buf)


def parse_reply(line):
    """'OK k=v k=v ...' / 'HOLD ...' / 'ERR ...' -> (verb, {k: v})"""
    parts = line.strip().split()
    verb = parts[0] if parts else "ERR"
    kv = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k] = v
    return verb, kv


def ask_sched(sock_path, line, timeout):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(sock_path)
    s.sendall((line + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
    s.close()
    return buf.decode(errors="replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sched-sock", default="/dev/shm/dapp_sched.sock")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--log", default="")
    ap.add_argument("--policy", default=os.path.join(HERE, "policy.yaml"))
    ap.add_argument("--policy-source", choices=["manual", "timetable", "live"], default="manual")
    ap.add_argument("--manual", default="26x=100/40,26s=100/40,26n=100/40", help="model=cap/gate,...")
    ap.add_argument("--timetable", default="")
    ap.add_argument("--ctrl", default=CB.DEFAULT_NAME)
    ap.add_argument("--ctrl-period-ms", type=float, default=5.0)
    ap.add_argument("--inflight", type=int, default=1, help="worker threads sending to dapp_sched at once")
    ap.add_argument("--shm-dir", default="/dev/shm")
    ap.add_argument("--sched-timeout", type=float, default=30.0)
    ap.add_argument("--rule", choices=["fixed", "target"], default=None, help="override policy.yaml")
    ap.add_argument("--target-e2e-ms", type=float, default=None)
    ap.add_argument("--overhead-ms", type=float, default=None)
    ap.add_argument("--ratio-26s-max", type=float, default=None)
    ap.add_argument("--ratio-26n-max", type=float, default=None)
    ap.add_argument("--window-s", type=float, default=None)
    ap.add_argument("--gate-closed-ms", type=float, default=None)
    a = ap.parse_args()

    cfg = dict(DEFAULT_POLICY)
    if yaml is not None and os.path.exists(a.policy):
        with open(a.policy) as f:
            cfg.update(yaml.safe_load(f) or {})
    for k, v in (("rule", a.rule), ("target_e2e_ms", a.target_e2e_ms), ("overhead_ms", a.overhead_ms),
                 ("ratio_26s_max", a.ratio_26s_max), ("ratio_26n_max", a.ratio_26n_max), ("window_s", a.window_s),
                 ("gate_closed_ms", a.gate_closed_ms)):
        if v is not None:
            cfg[k] = v
    models = list(cfg["models"])
    profiles = load_profiles(cfg, models) if cfg["rule"] == "target" else {}
    if cfg["rule"] == "target" and len(profiles) < len(models):
        print("warning: target rule without a profile for %s (dur_ms from %s)" % (
            [m for m in models if m not in profiles], cfg["profiles_dir"]), flush=True)

    if a.policy_source == "manual":
        source = ManualSource(a.manual, models)
    elif a.policy_source == "timetable":
        if not a.timetable:
            ap.error("--policy-source timetable needs --timetable")
        source = TimetableSource(a.timetable, models)
    else:
        source = LiveSource(a.ctrl, models)
    writer = None
    if a.policy_source != "live":
        writer = CtrlWriter(a.ctrl, source, models, a.ctrl_period_ms)
        writer.start()

    hist = History(cfg["window_s"], cfg["service_window"])
    hist_lock = threading.Lock()
    q = queue.Queue()
    stop = threading.Event()
    log = open(a.log, "w", newline="") if a.log else None
    logw = csv.DictWriter(log, fieldnames=CSV_FIELDS) if log else None
    if logw:
        logw.writeheader()
    log_lock = threading.Lock()
    seq = [0]
    seq_lock = threading.Lock()
    pid = os.getpid()
    stats = {"served": 0, "held": 0, "errors": 0, "max_depth": 0, "busy": 0}

    def shm_path():
        with seq_lock:
            seq[0] += 1
            return os.path.join(a.shm_dir, "dapq_%d_%d.jpg" % (pid, seq[0]))

    def worker():
        while not stop.is_set():
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            conn, req_id, sel, path, t_arrival, t_enqueue, t_shm, ahead = item
            t_deq = now_ns()
            with hist_lock:
                stats["busy"] += 1
            ent = source.entries(t_deq)
            with hist_lock:
                if sel != SEL_POLICY and 0 <= sel < len(models):
                    model, reason, preds = models[sel], "client sel %d" % sel, {}
                else:
                    model, reason, preds = choose_model(ent, hist, t_deq, cfg, profiles, ahead)
                hist.add(t_deq, model)
            cap, gate = ent.get(model, (0, 0))
            # RUN, re-sent on HOLD (gate closed / stale block) for up to hold_retry_ms
            n_hold = 0
            verb, kv = "ERR", {}
            t_sent = now_ns()
            t_limit = t_sent + int(cfg["hold_retry_ms"] * 1e6)
            while True:
                try:
                    verb, kv = parse_reply(ask_sched(a.sched_sock, "RUN model=%s %s" % (model, path), a.sched_timeout))
                except OSError as e:
                    verb, kv = "ERR", {"err": str(e).replace(" ", "_")}
                    break
                if verb != "HOLD":
                    break
                n_hold += 1
                if now_ns() > t_limit:
                    break
                time.sleep(0.002)
            try:
                os.unlink(path)
            except OSError:
                pass
            status = 0 if verb == "OK" else (2 if verb == "HOLD" else 1)
            n_det = int(kv.get("dets", 0)) if verb == "OK" else 0
            t_end = int(kv.get("t_end_ns", 0))
            t_start = int(kv.get("t_start_ns", 0))
            dec_ms = float(kv.get("dec", 0.0))
            gpu_ms = float(kv.get("ms", 0.0)) if verb == "OK" else 0.0
            t_reply = now_ns()
            tm = {"net_in": round((t_enqueue - t_arrival) / 1e6, 3), "queue": round((t_deq - t_enqueue) / 1e6, 3),
                  "decode": round(dec_ms, 3), "infer": round(gpu_ms, 3),
                  "post": round(max(0.0, (t_end - t_start) / 1e6 - dec_ms - gpu_ms), 3) if t_end and t_start else 0.0,
                  "server": round((t_reply - t_arrival) / 1e6, 3), "sched": round((t_reply - t_sent) / 1e6, 3)}
            body = json.dumps({"req_id": req_id, "status": status, "engine": model, "n_det": n_det, "dets": [],
                               "top": kv.get("top", ""), "img": [0, 0], "t_ms": tm, "reason": reason,
                               "pred_e2e_ms": {m: round(v, 1) for m, v in preds.items()}}).encode()
            try:
                conn.sendall(HDR.pack(MAGIC_R, req_id, status, len(body)) + body)
            except OSError:
                pass
            t_reply = now_ns()
            with hist_lock:
                stats["busy"] -= 1
                hist.add_service((t_reply - t_deq) / 1e6)
            stats["served"] += 1
            stats["held"] += n_hold
            if status:
                stats["errors"] += 1
            if logw:
                row = {"req_id": req_id, "t_arrival": t_arrival, "t_enqueue": t_enqueue, "t_dequeue": t_deq,
                       "t_shm_written": t_shm, "t_sent_to_sched": t_sent, "t_sched_recv": int(kv.get("t_recv_ns", 0)),
                       "t_gate_pass": int(kv.get("t_gate_pass_ns", 0)), "t_start": t_start, "t_end": t_end, "t_reply": t_reply,
                       "model": model, "cap_26x": ent.get("26x", (0, 0))[0], "cap_26s": ent.get("26s", (0, 0))[0],
                       "cap_26n": ent.get("26n", (0, 0))[0], "gate_slots": gate, "chosen_reason": reason, "queue_depth": ahead,
                       "n_det": n_det, "n_hold": n_hold, "status": status,
                       "pred_e2e_26x": "%.1f" % preds["26x"] if "26x" in preds else "",
                       "pred_e2e_26s": "%.1f" % preds["26s"] if "26s" in preds else "",
                       "pred_e2e_26n": "%.1f" % preds["26n"] if "26n" in preds else ""}
                with log_lock:
                    logw.writerow(row)
                    log.flush()

    def serve(conn, peer):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while not stop.is_set():
                first = conn.recv(HDR.size)
                if not first:
                    break
                t_arrival = now_ns()
                hdr = first + (recv_exact(conn, HDR.size - len(first)) if len(first) < HDR.size else b"")
                magic, req_id, sel, n = HDR.unpack(hdr)
                if magic != MAGIC_Q:
                    print("bad magic from %s, closing" % peer, flush=True)
                    break
                data = recv_exact(conn, n)
                t_enqueue = now_ns()
                path = shm_path()
                with open(path, "wb") as f:          # overlaps the inference in progress
                    f.write(data)
                t_shm = now_ns()
                ahead = q.qsize() + stats["busy"]    # requests ahead of this one: waiting + in service
                stats["max_depth"] = max(stats["max_depth"], ahead + 1)
                q.put((conn, req_id, sel, path, t_arrival, t_enqueue, t_shm, ahead))
        except (ConnectionError, OSError) as e:
            print("connection %s closed: %s" % (peer, e), flush=True)
        finally:
            conn.close()

    for _ in range(max(1, a.inflight)):
        threading.Thread(target=worker, daemon=True).start()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.bind, a.port))
    srv.listen(8)

    def _sig(*_):                      # SIGINT / SIGTERM: leave the accept loop, flush the csv, exit
        stop.set()
        try:
            srv.close()
        except OSError:
            pass
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    print("dispatcher: listening on %s:%d -> %s | policy %s rule %s (%s) models %s | inflight %d | log %s" % (
        a.bind, a.port, a.sched_sock, a.policy_source, cfg["rule"],
        {k: cfg[k] for k in ("window_s", "ratio_26s_max", "ratio_26n_max", "gate_closed_ms", "target_e2e_ms", "overhead_ms")},
        models, a.inflight, a.log or "-"), flush=True)
    try:
        while not stop.is_set():
            try:
                conn, addr = srv.accept()
            except OSError:
                break
            peer = "%s:%d" % addr
            print("client %s" % peer, flush=True)
            threading.Thread(target=serve, args=(conn, peer), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if writer:
            writer.stop.set()
        if log:
            log.close()
        print("dispatcher: served %d, HOLD replies %d, errors %d, max queue depth %d" % (
            stats["served"], stats["held"], stats["errors"], stats["max_depth"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
