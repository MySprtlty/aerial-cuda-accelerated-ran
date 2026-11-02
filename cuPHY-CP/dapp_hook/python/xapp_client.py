#!/usr/bin/env python3
"""xapp_client.py (2026-11-02): client library for the dApp agent protocol (docs/XAPP_INTERFACE.md).

    from xapp_client import DappAgent
    a = DappAgent("127.0.0.1", 9810)
    a.hello(role="ra", name="my-ra", subscribe=["estimate"], every=1, on_message=print)
    r = a.run(open("img.jpg", "rb").read(), model="26m", qos={"deadline_ms": 150})   # blocks until the result
    print(r["status"], r["latency_ms"], r["detections"])

Messages from the agent (estimate / sm_report / result / welcome) are JSON dicts; the receive thread calls
`on_message(dict)` for every pushed message and `run()` waits for the result with the matching req_id.
"""
import json
import socket
import threading
import time


class DappAgent:
    def __init__(self, host="127.0.0.1", port=9810, timeout=30.0):
        self.host, self.port, self.timeout = host, port, timeout
        self.sock = socket.create_connection((host, port), timeout=10.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)
        self._lock = threading.Lock()          # serialises writes
        self._results = {}                     # req_id -> dict
        self._cv = threading.Condition()
        self._seq = int(time.time() * 1000) & 0xFFFFFF
        self.on_message = None
        self.welcome = None
        self.latest_estimate = None
        self.latest_report = None
        self._rx = threading.Thread(target=self._recv_loop, daemon=True)
        self._rx.start()

    # ---- protocol -------------------------------------------------------------------------------------------------
    def _send_line(self, obj, payload=b""):
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode() + payload
        with self._lock:
            self.sock.sendall(data)

    def hello(self, role, name=None, subscribe=(), every=1, period_ms=1000, on_message=None, wait=5.0):
        self.on_message = on_message
        self._send_line({"type": "hello", "role": role, "name": name or role, "subscribe": list(subscribe), "every": every, "period_ms": period_ms})
        t0 = time.time()
        while self.welcome is None and time.time() - t0 < wait:
            time.sleep(0.01)
        return self.welcome

    def run(self, jpeg_bytes, model="26x", qos=None, want_dets=False, req_id=None, timeout=None):
        """Send one RUN and wait for its result dict (status 'success' | 'fail' | 'timeout')."""
        if req_id is None:
            self._seq += 1
            req_id = self._seq
        msg = {"type": "run", "req_id": req_id, "model": model, "qos": qos or {}, "image": "jpeg", "bytes": len(jpeg_bytes), "want_dets": bool(want_dets)}
        t_send = time.time_ns()
        self._send_line(msg, jpeg_bytes)
        with self._cv:
            ok = self._cv.wait_for(lambda: req_id in self._results, timeout=timeout or self.timeout)
            r = self._results.pop(req_id, None) if ok else {"type": "result", "req_id": req_id, "status": "timeout", "model": model}
        r["t_send_ns"] = t_send
        r["t_result_ns"] = time.time_ns()
        r["e2e_ms"] = (r["t_result_ns"] - t_send) / 1e6
        return r

    def run_async(self, jpeg_bytes, model="26x", qos=None, want_dets=False, req_id=None):
        """Send one RUN without waiting; the result arrives through on_message (type 'result')."""
        if req_id is None:
            self._seq += 1
            req_id = self._seq
        self._send_line({"type": "run", "req_id": req_id, "model": model, "qos": qos or {}, "image": "jpeg", "bytes": len(jpeg_bytes), "want_dets": bool(want_dets)}, jpeg_bytes)
        return req_id

    def get(self, what):
        self._send_line({"type": "get", "what": what})

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    # ---- receive --------------------------------------------------------------------------------------------------
    def _recv_loop(self):
        buf = b""
        while True:
            try:
                chunk = self.sock.recv(1 << 16)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                t = msg.get("type")
                if t == "welcome":
                    self.welcome = msg
                elif t == "estimate":
                    self.latest_estimate = msg
                elif t == "sm_report":
                    self.latest_report = msg
                if t == "result":
                    with self._cv:
                        self._results[msg.get("req_id")] = msg
                        self._cv.notify_all()
                if self.on_message:
                    try:
                        self.on_message(msg)
                    except Exception as e:  # a callback error must not kill the receiver
                        print("on_message error:", e)
        with self._cv:
            self._cv.notify_all()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="quick check: hello + one run")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=9810)
    ap.add_argument("--image"); ap.add_argument("--model", default="26x"); ap.add_argument("--estimates", type=int, default=3)
    a = ap.parse_args()
    seen = []
    c = DappAgent(a.host, a.port)
    print("welcome:", c.hello("ra", "xapp_client-test", ["estimate"], every=200, on_message=lambda m: seen.append(m) if m.get("type") == "estimate" else None))
    if a.image:
        r = c.run(open(a.image, "rb").read(), model=a.model)
        print("result:", {k: r[k] for k in ("status", "model", "sm", "latency_ms", "detections", "e2e_ms") if k in r})
    time.sleep(1.0)
    print("estimates received:", len(seen), seen[:a.estimates])
    c.close()
