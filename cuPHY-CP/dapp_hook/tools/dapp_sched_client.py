#!/usr/bin/env python3
"""
Send one request to dapp_sched and print the reply. Replaces `nc -U`, which the
Aerial container does not ship.

    dapp_sched_client.py [-s /tmp/dapp_sched.sock] RUN [image.jpg | milliseconds]
    dapp_sched_client.py -n 20 -i 0.5 RUN bus.jpg      # 20 requests, 0.5 s apart

Each request is one connection: write a line, read one line back.
"""
import argparse
import socket
import sys
import time


def ask(sock_path, line, timeout=30.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(sock_path)
    s.sendall((line.strip() + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
    s.close()
    return buf.decode(errors="replace").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--sock", default="/tmp/dapp_sched.sock")
    ap.add_argument("-n", "--count", type=int, default=1)
    ap.add_argument("-i", "--interval", type=float, default=0.0, help="seconds between requests")
    ap.add_argument("words", nargs="*", default=["RUN"])
    a = ap.parse_args()
    line = " ".join(a.words)
    for k in range(a.count):
        t0 = time.time()
        try:
            reply = ask(a.sock, line)
        except OSError as e:
            print("error: %s (%s)" % (e, a.sock))
            return 1
        print("%s   (round trip %.1f ms)" % (reply, (time.time() - t0) * 1e3))
        if k + 1 < a.count and a.interval > 0:
            time.sleep(a.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
