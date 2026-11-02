#!/usr/bin/env python3
"""slicing_stub.py (2026-11-02): stand-in for the DRL-based Network Slicing xApp. Subscribes to sm_report and prints /
logs the mean SM count YOLO got (docs/XAPP_INTERFACE.md §3).  slicing_stub.py [--host H] [--port 9810] [--period-ms 1000] [--csv F]"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xapp_client import DappAgent  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=9810)
ap.add_argument("--period-ms", type=int, default=1000); ap.add_argument("--csv"); ap.add_argument("--duration", type=float, default=0.0)
a = ap.parse_args()
w = None
if a.csv:
    f = open(a.csv, "w", newline=""); w = csv.writer(f); w.writerow(["t_ns", "window_ms", "requests", "sm_mean", "yolo_sm_time_mean", "ran_sm_time_mean", "sm_by_model"])

def on_msg(m):
    if m.get("type") != "sm_report":
        return
    print("[sm_report] t=%d window=%d ms requests=%s sm_mean=%s yolo_sm_time_mean=%s ran=%s by_model=%s" % (
        m.get("t_ns", 0) // 1000000000, m.get("window_ms"), m.get("requests"), m.get("sm_mean"), m.get("yolo_sm_time_mean"), m.get("ran_sm_time_mean"), m.get("sm_by_model")))
    if w:
        w.writerow([m.get("t_ns"), m.get("window_ms"), m.get("requests"), m.get("sm_mean"), m.get("yolo_sm_time_mean"), m.get("ran_sm_time_mean"), m.get("sm_by_model")]); f.flush()

c = DappAgent(a.host, a.port)
print("welcome:", c.hello("slicing", "slicing_stub", ["sm_report"], period_ms=a.period_ms, on_message=on_msg))
t0 = time.time()
try:
    while a.duration <= 0 or time.time() - t0 < a.duration:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
c.close()
