#!/usr/bin/env python3
"""
Export the YOLO26 detectors at a 1280x1280 input (for 4K sources: 3x downscale instead of 6x at 640),
batch 1 and 8, as <model>_1280_b<batch>.onnx next to the 640 ones, and write bus_1280.npy (the
letterboxed 1x3x1280x1280 float32 input the tenant worker copies H2D for every inference).

Runs on the host in the dapp_yolo venv (CPU torch):
    /home/troy/.venvs/dapp_yolo/bin/python export_1280.py [--models yolo26n yolo26s yolo26x] [--imgsz 1280]
    ... --batches 8 --nms-free [--max-det 300]     # end-to-end head: <model>_1280_b8_e2e.onnx, output [8, 300, 6]
Engines are then built inside the container: build_engines.sh with MODELS="yolo26n_1280 ..." (same cap loop).
"""
import argparse
import os
import pathlib
import shutil

import numpy as np


def letterbox_npy(jpg, size, out):
    from PIL import Image
    im = Image.open(jpg).convert("RGB")
    w, h = im.size
    r = size / max(w, h)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    a = np.asarray(canvas, dtype=np.float32) / 255.0          # HWC RGB
    a = np.ascontiguousarray(a.transpose(2, 0, 1))[None]      # 1x3xHxW
    np.save(out, a)
    print("%s: %s %s" % (out, a.shape, a.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolo26n", "yolo26s", "yolo26x"])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8])
    ap.add_argument("--nms-free", action="store_true", help="export the YOLO26 end-to-end (one-to-one, NMS-free) head: "
                    "output [N, max_det, 6] = xyxy, score, class; file suffix _e2e")
    ap.add_argument("--max-det", type=int, default=300, help="--nms-free: detections per image kept by the head's top-k")
    a = ap.parse_args()
    here = pathlib.Path(__file__).resolve().parent
    os.chdir(here)
    from ultralytics import YOLO
    for m in a.models:
        for b in a.batches:
            dst = here / ("%s_%d_b%d%s.onnx" % (m, a.imgsz, b, "_e2e" if a.nms_free else ""))
            if dst.exists() and dst.stat().st_size > 0:
                print("have", dst.name)
                continue
            model = YOLO("%s.pt" % m)
            kw = {}
            if a.nms_free:
                kw = {"nms": False, "max_det": a.max_det}      # ultralytics >= 8.3: nms=False = NMS-free end-to-end head
            p = model.export(format="onnx", imgsz=a.imgsz, batch=b, opset=17, simplify=True, dynamic=False, half=False, **kw)
            shutil.move(p, dst)
            print("onnx:", dst.name, dst.stat().st_size)
    npy = here / ("bus_%d.npy" % a.imgsz)
    if not npy.exists():
        letterbox_npy(here / "bus.jpg", a.imgsz, npy)
    print("EXPORT_1280_DONE")


if __name__ == "__main__":
    main()
