#!/usr/bin/env python3
"""
Export a YOLO detection model to ONNX for the dapp_sched TensorRT runner and
write reference detections for the bundled sample images, so the C++ pre/post
processing can be checked against ultralytics.

Runs on the host with a CPU-only torch; the TensorRT engine is then built
inside the Aerial container, where TensorRT lives:

    python3 export_yolo.py --model yolov8n.pt --imgsz 640
    docker exec c_aerial_troy /usr/src/tensorrt/bin/trtexec \
        --onnx=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/yolov8n.onnx \
        --saveEngine=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/yolov8n_fp16.engine --fp16

Everything this script produces is ignored by git (weights, ONNX, engines,
sample images, reference JSON); only the script is tracked.
"""
import argparse
import json
import os
import pathlib
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="ultralytics weights (downloaded on first use)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)  # weights and the ONNX file land next to this script

    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    model = YOLO(a.model)
    onnx_path = model.export(format="onnx", imgsz=a.imgsz, opset=17, simplify=True, dynamic=False, half=False)
    print("onnx:", onnx_path)

    # Sample images shipped with ultralytics.
    images = []
    for name in ("bus.jpg", "zidane.jpg"):
        src = pathlib.Path(ASSETS) / name
        if src.exists():
            shutil.copy(src, out / name)
            images.append(name)

    # Reference detections through the ONNX model (onnxruntime), which uses the
    # same square letterbox as the C++ runner; the .pt path pads to a rectangle.
    refs = {}
    try:
        onnx_model = YOLO(onnx_path, task="detect")
        for name in images:
            r = onnx_model.predict(str(out / name), imgsz=a.imgsz, conf=a.conf, iou=a.iou, device="cpu", verbose=False)[0]
            refs[name] = [
                {"cls": int(c), "name": r.names[int(c)], "conf": round(float(p), 4),
                 "xyxy": [round(float(v), 1) for v in b]}
                for c, p, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)
            ]
            print("%-12s %d detections: %s" % (name, len(refs[name]),
                  ", ".join("%s:%.2f" % (d["name"], d["conf"]) for d in refs[name][:6])))
    except Exception as e:  # onnxruntime missing or similar: not fatal for the export
        print("reference detections skipped:", e)

    with open(out / "reference_detections.json", "w") as f:
        json.dump(refs, f, indent=1)
    with open(out / "class_names.json", "w") as f:
        json.dump(model.names, f)


if __name__ == "__main__":
    main()
