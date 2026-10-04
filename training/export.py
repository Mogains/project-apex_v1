"""
Export the trained model so camera_turret.html can run it, and check it on real footage.

Run (after train.py):
  /Users/anmolbhatt/.pyenv/versions/3.10.13/bin/python training/export.py
Writes:
  models/bullseye.onnx
  models/bullseye.js     (same model, packed so the page works opened straight from disk)
Prints:
  how often it finds the page in the last 20% of each test video (frames it never trained on)
"""
import base64
import glob
import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
BEST = os.path.join(HERE, "runs", "bullseye", "weights", "best.pt")
OUT = os.path.join(HERE, "..", "models", "bullseye.onnx")
CONF = 0.45   # same as BULLSEYE_CONF in camera_turret.html


def check_videos(model):
    for video in sorted(glob.glob(os.path.join(HERE, "test_videos", "*"))):
        cap = cv2.VideoCapture(video)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        confs = []
        for i in range(int(n * 0.8), n, 2):   # held-out tail only
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                break
            r = model.predict(f, imgsz=640, conf=0.05, verbose=False)[0]
            confs.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
        c = np.array(confs)
        print(f"{os.path.basename(video)}: found in {round((c >= CONF).mean() * 100)}% of "
              f"{len(c)} held-out frames (median confidence {np.median(c):.2f})")


def main():
    model = YOLO(BEST)
    check_videos(model)
    onnx = model.export(format="onnx", imgsz=640, simplify=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    shutil.copy(onnx, OUT)
    write_js(OUT)
    print("wrote", os.path.normpath(OUT), "and", os.path.normpath(OUT[:-5] + ".js"))


def write_js(onnx_path):
    """Pack the model into a script file so camera_turret.html can load it when opened from disk."""
    with open(onnx_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    with open(onnx_path[:-5] + ".js", "w") as f:
        f.write(f'window.BULLSEYE_ONNX_B64 = "{b64}";\n')


if __name__ == "__main__":
    main()
