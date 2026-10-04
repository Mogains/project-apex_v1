"""
Train YOLOv8n on the generated bullseye dataset (run make_dataset.py first).

Run:
  /Users/anmolbhatt/.pyenv/versions/3.10.13/bin/python training/train.py
Result:
  training/runs/bullseye/weights/best.pt
"""
import argparse
import os

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps", help="mps = Apple GPU, cpu = slower fallback")
    args = ap.parse_args()

    model = YOLO(os.path.join(HERE, "..", "yolov8n.pt"))   # start from the pretrained nano model
    model.train(
        data=os.path.join(HERE, "dataset", "data.yaml"),
        imgsz=640,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=2,
        patience=10,          # stop early once it stops improving
        project=os.path.join(HERE, "runs"),
        name="bullseye",
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
