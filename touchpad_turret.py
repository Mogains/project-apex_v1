"""
Face tracker v3: smooth, SERVOS ONLY, fixed camera (Mac webcam).

Works with the touchpad_turret.ino sketch (X servo D8, Y servo D11).
It always sends fire = 0, so the DC motors never run.

Install once:
  /Users/anmolbhatt/.pyenv/versions/3.10.13/bin/python -m pip install ultralytics opencv-python pyserial numpy huggingface_hub

Run:
  /Users/anmolbhatt/.pyenv/versions/3.10.13/bin/python face_track_v3.py
  ... face_track_v3.py --no-servo     (detection only)
  ... face_track_v3.py --cam 1        (if the wrong camera opens)

Keys: q quit (recenters)   r pick a new face
      a / d  nudge pan   w / s  nudge tilt   (aim it right AT you, then leave them)
"""
import argparse
import glob
import os
import threading
import time

import cv2
import numpy as np
import serial
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ---------------- TUNING ----------------
FRAME_W, FRAME_H = 640, 480
IMGSZ = 320            # smaller = more frames per second = smoother (320 / 416)

SPAN_X = 60.0          # degrees of pan that match the camera's full width
SPAN_Y = 40.0          # degrees of tilt that match the camera's full height
INVERT_X = False       # flip if it pans the WRONG way
INVERT_Y = True        # flip if it tilts the WRONG way

X_MIN, X_MAX = 20, 160
Y_MIN, Y_MAX = 40, 130

# smoothing of the face position (adaptive: heavy when still, light when moving)
ALPHA_MIN = 0.15
ALPHA_MAX = 0.70
ALPHA_GAIN = 0.05      # how fast smoothing relaxes as the face moves (per degree)

# servo motion (runs 60x per second in its own thread, independent of detection speed)
SEND_HZ = 60
SERVO_TAU = 0.08       # seconds to glide ~63% to the target (lower = faster, higher = smoother)
MAX_DPS = 250.0        # speed limit, degrees per second
HYST_DEG = 0.8         # ignore sub-degree wobble so the servos don't buzz

MAX_JUMP_PX = 150
LOST_FRAMES = 20
# ----------------------------------------


def ensure_model(path):
    if os.path.exists(path):
        return path
    print("Getting face model (first run downloads it)...")
    return hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")


def find_port():
    ports = sorted(glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.wchusbserial*"))
    return ports[0] if ports else None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def adaptive(prev, new):
    a = clamp(ALPHA_MIN + ALPHA_GAIN * abs(new - prev), ALPHA_MIN, ALPHA_MAX)
    return prev + a * (new - prev)


def pick_face(faces, locked_center):
    if not faces:
        return None
    if locked_center is not None:
        best, best_d = None, MAX_JUMP_PX
        for f in faces:
            cx, cy = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2
            d = np.hypot(cx - locked_center[0], cy - locked_center[1])
            if d < best_d:
                best, best_d = f, d
        return best
    return max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))


class Mover(threading.Thread):
    """Glides the servos toward the latest target at a steady 60 Hz."""

    def __init__(self, ser):
        super().__init__(daemon=True)
        self.ser = ser
        self.lock = threading.Lock()
        self.tx, self.ty = 90.0, 90.0   # target from the tracker
        self.x, self.y = 90.0, 90.0     # eased position
        self.sent = [90, 90]
        self.running = True

    def set_target(self, x, y):
        with self.lock:
            self.tx, self.ty = x, y

    def run(self):
        last = time.time()
        while self.running:
            time.sleep(1.0 / SEND_HZ)
            now = time.time()
            dt = now - last
            last = now
            with self.lock:
                tx, ty = self.tx, self.ty

            k = 1.0 - np.exp(-dt / SERVO_TAU)
            cap = MAX_DPS * dt
            self.x += clamp((tx - self.x) * k, -cap, cap)
            self.y += clamp((ty - self.y) * k, -cap, cap)

            if abs(self.x - self.sent[0]) >= HYST_DEG:
                self.sent[0] = int(round(self.x))
            if abs(self.y - self.sent[1]) >= HYST_DEG:
                self.sent[1] = int(round(self.y))

            try:
                # always sends fire = 0, and doubles as the heartbeat
                self.ser.write(f"{self.sent[0]},{self.sent[1]},0\n".encode())
            except Exception as e:
                print("Serial error:", e)
                self.running = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbserial-110")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--model", default="yolov8n-face.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--no-servo", action="store_true")
    args = ap.parse_args()

    model = YOLO(ensure_model(args.model))

    ser, mover = None, None
    if not args.no_servo:
        port = args.port if os.path.exists(args.port) else find_port()
        if port is None:
            print("No Arduino port found. Run: ls /dev/cu.*   (or use --no-servo)")
            return
        print(f"Using port {port}")
        ser = serial.Serial(port, 115200, timeout=0.01)
        time.sleep(2)  # Uno resets when the port opens
        mover = Mover(ser)
        mover.start()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    trim_x, trim_y = 0.0, 0.0
    sm_x, sm_y = 90.0, 90.0
    locked_center = None
    lost = 0
    fps, t_prev = 0.0, time.time()

    print("Tracking. a/d/w/s = nudge aim, r = new face, q = quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed. Allow camera access for Terminal/VS Code, or try --cam 1")
                break
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            res = model(frame, imgsz=IMGSZ, conf=args.conf, verbose=False)[0]
            faces = [tuple(b) for b in res.boxes.xyxy.cpu().numpy()]
            target = pick_face(faces, locked_center)

            if target is not None:
                lost = 0
                cx, cy = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
                locked_center = (cx, cy)

                nx = cx / FRAME_W - 0.5
                ny = cy / FRAME_H - 0.5
                want_x = 90 + (1 if INVERT_X else -1) * nx * SPAN_X + trim_x
                want_y = 90 + (-1 if INVERT_Y else 1) * ny * SPAN_Y + trim_y
                want_x = clamp(want_x, X_MIN, X_MAX)
                want_y = clamp(want_y, Y_MIN, Y_MAX)

                sm_x = adaptive(sm_x, want_x)
                sm_y = adaptive(sm_y, want_y)
                if mover is not None:
                    mover.set_target(sm_x, sm_y)
            else:
                lost += 1   # face gone: hold still
                if lost > LOST_FRAMES:
                    locked_center = None

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

            for f in faces:
                cv2.rectangle(frame, (int(f[0]), int(f[1])), (int(f[2]), int(f[3])),
                              (150, 150, 150), 1)
            if target is not None:
                cv2.rectangle(frame, (int(target[0]), int(target[1])),
                              (int(target[2]), int(target[3])), (0, 255, 0), 2)
            status = "LOCKED" if target is not None else "SEARCHING"
            cv2.putText(frame, f"{status}  pan={sm_x:.0f} tilt={sm_y:.0f}  {fps:.0f} fps",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("face tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                locked_center = None
            elif key == ord("a"):
                trim_x -= 1
            elif key == ord("d"):
                trim_x += 1
            elif key == ord("w"):
                trim_y -= 1
            elif key == ord("s"):
                trim_y += 1
    finally:
        if mover is not None:
            mover.running = False
            mover.join(timeout=1.0)
        if ser is not None:
            try:
                ser.write(b"90,90,0\n")
                time.sleep(0.3)
            except Exception:
                pass
            ser.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped, servos centered.")


if __name__ == "__main__":
    main()