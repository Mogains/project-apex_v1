"""
Trackpad turret controller.

Move the pointer in the window  -> turret aims (X servo D8, Y servo D10)
Click                           -> DC motors (D5) spin
Hold the click                  -> motors keep spinning until you let go
q                               -> quit (servos recenter, motors off)

Install once:
  /Users/anmolbhatt/.pyenv/versions/3.10.13/bin/python -m pip install opencv-python pyserial numpy

Run:
  /Users/anmolbhatt/.pyenv/versions/3.10.13/bin/python touchpad_turret.py
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import serial

W, H = 600, 400

X_MIN, X_MAX = 20, 160
Y_MIN, Y_MAX = 40, 130
INVERT_X = True    # flip if the turret pans the WRONG way
INVERT_Y = False   # flip if it tilts the WRONG way
SEND_HZ = 30

state = {"x": W // 2, "y": H // 2, "fire": False}


def find_port():
    ports = sorted(glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.wchusbserial*"))
    return ports[0] if ports else None


def on_mouse(event, x, y, flags, param):
    state["x"] = max(0, min(W - 1, x))
    state["y"] = max(0, min(H - 1, y))
    if event == cv2.EVENT_LBUTTONDOWN:
        state["fire"] = True
    elif event == cv2.EVENT_LBUTTONUP:
        state["fire"] = False
    elif event == cv2.EVENT_MOUSEMOVE:
        state["fire"] = bool(flags & cv2.EVENT_FLAG_LBUTTON)


def angles():
    fx = state["x"] / (W - 1)
    fy = state["y"] / (H - 1)
    # pointer right -> lower pan angle, pointer down -> lower tilt angle (matches the face tracker)
    ax = (X_MIN + fx * (X_MAX - X_MIN)) if INVERT_X else (X_MAX - fx * (X_MAX - X_MIN))
    ay = (Y_MIN + fy * (Y_MAX - Y_MIN)) if INVERT_Y else (Y_MAX - fy * (Y_MAX - Y_MIN))
    return int(round(ax)), int(round(ay))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbserial-110")
    ap.add_argument("--no-serial", action="store_true", help="just test the window")
    args = ap.parse_args()

    ser = None
    if not args.no_serial:
        port = args.port if os.path.exists(args.port) else find_port()
        if port is None:
            print("No Arduino port found. Run: ls /dev/cu.*   (or use --no-serial)")
            return
        print(f"Using port {port}")
        ser = serial.Serial(port, 115200, timeout=0.01)
        time.sleep(2)  # Uno resets when the port opens

    cv2.namedWindow("turret")
    cv2.setMouseCallback("turret", on_mouse)
    print("Move the pointer to aim. Click / hold to run the motors. q = quit.")

    last_send = 0.0
    try:
        while True:
            ax, ay = angles()
            fire = state["fire"]

            now = time.time()
            if ser is not None and now - last_send >= 1.0 / SEND_HZ:
                ser.write(f"{ax},{ay},{1 if fire else 0}\n".encode())
                last_send = now

            img = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.line(img, (state["x"], 0), (state["x"], H), (70, 70, 70), 1)
            cv2.line(img, (0, state["y"]), (W, state["y"]), (70, 70, 70), 1)
            color = (0, 0, 255) if fire else (0, 255, 0)
            cv2.circle(img, (state["x"], state["y"]), 14, color, 2)
            if fire:
                cv2.rectangle(img, (0, 0), (W - 1, H - 1), (0, 0, 255), 6)
            label = "MOTORS ON" if fire else "motors off"
            cv2.putText(img, f"pan={ax} tilt={ay}  {label}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("turret", img)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    finally:
        if ser is not None:
            ser.write(b"90,90,0\n")
            time.sleep(0.3)
            ser.close()
        cv2.destroyAllWindows()
        print("Stopped. Servos centered, motors off.")


if __name__ == "__main__":
    main()