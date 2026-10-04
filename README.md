# Project Apex

A camera-controlled turret. A web page watches the webcam, finds a target (a **human face** or a hand-drawn **bullseye**), and aims two servos at it through an Arduino. It can spin a pair of DC motors to "fire", and it talks while it works, with funny voice lines made with ElevenLabs.

Everything runs on a laptop in Chrome. There's no server and no install for the page itself.

---

## What it does

- **Tracks a target with the webcam**
  - **Face mode** uses Google's MediaPipe face detector.
  - **Bullseye mode** uses a custom YOLOv8 model trained on our own hand-drawn bullseye. It runs inside the browser.
- **Aims the turret.** The page sends pan and tilt angles to the Arduino 30 times a second. Smoothing and a deadband keep the servos from jittering.
- **Fires**
  - Manually: hold **Space** or the **FIRE** button.
  - Automatically (bullseye mode): if the bullseye **moves for 3+ seconds, then holds still for 2 seconds**, the turret fires for 2 seconds. It re-arms the next time the bullseye moves.
- **Talks.** It says things when it spots a target, locks on, loses it and fires, plus a startup line and idle jokes for the judges. The 54 lines are pre-recorded MP3s, so it works offline and costs nothing to run.
- **Stays safe**
  - The motors are capped at 60% power and rev up slowly.
  - The servos are kept away from their end stops.
  - The motors switch off if the laptop stops talking to the Arduino.
  - The page reconnects by itself if the Arduino drops out.

---

## Hardware

| Part | Pin |
|---|---|
| Pan servo (X) | **D8** |
| Tilt servo (Y) | **D10** |
| DC motors, through a MOSFET (gate via a 220 Ω resistor) | **D5** (PWM) |

- The board is an Arduino Uno, or a clone with a CH340 USB chip.
- The servos and the motors each have their **own battery pack**, and **all grounds are tied together**.

**Recommended noise fixes.** These keep the motors from crashing the Arduino when they fire:
- A **flyback diode** across the motor (1N4007 or 1N5819), with the striped end on the motor's **+**.
- A **0.1 µF** ceramic capacitor across the motor terminals.
- A **470–1000 µF** capacitor across the servo battery.

---

## Quick start

1. **Upload the Arduino code.** Open `project_turrent.ino` in the Arduino IDE, choose your board and port, and click Upload.
2. **Open the page.** Either open `open_turret.py` in VS Code and click **▶ Run**, or double-click `camera_turret.html` in Finder. It must open in **Chrome** (or Edge), because Safari can't talk to the Arduino.
3. Pick **track: face** or **track: bullseye**.
4. Click **Start camera** and allow camera access.
5. Click **Connect Arduino** and pick the `usbserial` port.
6. Move a face or the bullseye in front of the camera and the turret follows it.

### Controls

| Control | What it does |
|---|---|
| **track** | Choose face or bullseye tracking |
| **FIRE** button / **Space** | Spin the motors while held |
| **auto-fire** | Turn bullseye auto-fire on or off |
| **invert X / invert Y** | Flip a direction if the turret moves the wrong way |
| **voice** | Turn the voice lines on or off |
| **Disconnect** | Center the servos, stop the motors, release the Arduino |

The readout at the top-right shows the pan/tilt angles, the motor state, `LOCKED`, and the auto-fire progress (`moving 1.8s` → `ARMED` → `firing in 1.2s` → `AUTO FIRE`).

---

## Project layout

```
camera_turret.html     the whole app: camera, tracking, aiming, firing, voice, Arduino link
open_turret.py         press Run in VS Code to open the page in Chrome
project_turrent.ino    Arduino code: servos, motor power and ramp, safety cut-offs
models/
  bullseye.onnx        trained bullseye detector
  bullseye.js          same model, packed so the page works opened straight from disk
voice/
  make_voice.py        all voice lines live here; generates the MP3s with ElevenLabs
  lines.js             list of lines and their MP3 files (generated)
  *.mp3                the recorded lines
training/
  bullseye_photo.jpg   photo of the bullseye
  label_video.py       auto-labels a video of the bullseye (no hand labeling)
  make_dataset.py      builds the training set from real frames and generated images
  train.py             trains YOLOv8n
  export.py            tests on held-out video, exports to models/
```

---

## How the bullseye detector was trained

We didn't label anything by hand.

1. **Film the target.** Record a short video of the bullseye moving around and put it in `training/test_videos/`.
2. **Auto-label** with `label_video.py`. In each frame it finds the purple center dot, then the white page around it, and draws the box. It throws away frames where the page outline isn't a clean rectangle, for example when it blends into a hand. It also saves the page cut out of each frame.
3. **Build the dataset** with `make_dataset.py`. This mixes the real frames with about 3,000 generated images. Each one has the real cut-out page pasted in at random sizes (14–600 px), angles, lighting, blur and noise. It adds **decoys** too (random circles, blank and lined paper) so the model learns *our* bullseye, not just "a circle" or "a sheet of paper". The last 20% of each video is held back for testing.
4. **Train** with `train.py`: YOLOv8n starting from pretrained weights, 640 px images, on the Apple GPU (`mps`).
5. **Export** with `export.py`. It reports how often the model finds the page in held-out frames, then writes `models/bullseye.onnx` and `models/bullseye.js`.

**Result:** the first model was trained only on the photo, and found the bullseye in **44%** of real video frames. The current model was retrained on the auto-labeled video and finds it in **100%** of held-out frames, at 0.94 median confidence.

To retrain, with `yolov8n.pt` in the project folder:

```bash
PY=/Users/anmolbhatt/.pyenv/versions/3.10.13/bin/python   # any Python with ultralytics, opencv-python, numpy
$PY training/label_video.py
$PY training/make_dataset.py
$PY training/train.py --epochs 30
$PY training/export.py
```

Optionally, put a video of the room **without** the bullseye in `training/backgrounds/`. This teaches the model what *isn't* a target.

---

## Voice lines (ElevenLabs)

- All lines live in `voice/make_voice.py`, grouped by moment: startup, idle, acquired, locked, lost and firing, separately for face and bullseye mode.
- The voice is **Callum, "Husky Trickster"** (`N2lVS1w4EtoT3dr4eOWO`).

To change what it says:
1. Create `elevenlabs_key.js` in the project folder containing `window.ELEVENLABS_KEY = "sk_...";`. This file is git-ignored, so the key never gets uploaded.
2. Edit the lines in `voice/make_voice.py`.
3. Run `voice/make_voice.py`. It only generates lines that don't have an MP3 yet, so you only pay for new lines. It deletes MP3s for lines you removed.

To change the voice, set `VOICE_ID` in both `voice/make_voice.py` and `camera_turret.html`, then run `make_voice.py` again.

If a line has no MP3, the page speaks it live with ElevenLabs (if a key is set) or with the built-in Mac voice.

---

## Tuning

**In `camera_turret.html`:**

| Setting | Default | What it does |
|---|---|---|
| `SMOOTH` | 0.15 | Aim smoothing. Higher is snappier, lower is smoother |
| `DEADBAND` | 3° | Ignore aim changes smaller than this (stops jitter) |
| `BULLSEYE_CONF` | 0.45 | Detection threshold. Raise it if it locks onto the wrong things |
| `AUTO_MOVE_MS` / `AUTO_STILL_MS` / `AUTO_FIRE_MS` | 3000 / 2000 / 2000 | Auto-fire timing |
| `MOVING_SPEED` / `STILL_SPEED` | 0.15 / 0.05 | What counts as moving or still (in frame-widths per second) |
| `LOCK_MS` | 1200 | How long a target must be held before "locked" |
| `IDLE_MIN_MS` / `IDLE_MAX_MS` | 25000 / 45000 | How often the idle jokes play |
| `X_MIN`…`Y_MAX` | 30–150 / 50–120 | Servo range. **Keep the same as the Arduino code** |

**In `project_turrent.ino`:**

| Setting | Default | What it does |
|---|---|---|
| `MAX_POWER_PCT` | 60 | Motor top speed, as a % of full power |
| `RAMP_UP_MS` | 1500 | Time for the motors to rev up to top speed |
| `X_MIN`…`Y_MAX` | 30–150 / 50–120 | Servo travel limits. Move a limit toward 90 if a servo buzzes or stalls |
| `STEP_X` / `STEP_Y` | 2 | Servo speed |
| `FAILSAFE_MS` | 500 | Motors stop if no command arrives for this long |

The Arduino talks over USB serial at **115200** baud, one line per command: `pan,tilt,fire\n`, for example `90,85,1`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| **"Resource busy" when connecting** | Another program has the Arduino open. Close the Arduino IDE's Serial Monitor and any other turret tab. |
| **Everything freezes when the motors fire** | That's electrical noise from the motors. Add the diode and capacitors from [Hardware](#hardware). The page reconnects on its own and the Arduino has a watchdog, but the diode is the real fix. |
| **Turret moves the wrong way** | Tick **invert X** or **invert Y**. |
| **Servo buzzes at one end** | Move that limit toward 90 in *both* the Arduino code and `camera_turret.html`. |
| **Motors won't start at 60%** | Raise `MAX_POWER_PCT`. |
| **Bullseye not detected** | It works best from a few steps away. Right up against the camera it fills the frame and gets missed. Add a video of your setup and retrain. |
| **No voice** | Make sure **voice** is ticked. Click the page once first, because browsers block sound until you interact with the page. |
