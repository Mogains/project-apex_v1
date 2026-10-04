// Trackpad turret.
// Reads "x,y,fire\n" over USB serial at 115200 (x, y = whole degrees, fire = 0 or 1).
//   D8  = X (pan) servo signal
//   D11 = Y (tilt) servo signal
//   D5  = DC motors (through the 220 ohm resistor to the MOSFET gate)
// Servos and motors run from their OWN battery packs. All grounds tied together.
#include <Servo.h>

const uint8_t PIN_X = 8;
const uint8_t PIN_Y = 11;
const uint8_t PIN_WHEELS = 5;

const int X_MIN = 20, X_MAX = 160;
const int Y_MIN = 40, Y_MAX = 130;

const int STEP_X = 2;                    // degrees per step, pan (higher = faster)
const int STEP_Y = 2;                    // degrees per step, tilt
const unsigned long STEP_MS = 3;
const unsigned long MIN_ON_MS = 400;     // a quick tap still spins the wheels this long
const unsigned long FAILSAFE_MS = 500;   // no messages for this long -> motors off

Servo xServo, yServo;

int curX = 90, curY = 90;
int tgtX = 90, tgtY = 90;

bool fireHeld = false;
unsigned long fireUntil = 0;
unsigned long lastMsg = 0;
unsigned long lastStep = 0;

char buf[24];
uint8_t idx = 0;

int moveToward(int cur, int tgt, int step) {
  if (abs(tgt - cur) <= step) return tgt;
  return cur + ((tgt > cur) ? step : -step);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_WHEELS, OUTPUT);
  digitalWrite(PIN_WHEELS, LOW);
  xServo.attach(PIN_X);
  yServo.attach(PIN_Y);
  xServo.write(curX);
  yServo.write(curY);
}

void loop() {
  unsigned long now = millis();

  // read "x,y,fire\n"
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      buf[idx] = 0;
      int x, y, f;
      if (sscanf(buf, "%d,%d,%d", &x, &y, &f) == 3) {
        tgtX = constrain(x, X_MIN, X_MAX);
        tgtY = constrain(y, Y_MIN, Y_MAX);
        fireHeld = (f != 0);
        if (fireHeld) fireUntil = now + MIN_ON_MS;
        lastMsg = now;
      }
      idx = 0;
    } else if (idx < sizeof(buf) - 1) {
      buf[idx++] = c;
    }
  }

  // motors: on while held, or briefly after a tap. Off if the laptop goes quiet.
  bool alive = (now - lastMsg) < FAILSAFE_MS;
  bool motorsOn = alive && (fireHeld || (long)(fireUntil - now) > 0);
  digitalWrite(PIN_WHEELS, motorsOn ? HIGH : LOW);

  // glide the servos toward the target
  if (now - lastStep >= STEP_MS) {
    lastStep = now;
    if (curX != tgtX) {
      curX = moveToward(curX, tgtX, STEP_X);
      xServo.write(curX);
    }
    if (curY != tgtY) {
      curY = moveToward(curY, tgtY, STEP_Y);
      yServo.write(curY);
    }
  }
}