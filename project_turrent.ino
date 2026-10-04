// Camera turret.
// Reads "x,y,fire\n" over USB serial at 115200 (x, y = whole degrees, fire = 0 or 1).
//   D8  = X (pan) servo signal
//   D10 = Y (tilt) servo signal
//   D5  = DC motors (PWM, through the 220 ohm resistor to the MOSFET gate)
// Servos and motors run from their OWN battery packs. All grounds tied together.
#include <Servo.h>
#include <avr/wdt.h>

const uint8_t PIN_X = 8;
const uint8_t PIN_Y = 10;
const uint8_t PIN_WHEELS = 5;            // must be a PWM pin (3, 5, 6, 9*, 10*, 11) — 9/10 are taken by Servo

// Servo travel limits. Kept inside the servos' end stops so they never strain / lock up.
// If a servo still buzzes or stalls at one end, pull that number toward 90.
// Keep these the same as X_MIN..Y_MAX in camera_turret.html.
const int X_MIN = 30, X_MAX = 150;
const int Y_MIN = 50, Y_MAX = 120;

// DC motor power
const int MAX_POWER_PCT = 60;            // top speed, % of full power
const unsigned long RAMP_UP_MS = 1500;   // time to rev from stopped to top speed

const int STEP_X = 2;                    // degrees per step, pan (higher = faster)
const int STEP_Y = 2;                    // degrees per step, tilt
const unsigned long STEP_MS = 3;
const unsigned long MIN_ON_MS = 400;     // a quick tap still spins the wheels this long
const unsigned long FAILSAFE_MS = 500;   // no messages for this long -> motors off

const int MAX_PWM = 255L * MAX_POWER_PCT / 100;

Servo xServo, yServo;

int curX = 90, curY = 90;
int tgtX = 90, tgtY = 90;

bool fireHeld = false;
unsigned long fireUntil = 0;
unsigned long lastMsg = 0;
unsigned long lastStep = 0;
unsigned long lastRamp = 0;
float power = 0;                         // current motor PWM, 0..MAX_PWM

char buf[24];
uint8_t idx = 0;

int moveToward(int cur, int tgt, int step) {
  if (abs(tgt - cur) <= step) return tgt;
  return cur + ((tgt > cur) ? step : -step);
}

void setup() {
  wdt_disable();
  Serial.begin(115200);
  pinMode(PIN_WHEELS, OUTPUT);
  analogWrite(PIN_WHEELS, 0);
  xServo.attach(PIN_X);
  yServo.attach(PIN_Y);
  xServo.write(curX);
  yServo.write(curY);
  // if motor noise ever freezes the board, the watchdog reboots it within half a second
  wdt_enable(WDTO_500MS);
}

void loop() {
  wdt_reset();
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
  if (!motorsOn) {
    power = 0;                           // stop right away
  } else if (now - lastRamp >= 10) {
    power = min((float)MAX_PWM, power + MAX_PWM * (now - lastRamp) / (float)RAMP_UP_MS);   // rev up slowly
  }
  if (now - lastRamp >= 10) lastRamp = now;
  analogWrite(PIN_WHEELS, (int)power);

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
