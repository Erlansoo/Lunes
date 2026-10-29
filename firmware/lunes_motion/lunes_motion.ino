// Lunes Motion: point-to-point motion firmware for an ATmega2560 board with the RAMPS 1.4 pinout (MKS Gen).
//
// Drives up to 5 axes through external step/dir drivers (DM542). One move at a time, trapezoidal speed profile,
// all axes synchronized with Bresenham so they arrive together. Positions are counted in steps; the host converts
// millimetres. Text commands in, one JSON object per line out.
//
// Safety lives here, where the agent cannot skip it: motors start disabled, rates and accelerations are capped,
// soft limits are checked before a move starts, a move stops by itself if the host goes silent, an axis never runs
// into its endstop, and the laser pointer switches itself off.
//
// Also on the board: a laser pointer (LASER), an MPU6050 motion sensor on I2C (IMU) and the switches (SW).

#include <Arduino.h>
#include <Wire.h>
#include <util/atomic.h>

const uint8_t N_AXES = 5;
const char AXIS_NAMES[N_AXES] = {'X', 'Y', 'Z', 'A', 'B'};
// RAMPS 1.4 driver sockets X, Y, Z, E0, E1.
const uint8_t STEP_PIN[N_AXES] = {54, 60, 46, 26, 36};
const uint8_t DIR_PIN[N_AXES] = {55, 61, 48, 28, 34};
const uint8_t EN_PIN[N_AXES] = {38, 56, 62, 24, 30};
const uint8_t PROBE_PIN = 18;  // Z_MIN header, active low with the internal pull-up
const uint8_t NO_PIN = 255;
// Endstops at each socket's minimum: X_MIN and Y_MIN headers, closing to ground (active low with pull-ups). One that
// is not wired reads open, so it never stops anything.
const uint8_t ENDSTOP_PIN[N_AXES] = {3, 14, NO_PIN, NO_PIN, NO_PIN};
const uint8_t LASER_PIN = 6;            // servo header; switches the laser module through a transistor (PWM)
const uint32_t LASER_MAX_MS = 60000;    // the laser goes off by itself a minute after the host last set it
const uint8_t MPU_ADDR = 0x68;          // MPU6050 with AD0 low, on SDA 20 / SCL 21

const uint32_t ISR_HZ = 20000;          // step timer rate; one step per axis per tick at most
const float RATE_CAP = 18000.0;         // steps/s, hard ceiling for any axis
const float START_RATE = 200.0;         // steps/s at the start and end of every move
const uint32_t HOST_TIMEOUT_MS = 1000;  // a move stops if the host sends nothing for this long
const uint32_t ONE_Q24 = 1UL << 24;     // speeds are steps per tick in Q24 fixed point
const uint8_t PULSE_TICKS = 12;         // step pulse width in Timer1 ticks (0.5 us each): 6 us

// Settings the host sends at connect time. Defaults are slow on purpose.
float max_rate[N_AXES] = {2000, 2000, 1000, 1000, 1000};   // steps/s
float max_accel[N_AXES] = {4000, 4000, 2000, 2000, 2000};  // steps/s^2
bool dir_invert[N_AXES] = {false, false, false, false, false};
bool limit_on[N_AXES] = {false, false, false, false, false};
int32_t limit_min[N_AXES], limit_max[N_AXES];
int8_t slave_of[N_AXES] = {-1, -1, -1, -1, -1};  // a slaved axis copies every step of its master (dual-motor Y)
bool en_active_low = true;  // stepstick drivers, and DM542 with ENA- to GND: a high enable pin disables the driver
bool enabled = false;
bool dry_run = false;       // count steps and time the profile without touching the step pins

// IDLE/ACCEL/CRUISE/DECEL run one coordinated move; JOG runs every axis at its own commanded velocity.
enum State : uint8_t { IDLE, ACCEL, CRUISE, DECEL, JOG };
const uint32_t JOG_DEADMAN_MS = 250;  // jog velocities fall to zero unless the host repeats JV this often
const char* alarm_reason = "";

// What ends a TOUCH-like move early: nothing, the probe closing (TOUCH), or the moving axis's endstop (SEEK).
enum Touch : uint8_t { TOUCH_NONE, TOUCH_PROBE, TOUCH_ENDSTOP };

// Shared with the step interrupt.
volatile State state = IDLE;
volatile int32_t pos[N_AXES];
volatile bool move_done = false, move_stopped = false, touch_hit = false, endstop_alarm = false;
volatile bool stop_req = false, finishing = false;
volatile Touch touch_mode = TOUCH_NONE;
volatile uint8_t pending = 0;  // axes to pulse at the next tick
volatile uint32_t v_q24;

uint32_t mv_steps[N_AXES], mv_n, steps_done, accel_steps, phase;
int8_t mv_sign[N_AXES];
int32_t bres[N_AXES];
uint32_t v_min_q24, v_max_q24, a_q24;

// Jog: signed velocities in Q24 steps per tick, ramped per axis.
// Read by the main loop only inside atomic blocks, so they need no volatile and the interrupt stays fast.
int32_t jog_target[N_AXES], jog_v[N_AXES], jog_a[N_AXES];  // jog_a is per ramp turn: N_AXES ticks of acceleration
uint32_t jog_abs[N_AXES], jog_phase[N_AXES];
uint8_t jog_mask = 0, jog_turn = 0;  // axes still moving or asked to move; axis whose ramp runs this tick

volatile uint8_t* step_port[N_AXES];
uint8_t step_mask[N_AXES];
volatile uint8_t* dir_port[N_AXES];
uint8_t dir_mask[N_AXES];
volatile uint8_t* probe_reg;
uint8_t probe_mask;
volatile uint8_t* endstop_reg[N_AXES];  // nullptr for a socket without an endstop
uint8_t endstop_mask[N_AXES];

uint32_t last_rx_ms = 0, last_jog_ms = 0;
uint8_t laser_level = 0;
uint32_t laser_on_ms = 0;
bool imu_ready = false;

inline bool endstop_closed(uint8_t i) { return endstop_reg[i] && !(*endstop_reg[i] & endstop_mask[i]); }

// True when an axis of the current move is heading for its minimum and its endstop has closed.
inline bool endstop_ahead() {
  for (uint8_t i = 0; i < N_AXES; i++) {
    if (mv_steps[i] && mv_sign[i] < 0 && endstop_closed(i)) return true;
  }
  return false;
}

// Direction pin write that is safe and fast inside the step interrupt.
inline void set_dir(uint8_t i, int8_t sign) {
  if ((sign > 0) != dir_invert[i]) *dir_port[i] |= dir_mask[i];
  else *dir_port[i] &= ~dir_mask[i];
}

// One tick of jog mode: every moving axis steps on its own phase accumulator, and one axis per tick, in turn, ramps
// its velocity toward its target. Doing all five ramps every tick took longer than the 50 us between ticks.
// Direction flips only after the velocity passes through zero, so the pin settles long before the next pulse.
inline void jog_tick() {
  uint8_t i = jog_turn;
  jog_turn = jog_turn + 1 == N_AXES ? 0 : jog_turn + 1;
  if (slave_of[i] < 0) {
    int32_t v = jog_v[i], tv = jog_target[i];
    if (v != tv) {
      if (v < tv) {
        v += jog_a[i];
        if (v > tv) v = tv;
      } else {
        v -= jog_a[i];
        if (v < tv) v = tv;
      }
      jog_v[i] = v;
      if (v) {
        int8_t sign = v > 0 ? 1 : -1;
        if (sign != mv_sign[i]) {
          mv_sign[i] = sign;
          set_dir(i, sign);
          jog_phase[i] = 0;
        }
      }
      jog_abs[i] = v > 0 ? v : -v;
    }
    if (v || tv) jog_mask |= 1 << i;
    else jog_mask &= ~(1 << i);
  }

  uint8_t m = 0;
  for (uint8_t k = 0; k < N_AXES; k++) {
    if (!(jog_mask & (1 << k))) continue;
    jog_phase[k] += jog_abs[k];
    if (jog_phase[k] >= ONE_Q24) {
      jog_phase[k] -= ONE_Q24;
      // Hard stop at a soft limit; guard_jog_limits() normally brakes before this is ever needed.
      if (limit_on[k] && ((mv_sign[k] > 0 && pos[k] >= limit_max[k]) || (mv_sign[k] < 0 && pos[k] <= limit_min[k]))) {
        continue;
      }
      if (mv_sign[k] < 0 && endstop_closed(k)) continue;  // against its endstop: no further
      m |= 1 << k;
    }
  }
  for (uint8_t s = 0; s < N_AXES; s++) {
    int8_t ms = slave_of[s];
    if (ms < 0) continue;
    if (mv_sign[s] != mv_sign[ms]) {
      mv_sign[s] = mv_sign[ms];
      set_dir(s, mv_sign[s]);
    }
    if (m & (1 << ms)) m |= 1 << s;
  }
  pending = m;
  if (!jog_mask) {
    state = IDLE;
    finishing = true;
  }
}

char line[96];
uint8_t line_len = 0;

ISR(TIMER1_COMPA_vect) {
  uint8_t out = pending;
  pending = 0;
  if (out) {
    for (uint8_t i = 0; i < N_AXES; i++) {
      if (out & (1 << i)) {
        if (!dry_run) *step_port[i] |= step_mask[i];
        pos[i] += mv_sign[i];
      }
    }
  }
  if (finishing) {
    finishing = false;
    move_done = true;
  }

  if (state == JOG) {
    jog_tick();
  } else if (state != IDLE) {
    bool probe = touch_mode == TOUCH_PROBE && !(*probe_reg & probe_mask);
    bool endstop = endstop_ahead();
    if (probe || endstop) {
      state = IDLE;
      if (probe || touch_mode == TOUCH_ENDSTOP) {
        touch_hit = true;
      } else {  // a plain move ran into an endstop: the host's idea of the position was wrong
        move_stopped = true;
        endstop_alarm = true;
      }
      move_done = true;
    } else {
      phase += v_q24;
      if (phase >= ONE_Q24) {
        phase -= ONE_Q24;
        uint8_t m = 0;
        for (uint8_t i = 0; i < N_AXES; i++) {
          if (mv_steps[i]) {
            bres[i] += mv_steps[i];
            if (bres[i] > 0) {
              bres[i] -= mv_n;
              m |= 1 << i;
            }
          }
        }
        pending = m;
        steps_done++;
        if (state == ACCEL) accel_steps++;
        uint32_t remaining = mv_n - steps_done;
        if (remaining == 0) {
          state = IDLE;
          finishing = true;
        } else if (state != DECEL && remaining <= accel_steps) {
          state = DECEL;
        }
      }
      if (stop_req && (state == ACCEL || state == CRUISE)) state = DECEL;
      if (state == ACCEL) {
        v_q24 += a_q24;
        if (v_q24 >= v_max_q24) {
          v_q24 = v_max_q24;
          state = CRUISE;
        }
      } else if (state == DECEL) {
        if (v_q24 > v_min_q24 + a_q24) {
          v_q24 -= a_q24;
        } else {
          v_q24 = v_min_q24;
          if (stop_req) {
            state = IDLE;
            move_stopped = true;
            finishing = true;
          }
        }
      }
    }
  }

  if (out && !dry_run) {
    while (TCNT1 < PULSE_TICKS) {
    }
    for (uint8_t i = 0; i < N_AXES; i++) {
      if (out & (1 << i)) *step_port[i] &= ~step_mask[i];
    }
  }
}

void print_pos() {
  int32_t p[N_AXES];
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    for (uint8_t i = 0; i < N_AXES; i++) p[i] = pos[i];
  }
  Serial.print(F("\"pos\":["));
  for (uint8_t i = 0; i < N_AXES; i++) {
    if (i) Serial.print(',');
    Serial.print(p[i]);
  }
  Serial.print(']');
}

void reply_ok(const char* cmd) {
  Serial.print(F("{\"ok\":\""));
  Serial.print(cmd);
  Serial.println(F("\"}"));
}

void reply_err(const char* why) {
  Serial.print(F("{\"err\":\""));
  Serial.print(why);
  Serial.println(F("\"}"));
}

void report_status() {
  static const char* const NAMES[] = {"idle", "accel", "cruise", "decel", "jog"};
  State s = state;
  uint32_t v = 0;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    if (s != JOG) {
      v = v_q24;
    } else {
      for (uint8_t i = 0; i < N_AXES; i++) {
        uint32_t a = jog_v[i] < 0 ? -jog_v[i] : jog_v[i];
        if (a > v) v = a;  // fastest axis
      }
    }
  }
  Serial.print(F("{\"st\":\""));
  Serial.print(NAMES[s]);
  Serial.print(F("\",\"alarm\":\""));
  Serial.print(alarm_reason);
  Serial.print(F("\",\"en\":"));
  Serial.print(enabled ? 1 : 0);
  Serial.print(F(",\"dry\":"));
  Serial.print(dry_run ? 1 : 0);
  Serial.print(F(",\"laser\":"));
  Serial.print(laser_level);
  Serial.print(F(",\"rate\":"));
  Serial.print(s == IDLE ? 0 : (uint32_t)((float)v * ISR_HZ / ONE_Q24));
  Serial.print(',');
  print_pos();
  Serial.println('}');
}

void set_laser(uint8_t level) {
  laser_level = level;
  laser_on_ms = millis();
  analogWrite(LASER_PIN, level);
}

// Wakes the MPU6050 and sets its ranges: +-2 g, +-250 deg/s, 44 Hz low-pass. False if nothing answers.
bool imu_begin() {
  static const uint8_t SETUP[][2] = {{0x6B, 0x01}, {0x1C, 0x00}, {0x1B, 0x00}, {0x1A, 0x03}};
  for (uint8_t k = 0; k < 4; k++) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(SETUP[k][0]);
    Wire.write(SETUP[k][1]);
    if (Wire.endTransmission() != 0) return false;
  }
  return true;
}

// {"imu":{"a":[x,y,z],"g":[x,y,z],"t":c}}: raw accelerometer (16384 per g) and gyro (131 per deg/s) counts, and the
// chip's temperature in hundredths of a degree.
void report_imu() {
  if (!imu_ready) imu_ready = imu_begin();
  int16_t v[7];
  bool got = false;
  if (imu_ready) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    got = Wire.endTransmission(false) == 0 && Wire.requestFrom(MPU_ADDR, (uint8_t)14) == 14;
    for (uint8_t k = 0; got && k < 7; k++) {
      uint8_t hi = Wire.read();
      v[k] = (int16_t)((hi << 8) | Wire.read());
    }
  }
  if (!got) {
    imu_ready = false;
    return reply_err("imu");
  }
  Serial.print(F("{\"imu\":{\"a\":["));
  for (uint8_t k = 0; k < 3; k++) {
    if (k) Serial.print(',');
    Serial.print(v[k]);
  }
  Serial.print(F("],\"g\":["));
  for (uint8_t k = 4; k < 7; k++) {
    if (k > 4) Serial.print(',');
    Serial.print(v[k]);
  }
  Serial.print(F("],\"t\":"));
  Serial.print((int32_t)v[3] * 100 / 340 + 3653);
  Serial.println(F("}}"));
}

void set_enabled(bool on) {
  enabled = on;
  for (uint8_t i = 0; i < N_AXES; i++) digitalWrite(EN_PIN[i], (on == en_active_low) ? LOW : HIGH);
}

void hard_stop() {
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    bool was_moving = state != IDLE;
    state = IDLE;
    pending = 0;
    finishing = false;
    for (uint8_t i = 0; i < N_AXES; i++) jog_v[i] = jog_target[i] = jog_abs[i] = 0;
    jog_mask = 0;
    if (was_moving) {
      move_stopped = true;
      move_done = true;
    }
  }
  set_laser(0);
  alarm_reason = "hard_stop";
}

int axis_index(char c) {
  for (uint8_t i = 0; i < N_AXES; i++) {
    if (AXIS_NAMES[i] == c) return i;
  }
  return -1;
}

// Starts a move to target (absolute steps). Rate and acceleration are for the dominant axis and get reduced so no
// axis exceeds its own limits.
bool start_move(const int32_t* target, float rate, Touch touch) {
  if (state != IDLE || finishing) return reply_err("busy"), false;
  if (alarm_reason[0]) return reply_err("alarm"), false;
  if (!enabled && !dry_run) return reply_err("disabled"), false;
  if (!(rate > 0)) return reply_err("rate"), false;

  int32_t cur[N_AXES];
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    for (uint8_t i = 0; i < N_AXES; i++) cur[i] = pos[i];
  }
  uint32_t n = 0;
  for (uint8_t i = 0; i < N_AXES; i++) {
    int8_t m = slave_of[i];
    // A slaved axis ignores its own target and repeats the master's move; the master's limits cover it.
    int32_t d = m >= 0 ? target[m] - cur[m] : target[i] - cur[i];
    if (m < 0 && limit_on[i] && (target[i] < limit_min[i] || target[i] > limit_max[i])) {
      return reply_err("limit"), false;
    }
    mv_sign[i] = d < 0 ? -1 : 1;
    mv_steps[i] = d < 0 ? -d : d;
    if (mv_steps[i] > n) n = mv_steps[i];
  }
  if (n == 0) {
    reply_ok("MOVE");
    move_stopped = false;
    touch_hit = false;
    move_done = true;
    return true;
  }

  float r = min(rate, RATE_CAP), a = 1e9;
  for (uint8_t i = 0; i < N_AXES; i++) {
    if (!mv_steps[i] || slave_of[i] >= 0) continue;
    float frac = (float)mv_steps[i] / n;
    r = min(r, max_rate[i] / frac);
    a = min(a, max_accel[i] / frac);
  }
  float r0 = min(r, START_RATE);

  for (uint8_t i = 0; i < N_AXES; i++) {
    if (mv_steps[i]) digitalWrite(DIR_PIN[i], ((mv_sign[i] > 0) != dir_invert[i]) ? HIGH : LOW);
  }
  delayMicroseconds(10);  // DM542 needs the direction set at least 5 us before the first pulse

  for (uint8_t i = 0; i < N_AXES; i++) bres[i] = -(int32_t)(n / 2);
  mv_n = n;
  steps_done = 0;
  accel_steps = 0;
  phase = 0;
  v_min_q24 = (uint32_t)(r0 * (ONE_Q24 / (float)ISR_HZ));
  v_max_q24 = (uint32_t)(r * (ONE_Q24 / (float)ISR_HZ));
  a_q24 = max((uint32_t)1, (uint32_t)(a * (ONE_Q24 / ((float)ISR_HZ * ISR_HZ))));
  if (v_min_q24 < 1) v_min_q24 = 1;
  reply_ok(touch == TOUCH_PROBE ? "TOUCH" : touch == TOUCH_ENDSTOP ? "SEEK" : "MOVE");
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    v_q24 = v_min_q24;
    stop_req = false;
    move_done = false;
    move_stopped = false;
    touch_hit = false;
    touch_mode = touch;
    state = (v_min_q24 >= v_max_q24) ? CRUISE : ACCEL;
  }
  return true;
}

// Reads "X100 Y-20 F500"-style words after the command name. Missing axes keep their current target.
bool parse_axes(char* p, int32_t* target, bool relative, float* rate) {
  int32_t cur[N_AXES];
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    for (uint8_t i = 0; i < N_AXES; i++) cur[i] = pos[i];
  }
  for (uint8_t i = 0; i < N_AXES; i++) target[i] = cur[i];
  *rate = 0;
  while (*p) {
    while (*p == ' ') p++;
    if (!*p) break;
    char k = *p++;
    char* end;
    if (k == 'F') {
      *rate = strtod(p, &end);
    } else {
      int i = axis_index(k);
      if (i < 0) return false;
      long v = strtol(p, &end, 10);
      target[i] = relative ? cur[i] + v : v;
    }
    if (end == p) return false;
    p = end;
  }
  return true;
}

void handle_line(char* s) {
  char* args = strchr(s, ' ');
  if (args) *args++ = 0;
  else args = s + strlen(s);

  if (!strcmp(s, "INFO")) {
    Serial.print(F("{\"fw\":\"lunes-motion\",\"ver\":\"0.2\",\"axes\":\"XYZAB\",\"isr_hz\":"));
    Serial.print(ISR_HZ);
    Serial.println('}');
  } else if (!strcmp(s, "MOVE") || !strcmp(s, "REL") || !strcmp(s, "TOUCH") || !strcmp(s, "SEEK")) {
    // TOUCH stops when the probe closes; SEEK X-40000 F600 stops when the endstop of the axis it drives closes
    int32_t target[N_AXES];
    float rate;
    bool relative = strcmp(s, "MOVE") != 0;
    Touch touch = !strcmp(s, "TOUCH") ? TOUCH_PROBE : !strcmp(s, "SEEK") ? TOUCH_ENDSTOP : TOUCH_NONE;
    if (!parse_axes(args, target, relative, &rate)) return reply_err("syntax");
    if (touch == TOUCH_ENDSTOP) {
      int32_t cur[N_AXES];
      ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
        for (uint8_t i = 0; i < N_AXES; i++) cur[i] = pos[i];
      }
      for (uint8_t i = 0; i < N_AXES; i++) {
        if (slave_of[i] >= 0 || target[i] == cur[i]) continue;
        if (!endstop_reg[i]) return reply_err("no_endstop");
        if (target[i] > cur[i]) return reply_err("endstops_are_at_min");
      }
    }
    start_move(target, rate, touch);
  } else if (!strcmp(s, "EN")) {
    if (state != IDLE) return reply_err("busy");
    set_enabled(atoi(args) != 0);
    reply_ok("EN");
  } else if (!strcmp(s, "ENPOL")) {
    if (state != IDLE) return reply_err("busy");
    en_active_low = args[0] != 'H';
    set_enabled(enabled);
    reply_ok("ENPOL");
  } else if (!strcmp(s, "DRY")) {
    if (state != IDLE) return reply_err("busy");
    dry_run = atoi(args) != 0;
    reply_ok("DRY");
  } else if (!strcmp(s, "ZERO")) {
    // ZERO zeroes every axis; ZERO X zeroes X and the axes slaved to it
    if (state != IDLE) return reply_err("busy");
    for (char* p = args; *p; p++) {
      if (*p != ' ' && axis_index(*p) < 0) return reply_err("axis");
    }
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
      for (uint8_t i = 0; i < N_AXES; i++) {
        int8_t m = slave_of[i] >= 0 ? slave_of[i] : i;
        if (!args[0] || strchr(args, AXIS_NAMES[m])) pos[i] = 0;
      }
    }
    reply_ok("ZERO");
  } else if (!strcmp(s, "SW")) {
    // {"sw":{"probe":0,"X":1,"Y":0}}: 1 = closed
    Serial.print(F("{\"sw\":{\"probe\":"));
    Serial.print(*probe_reg & probe_mask ? 0 : 1);
    for (uint8_t i = 0; i < N_AXES; i++) {
      if (!endstop_reg[i]) continue;
      Serial.print(F(",\""));
      Serial.print(AXIS_NAMES[i]);
      Serial.print(F("\":"));
      Serial.print(endstop_closed(i) ? 1 : 0);
    }
    Serial.println(F("}}"));
  } else if (!strcmp(s, "LASER")) {
    // LASER <0-255>: brightness; it goes out by itself LASER_MAX_MS after the last LASER
    long level = strtol(args, nullptr, 10);
    if (level < 0 || level > 255) return reply_err("laser");
    set_laser((uint8_t)level);
    reply_ok("LASER");
  } else if (!strcmp(s, "IMU")) {
    report_imu();
  } else if (!strcmp(s, "CFG")) {
    // CFG X R<rate> A<accel> I<0|1>
    if (state != IDLE) return reply_err("busy");
    int i = axis_index(args[0]);
    if (i < 0) return reply_err("axis");
    for (char* p = args + 1; *p; p++) {
      if (*p == 'R') max_rate[i] = min((float)strtod(p + 1, nullptr), RATE_CAP);
      else if (*p == 'A') max_accel[i] = strtod(p + 1, nullptr);
      else if (*p == 'I') dir_invert[i] = atoi(p + 1) != 0;
    }
    reply_ok("CFG");
  } else if (!strcmp(s, "LIM")) {
    // LIM X <min> <max>  or  LIM X OFF
    if (state != IDLE) return reply_err("busy");
    int i = axis_index(args[0]);
    if (i < 0) return reply_err("axis");
    char* p = args + 1;
    while (*p == ' ') p++;
    if (!strcmp(p, "OFF")) {
      limit_on[i] = false;
    } else {
      char* end;
      limit_min[i] = strtol(p, &end, 10);
      limit_max[i] = strtol(end, nullptr, 10);
      limit_on[i] = limit_min[i] < limit_max[i];
      if (!limit_on[i]) return reply_err("limit");
    }
    reply_ok("LIM");
  } else if (!strcmp(s, "JV")) {
    // JV X<steps/s> Y<steps/s> ... signed jog velocities; axes left out go to zero. Silent unless it fails, because
    // the host repeats it several times a second while a key is held (the deadman).
    if ((state != IDLE && state != JOG) || (state == IDLE && finishing)) return reply_err("busy");
    if (alarm_reason[0]) return reply_err("alarm");
    if (!enabled && !dry_run) return reply_err("disabled");
    int32_t tv[N_AXES] = {0, 0, 0, 0, 0};
    for (char* p = args; *p;) {
      while (*p == ' ') p++;
      if (!*p) break;
      int i = axis_index(*p++);
      char* end;
      float r = strtod(p, &end);
      if (i < 0 || end == p) return reply_err("syntax");
      p = end;
      if (slave_of[i] >= 0) continue;
      r = constrain(r, -max_rate[i], max_rate[i]);
      tv[i] = (int32_t)(r * (ONE_Q24 / (float)ISR_HZ));
    }
    last_jog_ms = millis();
    uint8_t asked = 0;
    for (uint8_t i = 0; i < N_AXES; i++) {
      if (tv[i]) asked |= 1 << i;
    }
    if (state == IDLE) {
      if (!asked) return;  // nothing to start
      for (uint8_t i = 0; i < N_AXES; i++) {
        // Each axis's ramp runs once every N_AXES ticks, so it takes N_AXES ticks' worth of acceleration.
        jog_a[i] = max((int32_t)1, (int32_t)(max_accel[i] * N_AXES * (ONE_Q24 / ((float)ISR_HZ * ISR_HZ))));
        jog_phase[i] = 0;
        mv_sign[i] = 0;  // forces the direction pin to be set on the first step
      }
    }
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
      for (uint8_t i = 0; i < N_AXES; i++) jog_target[i] = tv[i];
      jog_mask |= asked;
      if (state == IDLE) {
        for (uint8_t i = 0; i < N_AXES; i++) jog_v[i] = jog_abs[i] = 0;
        jog_mask = asked;
        jog_turn = 0;
        move_stopped = false;
        touch_hit = false;
        state = JOG;
      }
    }
  } else if (!strcmp(s, "SLAVE")) {
    // SLAVE A Y  (A copies Y)  or  SLAVE A OFF
    if (state != IDLE) return reply_err("busy");
    int i = axis_index(args[0]);
    if (i < 0) return reply_err("axis");
    char* p = args + 1;
    while (*p == ' ') p++;
    if (!strcmp(p, "OFF")) {
      slave_of[i] = -1;
    } else {
      int m = axis_index(p[0]);
      if (m < 0 || m == i || slave_of[m] >= 0) return reply_err("axis");
      for (uint8_t k = 0; k < N_AXES; k++) {
        if (slave_of[k] == i) return reply_err("axis");  // a master cannot be slaved itself
      }
      slave_of[i] = m;
      ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { pos[i] = pos[m]; }
    }
    reply_ok("SLAVE");
  } else if (!strcmp(s, "RESET")) {
    if (state != IDLE) return reply_err("busy");
    alarm_reason = "";
    reply_ok("RESET");
  } else if (s[0]) {
    reply_err("unknown");
  }
}

void stop_jog() {
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    for (uint8_t i = 0; i < N_AXES; i++) jog_target[i] = 0;
  }
}

// Brakes a jogging axis early enough to stop at its soft limit, and refuses to jog further into it.
void guard_jog_limits() {
  for (uint8_t i = 0; i < N_AXES; i++) {
    if (!limit_on[i] || slave_of[i] >= 0) continue;
    int32_t p, v, t;
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
      p = pos[i];
      v = jog_v[i];
      t = jog_target[i];
    }
    float vs = (float)v * ISR_HZ / ONE_Q24;               // steps/s, signed
    float stop = vs * vs / (2 * max_accel[i]) + fabs(vs) * 0.005 + 4;  // braking distance plus loop latency
    if ((t > 0 && p + stop >= limit_max[i]) || (t < 0 && p - stop <= limit_min[i])) {
      ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { jog_target[i] = 0; }
    }
  }
}

void setup() {
  for (uint8_t i = 0; i < N_AXES; i++) {
    pinMode(STEP_PIN[i], OUTPUT);
    digitalWrite(STEP_PIN[i], LOW);
    pinMode(DIR_PIN[i], OUTPUT);
    pinMode(EN_PIN[i], OUTPUT);
    step_port[i] = portOutputRegister(digitalPinToPort(STEP_PIN[i]));
    step_mask[i] = digitalPinToBitMask(STEP_PIN[i]);
    dir_port[i] = portOutputRegister(digitalPinToPort(DIR_PIN[i]));
    dir_mask[i] = digitalPinToBitMask(DIR_PIN[i]);
    pos[i] = 0;
  }
  set_enabled(false);
  pinMode(PROBE_PIN, INPUT_PULLUP);
  probe_reg = portInputRegister(digitalPinToPort(PROBE_PIN));
  probe_mask = digitalPinToBitMask(PROBE_PIN);
  for (uint8_t i = 0; i < N_AXES; i++) {
    endstop_reg[i] = nullptr;
    if (ENDSTOP_PIN[i] == NO_PIN) continue;
    pinMode(ENDSTOP_PIN[i], INPUT_PULLUP);
    endstop_reg[i] = portInputRegister(digitalPinToPort(ENDSTOP_PIN[i]));
    endstop_mask[i] = digitalPinToBitMask(ENDSTOP_PIN[i]);
  }
  pinMode(LASER_PIN, OUTPUT);
  set_laser(0);
  Wire.begin();
  Wire.setClock(400000);
  Wire.setWireTimeout(3000, true);  // a missing or stuck sensor must never hang the motion loop

  noInterrupts();
  TCCR1A = 0;
  TCCR1B = _BV(WGM12) | _BV(CS11);  // CTC, clock / 8 = 2 MHz
  OCR1A = 2000000UL / ISR_HZ - 1;
  TIMSK1 = _BV(OCIE1A);
  interrupts();

  Serial.begin(115200);
  Serial.println(F("{\"ev\":\"boot\",\"fw\":\"lunes-motion\",\"ver\":\"0.2\"}"));
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    last_rx_ms = millis();
    if (c == '?') {
      report_status();
    } else if (c == '!') {
      if (state == JOG) stop_jog();
      else stop_req = true;
    } else if (c == 0x18) {
      hard_stop();
      report_status();
    } else if (c == '\n' || c == '\r') {
      line[line_len] = 0;
      if (line_len) handle_line(line);
      line_len = 0;
    } else if (line_len < sizeof(line) - 1) {
      line[line_len++] = c;
    }
  }

  if (state == JOG) {
    if (millis() - last_jog_ms > JOG_DEADMAN_MS) stop_jog();
    guard_jog_limits();
  } else if (state != IDLE && !stop_req && millis() - last_rx_ms > HOST_TIMEOUT_MS) {
    stop_req = true;
    alarm_reason = "host_lost";
  }

  if (endstop_alarm) {
    endstop_alarm = false;
    alarm_reason = "endstop";
  }
  if (laser_level && millis() - laser_on_ms > LASER_MAX_MS) {
    set_laser(0);
    Serial.println(F("{\"ev\":\"laser_off\"}"));
  }

  if (move_done) {
    move_done = false;
    Serial.print(F("{\"ev\":\"done\",\"stopped\":"));
    Serial.print(move_stopped ? 1 : 0);
    Serial.print(F(",\"touch\":"));
    Serial.print(touch_hit ? 1 : 0);
    Serial.print(',');
    print_pos();
    Serial.println('}');
  }
}
