// Lunes Brazo: point-to-point motion firmware for the Creality E3 Free-runs board (STM32F401RE, four TMC2209).
//
// The same protocol as Lunes Motion (firmware/lunes_motion), so the PC drives both boards the same way: one move at
// a time, trapezoidal speed profile, all axes synchronized with Bresenham so they arrive together, jog with a
// deadman, positions counted in steps (the host converts degrees or millimetres). Text commands in, one JSON object
// per line out.
//
// What is new here is the drivers: the four TMC2209 sit on one UART (USART6, addresses 0 to 3), so their current,
// microsteps and chopper mode are set by this firmware instead of by a potentiometer. Each socket has a current cap
// for the motor Erlan put on it, so no command can burn a small motor: X and Y carry 1.8 A motors, Z a 1.2 A motor
// and E a 0.6 A motor.
//
// Safety lives here, where the agent cannot skip it: motors start disabled, they cannot be enabled until all four
// drivers answer and take their settings, rates and accelerations are capped, soft limits are checked before a move
// starts, a move stops by itself if the host goes silent, and the heater outputs of the board stay off.
//
// Pins from Creality's own Marlin configuration for this board (pins_CREALITY_CR4NTxxC10.h).

#include <Arduino.h>

const uint8_t N_AXES = 4;
const char AXIS_NAMES[N_AXES] = {'X', 'Y', 'Z', 'E'};  // the driver sockets as printed on the board
const PinName STEP_PIN[N_AXES] = {PC_2, PB_8, PB_6, PB_4};
const PinName DIR_PIN[N_AXES] = {PB_9, PB_7, PB_5, PB_3};
const uint32_t EN_PIN = PC3;  // shared by the four drivers, active low
const uint32_t STOP_PIN[3] = {PA5, PA6, PA7};  // X, Y and Z endstop headers
const uint32_t PROBE_PIN = PB0;                 // the BLTouch input: TOUCH stops when it goes low
const uint32_t FAN_PIN = PA0;
const uint32_t HEATER_PINS[2] = {PA1, PB10};    // hotend and bed MOSFETs: never used, held low

Uart Host(PA_10, PA_9);    // USART1, to the USB-serial chip of the board
Uart TmcBus(PC_7, PC_6);   // USART6, the TMC2209 UART line (TX through a resistor, RX on the line)
const uint32_t HOST_BAUD = 115200;
const uint32_t TMC_BAUD = 19200;    // what Creality uses on this board's long UART line

// Drivers. Sense resistors are 0.11 ohm on this board. Currents are RMS, in mA.
const float RSENSE = 0.11;
const uint16_t CURRENT_CAP[N_AXES] = {1500, 1500, 1000, 500};  // about 85 % of each motor's rating
uint16_t run_ma[N_AXES] = {1000, 1000, 800, 400};              // start well below the caps
uint8_t hold_pct[N_AXES] = {70, 70, 70, 70};  // an arm joint must hold against gravity, so more than a printer's 50 %
uint16_t microsteps[N_AXES] = {16, 16, 16, 16};
bool stealth[N_AXES] = {true, true, true, true};  // stealthChop: quiet; spreadCycle (CHOP X P) gives more torque fast
bool tmc_ok[N_AXES] = {false, false, false, false};

const uint32_t ISR_HZ = 40000;          // step timer rate; one step per axis per tick at most
const float RATE_CAP = 20000.0;         // steps/s, hard ceiling for any axis
const float START_RATE = 200.0;         // steps/s at the start and end of every move
const uint32_t HOST_TIMEOUT_MS = 1000;  // a move stops if the host sends nothing for this long
const uint32_t ONE_Q24 = 1UL << 24;     // speeds are steps per tick in Q24 fixed point
const uint32_t PULSE_CYCLES = 168;      // step pulse width: 2 us at 84 MHz (the TMC2209 needs 100 ns)

// Settings the host sends at connect time. Defaults are slow on purpose.
float max_rate[N_AXES] = {1600, 1600, 1600, 1600};    // steps/s: half a turn per second at 16 microsteps
float max_accel[N_AXES] = {3200, 3200, 3200, 3200};   // steps/s^2
bool dir_invert[N_AXES] = {false, false, false, false};
bool limit_on[N_AXES] = {false, false, false, false};
int32_t limit_min[N_AXES], limit_max[N_AXES];
bool enabled = false;
bool dry_run = false;  // count steps and time the profile without touching the step pins

// IDLE/ACCEL/CRUISE/DECEL run one coordinated move; JOG runs every axis at its own commanded velocity.
enum State : uint8_t { IDLE, ACCEL, CRUISE, DECEL, JOG };
const uint32_t JOG_DEADMAN_MS = 250;  // jog velocities fall to zero unless the host repeats JV this often
const char* alarm_reason = "";

// Shared with the step interrupt.
volatile State state = IDLE;
volatile int32_t pos[N_AXES];
volatile bool move_done = false, move_stopped = false, touch_hit = false;
volatile bool stop_req = false, finishing = false, touch_mode = false;
volatile uint8_t pending = 0;  // axes to pulse at the next tick
volatile uint32_t v_q24;

uint32_t mv_steps[N_AXES], mv_n, steps_done, accel_steps, phase;
int8_t mv_sign[N_AXES];
int32_t bres[N_AXES];
uint32_t v_min_q24, v_max_q24, a_q24;

// Jog: signed velocities in Q24 steps per tick, ramped every tick.
int32_t jog_target[N_AXES], jog_v[N_AXES], jog_a[N_AXES];
uint32_t jog_abs[N_AXES], jog_phase[N_AXES];
uint8_t jog_mask = 0;

GPIO_TypeDef* step_port[N_AXES];
uint32_t step_mask[N_AXES];
GPIO_TypeDef* dir_port[N_AXES];
uint32_t dir_mask[N_AXES];
GPIO_TypeDef* probe_port;
uint32_t probe_mask;
HardwareTimer* step_timer;

uint32_t last_rx_ms = 0, last_jog_ms = 0;

// Interrupts off for a block, restored on leaving it (the AVR ATOMIC_BLOCK).
struct Atomic {
  uint32_t primask;
  Atomic() : primask(__get_PRIMASK()) { __disable_irq(); }
  ~Atomic() { __set_PRIMASK(primask); }
};

// Direction pin write that is safe and fast inside the step interrupt.
inline void set_dir(uint8_t i, int8_t sign) {
  dir_port[i]->BSRR = ((sign > 0) != dir_invert[i]) ? dir_mask[i] : dir_mask[i] << 16;
}

// TMC2209 UART -------------------------------------------------------------------------------------------------------

uint8_t tmc_crc(const uint8_t* d, uint8_t n) {
  uint8_t crc = 0;
  for (uint8_t i = 0; i < n; i++) {
    uint8_t b = d[i];
    for (uint8_t k = 0; k < 8; k++) {
      crc = ((crc >> 7) ^ (b & 1)) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
      b >>= 1;
    }
  }
  return crc;
}

// Reads what arrives on the line within timeout_ms (our own bytes come back too: TX and RX share the line).
uint8_t tmc_collect(uint8_t* buf, uint8_t max, uint8_t want, uint32_t timeout_ms) {
  uint8_t n = 0;
  uint32_t t0 = millis();
  while (n < max && millis() - t0 < timeout_ms) {
    if (TmcBus.available()) buf[n++] = TmcBus.read();
    if (n >= want) break;
  }
  return n;
}

void tmc_write(uint8_t addr, uint8_t reg, uint32_t v) {
  uint8_t d[8] = {0x05, addr, (uint8_t)(reg | 0x80), (uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8),
                  (uint8_t)v, 0};
  d[7] = tmc_crc(d, 7);
  while (TmcBus.available()) TmcBus.read();
  TmcBus.write(d, 8);
  TmcBus.flush();
  uint8_t echo[8];
  tmc_collect(echo, 8, 8, 6);
  delay(1);
}

bool tmc_read(uint8_t addr, uint8_t reg, uint32_t* v) {
  uint8_t d[4] = {0x05, addr, reg, 0};
  d[3] = tmc_crc(d, 3);
  while (TmcBus.available()) TmcBus.read();
  TmcBus.write(d, 4);
  TmcBus.flush();
  uint8_t buf[16];
  uint8_t n = tmc_collect(buf, sizeof(buf), 12, 15);  // 4 bytes of echo, then the 8-byte reply
  for (uint8_t k = 0; k + 8 <= n; k++) {
    if (buf[k] == 0x05 && buf[k + 1] == 0xFF && buf[k + 2] == reg && tmc_crc(buf + k, 7) == buf[k + 7]) {
      *v = ((uint32_t)buf[k + 3] << 24) | ((uint32_t)buf[k + 4] << 16) | ((uint32_t)buf[k + 5] << 8) | buf[k + 6];
      return true;
    }
  }
  return false;
}

// Current scale for an RMS current, as the datasheet gives it; the low sense voltage range below about 0.6 A.
void current_bits(uint16_t ma, uint8_t* cs, bool* vsense) {
  float k = 32.0f * 1.41421f * (ma / 1000.0f) * (RSENSE + 0.02f);
  float c = k / 0.325f - 1;
  *vsense = c < 16;
  if (*vsense) c = k / 0.180f - 1;
  *cs = (uint8_t)constrain((int)(c + 0.5f), 0, 31);
}

uint8_t mres_bits(uint16_t ms) {
  uint8_t r = 8;  // full steps
  for (uint16_t m = 1; m < ms && r > 0; m <<= 1) r--;
  return r;
}

const uint8_t GCONF = 0x00, GSTAT = 0x01, IFCNT = 0x02, IOIN = 0x06, IHOLD_IRUN = 0x10, TPOWERDOWN = 0x11,
              TPWMTHRS = 0x13, CHOPCONF = 0x6C, DRV_STATUS = 0x6F;

// Writes every setting of one driver and checks, through its write counter, that all of them arrived.
bool tmc_setup(uint8_t i) {
  uint32_t before, after, ioin;
  if (!tmc_read(i, IOIN, &ioin) || (ioin >> 24) != 0x21) return false;  // 0x21: a TMC2209 answering
  if (!tmc_read(i, IFCNT, &before)) return false;
  uint8_t cs, hold;
  bool vsense;
  current_bits(min(run_ma[i], CURRENT_CAP[i]), &cs, &vsense);
  hold = (uint8_t)((cs + 1) * hold_pct[i] / 100);
  hold = hold ? hold - 1 : 0;
  tmc_write(i, GSTAT, 0x07);  // clear the reset and error flags
  // pdn_disable: the PDN pin is the UART; mstep_reg_select: microsteps from here; multistep_filt; spreadCycle or not
  tmc_write(i, GCONF, (1UL << 6) | (1UL << 7) | (1UL << 8) | (stealth[i] ? 0 : (1UL << 2)));
  // toff 4, hstrt 1, hend 2, tbl 24 clocks: Creality's chopper timing for 24 V; interpolation to 256 microsteps
  uint32_t chop = 4UL | (0UL << 4) | (5UL << 7) | (1UL << 15) | ((uint32_t)vsense << 17) |
                  ((uint32_t)mres_bits(microsteps[i]) << 24) | (1UL << 28);
  tmc_write(i, CHOPCONF, chop);
  tmc_write(i, IHOLD_IRUN, (uint32_t)hold | ((uint32_t)cs << 8) | (6UL << 16));
  tmc_write(i, TPOWERDOWN, 20);  // about 0.4 s still before dropping to the holding current
  tmc_write(i, TPWMTHRS, 0);     // stealthChop, when chosen, at every speed
  if (!tmc_read(i, IFCNT, &after)) return false;
  return (uint8_t)(after - before) == 6;
}

bool tmc_setup_all() {
  bool all = true;
  for (uint8_t i = 0; i < N_AXES; i++) {
    tmc_ok[i] = tmc_setup(i);
    all = all && tmc_ok[i];
  }
  return all;
}

// A driver that lost its motor power forgets its settings; this sees it in the reset flag.
bool tmc_was_reset(uint8_t i) {
  uint32_t g;
  return !tmc_read(i, GSTAT, &g) || (g & 1);
}

void report_tmc() {
  Host.print(F("{\"tmc\":["));
  for (uint8_t i = 0; i < N_AXES; i++) {
    uint32_t drv = 0, ioin = 0, gstat = 0;
    bool ok = tmc_read(i, IOIN, &ioin) && tmc_read(i, DRV_STATUS, &drv) && tmc_read(i, GSTAT, &gstat);
    uint8_t cs;
    bool vsense;
    current_bits(min(run_ma[i], CURRENT_CAP[i]), &cs, &vsense);
    if (i) Host.print(',');
    Host.print(F("{\"axis\":\""));
    Host.print(AXIS_NAMES[i]);
    Host.print(F("\",\"answers\":"));
    Host.print(ok ? 1 : 0);
    Host.print(F(",\"set\":"));
    Host.print(tmc_ok[i] ? 1 : 0);
    Host.print(F(",\"ma\":"));
    Host.print(min(run_ma[i], CURRENT_CAP[i]));
    Host.print(F(",\"cap\":"));
    Host.print(CURRENT_CAP[i]);
    Host.print(F(",\"hold_pct\":"));
    Host.print(hold_pct[i]);
    Host.print(F(",\"ustep\":"));
    Host.print(microsteps[i]);
    Host.print(F(",\"mode\":\""));
    Host.print(stealth[i] ? "stealth" : "spread");
    Host.print(F("\""));
    if (ok) {
      // Overtemperature (warning, shutdown), shorts to ground or supply, open coils (only valid while moving slowly)
      Host.print(F(",\"cs_actual\":"));
      Host.print((drv >> 16) & 31);
      Host.print(F(",\"otpw\":"));
      Host.print(drv & 1);
      Host.print(F(",\"ot\":"));
      Host.print((drv >> 1) & 1);
      Host.print(F(",\"short\":"));
      Host.print((drv >> 2) & 15 ? 1 : 0);
      Host.print(F(",\"open_a\":"));
      Host.print((drv >> 6) & 1);
      Host.print(F(",\"open_b\":"));
      Host.print((drv >> 7) & 1);
      Host.print(F(",\"standstill\":"));
      Host.print((drv >> 31) & 1);
      Host.print(F(",\"reset\":"));
      Host.print(gstat & 1);
      Host.print(F(",\"undervoltage\":"));
      Host.print((gstat >> 2) & 1);  // uv_cp: no motor power, or too little
    }
    Host.print('}');
  }
  Host.println(F("]}"));
}

// Motion ---------------------------------------------------------------------------------------------------------------

// One tick of jog mode: every axis ramps its velocity toward its target and steps on its own phase accumulator.
// Direction flips only after the velocity passes through zero, so the pin settles long before the next pulse.
inline void jog_tick() {
  uint8_t m = 0;
  for (uint8_t i = 0; i < N_AXES; i++) {
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
    if (!(jog_mask & (1 << i))) continue;
    jog_phase[i] += jog_abs[i];
    if (jog_phase[i] >= ONE_Q24) {
      jog_phase[i] -= ONE_Q24;
      // Hard stop at a soft limit; guard_jog_limits() normally brakes before this is ever needed.
      if (limit_on[i] && ((mv_sign[i] > 0 && pos[i] >= limit_max[i]) || (mv_sign[i] < 0 && pos[i] <= limit_min[i]))) {
        continue;
      }
      m |= 1 << i;
    }
  }
  pending = m;
  if (!jog_mask) {
    state = IDLE;
    finishing = true;
  }
}

char line[96];
uint8_t line_len = 0;

void step_isr() {
  uint32_t t0 = DWT->CYCCNT;
  uint8_t out = pending;
  pending = 0;
  if (out) {
    for (uint8_t i = 0; i < N_AXES; i++) {
      if (out & (1 << i)) {
        if (!dry_run) step_port[i]->BSRR = step_mask[i];
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
    if (touch_mode && !(probe_port->IDR & probe_mask)) {
      state = IDLE;
      touch_hit = true;
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
    while (DWT->CYCCNT - t0 < PULSE_CYCLES) {
    }
    for (uint8_t i = 0; i < N_AXES; i++) {
      if (out & (1 << i)) step_port[i]->BSRR = step_mask[i] << 16;
    }
  }
}

void print_pos() {
  int32_t p[N_AXES];
  {
    Atomic a;
    for (uint8_t i = 0; i < N_AXES; i++) p[i] = pos[i];
  }
  Host.print(F("\"pos\":["));
  for (uint8_t i = 0; i < N_AXES; i++) {
    if (i) Host.print(',');
    Host.print(p[i]);
  }
  Host.print(']');
}

void reply_ok(const char* cmd) {
  Host.print(F("{\"ok\":\""));
  Host.print(cmd);
  Host.println(F("\"}"));
}

void reply_err(const char* why) {
  Host.print(F("{\"err\":\""));
  Host.print(why);
  Host.println(F("\"}"));
}

void report_status() {
  static const char* const NAMES[] = {"idle", "accel", "cruise", "decel", "jog"};
  State s = state;
  uint32_t v = 0;
  {
    Atomic a;
    if (s != JOG) {
      v = v_q24;
    } else {
      for (uint8_t i = 0; i < N_AXES; i++) {
        uint32_t m = jog_v[i] < 0 ? -jog_v[i] : jog_v[i];
        if (m > v) v = m;  // fastest axis
      }
    }
  }
  Host.print(F("{\"st\":\""));
  Host.print(NAMES[s]);
  Host.print(F("\",\"alarm\":\""));
  Host.print(alarm_reason);
  Host.print(F("\",\"en\":"));
  Host.print(enabled ? 1 : 0);
  Host.print(F(",\"dry\":"));
  Host.print(dry_run ? 1 : 0);
  Host.print(F(",\"rate\":"));
  Host.print(s == IDLE ? 0 : (uint32_t)((float)v * ISR_HZ / ONE_Q24));
  Host.print(F(",\"stops\":["));  // endstop headers X, Y, Z and the probe input, raw levels
  for (uint8_t k = 0; k < 3; k++) {
    Host.print(digitalRead(STOP_PIN[k]));
    Host.print(',');
  }
  Host.print(digitalRead(PROBE_PIN));
  Host.print(F("],"));
  print_pos();
  Host.println('}');
}

void set_enabled(bool on) {
  enabled = on;
  digitalWrite(EN_PIN, on ? LOW : HIGH);
}

void hard_stop() {
  {
    Atomic a;
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
bool start_move(const int32_t* target, float rate, bool touch) {
  if (state != IDLE || finishing) return reply_err("busy"), false;
  if (alarm_reason[0]) return reply_err("alarm"), false;
  if (!enabled && !dry_run) return reply_err("disabled"), false;
  if (!(rate > 0)) return reply_err("rate"), false;

  int32_t cur[N_AXES];
  {
    Atomic a;
    for (uint8_t i = 0; i < N_AXES; i++) cur[i] = pos[i];
  }
  uint32_t n = 0;
  for (uint8_t i = 0; i < N_AXES; i++) {
    int32_t d = target[i] - cur[i];
    if (limit_on[i] && (target[i] < limit_min[i] || target[i] > limit_max[i])) return reply_err("limit"), false;
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

  float r = min(rate, RATE_CAP), acc = 1e9;
  for (uint8_t i = 0; i < N_AXES; i++) {
    if (!mv_steps[i]) continue;
    float frac = (float)mv_steps[i] / n;
    r = min(r, max_rate[i] / frac);
    acc = min(acc, max_accel[i] / frac);
  }
  float r0 = min(r, START_RATE);

  for (uint8_t i = 0; i < N_AXES; i++) {
    if (mv_steps[i]) set_dir(i, mv_sign[i]);
  }
  delayMicroseconds(5);  // direction settles before the first pulse

  for (uint8_t i = 0; i < N_AXES; i++) bres[i] = -(int32_t)(n / 2);
  mv_n = n;
  steps_done = 0;
  accel_steps = 0;
  phase = 0;
  v_min_q24 = (uint32_t)(r0 * (ONE_Q24 / (float)ISR_HZ));
  v_max_q24 = (uint32_t)(r * (ONE_Q24 / (float)ISR_HZ));
  a_q24 = max((uint32_t)1, (uint32_t)(acc * (ONE_Q24 / ((float)ISR_HZ * ISR_HZ))));
  if (v_min_q24 < 1) v_min_q24 = 1;
  reply_ok(touch ? "TOUCH" : "MOVE");
  {
    Atomic a;
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
  {
    Atomic a;
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
    Host.print(F("{\"fw\":\"lunes-brazo\",\"ver\":\"0.1\",\"axes\":\"XYZE\",\"isr_hz\":"));
    Host.print(ISR_HZ);
    Host.println('}');
  } else if (!strcmp(s, "MOVE") || !strcmp(s, "REL") || !strcmp(s, "TOUCH")) {
    int32_t target[N_AXES];
    float rate;
    bool relative = strcmp(s, "MOVE") != 0;
    if (!parse_axes(args, target, relative, &rate)) return reply_err("syntax");
    start_move(target, rate, !strcmp(s, "TOUCH"));
  } else if (!strcmp(s, "EN")) {
    if (state != IDLE) return reply_err("busy");
    bool on = atoi(args) != 0;
    // The drivers are set up again first: they forget everything when the motor power was off.
    if (on && !tmc_setup_all()) {
      set_enabled(false);
      return reply_err("tmc");
    }
    set_enabled(on);
    reply_ok("EN");
  } else if (!strcmp(s, "DRY")) {
    if (state != IDLE) return reply_err("busy");
    dry_run = atoi(args) != 0;
    reply_ok("DRY");
  } else if (!strcmp(s, "ZERO")) {
    if (state != IDLE) return reply_err("busy");
    {
      Atomic a;
      for (uint8_t i = 0; i < N_AXES; i++) pos[i] = 0;
    }
    reply_ok("ZERO");
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
  } else if (!strcmp(s, "CUR")) {
    // CUR X <mA> [H<hold %>]: run current, capped for the motor on that socket
    if (state != IDLE) return reply_err("busy");
    int i = axis_index(args[0]);
    if (i < 0) return reply_err("axis");
    char* end;
    long ma = strtol(args + 1, &end, 10);
    if (end == args + 1 || ma <= 0) return reply_err("syntax");
    if (ma > CURRENT_CAP[i]) return reply_err("cap");
    char* h = strchr(end, 'H');
    if (h) hold_pct[i] = constrain(atoi(h + 1), 10, 100);
    run_ma[i] = ma;
    tmc_ok[i] = tmc_setup(i);
    if (!tmc_ok[i]) return reply_err("tmc");
    reply_ok("CUR");
  } else if (!strcmp(s, "MS")) {
    // MS X <1..256>: microsteps; the host must rescale its steps per degree
    if (state != IDLE) return reply_err("busy");
    int i = axis_index(args[0]);
    if (i < 0) return reply_err("axis");
    long ms = atol(args + 1);
    if (ms < 1 || ms > 256 || (ms & (ms - 1))) return reply_err("syntax");
    microsteps[i] = ms;
    tmc_ok[i] = tmc_setup(i);
    if (!tmc_ok[i]) return reply_err("tmc");
    reply_ok("MS");
  } else if (!strcmp(s, "CHOP")) {
    // CHOP X S (stealthChop, quiet) or CHOP X P (spreadCycle, more torque at speed)
    if (state != IDLE) return reply_err("busy");
    int i = axis_index(args[0]);
    if (i < 0) return reply_err("axis");
    char* p = args + 1;
    while (*p == ' ') p++;
    if (*p != 'S' && *p != 'P') return reply_err("syntax");
    stealth[i] = *p == 'S';
    tmc_ok[i] = tmc_setup(i);
    if (!tmc_ok[i]) return reply_err("tmc");
    reply_ok("CHOP");
  } else if (!strcmp(s, "TMC")) {
    // TMC: what each driver says; TMC SET: write their settings again first
    if (!strcmp(args, "SET")) {
      if (state != IDLE) return reply_err("busy");
      tmc_setup_all();
    }
    report_tmc();
  } else if (!strcmp(s, "FAN")) {
    digitalWrite(FAN_PIN, atoi(args) ? HIGH : LOW);
    reply_ok("FAN");
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
    int32_t tv[N_AXES] = {0, 0, 0, 0};
    for (char* p = args; *p;) {
      while (*p == ' ') p++;
      if (!*p) break;
      int i = axis_index(*p++);
      char* end;
      float r = strtod(p, &end);
      if (i < 0 || end == p) return reply_err("syntax");
      p = end;
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
        jog_a[i] = max((int32_t)1, (int32_t)(max_accel[i] * (ONE_Q24 / ((float)ISR_HZ * ISR_HZ))));
        jog_phase[i] = 0;
        mv_sign[i] = 0;  // forces the direction pin to be set on the first step
      }
    }
    {
      Atomic a;
      for (uint8_t i = 0; i < N_AXES; i++) jog_target[i] = tv[i];
      jog_mask |= asked;
      if (state == IDLE) {
        for (uint8_t i = 0; i < N_AXES; i++) jog_v[i] = jog_abs[i] = 0;
        jog_mask = asked;
        move_stopped = false;
        touch_hit = false;
        state = JOG;
      }
    }
  } else if (!strcmp(s, "RESET")) {
    if (state != IDLE) return reply_err("busy");
    alarm_reason = "";
    reply_ok("RESET");
  } else if (s[0]) {
    reply_err("unknown");
  }
}

void stop_jog() {
  Atomic a;
  for (uint8_t i = 0; i < N_AXES; i++) jog_target[i] = 0;
}

// Brakes a jogging axis early enough to stop at its soft limit, and refuses to jog further into it.
void guard_jog_limits() {
  for (uint8_t i = 0; i < N_AXES; i++) {
    if (!limit_on[i]) continue;
    int32_t p, v, t;
    {
      Atomic a;
      p = pos[i];
      v = jog_v[i];
      t = jog_target[i];
    }
    float vs = (float)v * ISR_HZ / ONE_Q24;                            // steps/s, signed
    float stop = vs * vs / (2 * max_accel[i]) + fabs(vs) * 0.005 + 4;  // braking distance plus loop latency
    if ((t > 0 && p + stop >= limit_max[i]) || (t < 0 && p - stop <= limit_min[i])) {
      Atomic a;
      jog_target[i] = 0;
    }
  }
}

void setup() {
  for (uint32_t pin : HEATER_PINS) {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, LOW);
  }
  pinMode(FAN_PIN, OUTPUT);
  digitalWrite(FAN_PIN, LOW);
  pinMode(EN_PIN, OUTPUT);
  set_enabled(false);
  for (uint8_t i = 0; i < N_AXES; i++) {
    pinMode(pinNametoDigitalPin(STEP_PIN[i]), OUTPUT);
    pinMode(pinNametoDigitalPin(DIR_PIN[i]), OUTPUT);
    step_port[i] = get_GPIO_Port(STM_PORT(STEP_PIN[i]));
    step_mask[i] = 1UL << STM_PIN(STEP_PIN[i]);
    dir_port[i] = get_GPIO_Port(STM_PORT(DIR_PIN[i]));
    dir_mask[i] = 1UL << STM_PIN(DIR_PIN[i]);
    step_port[i]->BSRR = step_mask[i] << 16;
    pos[i] = 0;
  }
  for (uint32_t pin : STOP_PIN) pinMode(pin, INPUT_PULLUP);
  pinMode(PROBE_PIN, INPUT_PULLUP);
  PinName probe = digitalPinToPinName(PROBE_PIN);
  probe_port = get_GPIO_Port(STM_PORT(probe));
  probe_mask = 1UL << STM_PIN(probe);

  // The cycle counter times the step pulses.
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  Host.begin(HOST_BAUD);
  TmcBus.begin(TMC_BAUD);
  delay(100);  // the drivers start up with the board
  bool drivers = tmc_setup_all();

  // TIM3: its pins on this board are plain step and direction outputs, so no PWM shares the timer
  step_timer = new HardwareTimer(TIM3);
  step_timer->setOverflow(ISR_HZ, HERTZ_FORMAT);
  step_timer->attachInterrupt(step_isr);
  step_timer->resume();

  Host.print(F("{\"ev\":\"boot\",\"fw\":\"lunes-brazo\",\"ver\":\"0.1\",\"drivers\":\""));
  for (uint8_t i = 0; i < N_AXES; i++) {
    if (tmc_ok[i]) Host.print(AXIS_NAMES[i]);
  }
  Host.print(F("\",\"drivers_ok\":"));
  Host.print(drivers ? 1 : 0);
  Host.println('}');
}

void loop() {
  while (Host.available()) {
    char c = Host.read();
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

  // Every second while still and enabled: a driver that lost its motor power forgot its current and microsteps.
  // Its motor is switched off rather than left running on the defaults.
  static uint32_t last_check = 0;
  if (enabled && state == IDLE && millis() - last_check > 1000) {
    last_check = millis();
    for (uint8_t i = 0; i < N_AXES; i++) {
      if (tmc_was_reset(i)) {
        set_enabled(false);
        alarm_reason = "tmc_reset";
        Host.print(F("{\"ev\":\"tmc_reset\",\"axis\":\""));
        Host.print(AXIS_NAMES[i]);
        Host.println(F("\"}"));
        break;
      }
    }
  }

  if (move_done) {
    move_done = false;
    Host.print(F("{\"ev\":\"done\",\"stopped\":"));
    Host.print(move_stopped ? 1 : 0);
    Host.print(F(",\"touch\":"));
    Host.print(touch_hit ? 1 : 0);
    Host.print(',');
    print_pos();
    Host.println('}');
  }
}
