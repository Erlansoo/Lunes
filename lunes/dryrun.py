"""A dry run of a microcontroller's own code on its simulated circuit: no board, nothing flashed.

The code Lunes wrote for the board (the part's "code", a whole Arduino sketch) is preprocessed by arduino-cli, as
the IDE does (it adds the function prototypes), and compiled for this PC against a small stand-in for the Arduino
core: pinMode, digitalRead/Write, analogRead/Write, millis/micros/delay, Serial, String, map, constrain, random,
tone. Time is simulated, so a two-second run takes a moment.

The run is a sequence of situations ("cases": the window closed, then open, then closed again), one after another
without resetting the board, so what the code remembers carries over (a latched alarm, hysteresis, a counter). For
each case ngspice simulates the circuit with the case's changes (a pot's position, a source's volts) and the pins
as the code drives them; the voltage on each pin becomes what analogRead and digitalRead return (with a count of
noise, and anywhere in its swing when the net moves). What the code then does with its outputs goes back into the
circuit and the case is simulated again, until the pins and the circuit agree (an output that feeds an input).

Returned per case: what each input read, what each output did (its level, or how often it toggled), the LEDs'
currents, what the code printed, and any expectation that failed.

Safety: the code runs on this PC, so it must only use the Arduino API. It may include Arduino.h and the C headers
the AVR core has (math, string, stdlib, stdint, ctype, avr/pgmspace); anything that reaches files, processes or the
network is refused before compiling, and the run is limited to a few seconds of CPU and 512 MB. On Windows it
compiles and runs in WSL (Ubuntu), on Linux directly.
"""
import copy
import hashlib
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

from lunes import mcu

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "runs" / "dryrun"
WSL_DISTRO = "Ubuntu-24.04"
ALLOWED_INCLUDES = {"Arduino.h", "math.h", "string.h", "stdlib.h", "stdint.h", "ctype.h", "avr/pgmspace.h",
                    "stdbool.h", "limits.h"}
FORBIDDEN = re.compile(r"(?<![.\w>])(system|popen|fork|exec[lv]p?e?|fopen|freopen|open|unlink|remove|rename|socket|connect|"
                       r"ofstream|ifstream|fstream|asm|__asm__|dlopen|mmap|ptrace|kill|getenv|chdir|_exit)\s*\(|"
                       r"\b(std::filesystem|FILE\s*\*)")
PIN_NUMBERS = {**{f"D{i}": i for i in range(14)}, **{f"A{i}": 14 + i for i in range(8)}}

HEADER = r"""
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <string>
typedef uint8_t byte;
typedef bool boolean;
typedef uint16_t word;
#define HIGH 1
#define LOW 0
#define INPUT 0
#define OUTPUT 1
#define INPUT_PULLUP 2
#define LED_BUILTIN 13
#define A0 14
#define A1 15
#define A2 16
#define A3 17
#define A4 18
#define A5 19
#define A6 20
#define A7 21
#define DEC 10
#define HEX 16
#define OCT 8
#define BIN 2
#define CHANGE 1
#define FALLING 2
#define RISING 3
#define DEFAULT 1
#define EXTERNAL 0
#define INTERNAL 3
#define PI 3.1415926535897932384626433832795
#define HALF_PI 1.5707963267948966192313216916398
#define TWO_PI 6.283185307179586476925286766559
#define DEG_TO_RAD 0.017453292519943295769236907684886
#define RAD_TO_DEG 57.295779513082320876798154814105
#define PROGMEM
#define F(s) (s)
#define PSTR(s) (s)
#define pgm_read_byte(a) (*(const uint8_t *)(a))
#define pgm_read_word(a) (*(const uint16_t *)(a))
#define constrain(x, a, b) ((x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))
#define sq(x) ((x) * (x))
#define radians(d) ((d) * DEG_TO_RAD)
#define degrees(r) ((r) * RAD_TO_DEG)
#define bitRead(v, b) (((v) >> (b)) & 0x01)
#define bitSet(v, b) ((v) |= (1UL << (b)))
#define bitClear(v, b) ((v) &= ~(1UL << (b)))
#define bitWrite(v, b, x) ((x) ? bitSet(v, b) : bitClear(v, b))
#define bit(b) (1UL << (b))
#define lowByte(w) ((uint8_t)((w) & 0xff))
#define highByte(w) ((uint8_t)((w) >> 8))
#define digitalPinToInterrupt(p) ((p) == 2 ? 0 : ((p) == 3 ? 1 : -1))
#ifndef abs
#define abs(x) ((x) > 0 ? (x) : -(x))
#endif
template <class A, class B> auto min(A a, B b) -> decltype(a < b ? a : b) { return a < b ? a : b; }
template <class A, class B> auto max(A a, B b) -> decltype(a > b ? a : b) { return a > b ? a : b; }

unsigned long millis();
unsigned long micros();
void delay(unsigned long ms);
void delayMicroseconds(unsigned int us);
void pinMode(uint8_t pin, uint8_t mode);
void digitalWrite(uint8_t pin, uint8_t value);
int digitalRead(uint8_t pin);
int analogRead(uint8_t pin);
void analogWrite(uint8_t pin, int value);
void analogReference(uint8_t mode);
void tone(uint8_t pin, unsigned int hz, unsigned long ms = 0);
void noTone(uint8_t pin);
unsigned long pulseIn(uint8_t pin, uint8_t state, unsigned long timeout = 1000000UL);
void attachInterrupt(uint8_t n, void (*f)(), int mode);
void detachInterrupt(uint8_t n);
inline void interrupts() {}
inline void noInterrupts() {}
inline void yield() {}
long random(long hi);
long random(long lo, long hi);
void randomSeed(unsigned long seed);
inline long map(long x, long a, long b, long c, long d) { return (x - a) * (d - c) / (b - a) + c; }

class String {
 public:
  std::string s;
  String() {}
  String(const char *c) : s(c ? c : "") {}
  String(const std::string &c) : s(c) {}
  String(char c) : s(1, c) {}
  String(int v, int base = DEC) { s = num((long)v, base); }
  String(unsigned int v, int base = DEC) { s = num((long)v, base); }
  String(long v, int base = DEC) { s = num(v, base); }
  String(unsigned long v, int base = DEC) { s = num((long)v, base); }
  String(double v, int digits = 2) { char b[64]; snprintf(b, sizeof b, "%.*f", digits, v); s = b; }
  static std::string num(long v, int base) {
    if (base == DEC) return std::to_string(v);
    std::string o; unsigned long u = (unsigned long)v;
    do { o.insert(o.begin(), "0123456789ABCDEF"[u % base]); u /= base; } while (u);
    return o;
  }
  unsigned int length() const { return s.size(); }
  const char *c_str() const { return s.c_str(); }
  long toInt() const { return atol(s.c_str()); }
  float toFloat() const { return atof(s.c_str()); }
  char charAt(unsigned int i) const { return i < s.size() ? s[i] : 0; }
  char operator[](unsigned int i) const { return charAt(i); }
  int indexOf(char c, unsigned int from = 0) const { auto p = s.find(c, from); return p == std::string::npos ? -1 : (int)p; }
  int indexOf(const String &t, unsigned int from = 0) const { auto p = s.find(t.s, from); return p == std::string::npos ? -1 : (int)p; }
  String substring(unsigned int a) const { return a < s.size() ? String(s.substr(a)) : String(); }
  String substring(unsigned int a, unsigned int b) const { return a < s.size() && b > a ? String(s.substr(a, b - a)) : String(); }
  bool startsWith(const String &t) const { return s.rfind(t.s, 0) == 0; }
  bool endsWith(const String &t) const { return s.size() >= t.s.size() && s.compare(s.size() - t.s.size(), t.s.size(), t.s) == 0; }
  bool equals(const String &t) const { return s == t.s; }
  bool equalsIgnoreCase(const String &t) const { if (s.size() != t.s.size()) return false; for (size_t i = 0; i < s.size(); i++) if (tolower(s[i]) != tolower(t.s[i])) return false; return true; }
  void trim() { size_t a = s.find_first_not_of(" \t\r\n"); size_t b = s.find_last_not_of(" \t\r\n"); s = a == std::string::npos ? "" : s.substr(a, b - a + 1); }
  void toUpperCase() { for (auto &c : s) c = toupper(c); }
  void toLowerCase() { for (auto &c : s) c = tolower(c); }
  void replace(const String &a, const String &b) { if (a.s.empty()) return; size_t p = 0; while ((p = s.find(a.s, p)) != std::string::npos) { s.replace(p, a.s.size(), b.s); p += b.s.size(); } }
  bool concat(const String &t) { s += t.s; return true; }
  String &operator+=(const String &t) { s += t.s; return *this; }
  bool operator==(const String &t) const { return s == t.s; }
  bool operator!=(const String &t) const { return s != t.s; }
  bool operator<(const String &t) const { return s < t.s; }
};
inline String operator+(const String &a, const String &b) { return String(a.s + b.s); }
inline String operator+(const String &a, const char *b) { return String(a.s + b); }
inline String operator+(const char *a, const String &b) { return String(std::string(a) + b.s); }
inline String operator+(const String &a, char b) { return String(a.s + b); }
inline String operator+(const String &a, int b) { return a + String(b); }
inline String operator+(const String &a, long b) { return a + String(b); }
inline String operator+(const String &a, unsigned long b) { return a + String(b); }
inline String operator+(const String &a, double b) { return a + String(b); }

class HardwareSerial {
 public:
  void begin(unsigned long baud, int config = 0);
  void end() {}
  operator bool() const { return true; }
  int available();
  int read();
  int peek();
  void flush() {}
  void setTimeout(unsigned long) {}
  size_t write(uint8_t c);
  size_t write(const char *t) { size_t n = 0; while (*t) n += write((uint8_t)*t++); return n; }
  size_t print(const String &t) { return write(t.c_str()); }
  size_t print(const char *t) { return write(t); }
  size_t print(char c) { return write((uint8_t)c); }
  size_t print(int v, int base = DEC) { return print(String((long)v, base)); }
  size_t print(unsigned int v, int base = DEC) { return print(String((long)v, base)); }
  size_t print(long v, int base = DEC) { return print(String(v, base)); }
  size_t print(unsigned long v, int base = DEC) { return print(String(v, base)); }
  size_t print(unsigned char v, int base = DEC) { return print(String((long)v, base)); }
  size_t print(double v, int digits = 2) { return print(String(v, digits)); }
  size_t println() { return write((uint8_t)'\n'); }
  template <class T> size_t println(const T &v) { size_t n = print(v); return n + println(); }
  template <class T> size_t println(const T &v, int f) { size_t n = print(v, f); return n + println(); }
  String readString();
  String readStringUntil(char end);
  long parseInt();
  float parseFloat();
};
extern HardwareSerial Serial;
"""

HARNESS = r"""
// The stand-in for the board: simulated time, pins read from the circuit's simulation, events on stdout.
static const int NPIN = 22;
struct Seg { unsigned long long start_us; double lo[NPIN], hi[NPIN], share[NPIN]; std::string serial; };
static std::vector<Seg> segs;
static int seg = -1;
static unsigned long long t_us = 0, end_us = 0;
static int mode_[NPIN], out_[NPIN], pwm_[NPIN], toned_[NPIN], touched_[NPIN];
static unsigned long long high_us[NPIN], last_us[NPIN];
static int toggles[NPIN];
static std::string rx, line;
static long serial_lines = 0;
static unsigned long long rng = 88172645463325252ULL;
static unsigned long long xr() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double ur() { return (xr() >> 11) * (1.0 / 9007199254740992.0); }
static int is_high(int p) { return mode_[p] == OUTPUT && (out_[p] || pwm_[p] > 0); }
static void account(int p) { if (is_high(p)) high_us[p] += t_us - last_us[p]; last_us[p] = t_us; }
static void close_seg() {
  for (int p = 0; p < NPIN; p++) {
    if (!touched_[p]) continue;
    account(p);
    unsigned long long span = t_us - segs[seg].start_us;
    printf("F %d %d %d %d %d %.4f %d\n", seg, p, mode_[p], out_[p], pwm_[p], span ? (double)high_us[p] / span : 0.0, toggles[p]);
    high_us[p] = 0; toggles[p] = 0;
  }
  if (!line.empty()) { printf("S %d %s\n", seg, line.c_str()); line.clear(); }
  printf("E %d %llu\n", seg, t_us);
}
static void finish() { close_seg(); fflush(stdout); exit(0); }
static void open_seg(int k) { seg = k; rx += segs[k].serial; for (int p = 0; p < NPIN; p++) last_us[p] = t_us; }
static void advance(unsigned long long us) {
  unsigned long long target = t_us + us;
  while (seg + 1 < (int)segs.size() && target >= segs[seg + 1].start_us) {
    t_us = segs[seg + 1].start_us; close_seg(); open_seg(seg + 1);
  }
  t_us = target;
  if (t_us >= end_us) { t_us = end_us; finish(); }
}
unsigned long millis() { advance(4); return (unsigned long)(t_us / 1000); }
unsigned long micros() { advance(4); return (unsigned long)t_us; }
void delay(unsigned long ms) { advance((unsigned long long)ms * 1000); }
void delayMicroseconds(unsigned int us) { advance(us); }
static int okpin(uint8_t p) { return p < NPIN; }
void pinMode(uint8_t p, uint8_t m) { if (!okpin(p)) return; account(p); touched_[p] = 1; mode_[p] = m; }
void digitalWrite(uint8_t p, uint8_t v) {
  if (!okpin(p)) return; account(p); touched_[p] = 1;
  if (mode_[p] != OUTPUT) { mode_[p] = v ? INPUT_PULLUP : INPUT; return; }  // as on the AVR: writing an input sets its pull-up
  int now = v ? 1 : 0; if (now != out_[p] || pwm_[p]) toggles[p]++;
  out_[p] = now; pwm_[p] = 0; advance(5);
}
int digitalRead(uint8_t p) {
  if (!okpin(p)) return 0; advance(5);
  if (mode_[p] == OUTPUT) return out_[p];
  double s = segs[seg].share[p]; return ur() < s ? HIGH : LOW;
}
int analogRead(uint8_t p) {
  if (p < 8) p += 14; if (!okpin(p)) return 0; advance(112);
  double lo = segs[seg].lo[p], hi = segs[seg].hi[p];
  double v = lo + (hi - lo) * ur() + (ur() - 0.5) * 2.0;  // anywhere in its swing, and a count of noise
  int c = (int)lround(v); return c < 0 ? 0 : (c > 1023 ? 1023 : c);
}
void analogWrite(uint8_t p, int v) {
  if (!okpin(p)) return; account(p); touched_[p] = 1; mode_[p] = OUTPUT;
  v = v < 0 ? 0 : (v > 255 ? 255 : v);
  if (v != pwm_[p] || out_[p]) toggles[p]++;
  if (v == 0 || v == 255) { out_[p] = v ? 1 : 0; pwm_[p] = 0; } else { pwm_[p] = v; out_[p] = 0; }
}
void analogReference(uint8_t) {}
void tone(uint8_t p, unsigned int hz, unsigned long) { if (okpin(p)) { touched_[p] = 1; toned_[p] = hz; mode_[p] = OUTPUT; printf("T %d %d %u\n", seg, p, hz); } }
void noTone(uint8_t p) { if (okpin(p)) { toned_[p] = 0; printf("T %d %d 0\n", seg, p); } }
unsigned long pulseIn(uint8_t, uint8_t, unsigned long timeout) { advance(timeout); return 0; }
void attachInterrupt(uint8_t n, void (*)(), int) { printf("I %d %d\n", seg, n); }
void detachInterrupt(uint8_t) {}
long random(long hi) { return hi > 0 ? (long)(xr() % (unsigned long long)hi) : 0; }
long random(long lo, long hi) { return hi > lo ? lo + random(hi - lo) : lo; }
void randomSeed(unsigned long s) { rng = s ? s * 2654435761ULL : 88172645463325252ULL; }
HardwareSerial Serial;
void HardwareSerial::begin(unsigned long baud, int) { printf("B %d %lu\n", seg, baud); }
int HardwareSerial::available() { advance(2); return rx.size(); }
int HardwareSerial::read() { advance(2); if (rx.empty()) return -1; int c = (unsigned char)rx[0]; rx.erase(0, 1); return c; }
int HardwareSerial::peek() { return rx.empty() ? -1 : (unsigned char)rx[0]; }
size_t HardwareSerial::write(uint8_t c) {
  if (c == '\n') { if (serial_lines++ < 4000) printf("S %d %s\n", seg, line.c_str()); line.clear(); }
  else if (c != '\r' && line.size() < 300) line += (char)c;
  advance(87);  // a byte at 115200 baud
  return 1;
}
String HardwareSerial::readString() { String o(rx); rx.clear(); return o; }
String HardwareSerial::readStringUntil(char e) { auto p = rx.find(e); String o(rx.substr(0, p)); rx = p == std::string::npos ? "" : rx.substr(p + 1); return o; }
long HardwareSerial::parseInt() { size_t i = 0; while (i < rx.size() && !isdigit(rx[i]) && rx[i] != '-') i++; rx.erase(0, i); size_t n = 0; long v = strtol(rx.c_str(), nullptr, 10); while (n < rx.size() && (isdigit(rx[n]) || (n == 0 && rx[n] == '-'))) n++; rx.erase(0, n); return v; }
float HardwareSerial::parseFloat() { char *e; float v = strtof(rx.c_str(), &e); rx.erase(0, e - rx.c_str()); return v; }

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "r"); if (!f) return 2;
  char kind[8];
  while (fscanf(f, "%7s", kind) == 1) {
    if (!strcmp(kind, "SEG")) { Seg s{}; fscanf(f, "%llu", &s.start_us); for (int p = 0; p < NPIN; p++) { s.lo[p] = 0; s.hi[p] = 0; s.share[p] = 0.5; } segs.push_back(s); }
    else if (!strcmp(kind, "PIN")) { int p; double lo, hi, sh; fscanf(f, "%d %lf %lf %lf", &p, &lo, &hi, &sh); if (okpin(p)) { segs.back().lo[p] = lo; segs.back().hi[p] = hi; segs.back().share[p] = sh; } }
    else if (!strcmp(kind, "RX")) { char hex[4096]; fscanf(f, "%4095s", hex); for (size_t i = 0; hex[i] && hex[i + 1]; i += 2) { unsigned v; sscanf(hex + i, "%2x", &v); segs.back().serial += (char)v; } }
    else if (!strcmp(kind, "END")) fscanf(f, "%llu", &end_us);
  }
  fclose(f);
  if (segs.empty()) return 2;
  open_seg(0);
  setup();
  for (;;) { loop(); advance(10); }
}
"""


def _wsl_path(p):
    p = Path(p).resolve()
    return f"/mnt/{p.drive[0].lower()}{p.as_posix()[2:]}"


def _shell(script, timeout):
    """Runs a bash script on Linux, or in WSL on Windows: (returncode, output)."""
    if shutil.which("g++") and not shutil.which("wsl"):
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout)
    else:
        r = subprocess.run(["wsl", "-d", WSL_DISTRO, "--", "bash", "-c", script], capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _path(p):
    return _wsl_path(p) if shutil.which("wsl") else str(Path(p).resolve())


def safety(code):
    """Problems that keep the code from running on this PC: includes outside the Arduino core, and calls that reach
    files, processes or the network. [] when it may run."""
    problems = []
    for inc in re.findall(r"#\s*include\s*[<\"]([^>\"]+)[>\"]", code):
        if inc not in ALLOWED_INCLUDES:
            problems.append(f"#include <{inc}>: the dry run has only the Arduino core, not this library; the board "
                            f"compile still checks it, but its behaviour cannot be tried here")
    m = FORBIDDEN.search(re.sub(r"//[^\n]*|/\*.*?\*/|\"(\\.|[^\"\\])*\"", "", code, flags=re.S))
    if m:
        problems.append(f"'{m.group(0).strip()}' is not something a sketch does: the dry run does not run it")
    return problems


def build(code, kind="ARDUINO_NANO"):
    """Preprocesses the sketch with arduino-cli and compiles it for this PC: {"ok", "exe"} or {"ok": False,
    "errors"}. Cached by the code's hash."""
    if not mcu.BOARDS[kind]["fqbn"].startswith("arduino:avr"):
        return {"ok": False, "errors": [f"the dry run knows the Uno and the Nano; a {mcu.BOARDS[kind]['name']} is "
                                        f"only compiled"]}
    problems = safety(code)
    if problems:
        return {"ok": False, "errors": problems}
    key = hashlib.sha1(code.encode()).hexdigest()[:12]
    folder = WORK / key
    exe = folder / "sketch_host"
    if exe.exists():
        return {"ok": True, "exe": exe}
    name = f"dry_{key}"
    src = folder / name
    src.mkdir(parents=True, exist_ok=True)
    (src / f"{name}.ino").write_text(code, encoding="utf-8")
    rc, pre = mcu._cli("compile", "--fqbn", mcu.BOARDS[kind]["fqbn"], "--preprocess", str(src), timeout=120)
    if rc != 0:
        return {"ok": False, "errors": [l for l in pre.splitlines() if "error" in l.lower()][:8] or pre.splitlines()[-8:]}
    pre = re.sub(r'#line (\d+) "[^"]*"', r'#line \1 "sketch.ino"', pre)
    (folder / "Arduino.h").write_text(HEADER, encoding="utf-8")
    (folder / "avr").mkdir(exist_ok=True)
    (folder / "avr" / "pgmspace.h").write_text("#pragma once\n", encoding="utf-8")
    (folder / "sketch.cpp").write_text(pre + "\n", encoding="utf-8")
    (folder / "harness.cpp").write_text('#include <vector>\n#include <map>\n#include "Arduino.h"\nvoid setup();\n'
                                        'void loop();\n' + HARNESS, encoding="utf-8")
    d = _path(folder)
    rc, out = _shell(f"cd '{d}' && g++ -std=gnu++17 -O1 -w -I. -o sketch_host sketch.cpp harness.cpp 2>&1", 120)
    if rc != 0:
        errors = [l for l in out.splitlines() if "error" in l][:8] or out.splitlines()[-8:]
        return {"ok": False, "errors": errors}
    return {"ok": True, "exe": exe}


def _pin_inputs(part, nets):
    """{pin number: (lowest counts, highest counts, share of reads HIGH)} for every connected pin of the board,
    from the simulated nets."""
    board = mcu.BOARDS[part["kind"]]
    out = {}
    for pin, net in part["pins"].items():
        if pin not in PIN_NUMBERS or net not in nets:
            continue
        m = nets[net]
        lo, hi = mcu._counts(m.get("min_v", m["mean_v"]), board), mcu._counts(m.get("max_v", m["mean_v"]), board)
        level, low_l, high_l = (mcu._level(m["mean_v"], board), mcu._level(m.get("min_v", m["mean_v"]), board),
                                mcu._level(m.get("max_v", m["mean_v"]), board))
        if m.get("frequency_hz") and low_l == 0 and high_l == 1:
            share = float(m.get("duty") or 0.5)
        else:
            share = 0.5 if level is None else float(level)
        out[PIN_NUMBERS[pin]] = (lo, hi, share)
    return out


def _run(exe, segments, total_us):
    """Runs the compiled sketch over the segments: [{"pins": {number: (lo, hi, share)}, "serial": text, "start_us"}]."""
    lines = []
    for s in segments:
        lines.append(f"SEG {s['start_us']}")
        for p, (lo, hi, share) in s["pins"].items():
            lines.append(f"PIN {p} {lo} {hi} {share}")
        if s.get("serial"):
            lines.append(f"RX {s['serial'].encode().hex()}")
    lines.append(f"END {total_us}")
    inputs = exe.parent / "inputs.txt"
    inputs.write_text("\n".join(lines) + "\n", encoding="utf-8")
    d = _path(exe.parent)
    rc, out = _shell(f"cd '{d}' && ulimit -t 8 -v 524288 && timeout 10 ./sketch_host inputs.txt", 30)
    return rc, out


def _parse(out, n):
    segs = [{"pins": {}, "serial": [], "tones": {}, "ended": False} for _ in range(n)]
    for raw in out.splitlines():
        kind, _, rest = raw.partition(" ")
        if kind == "S":
            k, _, text = rest.partition(" ")
            if k.lstrip("-").isdigit() and 0 <= int(k) < n:
                segs[int(k)]["serial"].append(text)
        elif kind == "F":
            k, p, mode, out_, pwm, share, toggles = rest.split()
            segs[int(k)]["pins"][int(p)] = {"mode": int(mode), "out": int(out_), "pwm": int(pwm),
                                            "high_share": float(share), "toggles": int(toggles)}
        elif kind == "T":
            k, p, hz = rest.split()
            segs[int(k)]["tones"][int(p)] = int(hz)
        elif kind == "E":
            segs[int(rest.split()[0])]["ended"] = True
    return segs


NAMES = {v: k for k, v in PIN_NUMBERS.items()}


def _program_from(part, seg, ms):
    """The pin program that matches what the code did with each pin in a segment, to simulate the circuit with it."""
    board = mcu.BOARDS[part["kind"]]
    program = dict(part.get("program") or {})
    for pin in part["pins"]:  # a pin the code never touches stays an input at reset: it floats
        if pin in PIN_NUMBERS and PIN_NUMBERS[pin] not in seg["pins"] and mcu.mode_of(program.get(pin))[0] in mcu.OUTPUTS:
            program[pin] = "input"
    for number, st in seg["pins"].items():
        pin = NAMES.get(number)
        if pin not in part["pins"]:
            continue
        if st["mode"] == 1:  # OUTPUT
            if st["pwm"]:
                program[pin] = {"pwm": round(st["pwm"] / 255, 3)} if pin in board["pwm"] else (
                    "high" if st["pwm"] >= 128 else "low")
            elif st["toggles"] >= 4:  # it blinks: simulate the share of time it is HIGH as a PWM, steady enough
                hz = st["toggles"] / 2 / (ms / 1000)
                program[pin] = {"square": round(min(hz, 5000), 3), "duty": round(st["high_share"], 3)} \
                    if hz >= 20 else ("high" if st["out"] else "low")
            else:
                program[pin] = "high" if st["out"] else "low"
        else:
            current = mcu.mode_of(program.get(pin))[0]
            if st["mode"] == 2:
                program[pin] = "input_pullup"
            elif current not in ("input", "analog_in"):
                program[pin] = "input"
    return program


def _apply(circuit, changes):
    """A copy of the circuit with a case's changes: {ref: {field: value}} merged into the parts."""
    c = copy.deepcopy(circuit)
    for ref, fields in (changes or {}).items():
        part = next((p for p in c["parts"] if p["ref"] == ref), None)
        if part is None:
            raise ValueError(f"the circuit has no part {ref}")
        if not isinstance(fields, dict):
            fields = {"value": fields}
        part.update(fields)
    return c


def _check(expect, row):
    """The expectations of a case that the dry run did not meet."""
    failed = []
    for key, want in (expect or {}).items():
        if key == "serial_contains":
            if not any(str(want) in line for line in row["serial"]):
                failed.append(f"nothing printed contains {want!r}")
        elif key in row["outputs"] or key in row["untouched"]:
            got = row["outputs"][key]["level"] if key in row["outputs"] else "not driven by the code (it floats)"
            if not got.lower().startswith(str(want).lower()):
                failed.append(f"{key} should be {want}, it is {got}")
        elif key in row["leds"]:
            on = row["leds"][key]["on"]
            if (str(want).lower() in ("on", "true", "1", "encendido")) != on:
                failed.append(f"{key} should be {want}, it is {'on' if on else 'off'} "
                              f"({row['leds'][key]['ma']} mA)")
        else:
            failed.append(f"{key}: not a pin, LED or serial_contains of this circuit")
    return failed


def dry_run(circuit, ref, cases, ms=1000, simulate=None):
    """Runs the board's code over the cases in order, without a reset between them. cases: [{"label", "set":
    {ref: {field: value}}, "serial": text sent to the board at the start, "expect": {pin: HIGH|LOW|blinks, LED: on|off,
    "serial_contains": text}, "ms"}]. simulate(circuit) -> lunes.simulator.expected's result (injected in tests)."""
    if simulate is None:
        from lunes.simulator import expected as simulate
    part = next((p for p in circuit["parts"] if p["ref"] == ref), None)
    if part is None or part["kind"] not in mcu.BOARDS:
        return {"ok": False, "error": f"no board {ref} in the circuit"}
    code = part.get("code") or mcu.sketch(part, circuit.get("name", "circuit"))
    built = build(code, part["kind"])
    if not built["ok"]:
        return {"ok": False, "stage": "compile for the dry run", "errors": built["errors"]}
    cases = cases or [{"label": "as designed"}]
    spans = [int(float(c.get("ms", ms)) * 1000) for c in cases]
    starts = [sum(spans[:k]) for k in range(len(cases))]
    try:
        benches = [_apply(circuit, c.get("set")) for c in cases]
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    programs = [copy.deepcopy(part.get("program") or {}) for _ in cases]
    cache, sims, segs, passes, rc, out = {}, [None] * len(cases), None, 0, 0, ""
    for passes in range(1, 5):
        for k, bench in enumerate(benches):
            board = next(p for p in bench["parts"] if p["ref"] == ref)
            board["program"] = programs[k]
            board.pop("code", None)
            key = json.dumps(bench, sort_keys=True)
            if key not in cache:
                cache[key] = simulate(bench)
            sims[k] = cache[key]
        segments = [{"start_us": starts[k], "pins": _pin_inputs(part, sims[k]["nets"]),
                     "serial": cases[k].get("serial", "")} for k in range(len(cases))]
        rc, out = _run(built["exe"], segments, sum(spans))
        segs = _parse(out, len(cases))
        new = [_program_from(part, segs[k], spans[k] / 1000) for k in range(len(cases))]
        if new == programs:
            break
        programs = new
    settled = new == programs
    if rc not in (0,) and not segs[-1]["ended"]:
        stuck = next((k for k, s in enumerate(segs) if not s["ended"]), 0)
        why = ("it ran out of CPU time: an endless loop that never reads a pin or the time (a while loop "
               "waiting for something that cannot change)" if rc in (9, 24, 124, 137, 152) else
               f"it crashed (exit code {rc}): an array out of bounds, a null pointer, or dividing by zero")
        return {"ok": False, "stage": "run", "error": f"the code stopped answering in case {stuck + 1} "
                                                      f"({cases[stuck].get('label', '')}): {why}",
                "output": out.splitlines()[-6:]}
    rows = []
    board_def = mcu.BOARDS[part["kind"]]
    for k, case in enumerate(cases):
        sim, seg = sims[k], segs[k]
        inputs, outputs = {}, {}
        for pin, net in part["pins"].items():
            if pin not in PIN_NUMBERS or net not in sim["nets"]:
                continue
            st = seg["pins"].get(PIN_NUMBERS[pin])
            v = sim["nets"][net]["mean_v"]
            if st and st["mode"] == 1:
                if st["toggles"] >= 4:
                    hz = st["toggles"] / 2 / (spans[k] / 1e6)
                    level = f"blinks about {hz:.3g} Hz, HIGH {st['high_share']:.0%} of the time"
                elif st["pwm"]:
                    level = f"PWM {st['pwm']}/255"
                else:
                    level = "HIGH" if st["out"] else "LOW"
                outputs[pin] = {"net": net, "level": level, "volts": round(v, 2) + 0.0}
                if PIN_NUMBERS[pin] in seg["tones"] and seg["tones"][PIN_NUMBERS[pin]]:
                    outputs[pin]["tone_hz"] = seg["tones"][PIN_NUMBERS[pin]]
            else:
                lvl = mcu._level(v, board_def)
                inputs[pin] = {"net": net, "volts": round(v, 3) + 0.0}
                if pin.startswith("A"):
                    inputs[pin]["analogRead"] = mcu._counts(v, board_def)
                else:
                    inputs[pin]["digitalRead"] = {1: "HIGH", 0: "LOW"}.get(lvl, "either (between the thresholds)")
        leds = {p["ref"]: {"ma": round(sim["currents"].get(p["ref"], {}).get("mean_a", 0.0) * 1e3, 2)}
                for p in circuit["parts"] if p["kind"] == "LED"}
        for led in leds.values():
            led["on"] = led["ma"] >= 0.5
        untouched = [pin for pin in part["pins"] if pin in PIN_NUMBERS and PIN_NUMBERS[pin] not in seg["pins"]
                     and mcu.mode_of((part.get("program") or {}).get(pin))[0] in mcu.OUTPUTS]
        row = {"case": case.get("label", f"case {k + 1}"), "set": case.get("set") or {}, "inputs": inputs,
               "outputs": outputs, "leds": leds, "serial": seg["serial"][:12], "untouched": untouched}
        if untouched:
            row["warning"] = (f"the circuit's program drives {', '.join(untouched)} but the code never sets "
                              f"{'it' if len(untouched) == 1 else 'them'}: on the board they float")
        if len(seg["serial"]) > 12:
            row["serial_lines"] = len(seg["serial"])
        if case.get("expect"):
            row["failed"] = _check(case["expect"], {**row, "serial": seg["serial"]})
        for key in ("untouched", "failed", "set"):
            if not row.get(key):
                row.pop(key, None)
        rows.append(row)
    result = {"ok": True, "board": ref, "cases": rows, "passes": passes, "settled": settled,
              "all_expectations_met": all(not r.get("failed") for r in rows),
              "note": "simulated time, from one reset, case after case; ints are 32 bits here and 16 on the AVR "
                      "(values past 32767 overflow on the board), and libraries beyond the core are not simulated"}
    if not settled:
        result["warning"] = ("the outputs and the circuit never agreed (an output that changes an input that changes "
                             "it back): the code oscillates through the circuit")
    if not part.get("code"):
        result["note"] += "; this is Lunes' own pin-program sketch: the board has no code of its own yet"
    return result
