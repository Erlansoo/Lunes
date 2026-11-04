# Case study: the real bench on the morning of 3 November 2026

What Lunes did on Erlan's bench between about 09:59 and 12:00 on 3 November 2026, written for the hackathon judges
from the files the session left behind. Every number here is read from one of those files or from re-running the
scripts named below on the saved pictures and the simulated bench; where a number was said only out loud or on the
screen and reached no file, it is left out and the text says so. What failed is written down with what was done about
it, because the point of the morning was to find out whether the physical loop works, not to make it look as if it did.

Sources, all in this repository unless said otherwise: `runs/api_calls.jsonl` and `runs/hume_calls.jsonl` (every
model call, with model, purpose, effort and seconds), `runs/verifier.jsonl` (every answer Lunes gave, with the
verifier's check), `runs/machine_events.jsonl`, `config/machine.json`, `config/inventory.json`, `runs/circuit.json`
and `runs/circuits/`, `runs/sketches/`, `runs/fault_dictionaries/`, `runs/reports/`, `lunes/report.py`,
`lunes/inventory.py`, `lunes/bench.py`, `lunes/hume.py`, `tests/`, `.github/workflows/tests.yml`, and the scripts the
operator's Claude Code session wrote outside the repository to drive the bench and the simulated bench
(`calibrate_bench_now.py`, `teach_height_now.py`, `find_chip_xy.py`, `check_bench_cam.py`, `ring_debug.py`,
`lm393_nano_demo.py`, `report_demo.py`, `circuits_demo.py`, `ci_test2.sh`). The web lab's conversation column is
kept in memory (the last 500 events) and printed to the console; it is not written to disk, so the spoken exchange
of 10:48 to 10:52 is reported as it was seen on that page, with the model-call log as its trace on disk.

## The bench

- **The machine.** A hand-built CNC with three axes (Y is a belt axis with two motors) driven by the motion firmware
  written for this project. The USB microscope is its head: 640 x 480 pixels, 38.37 px/mm in X and 38.21 px/mm in Y
  (`config/machine.json`, `microscope`), with a ring light round the lens. Z = 0 is the focused surface of a board on
  the bed; the breadboard focuses at Z 13.7 and a chip on it at Z 16.5 to 17.1. A 5 mW laser pointer rides on the
  carriage and marks holes and pins with a dot checked by the microscope.
- **The eyes.** A 4K webcam beside the bench (EMEET S800, 3840 x 2160) sees the whole table; an Xbox 360 Kinect, read
  through Lunes' own USB driver, gives depth and checks the head's position after every move.
- **The instruments.** A Hantek 6022BE two-channel USB oscilloscope driven by Lunes' own driver; a KORAD KA3010D bench
  supply with no USB, so Lunes reads its red seven-segment display with the bench camera, segment by segment, with no
  model; an Arduino Nano, which Lunes programs and reads as a second instrument.
- **On the breadboard that morning:** an LM393P (the chip the bench camera saw at 11:41, `runs/bench_world.json`), and
  earlier a 555, the chip the bench camera's height was taught from at 10:20.
- **The Meta models that ran** (`runs/api_calls.jsonl`, 3 November 09:00 to 12:30): Muse Spark 1.3 as the conductor of
  the conversation at minimal effort, 37 calls, the first at 10:42:24; Muse Wonder 1.0 as the worker that uses the
  bench tools, at medium effort, 32 calls, the first at 10:50:52; both through the Meta Model API. The ears
  (Omnilingual ASR) and the voice (MMS-TTS Spanish) run on the bench PC and leave no API record. Muse Glimmer's open
  weights stand in for Spark and Wonder when there is no internet; they were not needed that morning. The camera
  recalibration below used no model at all: finding the ring light and the chip in a picture is code.

## The bench camera recalibration (09:59 to 10:21)

The bench camera turns pixels of its picture into machine millimetres through a 3 x 4 camera projection fitted to
pairs of (where the head's ring light appears in the picture, where the machine says the head is). The camera had
been moved since the fit of 2 November, so its millimetres could not be trusted: `check_bench_cam.py` asks for the
same thing `reach_in_bench` asks before every move, that the ring appear within 40 px (`MOVED_PX`) of where the old
projection puts it for the head's known position.

**The route.** `calibrate_bench` drives the head to 16 places (`plan_route`: a 4 x 3 grid over the work area at the low
height, then the four corners at the high height, two heights because a camera cannot be fitted from points on one
plane) and looks for the ring at each. `runs/machine_events.jsonl` shows two routes that morning, 10:04 to 10:06 and
10:10 to 10:12, each with several `host_lost` events: the motion board stops a move when it loses the PC for a moment,
and Lunes finishes the move itself, so the position stays exact. The first route is the one that missed the ring (see
"What went wrong"); the second gave the fit.

**The fit**, saved in `config/machine.json` under `bench_camera` at 10:20:

| | |
|---|---|
| ring pairs used | 11 of the 16 looks (the other 5 showed no ring and were skipped; the fit needs at least 8) |
| heights of the pairs | focus Z 3.0 (8 points) and Z 47.3 (3 points) |
| error of each pair, mm | 0.9, 0.5, 0.2, 1.4, 0.5, 0.5, 0.2, 0.2, 1.0, 0.8, 0.0 |
| typical error (median) | 0.5 mm |
| worst error | 1.4 mm |
| held out, each pair located without itself | 0.74 mm |

**The height of the ring.** The ring is not the surface: it hangs over whatever the microscope focuses on. That height
is taught from one thing both cameras see. `teach_height_now.py` centred the microscope on the 555 on the breadboard
(`_center_chip`: focus, look, move by what is left, look again), then `teach_bench_height(chip=true)` found the same
chip in the bench camera's 4K picture by itself (`chip_in_picture`, a dark rectangle on the white board) and kept the
one height that puts that chip on the camera's ray to it. Saved as `taught`: the chip at column 2349, row 1847 of the
bench picture, with the microscope on it at machine X -2.62, Y -73.62, focus Z 17.10; the ring is **101.26 mm** over
the surface in focus (`ring_above_mm`). The night before, the same figure had been 98.06 mm (noted in
`find_chip_xy.py`, which used it as a guess to put the chip inside the microscope's view before teaching). The
`note` in the config says what the numbers mean, so a reader of the file needs no code.

## The workshop inventory (`config/inventory.json`, `lunes/inventory.py`)

Erlan listed what is on his shelves on 3 November and Lunes reads the file whenever it changes:

- resistors, ohm: 1, 220, 330, 1000, 1500, 2200, 4850 (a measured value, probably sold as 4.7k), 10000, 100000, 150000,
  1000000;
- capacitors: 100 nF ceramic (marked 104) and 220 nF tantalum (marked 224, polarised);
- potentiometers: 5k, 10k, 20k (the 20k marked 203), and one multi-turn pot whose value is not read yet;
- 22 parts, chips, boards and modules, each marked `simulated: true` when Lunes has a SPICE model of it and can design
  and diagnose with it (NE555, LM393, 2N2222, 1N4148-type diodes, red and green LEDs, Arduino Nano, STM32 Blue Pill)
  or `false` when it cannot yet (MCP6002, SN74LS74AN, 1N5400, zeners, Arduino Mega, ESP8266, ADS1115, MPU6050,
  servo and stepper motors, stepper drivers, buck and boost modules, industrial joysticks).

Two rules in `lunes/inventory.py`: a design lands on a stocked value when one lies within **20 %** of the ideal
(`STOCK_TOLERANCE`), otherwise the standard E12 or E6 series decides; and a designed value within 3 % of a stocked one
counts as on the shelf (`ON_SHELF`). The 555 astable at 650 Hz, run against this inventory (`TEMPLATES["astable_555"]`
with `frequency_hz=650`, red LED):

| part | value | on the shelf |
|---|---|---|
| R1 | 2.2k | yes |
| R2 | 10k | yes |
| C1 | 100 nF | yes |
| C2 (pin 5, keeps noise out of the timing) | 10 nF | **no**; the nearest there is 100 nF |
| R3 (the LED's resistor) | 180 ohm | **no**; the nearest there is 220 ohm |

With those values the formula gives 648.6 Hz and a duty of 55 % (`design_notes`). The conductor's briefing carries
a one-paragraph summary of the inventory (`inventory.summary()`), so "what resistors do I have" needs no tool, and
the `inventory` tool answers the rest, with `search` to keep only what mentions a word or a value.

## The live exercise, 10:48 to 10:52

As seen in the web lab's conversation column: Erlan asked for a circuit with the "LM33". Lunes answered that there
is no LM33 in the inventory and offered the LM393. Erlan then asked for "a comparator". The agent used the tools
`inventory`, `chip_catalog`, `circuit_guide` and `circuit_set`, the last one three times until the review before
simulating passed, and delivered a comparator.

What the files hold of it:

- `runs/api_calls.jsonl`: conductor calls (Muse Spark) at 10:48:41, 10:49:00, 10:49:46, 10:50:29 and 10:50:44, then
  six worker turns (Muse Wonder, medium effort) at 10:50:52, 10:50:55, 10:51:05, 10:51:09, 10:51:20 and 10:51:43,
  of 2.6 to 10.7 s each, then the conductor again at 10:51:45 and 10:51:47.
- `runs/verifier.jsonl`, 10:51:43, Lunes' answer, checked "supported by the code check" with nothing missing (every
  number in it appeared in a tool's result), translated from the Spanish: *"Done, a comparator with an LM393 at 5 V,
  simulated with no warnings and every goal met: two 10k resistors set the reference at 2.4992 V on pin 3, and a 10k
  potentiometer takes the input to pin 2, here 3.4998 V. As the input is above the threshold the output sits at
  0.2768 V with its 10k pull-up and lights the red LED through 330 ohm; if the potentiometer goes below the reference
  the output rises to 5 V and the LED goes off."*
- `runs/circuit.json` and `runs/circuits/20261103-1412-comparador-lm393-umbral-2-5v.json`, the circuit on the bench,
  "Comparador LM393 umbral 2.5V", 5 V: U1 LM393 (OUT1 to OUT, IN1- to VIN, IN1+ to VREF, the unused second comparator's
  inputs tied to GND and VCC), R1 and R2 10k from VCC to VREF to GND, RV1 10k pot at position 0.3 driving VIN, R3 10k
  pull-up on OUT, R4 330 ohm from VCC to the LED's anode, D1 red LED from LED_A to OUT. The simulation in the file
  (ngspice, 0.02 s simulated): VIN 3.4998 V, VREF 2.5005 V, OUT 0.2768 V, LED_A 2.1106 V, VCC 5.0 V, GND 0 V; supply
  current 10.98 mA, LED current 8.756 mA, R4 at 25.3 mW, no warnings. The file was written again at 14:12, when the
  breadboard plan was added to it (see below); its VREF reads 2.5005 V where the 10:51 answer said 2.4992 V, a
  difference of 1.3 mV between two simulations of the same divider.
- The breadboard plan kept in that file: U1 with pin 1 in e60, RV1 in e46 to e48, the resistors from the rails into
  rows 51, 52, 53 and 60, the LED in c53 and c54, ten wires each with a colour, a probe hole for every net (OUT at
  c60, VIN at c61, VREF at c62, GND and VCC on the rails at row 3), and 18 steps said in Spanish, the last one the
  supply: *"with the supply off and at 5 V: positive to the + rail, negative to the - rail"*.
- `runs/fault_dictionaries/80c8dc25a1bc5dcd.json`, 11:40: the 34 faults the simulator worked out for this comparator
  (the chip removed, dead, or any of its pins not making contact; each resistor open, shorted, ten times or a tenth of
  its value; the pot open, its wiper open, turned all the way either way; the LED open, shorted or reversed), plus the
  healthy case, ready for a diagnosis by measurement.

## The simulated bench the same morning (`lm393_nano_demo.py`, 10:35; re-run while writing this)

The same tools, with no hardware, on a circuit written as the model writes any circuit (`circuit_set`): a dark
detector, an LDR against a pot's reference into an LM393, the LED on when dark, read by an Arduino Nano on A0 and D2.

- Review: no errors, no warnings. Simulated: SENS 0.05 V, REF 1.50 V, OUT 0.28 V, LED_A 2.11 V.
- `mcu_program` wrote the sketch (`runs/sketches/lunes_dark_detector_LM393_with_an_LDR_read_by_a_Nano_U2/`) and
  built it with arduino-cli for `arduino:avr:nano`: **"Sketch uses 4334 bytes (14%)"**, the same 4334 bytes the
  built `.hex` holds. It was put on the simulated board, not on a bench, and the result says so.
- `mcu_read`: A0 on OUT expected 57 counts (0.277 V), read 57.2 counts (0.28 V) with a swing of 0.015 V; D2 on OUT
  expected low all the time, read low all the time; `all_as_simulated: true`.
- The simulator knows **34 faults** for this circuit (`runs/fault_dictionaries/d198941b8f139b7a.json`, 10:37, with
  the healthy case as a 35th entry).
- With the LM393's VCC leg out of its row (`U1:pin8_open`) planted on the simulated bench: the Nano's reading no
  longer matched the simulation, the diagnosis asked for the scope on **VCC**, then **SENS**, then **REF**, and reached
  a verdict after those three: *"U1 missing or not making contact, or U1 pin 1 (OUT1) not making contact, or U1 pin 8
  (VCC) not making contact, or R3 shorted"*, four faults that read the same at every net reached, the planted one
  among them, to be told apart by a look. Nothing wrong was named.

Two more scripts of the morning exercised the report (`report_demo.py`, 10:48: the 555 from the shelf, the simulated
Nano reading it, a planted fault, then the `report` tool) and the saved circuits (`circuits_demo.py`, 14:09: list,
load, the plan surviving a restart, `holes_connected`). Their printed output was not kept, so their numbers are not
quoted here; the tests in `tests/test_simulated_bench.py` cover the same ground and passed (below).

## The report (`lunes/report.py`)

A Markdown report of the circuit on the bench, written to `runs/reports/<slug>-<date>.md` and copied to `latest.md`,
which the web lab shows on its report page. It is written by itself when a diagnosis reaches a verdict
(`diagnose_measure`) and by the `report` tool when Erlan asks. Everything in it is read off Hume's memory of the
session; nothing is computed anew, so the report says only what the tools said. Its sections:

1. **Parts**, each with what the shelf says ("on the shelf", or "not on the shelf; the nearest there is 220").
2. **Simulation (ngspice)**: what every net should read (shape, mean, peak to peak, frequency, duty).
3. **Measured on the bench**: the scope's and the multimeter's readings each named, the supply as read on its display
   by the bench camera, and the board's own inputs beside their simulated values.
4. **Simulation against the bench**, with the criterion stated in the report itself: frequency within **8 %**, duty
   within **6 points**, volts within **5 % plus 0.15 V** (`TOLERANCE`), and the allowed band printed beside every
   difference, so each "agrees" or "**no**" can be checked by the reader.
5. **Diagnosis**: every remaining explanation with its probability, and the verdict when one reached 90 %.
6. **Pictures** (the board's mosaic, the chip) and the verifier's note, when there are any.

The first reports written on the real bench are `runs/reports/555-astable-at-1000-hz-with-a-red-led-20261103-1137.md`
and `-1144.md`, from the rehearsal described next: three scope readings at 104 Hz against 948 Hz simulated, every
voltage inside the criterion, and the verdict "R2 ten times too big" at 93 %. `tests/test_simulated_bench.py` checks
the sections, the two instruments named, the criterion, and that 8 % in frequency and 5 points of duty agree while
0.5 V does not.

## What went wrong, and what was done about it the same day

1. **The ring finder missed the ring in daylight.** `ring_in_picture` kept pixels brighter than 235 (at the finder's
   960 px scale). At night the lit ring saturates; in daylight the webcam shortens its exposure and the ring comes
   out just under that. Re-running the old and the new finder on the five 4K pictures saved that morning: in
   `bench_X204_Y-208_Z47.png` and `bench_X62_Y-82_Z19.png` the old finder found nothing and the new one found the
   ring; in both, the ring's lit pixels average **234** with the brightest at 236 to 237, against the threshold of
   235. The fix (`lunes/bench.py`): a second pass 20 lower, with the same shape, tint and dark-tube checks, taken only
   when the first finds nothing.
2. **0.4 s after a move was not enough for the webcam.** On the first route the ring was missed at **11 of the 16**
   places where it was seen a second later: the webcam hands over a picture taken before the carriage settled.
   `calibrate_bench_now.py` wrapped each look in a 1 s wait and two fresh frames, and `lunes/hume.py` now waits 1.0 s
   (the comment in the code records the 11 of 16). Even so, five of the sixteen looks on the second route had no
   ring and were skipped; the fit used 11 pairs.
3. **The supply's display, read by camera, alternated between a reading and "no veo los dos números rojos de la
   fuente en la imagen"** (`lunes/supply.py`: the two red windows were not found in that picture). What the code does
   about it: `supply_look` takes up to two more pictures 0.3 s apart when a digit reads half lit (the digits are
   multiplexed LEDs), and `supply_state` keeps the last good reading with its age and tells the model that what it
   remembers may have changed. That retry entered in the commit of 01:14 that day; no file records whether the
   alternation was gone by midday. `runs/supply.json` holds the latest reading, taken at 14:16: 5.0 V, 0.5 A, output
   off.
4. **The Kinect held the machine for a false reason.** Its check of the head's position had a 5 mm tolerance in X
   while a corner reading of its own calibration missed by 5.7 mm. The tolerance now takes the worst held-out miss
   of the calibration itself, 12.6 mm in X (`config/machine.json`, `kinect_head`, note dated Nov 3), and the next
   `calibrate_kinect` sets it from its own misses.
5. **Two plans of one LM393 circuit.** In the rehearsal of 11:29 to 11:47 Lunes compared Erlan's build against a plan
   that was not the one he had built from. `runs/verifier.jsonl`, 11:39:41: *"I saw an LM393P with pin 1 in f60, but
   the plan expects it in e60 with the notch to the left, so it is backwards: put it in e60 and tell me"*; 11:41:42,
   the same again; 11:44:30, after Erlan objected: *"You are right: I remember the earlier design was a 555 astable
   with an LED, different from the current LM393 comparator, and mosaic 18 showed that earlier layout."* The cause
   was a circuit laid out again, or designed again, after the build had started. What was added (`lunes/hume.py`,
   `lunes/knowledge/circuits.md`): the breadboard plan is saved with the circuit (`runs/circuit.json`, and a copy
   per circuit in `runs/circuits/`) and comes back after a restart, never laid out again; `circuit_list` lists the
   saved circuits with their parts and whether they have a plan, `circuit_load` puts one back on the bench with its
   plan, and the guide now says: once Erlan started building, do not design or lay out again, and if it is unclear
   which circuit is on the board, list them and let him pick. `holes_connected` says whether two holes are the same
   electrical node (the same strip of five, or the same rail), so a lead in another hole of the plan's strip is
   reported as right, with the hole it is in, instead of as a mistake.
6. **The probe on the wrong net, and a flat trace with the supply off** (tests dated Nov 3 in
   `tests/test_simulated_bench.py`): a moving signal read on a net the simulation says is steady is refused as the
   probe elsewhere, and nothing enters the diagnosis; a flat 0 V while the camera last read the supply's output off
   says so first instead of blaming the circuit.
7. **The web lab** (`lunes/web/lab.html`, `lunes/dashboard.py`, `lunes/web/report.html`), the same day: a text box to
   type to Lunes beside the microphone and a button to mute the microphone (a fridge and dogs next to the bench);
   zoom on the breadboard drawing from 25 % to 400 %; a "Proyectos" list of the saved circuits with a drawn thumbnail
   of each plan, to put one back on the bench with a click; the report page; marks for where the probe tip and the
   ground clip go; rail holes said as "rail +, row 18". Bugs fixed with it: the page lost the first events after Lunes
   restarted, the conversation log grew without bound (now 500 nodes), the tap mark on the microscope picture fell in
   the wrong place with zoom, and every image load the browser aborted printed a full traceback in the console (10 in
   5 min before, 0 after). The Lunes running on the bench loaded the earlier `dashboard.py`, so these need one
   restart to show.
8. **Freeing the GPU** (`tools/talk.py`): Lunes loads Muse Glimmer with `keep_alive -1` so the bench never waits for
   it twice, and Ollama then kept it in the GPU after Lunes exited (about 16 GB of VRAM taken all night). `talk.py`
   now asks Ollama to unload it on every exit.

## The test suite

`python tests/run_all.py` runs the 12 files that need no hardware: `test_geometry.py`, `test_scope_auto.py`,
`test_supply.py`, `test_breadboard.py`, `test_chip.py`, `test_bench.py`, `test_calc.py`, `test_parts.py`,
`test_catalog.py`, `test_voice.py`, `test_lab.py`, `test_simulated_bench.py`. `test_bench.py` covers the bench camera
of this case study: the ring found in a drawn webcam picture, the fit from the ring at two heights, the height taught
from one thing seen by both cameras (a point that is not the thing refused), points on the table and above it located,
the height kept through a new fit. `test_parts.py` covers the inventory: a 555 at 650 Hz from the shelf lands on 2.2k,
10k and 100 nF, R1 and C1 reported on the shelf, and the conductor's summary names values and parts.

`.github/workflows/tests.yml` runs them on every push on `ubuntu-latest`: Python 3.12, `apt-get install ngspice
libportaudio2`, arduino-cli with the `arduino:avr` core, `pip install -r requirements-ci.txt`. The workflow file was
written that morning and is in the working tree; at the time of writing it has not been pushed. The same steps were
run in WSL Ubuntu on the bench PC at 09:44 (`ci_test2.sh`, log `wsl_tests.log`): **all passed**, 12 files, in these
seconds:

| test file | s | test file | s |
|---|---|---|---|
| test_geometry.py | 1 | test_calc.py | 2 |
| test_scope_auto.py | 2 | test_parts.py | 0 |
| test_supply.py | 3 | test_catalog.py | 29 |
| test_breadboard.py | 3 | test_voice.py | 0 |
| test_chip.py | 9 | test_lab.py | 48 |
| test_bench.py | 2 | test_simulated_bench.py | 168 |

The same `run_all.py` passed four times on Windows during the web lab's work that day (10:57, 11:12, 11:42 and 14:15,
in the session's notes), the last after the Proyectos list and the breadboard zoom.

## For the 3-minute video

What of the above is on camera, if the screen was being recorded, and what is not:

- **In the web lab's conversation column**, 10:48 to 10:52: Erlan's "LM33" line, Lunes' answer that there is no LM33 in
  the inventory and the offer of the LM393, the request for "a comparator", the tool names as they ran (`inventory`,
  `chip_catalog`, `circuit_guide`, `circuit_set` three times), and the final answer with its voltages. The column
  also shows the time of each line.
- **In the "Circuito y diagnóstico" panel**: the comparator appearing as parts and simulated nets after `circuit_set`.
  The breadboard plan in the file is dated 14:12, so the plan drawing was not on the page during the 10:48 exchange;
  it is there now, and with the new page also in the Proyectos menu as a thumbnail and on the report page.
- **The bench camera's view** during the two calibration routes (10:04 to 10:06 and 10:10 to 10:12): the head
  crossing the table at two heights with the ring lit. The five saved 4K looks (`looks/` in the session's scratch
  folder) and `bench_check2.jpg` (10:21, the ring where the new projection expects it) are stills from it.
- **Not on camera**: the simulated-bench scripts (console output only), the WSL test run (a log), and the
  recalibration numbers (a JSON file). These can be shown as text or left to the README.
- **Can be re-staged live** without changing anything: the comparator is saved and loads from Proyectos with its
  plan, so the drawing, the probe marks and the report page can be shown on the real page after restarting Lunes
  once; the "LM33" question can be asked again, since the inventory file is the same.
