# Lunes

**Lunes** ("Monday" in Spanish) is an AI assistant for the electronics bench. Tony Stark had FRIDAY; a repair
technician in Cochabamba gets Lunes, the one who shows up to work on Monday.

You put a broken board on the bench and say "it doesn't work". Lunes looks at the board through a USB microscope,
tells you exactly where to probe, reads your instruments, compares every measurement against a simulation of the
circuit, picks the next most informative measurement, and finds the faulty component. The hands are yours; the
reasoning and the eyes are Lunes's.

Built solo for the **Meta Global AI Developer Hackathon 2026** (Indie track, *Push the Frontier*).

> Status: work in progress. The hackathon runs Oct 26 – Nov 4, 2026.

## Try it

No hardware and no API key: on Windows with Python 3.14, and ngspice as in [Installing ngspice](#installing-ngspice-once-per-pc):

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python tools/lab_preview.py
```

Then open http://127.0.0.1:8765: the web lab diagnoses a 555 with a planted fault on a simulated bench.
`.venv\Scripts\python tests/run_all.py` runs every test that needs no hardware.

With the bench: copy `.env.example` to `.env`, paste your Meta Model API key from dev.meta.ai, and run
`python tools/talk.py COM8 --spark --mic <part of your microphone's name>`. The work is done by Muse Wonder
reasoning at medium effort (`--brain muse-spark-1.3 --effort low` for the faster, smaller model); the conductor of
the conversation stays on Muse Spark so it answers in a couple of seconds. To try Lunes' voice and tools without the
scope or an Arduino, add `--osciloscopio simulado --arduino simulado`. Never turn the microscope's focus wheel:
Lunes focuses by moving Z, and every height it knows (the table, a board, the laser's landing, the bench camera's
ring height) hangs on the lens' working distance. The CNC, the microscope and the Ollama model are
set up as this README describes.

## Where it stands

- **Works today:**
  - **Hands:** own firmware drives the bench CNC, with the microscope as its head.
  - **Eyes:** Lunes measures the board, builds a mosaic of it, finds components by name (SAM 3.1), goes to one and
    reads its marking.
  - **Aim:** it puts the microscope exactly on a point it sees, or on a corner of the board. The model says what; the
    code finds the board's edges and turns pixels into millimetres, and checks where it landed with a last picture.
  - **Wide view:** a camera beside the bench (an EMEET S800, 4K over USB) sees the whole work area,
    which the microscope cannot; the Kinect adds depth.
  - **Reaching a thing fast, with every camera** (`lunes/bench.py`). The bench camera is calibrated to the
    machine's millimetres by itself (`calibrate_bench`): the head's ring light is found in its picture over the whole
    table at two heights far apart and a camera projection is fitted. One thing seen by both then teaches it how
    high the ring hangs over what the microscope focuses on (about 90 mm): the microscope centred on the chip, SAM
    3.1 finding the same chip in the bench picture, `teach_bench_height` (a point that is not the thing is refused).
    `reach_in_bench` then takes the head to anything the bench camera sees (a point of its picture, or what
    `find_in_bench` finds): the bench camera gives the millimetres, the Kinect checks that the head got there, and
    down at the surface the laser's dot is put on the point and seen by the microscope, where it lands telling the
    height of what is really there (a dot lost in a hole or on a part's side is tried 4 mm beside). An approximation
    by design, a few mm on a hand-built machine; a disagreement is reported, never corrected on its own, and the
    microscope settles the rest.
  - **Self-check:** the Kinect checks every move against where the machine thinks it is. Lunes drives the Kinect
    with its own USB driver, written for this project.
    - It is calibrated as a camera over the whole table (`lunes/head.py`). The Kinect measures no depth on the
      black microscope tube, so the ring light's place in the picture, at the height the machine knows, gives X
      and Y: 0.3 mm typical error, a 5 mm tolerance.
    - When it disagrees, the microscope looks for itself in the board's mosaic as a second opinion. Every tool
      that moves refuses until Erlan says the machine may go on.
    - A move the motion board stops because it lost the PC for a moment is finished by itself. The board counts
      every step, so the position stays exact.
  - **Research:** it looks up datasheets on the web, with sources.
  - **Conversation:** you talk to it like a call: it answers at once, works in the background and tells the result.
- **Working on the real scope:** the oscilloscope.
  - Lunes drives a Hantek 6022BE (two channels, 20 MHz) with its own USB driver (`lunes/hantek.py`).
  - **First real run, Oct 31, on the scope's calibration output:** both channels read 2 V square waves, and Auto
    read all seven frequencies tried, from 32 Hz to 100 kHz, with no error. The self-test passes on both.
  - **What the real scope taught:** a capture starts with leftovers of the last one in the scope's FIFO, which
    left a hole in its first samples and broke the reading of some frequencies. Lunes now skips past them.
  - **The zero of each range:** the scope keeps it in whole counts, up to 0.15 V off at the tip of a x10 probe.
    `scope_zero` measures it to a fraction of a count and keeps it in `config/machine.json`, with nothing
    written to the scope. On this one the low level of every range now reads within 10 mV of 0 V.
  - Like the Auto button of a bench scope, the code picks the range and the sample rate, then measures levels,
    frequency, duty cycle, rise time, the signal's shape and, for serial data, the likely baud rate
    (`lunes/signals.py`).
  - Notes say how far to trust each reading (it clips, too few samples per period, 0 V and flat, mains hum), and a
    picture of the screen goes to the model with the numbers.
  - The model judges the circuit from those numbers. It never reads a value off a picture.
  - A self-test on the scope's own calibration output catches a probe set to x1 while Lunes counts x10, or the
    other way round.
  - A simulated scope with the same interface (`--osciloscopio simulado`) tries all of this without the real one.
- **Working on the simulated bench, first test on a real breadboard on Oct 30:** diagnosis by measurement.
  - **The circuit as data:** Lunes keeps the circuit on the bench as parts, values and nets (`lunes/circuit.py`),
    one Erlan describes or one Lunes designs.
  - **Simulation:** ngspice simulates it, and each net is read with the same measurements the scope uses
    (`lunes/simulator.py`), so a bench reading and a simulated one compare number to number.
  - **Own part models:** the 555 (analogue comparators and a digital latch) and the op-amp are behavioural models
    written for Lunes (`lunes/spice/models.lib`).
  - **Fault dictionary:** the simulator also runs every typical fault of every part, as parts fail on a bench: open,
    shorted, ten times the value (a colour code read wrong), backwards, a dead chip.
  - **Choosing what to measure:** the engine (`lunes/diagnosis.py`) weighs each reading against every hypothesis
    with Bayes' rule, then picks the net whose reading separates the remaining ones best.
  - **The verdict:** it names a fault only at 90 %. Faults that read the same everywhere are reported together, to
    be told apart by a look. A "none of these" hypothesis keeps it from naming the nearest wrong fault.
  - **Tested on all 24 faults of a 555 blinker:** 20 found, 4 left honestly undecided, none wrong, 2.6
    measurements on average.
  - **Muse Spark end to end, on the simulated bench:** it designed the blinker itself (977 Hz simulated for the
    1 kHz asked), then found a resistor ten times too big in two measurements.
  - Most faults (a shorted capacitor, a dead MOSFET) look exactly like a good part, so this works from
    measurements, not from pictures.
- **Design and review, like a senior engineer:**
  - The model decides what is wanted; the code designs it by formula with standard values (E12 resistors, E6
    capacitors) and the simulation must meet its goals (`lunes/designs.py`). Templates: an LED, a 555 astable
    fixed or tuned over a range with a pot, a transistor or MOSFET switch, a fixed or adjustable divider, an RC
    filter, an op-amp amplifier, a zener or three-pin linear regulator, a light sensor, a button lighting an LED.
    Every test design meets its goals.
  - The parts it knows (`lunes/circuit.py`, models in `lunes/spice/models.lib`): resistors, capacitors, coils,
    diodes and Schottky diodes, zeners, LEDs, bipolar transistors, MOSFETs (logic level and not), potentiometers,
    switches, light and heat sensors, 7805/7812/LM1117 regulators, the 555, an op-amp, extra sources and Arduino
    boards. Each breaks the way it does on a bench (a pot turned all the way or with a worn wiper, a welded
    switch, a sensor covered, a regulator with its input on its output) so the diagnosis can tell them apart.
  - Every other number is worked out by the code too (`lunes/calc.py`, the `calculate` tool): the model writes what
    an engineer writes on paper ("12 V / 20 mA", "1/(2*pi*10 kohm*100 nF)") and Pint works it out with units, in
    the unit that fits (600 Ω, 159.2 Hz); a formula is solved for its unknown by SymPy ("P = V**2 / R" for R) with
    the known values' units carried through. A mix of units that makes no sense is an error, not a number, and
    nothing but arithmetic is ever run. The model never does arithmetic in its head, and the verifier removes any
    number that came from no tool.
  - **Chips are data, not code** (`lunes/catalog.py`, `lunes/catalog/*.json`). A chip is a file with its pins, supply,
    a SPICE subcircuit and a test circuit, and the simulation, the review, the faults, the breadboard and the
    diagnosis all work with it with no line written for it. Asked for a chip it does not know, Lunes finds the
    datasheet, writes the entry and the code runs its test: only what passes enters, marked as learnt. It learnt the
    LM339 that way, in 48 s. A chip is data, so what the model writes never reaches the computer as code. Measured in
    `docs/unseen.md`: 8 sentences with no template behind them, judged by the simulator with criteria the model
    never sees, 7 work. The eighth is the honest limit: a NAND oscillator ran out of steps.
  - A review before simulating catches nets with no DC path to ground, a shorted supply and parts in one row
    (`lunes/review.py`). One after simulating catches an LED past 30 mA, a resistor past 1/4 W, a transistor past
    its rating, and a net outside the supply.
  - ngspice cannot hang Lunes. Every run has a time limit, and a run that does not converge is retried with more
    tolerant settings. Its errors come back as sentences in the circuit's own words, and a result from a singular
    matrix is refused, not trusted.
  - A guide of what a bench engineer keeps in mind (pinouts, typical faults, limits) is read by Lunes when it needs
    it (`lunes/knowledge/circuits.md`).
  - Lunes checks its own parts (`system_check`: a simulator self-test, the scope, the Kinect, the cameras, the
    CNC's alarm) and restarts the one that fails (`system_restart`), telling Erlan what happened.
  - A verifier checks every answer before it is said (`lunes/verifier.py`):
    - The code first: each number, in figures or in words, must appear in what Erlan said, what the tools returned
      or the memory, allowing for rounding and units (0.0147 A said as 14.7 mA).
    - Only when one is missing, or the answer claims to have seen or measured something with no tool run, Muse
      Spark (or Glimmer offline) fixes just that: an invented board size becomes the measured one, an unmeasured
      "the LED is on" becomes "I did not measure it".
    - Answers backed by their tools pass in milliseconds.
- **Laser pointer, working on the real bench:** Lunes points at things for Erlan with a red dot (`lunes/laser.py`).
  - A 5 mW laser sits on the carriage, not on Z, driven by the motion board's PWM on D6 at 31 kHz.
  - It calibrates itself with the microscope. The dot is found against a picture with the laser off. The brightness
    steps down past the levels that glow over the whole board, and stops just above where the diode starts to lase:
    PWM 29 of 255, a 0.4 mm dot.
  - The beam is slanted, 23.6 degrees on this bench, so where it lands depends on the height of the surface.
    Calibrated on two surfaces (the table, and a 4 mm block measured with a caliper), Lunes knows the slant, and it
    works both ways:
    - `laser_point` puts the dot on what Lunes means: a point of a microscope picture, of the board's mosaic, or in
      mm. It focuses on the surface first (the beam is slanted, so its landing depends on the height), the
      microscope then checks where the dot really landed, the head moves again for what is left, and the answer
      says `confirmed` only when the dot was seen on the point; otherwise it says how far it is, and Lunes must say
      so. For a chip's pin 1, `chip_pin: 1` makes the code find the chip and its leg (`see_chip`, from the DIP's
      2.54 mm grid), put the dot on the chip's flat top beside that leg and confirm it is seen on the body; the leg
      itself is lower and shiny, so the dot stays where it can be checked. The model never picks a pin by eye.
    - `laser_height` tells how high the surface under the dot is, with no focusing and no fourth axis. It measured
      the 4 mm block as 4.02 mm.
  - Measured error on a board: 0.12 mm.
  - `tools/laser_aim.py` is a live microscope view with a cross, for aiming the laser by hand.
- **Working on the simulated bench, first test on a real board when one is plugged in: an Arduino in the circuit**
  (`lunes/mcu.py`).
  - An Uno, a Nano or a Blue Pill is a part of the circuit, with a program for its pins. ngspice simulates that
    program with the circuit: each output is a source behind the pin's own resistance, and each input has the
    chip's protection diodes, so 12 V through a resistor sits at about 5.5 V, as on the real pin.
  - **The same program goes on the board.** `mcu_program` writes it as a sketch, builds it with arduino-cli and
    uploads it. It never touches the CNC's port, and without a port it uses the only Arduino plugged in or asks.
  - **The board measures too.** Between reports, the sketch reads every input as fast as it can. An analogue
    input reports the mean, lowest and highest of its net; a digital one how much of the time it reads HIGH.
    `mcu_read` puts those beside the simulation.
  - **Diagnosis without the probe:** while a diagnosis runs, each net on an analogue input goes to the engine like a
    measurement. A reading stuck at 0 or at 5 V counts as "at most" or "at least", as on a scope that clips. The
    engine then asks for the scope only where the board cannot see.
  - **Serial monitor:** `serial_monitor` reads what any sketch prints, like the Arduino IDE's monitor. It works at
    a given baud rate or finds it (`auto` keeps the rate whose bytes read as text). It can send a line to sketches
    that take commands, and can restart the board to catch its start message.
  - Design templates: a blinking LED, PWM into an RC filter (the board reads its own output back), a transistor
    switch, reading 12 V through a divider. All meet their goals in ngspice, and all their sketches compile.
  - **Tested on a simulated board** (`--arduino simulado`, with `--falla R1:open` to plant a fault), on the PWM
    read-back and the divider:
    - Seven cases, and none named wrong.
    - Five planted faults were named, each with the faults that read the same at every net (for example "R1 open
      or C1 shorted", which a look tells apart). It took one to three readings.
    - The healthy circuit was confirmed by the board alone, in one reading.
    - One stayed undecided: R2 open against R2 ten times too big. With the pin clamped, both read about 5.5 V.
- **The web lab** (`lunes/web/lab.html`, `lunes/lab.py`): one live page of the whole bench at
  http://127.0.0.1:8765, in Spanish or English.
  - The microscope with its scale bar and position, the bench camera, the Kinect and the last picture the model
    looked at.
  - The scope's traces and readings, and the machine on a map of its table, with the Kinect's check.
  - The circuit, what the engine thinks is wrong, where to probe next, and each net simulated against measured.
  - The Arduino's readings and serial lines, and the conversation with the tools Lunes used.
  - Nothing leaves the PC, and the page loads nothing from outside it.
  - `python tools/lab_preview.py` shows it on a simulated bench with no hardware: a 555 with a planted fault,
    diagnosed by ngspice and the simulated scope. No model is called, and the conversation is scripted.
- **Guided by the laser, on a breadboard** (`lunes/breadboard.py`). Tried so far with the laser simulated.
  - `breadboard_plan` lays any circuit Lunes designs out on a breadboard by rule, then rebuilds the connections from
    the holes alone and refuses a plan that is not the circuit.
  - `guide_step` shows each lead, wire and supply connection with the laser's dot, in words. During a diagnosis,
    `guide_net` shows where the probe goes for the net the engine asks for, and where the ground clip goes.
- **What Lunes has in mind** (`note_on_map`, the web lab's map). The chip it saw, with its notch and pin 1, the laser's last
  point, the last mosaic and its notes stay in machine mm across restarts, and the page draws them where they are.
- **A chip and its pin 1** (`lunes/chip.py`), tried on the real 555 on the breadboard.
  - `see_chip` finds the DIP's body, its angle and which end has the notch or the pin-1 dent, by image processing with
    no model, and says how sure it is. A body that runs off the picture's edge is photographed whole, four pictures
    lined up by their content. On the real 555 it found the body (10 x 9 mm with its pins) and the notch's end, at 41 %
    sure: a chip whose notch is moulded in relief, as dark as the body.
  - Under 60 % sure, or with no notch seen, Lunes asks instead of guessing. `confirm_pin1` takes Erlan's answer four
    ways: a tap on the microscope's picture in the web lab, a wooden stick laid with its tip on the pin (the code finds
    the tip), the laser's dot moved onto the pin, or the hole's name.
  - With the breadboard placed and a plan, it says which hole pin 1 is in and whether that is the plan's; a chip the
    wrong way round is reported with both holes.
- **Eyes on the bench supply** (`lunes/supply.py`). The KORAD KA3010D has no USB, so Lunes reads its panel with the bench
  camera, segment by segment, with no model. A panel it cannot read is no reading.
  - It looks every few seconds, remembers the last reading with its age, and says when it changes.
  - Designing a circuit or starting a diagnosis compares the supply with what the circuit needs, in the tool itself: the
    real panel at 24 V stopped a 5 V circuit. Out of view, Lunes says it does not know what the supply is set to.
- **Tests with no hardware:** `python tests/run_all.py`, about three minutes. It runs eleven files:
  - `test_geometry.py`:
    - the laser dot, found to a tenth of a pixel, and a glow refused as too bright;
    - laser triangulation;
    - the Kinect's camera fit over the whole table, at positions it never saw too.
  - `test_scope_auto.py`: the scope's Auto on 20 kinds of signal with each probe setting, and the self-test catching
    a probe at x1 while Lunes counts x10, and the other way round. The signals are:
    - square waves from 50 Hz to 1 MHz;
    - PWM, and logic at 3.3, 5 and 12 V;
    - sines and steady levels;
    - serial data at 9600 and 115200 baud.
  - `test_supply.py`: every digit of the panel on drawn panels, unreadable panels, and the check before powering.
  - `test_chip.py`: drawn DIPs turned any way, the notch end and pin 1, a body with no notch refused, a pointer's tip.
  - `test_breadboard.py`: every design laid out and rebuilt from its holes, holes in mm with the board turned, the guide.
  - `test_bench.py`: the ring light found in a drawn webcam picture (the supply's digits refused), the bench camera
    fitted from the ring at two heights, the ring's height learnt from one thing seen by both (a point that is not
    the thing refused), points on the table and above it located, the height kept through a new fit, and what is
    refused.
  - `test_calc.py`: expressions with units as written on paper (600 Ω, 159.2 Hz, 2 sqrt(2) V), units that do not
    add up refused, Python in the text refused, and formulas solved for their unknown (an RC cutoff for C, a
    quadratic's two roots, a missing value asked for).
  - `test_catalog.py`: every chip of the catalog passes its own test and lays out on a breadboard, a learnt chip enters
    only if it passes, and a subcircuit carrying anything but electricity (`.control`, `.include`, shell) is refused.
  - `test_parts.py`: the newer parts accepted when written right and refused when not, their SPICE lines healthy
    and with each fault, the review catching a zener across the supply, a floating gate, a regulator fed too
    little and a MOSFET half driven, and every new design laid out on the breadboard.
  - `test_lab.py`: what the web lab receives at each stage of a session, and that the page loads nothing from
    outside the PC.
  - `test_simulated_bench.py`: the Arduino templates and their sketches, the simulated board through Lunes' own
    tools, and the 555 blinker's 24 faults.
- **Measured with the model at the wheel** (`tools/reliability.py`, record in [docs/reliability.md](docs/reliability.md)):
  the same seven cases, but the model drives the tools from a job in Spanish, as it does with Erlan (design the
  circuit, program the made-up board, read it, ask for the probe net by net, name the fault). Three rounds on
  Nov 2, 2026: 21 runs, 18 found, 3 unsure, 0 wrong. The three unsure are the divider's lower resistor open, which
  reads the same as it being ten times larger at the nets the board can reach: the engine stays at 76 % and Lunes
  says so, pointing at R2, instead of guessing. Muse Spark at low effort and Muse Wonder at medium scored the same,
  because the verdict comes from the engine and the simulation, not from the model: Spark about 27 s and 133 k
  tokens a run, Wonder about 38 s and 164 k.
- **Being built:** the probe held by the machine.

## Built from scratch

- **Everything in this repository was written from scratch during the hackathon, starting Oct 27, 2026,** except the
  Hantek 6022BE's firmware in `firmware/hantek6022be`, which is the OpenHantek6022 project's, shipped unchanged.
- Lunes does not use any earlier project by the author. It depends on the libraries listed below and on Meta models.
- The demo video opens by saying the same thing.

## Open-source dependencies

| Dependency | Used for | License |
|---|---|---|
| [pyserial](https://github.com/pyserial/pyserial) | Talking to the motion board from the PC tools | BSD-3-Clause |
| [OpenCV](https://github.com/opencv/opencv-python) | Microscope capture, focus measure, image-shift calibration and finding the laser dot | Apache-2.0 / MIT |
| [NumPy](https://github.com/numpy/numpy) | Image arithmetic | BSD-3-Clause |
| [openai](https://github.com/openai/openai-python) Python SDK | Calling the Meta Model API, which speaks the same protocol | Apache-2.0 |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Running Meta's Omnilingual ASR ear and MMS-TTS Spanish voice on the bench PC | Apache-2.0 |
| NVIDIA CUDA runtime wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cufft-cu12`, `nvidia-curand-cu12`, `nvidia-cuda-runtime-cu12`) | The CUDA libraries sherpa-onnx needs to run the fast ear on the GPU | NVIDIA software license, free, closed source; installed by pip, not redistributed here |
| [comtypes](https://github.com/enthought/comtypes) | Listing the cameras by name (Windows DirectShow), so the microscope and the bench camera never swap | MIT |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) | Microphone and speaker access | MIT |
| [Ollama](https://github.com/ollama/ollama) | Running Muse Glimmer's open weights on the bench PC | MIT |
| [python-libusb1](https://github.com/vpelletier/python-libusb1), with [libusb](https://github.com/libusb/libusb) inside | USB access for Lunes' own Kinect driver (`lunes/kinect.py`) | LGPL-2.1-or-later |
| Open firmware for the Hantek 6022BE, from [OpenHantek6022](https://github.com/OpenHantek/OpenHantek6022) / [Hantek6022API](https://github.com/Ho-Ro/Hantek6022API) (`firmware/hantek6022be`) | The program Lunes loads into the scope at every plug-in (the scope keeps none) | GPL-3.0, shipped unchanged with its source named |
| [ngspice](https://ngspice.sourceforge.io) 41 | Simulating the circuit on the bench and each of its faults | BSD-3-Clause (downloaded into `ngspice/`, not shipped here) |
| [Arduino AVR core](https://github.com/arduino/ArduinoCore-avr) and [arduino-cli](https://github.com/arduino/arduino-cli) | Building the motion firmware, and the sketches for an Arduino in the circuit | LGPL-2.1 / GPL-3.0 |

The motion firmware (`firmware/lunes_motion`), its flasher, the Kinect driver and the PC tools in `tools/` are
written for this project. Every library Lunes installs is open source except NVIDIA's CUDA runtime, which runs the
fast ear on the graphics card (the organizers confirmed that closed libraries are fine when their licenses are
followed; the rule is that every model is a Meta model). The rest of the closed software is the PC's own: Windows,
its device drivers and NVIDIA's graphics driver. The Kinect protocol follows what the
[OpenKinect](https://github.com/OpenKinect/libfreenect) project documented.

### Setting up the Kinect (Xbox 360, once per PC)

Lunes reads the Kinect's camera through WinUSB, the generic USB driver that comes with Windows; no Kinect SDK is
needed. Device Manager will not put a generic driver on a device it knows, so the one-time switch is done with
[Zadig](https://zadig.akeo.ie) (open source, GPL-3.0). With the Kinect plugged in (USB and its 12 V adapter):

1. Open Zadig → Options → List All Devices.
2. Pick "Xbox NUI Camera" and check that its USB ID reads 045E 02AE (not the audio device, not the CNC's FT232R).
3. Target driver WinUSB → Replace Driver.

`python tools/kinect_view.py` then shows colour and depth. To undo it, uninstall that device in Device Manager and
plug the Kinect in again.

### Installing ngspice (once per PC)

Lunes runs ngspice's console build from `ngspice/bin/ngspice_con.exe`. Either unpack the official Windows build from
[ngspice.sourceforge.io](https://ngspice.sourceforge.io/download.html) there, or take conda-forge's build of the
same program: from [ngspice-exe](https://anaconda.org/conda-forge/ngspice-exe), unpack the `Library` folder of the
win-64 package into `ngspice/`. Its XSPICE code models (`ngspice/lib/ngspice/*.cm`) are needed for the 555 model.

### Setting up the Hantek 6022BE (once per PC)

The scope talks to Lunes through WinUSB too, and it has two USB identities: empty when plugged in (USB ID 04B4 6022),
and with Lunes' firmware loaded a second later (04B5 6022). Each needs WinUSB once, with Zadig:

If Hantek's own driver is installed, remove it first, or it loads Hantek's firmware at every plug-in and the
identity 04B4 goes by too fast for Zadig. With the scope unplugged, in a terminal opened as administrator, find
its name with `pnputil /enum-drivers` and run `pnputil /delete-driver oemNN.inf /uninstall /force`. To use
Hantek's software again, update the driver from its folder in Device Manager.

1. Plug the scope in → Zadig → Options → List All Devices → the Hantek with USB ID 04B4 6022 → WinUSB → Replace
   Driver.
2. Run `python tools/scope_view.py`. Lunes loads the firmware and the scope comes back as 04B5 6022. If it says
   the scope did not come back, do step 1 again for 04B5 6022, then run it again.
3. Tell Lunes what each probe's switch is set to (x1 or x10), or run its self-test: the probe tip on the scope's
   calibration pin, the ground clip on the ground pin beside it.

The scope's ground is the PC's ground through the USB: never clip it to anything tied to the mains.

## Meta models

Every model Lunes runs is a Meta model. What is not a model (finding speech by its loudness, telling Lunes' own
voice from Erlan's by the words it just said, deciding when to tell how the work goes, stitching and aligning the
mosaics) is code written for this project.

Talking to Lunes works like a call: the conductor answers in about two seconds and hands each job to the worker, which
runs in the background while the conversation goes on; Lunes tells how it goes at the milestones, not step by step.
The design is in [docs/diseno-conversacion.md](docs/diseno-conversacion.md).

| Model | Role in Lunes | State |
|---|---|---|
| Muse Spark | The conductor, at minimal reasoning: Lunes' side of the conversation, deciding by judgment what Erlan wants and whether to answer, look or start a job. And the worker: the bench agent that plans, uses the machine, reads chips and boards, and searches the web for datasheets | in use |
| Muse Glimmer (open weights) | The same conductor and worker on the bench PC (RTX 3090), when there is no internet and in private mode | in use |
| Omnilingual ASR 300M and 1B (open weights, Apache-2.0) | Lunes' ears, on the bench PC: the 300M on the GPU hears everything at once, the 1B on the CPU gets a job's exact words (part numbers) | in use |
| MMS-TTS Spanish (open weights, [`facebook/mms-tts-spa`](https://huggingface.co/facebook/mms-tts-spa), CC-BY-NC-4.0) | Lunes' spoken voice, on the bench PC | in use |
| Muse Voice Transcribe | A second ear through the API, for noisy rooms and bench names | available, not used by default |
| SAM 3.1 | Finds components by name on the board mosaic and in the bench camera's and the Kinect's views, and tracks them in recorded video | in use |
| Muse Wonder | Short video clips from the bench | planned |

## Principles

- **Verifiable, not opinion:** every conclusion links back to the measurements that support it. The expected
  readings will come from Lunes's own circuit simulator, written for this project, which the agent does not control
  (being built).
- **The model decides what to measure; geometry decides where.** Probe positions come from the board map and
  calibration, never from the language model.
- **It checks its own hands:** after each move the Kinect sees where the microscope really is and compares it with
  where the machine thinks it is. A disagreement is told to the person, never acted on alone, so a check can never
  start a loop.
- **Safety first:** a machine never touches a powered board; powered measurements are done by a person, with a
  current-limited supply.

## Development tools

Claude Code (Anthropic) is used as a coding assistant while building. The AI inside the product runs on Meta models.

## License

[MIT](LICENSE). The Hantek 6022BE's firmware in `firmware/hantek6022be` is the OpenHantek6022 project's and stays under
its own GPL-3.0 license; it is a separate program that Lunes loads into the scope.

## Author

Erlan Medrano Cruz, Electronics and Telecommunications Engineering, Universidad Privada Boliviana (Cochabamba,
Bolivia).
