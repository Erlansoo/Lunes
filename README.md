# Lunes

**Lunes** ("Monday" in Spanish) is an AI assistant for the electronics bench. Tony Stark had FRIDAY; a repair
technician in Cochabamba gets Lunes, the one who shows up to work on Monday.

You put a broken board on the bench and say "it doesn't work". Lunes looks at the board through a USB microscope,
tells you exactly where to probe, reads your instruments, compares every measurement against a simulation of the
circuit, picks the next most informative measurement, and finds the faulty component. The hands are yours; the
reasoning and the eyes are Lunes's.

Built solo for the **Meta Global AI Developer Hackathon 2026** (Indie track, *Push the Frontier*).

> Status: work in progress. The hackathon runs Oct 26 – Nov 4, 2026.

## Where it stands

- **Works today:**
  - **Hands:** own firmware drives the bench CNC, with the microscope as its head.
  - **Eyes:** Lunes measures the board, builds a mosaic of it, finds components by name (SAM 3.1), goes to one and
    reads its marking.
  - **Aim:** it puts the microscope exactly on a point it sees, or on a corner of the board. The model says what; the
    code finds the board's edges and turns pixels into millimetres, and checks where it landed with a last picture.
  - **Wide view:** a camera over the bench (a Sony ZV-E10 II, as a plain USB webcam) sees the whole work area,
    which the microscope cannot; the Kinect adds depth.
  - **Self-check:** the Kinect checks every move against where the machine thinks it is. Lunes drives the Kinect
    with its own USB driver, written for this project.
  - **Research:** it looks up datasheets on the web, with sources.
  - **Conversation:** you talk to it like a call: it answers at once, works in the background and tells the result.
- **Ready, first test on the real scope on Oct 30:** the oscilloscope.
  - Lunes drives a Hantek 6022BE (two channels, 20 MHz) with its own USB driver (`lunes/hantek.py`).
  - Like the Auto button of a bench scope, the code picks the range and the sample rate, then measures levels,
    frequency, duty cycle, rise time, the signal's shape and, for serial data, the likely baud rate
    (`lunes/signals.py`).
  - Notes say how far to trust each reading (it clips, too few samples per period, 0 V and flat, mains hum), and a
    picture of the screen goes to the model with the numbers.
  - The model judges the circuit from those numbers. It never reads a value off a picture.
  - A self-test on the scope's own calibration output catches a probe set to x1 while Lunes counts x10, or the
    other way round.
  - Tested so far on a simulated scope with the same interface (`--osciloscopio simulado`).
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
    capacitors) and the simulation must meet its goals (`lunes/designs.py`). Templates: an LED, a 555 astable,
    a transistor switch, a divider, an RC filter, an op-amp amplifier. All twelve test designs meet their goals.
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
      mm. The microscope then checks where the dot really landed, and the head moves again for the height of what
      it hit.
    - `laser_height` tells how high the surface under the dot is, with no focusing and no fourth axis. It measured
      the 4 mm block as 4.02 mm.
  - Measured error on a board: 0.12 mm.
  - `tools/laser_aim.py` is a live microscope view with a cross, for aiming the laser by hand.
- **Being built: Arduino in the simulation** (`lunes/mcu.py`).
  - An Uno, a Nano or a Blue Pill is a part of the circuit, with a program for its pins. The same program is
    simulated with the circuit, written as a sketch and built with arduino-cli. The board's readings come back
    over USB serial, to compare with the simulation.
  - Design templates: a blinking LED, PWM into an RC filter, a transistor switch, reading a divider.
  - Not yet a tool Lunes calls on its own. Flashing refuses the CNC's port.
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

## Author

Erlan Medrano Cruz, Electronics and Telecommunications Engineering, Universidad Privada Boliviana (Cochabamba,
Bolivia).
