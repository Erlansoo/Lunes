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
- **Being built:** the diagnosis itself.
  - Lunes asks for measurements at points it shows you.
  - It compares each reading with what the circuit should give.
  - It picks the next most informative measurement until the faulty part is found.
  - Most faults (a shorted capacitor, a dead MOSFET) look exactly like a good part, so this step works from
    measurements, not from pictures.

## Built from scratch

- **Everything in this repository was written from scratch during the hackathon, starting Oct 27, 2026.**
- Lunes does not use any earlier project by the author. It depends only on public open-source libraries, listed
  here as they are added, and on Meta models.
- The demo video opens by saying the same thing.

## Open-source dependencies

| Dependency | Used for | License |
|---|---|---|
| [pyserial](https://github.com/pyserial/pyserial) | Talking to the motion board from the PC tools | BSD-3-Clause |
| [OpenCV](https://github.com/opencv/opencv-python) | Microscope capture, focus measure and image-shift calibration | Apache-2.0 / MIT |
| [NumPy](https://github.com/numpy/numpy) | Image arithmetic | BSD-3-Clause |
| [openai](https://github.com/openai/openai-python) Python SDK | Calling the Meta Model API, which speaks the same protocol | Apache-2.0 |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Running Meta's Omnilingual ASR ear and MMS-TTS Spanish voice on the bench PC | Apache-2.0 |
| NVIDIA CUDA runtime wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cufft-cu12`, `nvidia-curand-cu12`, `nvidia-cuda-runtime-cu12`) | The CUDA libraries sherpa-onnx needs to run the fast ear on the GPU | NVIDIA software license, free, closed source; installed by pip, not redistributed here |
| [comtypes](https://github.com/enthought/comtypes) | Listing the cameras by name (Windows DirectShow), so the microscope and the bench camera never swap | MIT |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) | Microphone and speaker access | MIT |
| [Ollama](https://github.com/ollama/ollama) | Running Muse Glimmer's open weights on the bench PC | MIT |
| [python-libusb1](https://github.com/vpelletier/python-libusb1), with [libusb](https://github.com/libusb/libusb) inside | USB access for Lunes' own Kinect driver (`lunes/kinect.py`) | LGPL-2.1-or-later |
| [Arduino AVR core](https://github.com/arduino/ArduinoCore-avr) and [arduino-cli](https://github.com/arduino/arduino-cli) | Building the motion firmware | LGPL-2.1 / GPL-3.0 |

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
