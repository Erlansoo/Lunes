# Lunes

**Lunes** ("Monday" in Spanish) is an AI assistant for the electronics bench. Tony Stark had FRIDAY; a repair
technician in Cochabamba gets Lunes, the one who shows up to work on Monday.

You put a broken board on the bench and say "it doesn't work". Lunes looks at the board through a USB microscope,
tells you exactly where to probe, reads your instruments, compares every measurement against a simulation of the
circuit, picks the next most informative measurement, and finds the faulty component. The hands are yours; the
reasoning and the eyes are Lunes's.

Built solo for the **Meta Global AI Developer Hackathon 2026** (Indie track, *Push the Frontier*).

> Status: work in progress. The hackathon runs Oct 26 – Nov 4, 2026.

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
| NVIDIA CUDA wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cufft-cu12`, `nvidia-curand-cu12`, `nvidia-cuda-runtime-cu12`) | The CUDA libraries sherpa-onnx needs to run the fast ear on the GPU | NVIDIA software license, free; installed by pip, not redistributed here |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) | Microphone and speaker access | MIT |
| [Ollama](https://github.com/ollama/ollama) | Running Muse Glimmer's open weights on the bench PC | MIT |
| Kinect for Windows SDK 1.8 (Microsoft) | The Kinect bridge in `kinect/` (colour, depth, tilt) | Microsoft SDK license, free download; not redistributed here |
| [Arduino AVR core](https://github.com/arduino/ArduinoCore-avr) and [arduino-cli](https://github.com/arduino/arduino-cli) | Building the motion firmware | LGPL-2.1 / GPL-3.0 |

The motion firmware (`firmware/lunes_motion`), its flasher and the PC tools in `tools/` are written for this project.

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
| Omnilingual ASR 300M and 1B (open weights, Apache-2.0) | Lunes' ears, on the bench PC: the 300M on the GPU hears everything at once, the 1B gets a job's exact words (part numbers) | in use |
| MMS-TTS Spanish (open weights, [`facebook/mms-tts-spa`](https://huggingface.co/facebook/mms-tts-spa), CC-BY-NC-4.0) | Lunes' spoken voice, on the bench PC | in use |
| Muse Voice Transcribe | A second ear through the API, for noisy rooms and bench names | available, not used by default |
| SAM 3.1 | Finds components by name on the board mosaic and in the Kinect's view, and tracks them in recorded video | in use |
| Muse Wonder | Short video clips from the bench | planned |

## Principles

- **Verifiable, not opinion:** every conclusion links back to the measurements that support it. The fault
  hypotheses come from Lunes's own circuit simulator, written for this project, which the agent does not control.
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
