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
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with Whisper large-v3-turbo and its Silero voice detector | Lunes' ear: speech to text on the bench PC's GPU | MIT |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) with the `es_MX-ald-medium` voice | Lunes' spoken voice, local | GPL-3.0 (voice: see its model card) |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with the WeSpeaker ResNet34 speaker model | Telling Erlan's voice from other people's, so Lunes only obeys him | Apache-2.0 |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) | Microphone and speaker access | MIT |
| [Ollama](https://github.com/ollama/ollama) | Running Muse Glimmer's open weights on the bench PC | MIT |
| Kinect for Windows SDK 1.8 (Microsoft) | The Kinect bridge in `kinect/` (colour, depth, tilt) | Microsoft SDK license, free download; not redistributed here |
| [Arduino AVR core](https://github.com/arduino/ArduinoCore-avr) and [arduino-cli](https://github.com/arduino/arduino-cli) | Building the motion firmware | LGPL-2.1 / GPL-3.0 |

The motion firmware (`firmware/lunes_motion`), its flasher and the PC tools in `tools/` are written for this project.

## Meta models (planned)

| Model | Role in Lunes |
|---|---|
| Muse Spark | Multimodal reasoning and the diagnosis agent: plans, explains, reads instrument screens |
| SAM 3.1 | Segments and tracks components, pads and the probe tip in the microscope video |
| Muse Voice Transcribe | Hands-free voice while both hands hold the probes |
| Muse Glimmer (open weights) | Fast local loop on the bench PC (RTX 3090) |

## Principles

- **Verifiable, not opinion:** every conclusion links back to the measurements that support it. The fault
  hypotheses come from Lunes's own circuit simulator, written for this project, which the agent does not control.
- **The model decides what to measure; geometry decides where.** Probe positions come from the board map and
  calibration, never from the language model.
- **Safety first:** a machine never touches a powered board; powered measurements are done by a person, with a
  current-limited supply.

## Development tools

Claude Code (Anthropic) is used as a coding assistant while building. The AI inside the product runs on Meta models.

## Author

Erlan Medrano Cruz, Electronics and Telecommunications Engineering, Universidad Privada Boliviana (Cochabamba,
Bolivia).
