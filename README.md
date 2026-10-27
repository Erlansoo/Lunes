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
