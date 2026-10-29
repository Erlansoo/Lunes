# Firmware for the Hantek 6022BE

The 6022BE keeps no firmware of its own: its FX2 microcontroller starts empty every time it is plugged in, and the
PC loads a program into its RAM. Lunes' driver (`lunes/hantek.py`) loads `dso6022be-firmware.hex` at every start.

- **What it is:** the open firmware for the 6022BE written by the OpenHantek6022 / Hantek6022API projects, version
  0x0210. It is not Lunes' code.
- **Source:** [Ho-Ro/Hantek6022API](https://github.com/Ho-Ro/Hantek6022API), commit
  `cf6a1032f1064397b98da6f297f7ba6f71b5fb78`, file `PyHT6022/Firmware/HEX/dso6022be-firmware.hex`. The corresponding
  source code is in that repository, under `Firmware/DSO6022BE`.
- **License:** GNU General Public License version 3 (`LICENSE-GPL-3.0.txt`), as the source repository states. It is
  shipped here unchanged, as a separate program the scope runs.
