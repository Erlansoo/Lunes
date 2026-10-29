# Lunes' circuit guide

What a senior bench engineer keeps in mind. Lunes reads one section when it needs it (circuit_guide). The numbers
are typical values for parts on a hobby bench. The code still computes and the simulator still checks; this guide
is for judgement, not for arithmetic.

## method
- The model decides what is wanted; the code designs it by formula (circuit_design) or Lunes writes the parts
  (circuit_set), then ngspice simulates and the review checks it. Never give a value that no tool computed or
  measured.
- Every design comes with goals: frequency, current, voltage. circuit_set with a spec says whether the simulation
  meets them. A design is handed to Erlan only when every goal passes and the review raises nothing serious.
- Review warnings are real problems: an LED above 30 mA, a resistor above 1/4 W, a net beyond the supply. Fix them
  before handing over.
- If ngspice refuses a circuit, its message says why in the circuit's words. Fix the circuit; do not retry the same
  thing.
- Simulation is not the bench. Part tolerances (5 % resistors, 10 to 20 % capacitors) shift frequencies by 10 % or
  more, so measure on the bench and compare.

## 555
- Astable: f = 1.44 / ((R1 + 2 R2) C). Duty = (R1 + R2) / (R1 + 2 R2), always above 50 %. For about 50 %, keep R1
  much smaller than R2, with R1 at least 1 k so the discharge pin survives.
- Pin 4 (RESET) to VCC, always; floating, it can stop the chip. Put 10 nF from pin 5 (CTRL) to GND.
- Pins 2 and 6 swing between 1/3 and 2/3 of VCC in an astable: a triangle-like wave on the timing capacitor.
- The output high is about 1.5 V below VCC (bipolar NE555): 3.3 to 3.5 V at 5 V. It sources or sinks up to 200 mA.
- Supply 4.5 to 16 V. Put 100 nF from pin 8 to GND right at the chip; a 555 makes spikes on the supply when it
  switches.
- Common faults:
  - Output stuck high: timing capacitor shorted, or R2 open.
  - Output stuck low: chip dead, CTRL shorted, or pin 4 at ground.
  - Wrong frequency: a resistor or capacitor of the wrong value (a colour code read wrong).
  - A big C1 (above 1 µF) is an electrolytic: its minus goes to GND.

## led
- R = (V - Vf) / I. Vf: red about 1.8 V, yellow 2.0, green 2.1, blue and white 3.0 to 3.3.
- 5 to 15 mA is plenty for an indicator; 20 mA is the usual maximum for a 5 mm LED. Above 30 mA it dies early.
- The long leg is the anode (+). The flat side of the rim is the cathode (−). Backwards, it does not light and
  takes about 5 V in reverse.
- An LED never goes straight on a supply or an output pin: it always needs its resistor.

## transistor
- An NPN as a switch: the emitter to GND, the load between VCC and the collector, the base through Rb from the
  driving pin.
- For a solid switch, give the base a tenth of the collector current (forced beta 10):
  Rb = (Vpin - 0.7) / (Ic / 10).
- Saturated, Vce is about 0.1 to 0.3 V. If the collector stays at a few volts while "on", the base gets too little
  current.
- 2N2222: up to 600 mA. BC547 and 2N3904: up to 100 to 200 mA. Pinout: BC547 is C-B-E with the flat face to you;
  2N2222 in TO-92 is E-B-C. Check the datasheet; it is the most common mistake on a breadboard.
- A relay or a motor needs a diode across it (cathode to +V), or its kick kills the transistor.
- PNP switches the high side: its emitter goes to VCC, and the base is pulled low to turn it on.

## opamp
- Non-inverting gain = 1 + Rf / Rg. Inverting gain = −Rf / Rin.
- An LM358 on one supply reaches ground at its inputs and output. Its output tops out about 1.5 V below the
  supply. A signal around 0 V needs an offset (a divider to half the supply).
- Gain times frequency is about 1 MHz for the LM358: a gain of 10 is good up to about 100 kHz.
- Unused op-amp: tie IN+ to a fixed voltage, and IN− to the output.

## rc
- Time constant τ = R C. A capacitor charges to 63 % in τ and is full in 5 τ.
- First-order filter: fc = 1 / (2 π R C). At fc the output is 70.7 % (−3 dB). The slope beyond is 20 dB per decade.
- A high-pass output sits around 0 V. Bias it with a divider before it reaches a chip input on one supply.
- Ceramic capacitors from 10 pF to 1 µF; electrolytics above, polarised.

## divider
- Vout = V · R2 / (R1 + R2). It holds only into a load much larger than R2 (ten times at least).
- A divider is not a supply: for current, use a regulator.
- To read 0 to 12 V with a 5 V ADC: R1 = 15 k, R2 = 10 k gives 4.8 V at 12 V.

## diodes
- 1N4007: rectifier, 1 A, 0.7 to 1 V drop. 1N4148: small signal, 300 mA, fast. 1N5819: Schottky, 0.3 to 0.4 V,
  for reverse-polarity protection and fast switching.
- The band is the cathode.

## supply
- Decouple each chip with 100 nF from its supply pin to GND, as close as possible. Add 10 to 100 µF where the
  supply enters the breadboard.
- USB gives 5 V at 500 mA (900 mA on USB 3). Above that, use a bench supply.
- A 9 V battery gives about 50 mA. AA batteries are 1.5 V each when new and 1.2 V when tired.

## breadboard
- The five holes of a column (a to e, f to j) are one net. The centre channel separates the two sides. A DIP chip
  straddles the channel.
- The rails along the edges are often split in the middle on long boards: bridge them, or half the board has no
  supply.
- Common mistakes:
  - two leads in the same column (the part does nothing);
  - a lead one column off;
  - a jumper on the wrong rail;
  - a chip turned round (pin 1 at the notch);
  - the ground of the scope or of the supply not connected.

## simulation
- "No DC path to ground": a net reached only through capacitors or chip inputs. Connect it, or add a resistor
  to GND (1 M is enough).
- "Loop of sources or coils": a coil or a second source straight across a source. A coil at DC is a short.
- "Unknown model or chip": only the parts Lunes models are known: R, C, L, D (1N4148, 1N4007, 1N5819), LED, NPN
  (2N2222, 2N3904, BC547), PNP (2N3906, BC557), NE555, OPAMP and sources.
- "Could not follow the circuit in time": an edge too fast, or values that make it race. Check for a missing
  timing capacitor or absurd values.
- "Simulated with relaxed settings": the numbers are a little less exact; fine for a bench comparison.
- The models are behavioural for the 555 and the op-amp: thresholds, levels and timing as measured, not the
  transistors inside.

## diagnosis
- Supply first, at the chip's own pin: half of all faults are no supply, a loose rail or a missing ground.
- Measure where the readings separate the remaining causes best: diagnose_start and diagnose_measure choose the
  point. Say why to Erlan in one sentence.
- Name a fault only when it is clear (90 %). If several read the same (a resistor or a capacitor ten times off
  both change the frequency), ask to look at those parts: colour codes, markings, a lead off its column.
- A frequency 10 % off is often tolerance, not a fault; a factor of ten never is.
- If nothing explains the readings, say so, and check the wiring against the plan, row by row.

## measurement
- The scope's ground clip goes to the circuit's ground. The scope's ground is the PC's ground through the USB: never
  near the mains.
- x10 probe: 10 MΩ load and up to 50 V; x1: small signals, but it loads the circuit and has less bandwidth.
- A multimeter gives the mean of a fast signal: a 5 V PWM at 30 % reads about 1.5 V.
- A long ground lead rings on fast edges: overshoot of 20 % or more on a square wave is often the probe, not the
  circuit.

## arduino
- Uno or Nano (ATmega328P):
  - Pins give 5 V, 20 mA each recommended, 40 mA absolute maximum, 200 mA for the whole chip.
  - PWM on pins 3, 5, 6, 9, 10, 11 (about 490 Hz; pins 5 and 6 about 980 Hz).
  - analogRead is 10 bits over 0 to 5 V.
  - The internal pull-up is about 35 k.
- An output pin never drives a motor, relay or LED strip directly: use a transistor (see transistor).
- Inputs left floating read noise: use INPUT_PULLUP, or a resistor to GND or VCC.
- An STM32 runs at 3.3 V and many of its pins are not 5 V tolerant: a divider or a level shifter for 5 V signals.
