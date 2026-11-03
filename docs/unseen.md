# Circuits nobody prepared, with Muse Wonder at effort medium

Made by `tools/unseen.py` on 2026-11-02 18:49. Each case is one sentence in Spanish with no template behind it, and the last ones name a chip that was not in the catalog, which the model must learn from its datasheet. The verdict is the code's: it simulates what the model left and checks what the sentence asked for, with criteria the model never sees.

| Case | Works | Learnt | Tools | Tokens | Time |
|---|---|---|---|---|---|
| oscillator with a Schmitt inverter | yes | - | 5 | 101,339 | 22 s |
| comparator with a reference | yes | - | 5 | 64,675 | 24 s |
| adjustable regulator | yes | - | 6 | 85,447 | 48 s |
| amplifier | yes | - | 4 | 89,417 | 42 s |
| Arduino through a driver | yes | - | 4 | 85,501 | 34 s |
| two NAND gates | no: the simulation does not do what was asked | - | 19 | 444,270 | 237 s |
| isolated signal | yes | - | 10 | 162,451 | 129 s |
| a chip it must learn: quad comparator | yes | LM339 | 7 | 187,563 | 48 s |

**7 of 8 work.** Tokens in all: 1,220,663.
