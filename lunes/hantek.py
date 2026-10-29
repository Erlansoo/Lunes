"""Lunes' own driver for the Hantek 6022BE USB oscilloscope: settings, captures in volts, and its calibration output.

The scope keeps no firmware: its FX2 microcontroller starts empty at every plug-in, and this driver loads the open
firmware from firmware/hantek6022be (GPL-3.0, see its README) before talking to it. Commands are vendor control
requests, the ones the OpenHantek6022 project documented (0xE0/0xE1 range of CH1/CH2, 0xE2 sample rate, 0xE3 start
and stop, 0xE4 number of channels, 0xE6 calibration output, 0xA2 EEPROM), and samples arrive on bulk endpoint 0x86,
interleaved CH1, CH2 when both channels run. It goes through libusb (python-libusb1), so Windows needs WinUSB on the
scope, once for each of its two USB identities: empty (04B4:6022) and with the firmware (04B5:6022); see the README.

The hardware: two DC-coupled channels, 1 MOhm, 8-bit ADC, 20 MHz of bandwidth. Input ranges of +-5, 2.5, 1 and
0.5 V at the BNC, times the probe's factor (x10 on the probe: +-50 V ... +-5 V at the tip). Sample rates from 20 kS/s
to 24 MS/s with CH1 alone, 12 MS/s with both. There is no hardware trigger and no offset: a capture is a block of
samples, triggered in software afterwards, and a range must hold the signal's whole swing around 0 V.

SimulatedScope has the same interface with signals made up in software, so Lunes' measuring can be tested without
the scope on the bench.
"""
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIRMWARE = ROOT / "firmware" / "hantek6022be" / "dso6022be-firmware.hex"
VENDOR_EMPTY, VENDOR_LOADED, PRODUCT = 0x04B4, 0x04B5, 0x6022
RANGES_AT_BNC = (5.0, 2.5, 1.0, 0.5)  # volts, either way of 0
GAIN_CODE = {5.0: 1, 2.5: 2, 1.0: 5, 0.5: 10}  # what the firmware takes for each range
CAL_BYTE = {10: 0, 5: 6, 2: 8, 1: 14}  # where each gain code's CH1 calibration sits in the EEPROM block (CH2: +1)
DEFAULT_GAIN = {10: 1.0, 5: 0.99, 2: 1.01, 1: 1.01}  # the front end's own error before calibration
RATE_CODE = {20e3: 102, 32e3: 103, 40e3: 104, 50e3: 105, 64e3: 106, 100e3: 110, 128e3: 113, 200e3: 120, 400e3: 140,
             500e3: 150, 640e3: 164, 1e6: 1, 2e6: 2, 3e6: 3, 4e6: 4, 5e6: 5, 6e6: 6, 8e6: 8, 10e6: 10, 12e6: 12,
             15e6: 15, 16e6: 16, 24e6: 24}
MAX_RATE = {1: 24e6, 2: 12e6}  # what USB keeps up with
BULK_EP, PACKET = 0x86, 512
SKIP_BYTES = 2 * PACKET  # the first packets of a capture are unsettled


def rates(channels):
    return sorted(r for r in RATE_CODE if r <= MAX_RATE[channels])


def ranges(probe):
    """Input ranges at the probe tip, widest first."""
    return [r * probe for r in RANGES_AT_BNC]


def _firmware_records(path=FIRMWARE):
    """(address, bytes) to write into the FX2's memory: hold its CPU, the firmware, then let it run."""
    records = [(0xE600, b"\x01")]
    for line in path.read_text(encoding="ascii").split():
        raw = bytes.fromhex(line[1:])
        length, address, kind = raw[0], int.from_bytes(raw[1:3], "big"), raw[3]
        if sum(raw) & 0xFF:
            raise ValueError(f"bad checksum in {path.name}: {line}")
        if kind == 1:
            break
        if kind == 0:
            records.append((address, raw[4:4 + length]))
    records.append((0xE600, b"\x00"))
    return records


class Capture:
    """One block of samples: volts per channel, the sample rate and the settings it was taken with."""

    def __init__(self, rate, volts, range_v, clipped):
        self.rate, self.volts, self.range_v, self.clipped = rate, volts, range_v, clipped

    @property
    def samples(self):
        return len(next(iter(self.volts.values())))


class _Settings:
    """What both the real and the simulated scope keep: probe factors, ranges, sample rate, channels."""

    def __init__(self, probe):
        self.probe = {"CH1": float(probe.get("CH1", 10)), "CH2": float(probe.get("CH2", 10))}
        self.range_v = {"CH1": ranges(self.probe["CH1"])[0], "CH2": ranges(self.probe["CH2"])[0]}
        self.rate, self.channels = 1e6, 2

    def _check(self, rate, range_v, channels):
        channels = channels or self.channels
        if channels not in (1, 2):
            raise ValueError("channels must be 1 (CH1 alone) or 2")
        rate = rate or min(self.rate, MAX_RATE[channels])
        if rate not in RATE_CODE or rate > MAX_RATE[channels]:
            raise ValueError(f"sample rate must be one of {[f'{r:g}' for r in rates(channels)]} S/s")
        new = dict(self.range_v)
        for ch, v in (range_v or {}).items():
            allowed = ranges(self.probe[ch])
            if not any(abs(v - a) < 1e-6 for a in allowed):
                raise ValueError(f"{ch} range must be one of {allowed} V with a x{self.probe[ch]:g} probe")
            new[ch] = float(v)
        return rate, new, channels

    def gain_code(self, ch, range_v=None):
        return GAIN_CODE[round((range_v or self.range_v[ch]) / self.probe[ch], 3)]


class Scope(_Settings):
    def __init__(self, probe=None):
        """probe: {"CH1": 10, "CH2": 10}, the factor set on each probe's switch (1 or 10)."""
        import usb1
        super().__init__(probe or {})
        self.usb1 = usb1
        self.ctx = usb1.USBContext()
        self.ctx.open()
        device = self._find(VENDOR_LOADED)
        if device is None:
            empty = self._find(VENDOR_EMPTY)
            if empty is None:
                self.ctx.close()
                raise RuntimeError("the Hantek 6022BE is not on the USB")
            self._load_firmware(empty)
            t0 = time.time()
            while device is None and time.time() - t0 < 8:  # it leaves the USB and comes back as another device
                time.sleep(0.25)
                device = self._find(VENDOR_LOADED)
            if device is None:
                self.ctx.close()
                raise RuntimeError("the Hantek took its firmware but did not come back: give its second identity "
                                   "(USB ID 04B5 6022) the WinUSB driver with Zadig, as the README says")
        self.handle = self._open(device)
        self.handle.setInterfaceAltSetting(0, 0)  # bulk transfers
        self.firmware_version = device.getbcdDevice()
        self.serial = device.getSerialNumber()
        self.offset, self.gain = self._calibration()
        self.configure(rate=1e6, range_v=self.range_v, channels=2)

    def _find(self, vendor):
        for d in self.ctx.getDeviceIterator(skip_on_error=True):
            if (d.getVendorID(), d.getProductID()) == (vendor, PRODUCT):
                return d
        return None

    def _open(self, device):
        try:
            handle = device.open()
            handle.claimInterface(0)
            return handle
        except (self.usb1.USBErrorNotSupported, self.usb1.USBErrorAccess, self.usb1.USBErrorNotFound) as e:
            self.ctx.close()
            ids = f"{device.getVendorID():04X} {device.getProductID():04X}"
            raise RuntimeError(f"Windows has the Hantek (USB ID {ids}) on another driver: give it WinUSB with "
                               f"Zadig, as the README says ({e})")

    def _load_firmware(self, device):
        handle = self._open(device)
        try:
            for address, data in _firmware_records():
                handle.controlWrite(0x40, 0xA0, address, 0, data, timeout=1000)
        finally:
            handle.close()

    def _vendor(self, request, value):
        self.handle.controlWrite(0x40, request, 0, 0, bytes([value]), timeout=1000)

    def _calibration(self):
        """Offset (ADC counts) and gain per channel and gain code, from the factory calibration in the EEPROM and
        the finer correction OpenHantek's calibration may have added."""
        block = bytes(self.handle.controlRead(0xC0, 0xA2, 0x08, 0, 80, timeout=1000))
        offset, gain = {"CH1": {}, "CH2": {}}, {"CH1": {}, "CH2": {}}
        for code, at in CAL_BYTE.items():
            for k, ch in enumerate(("CH1", "CH2")):
                offset[ch][code] = block[at + k] - 128.0
                gain[ch][code] = DEFAULT_GAIN[code]
                fine, g = block[48 + at + k], block[32 + at + k]
                if len(block) >= 64 and fine not in (0, 255):
                    offset[ch][code] += (fine - 128) / 250
                if len(block) >= 64 and g not in (0, 255):
                    gain[ch][code] *= 1 + (g - 128) / 500
        return offset, gain

    def configure(self, rate=None, range_v=None, channels=None):
        """Sample rate in samples per second, input range per channel at the probe tip ({"CH1": 5.0}), and the
        number of channels (1: CH1 alone, which allows twice the sample rate). What is not given stays."""
        rate, new, channels = self._check(rate, range_v, channels)
        self._vendor(0xE4, channels)
        self._vendor(0xE2, RATE_CODE[rate])
        self._vendor(0xE0, self.gain_code("CH1", new["CH1"]))
        self._vendor(0xE1, self.gain_code("CH2", new["CH2"]))
        self.rate, self.range_v, self.channels = rate, new, channels

    def capture(self, samples=20000):
        """A block of samples per channel, in volts at the probe tip."""
        size = -(-(samples * self.channels + SKIP_BYTES) // PACKET) * PACKET
        self._vendor(0xE3, 1)
        try:
            raw = self.handle.bulkRead(BULK_EP, size, timeout=int(1000 + 2000 * size / self.rate))
        finally:
            self._vendor(0xE3, 0)
        adc = np.frombuffer(bytes(raw), np.uint8)[SKIP_BYTES:]
        names = ("CH1", "CH2")[:self.channels]
        volts, clipped = {}, {}
        for k, ch in enumerate(names):
            counts = adc[k::self.channels][:samples]
            code = self.gain_code(ch)
            volts_per_count = 5.12 * self.gain[ch][code] * self.probe[ch] / (code * 128)
            volts[ch] = (counts.astype(np.float64) - 128 - self.offset[ch][code]) * volts_per_count
            clipped[ch] = float(np.mean((counts <= 1) | (counts >= 254)))
        return Capture(self.rate, volts, dict(self.range_v), clipped)

    def calibration_output(self, hz=1000):
        """Sets the square wave on the scope's calibration pins (32 Hz to 100 kHz, quantised as the firmware
        does); returns the frequency it will really put out."""
        hz = float(hz)
        if not 32 <= hz <= 100e3:
            raise ValueError("the calibration output goes from 32 Hz to 100 kHz")
        if hz < 1000:
            code = int((hz + 5) // 10) + 100
            coded = 32.0 if code == 103 else 10.0 * (code - 100)
        elif hz < 5750:
            code = min(int((hz + 50) // 100) + 200, 255)
            coded = 100.0 * (code - 200)
        else:
            code = int((hz + 500) // 1000)
            coded = 1000.0 * code
        self._vendor(0xE6, code)
        return 2e6 / round(2e6 / coded)  # the firmware divides a 2 MHz clock

    def info(self):
        return {"model": "Hantek 6022BE", "serial": self.serial, "firmware": f"{self.firmware_version:#06x}",
                "probe": self.probe, "range_v": self.range_v, "rate": self.rate, "channels": self.channels}

    def close(self):
        try:
            self.handle.releaseInterface(0)
            self.handle.close()
        finally:
            self.ctx.close()


class SimulatedScope(_Settings):
    """The scope's behaviour on made-up signals: each channel is a function of time (s, numpy array) giving volts
    at the probe tip. Quantised to 8 bits and clipped at the range like the real one, with a little noise."""

    def __init__(self, signals=None, probe=None, noise_v=0.01, seed=1):
        super().__init__(probe or {})
        self.signals = signals or {"CH1": square(1000, 0.0, 2.0), "CH2": lambda t: np.zeros_like(t)}
        self.noise_v, self.rng, self.t0 = noise_v, np.random.default_rng(seed), 0.0
        self.calibration_hz, self.serial, self.firmware_version = 1000.0, "simulated", 0x0210

    def configure(self, rate=None, range_v=None, channels=None):
        self.rate, self.range_v, self.channels = self._check(rate, range_v, channels)

    def capture(self, samples=20000):
        self.t0 += 0.0137  # each capture starts at another moment, like the real one
        t = self.t0 + np.arange(samples) / self.rate
        volts, clipped = {}, {}
        for ch in ("CH1", "CH2")[:self.channels]:
            full = self.range_v[ch] * 1.024  # the ADC spans 5.12 V for a 5 V range
            v = self.signals[ch](t) + self.rng.normal(0, self.noise_v, samples)
            counts = np.clip(np.round(v / full * 128) + 128, 0, 255)
            volts[ch] = (counts - 128) * full / 128
            clipped[ch] = float(np.mean((counts <= 1) | (counts >= 254)))
        return Capture(self.rate, volts, dict(self.range_v), clipped)

    def calibration_output(self, hz=1000):
        self.calibration_hz = float(hz)
        return self.calibration_hz

    def info(self):
        return {"model": "simulated 6022BE", "probe": self.probe, "range_v": self.range_v, "rate": self.rate,
                "channels": self.channels}

    def close(self):
        pass


def square(hz, low, high, duty=0.5, rise_s=0.0):
    """A square wave as a function of time, for SimulatedScope."""
    def f(t):
        phase = (t * hz) % 1.0
        v = np.where(phase < duty, high, low).astype(np.float64)
        if rise_s and len(t) > 1:  # first-order edges, like an RC on the line
            k = 1 - np.exp(-(t[1] - t[0]) / rise_s)
            for i in range(1, len(v)):
                v[i] = v[i - 1] + (v[i] - v[i - 1]) * k
        return v
    return f


def sine(hz, amplitude, offset=0.0):
    return lambda t: offset + amplitude * np.sin(2 * np.pi * hz * t)
