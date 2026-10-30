"""Lunes' own driver for the Xbox 360 Kinect: colour and depth frames as NumPy arrays, straight from USB.

It talks to the Kinect's camera through libusb (python-libusb1), so no closed SDK is involved. Windows only needs its
own generic WinUSB driver on the camera ("Kinect for Windows Camera", USB 045E:02AE), chosen once in Device Manager
(see the README). The protocol is the one the OpenKinect project documented: commands travel in control transfers
("GM" out, "RB" back), and each stream arrives in isochronous packets with a 12-byte header that marks the first,
middle and last packet of a frame.

Depth comes as 11-bit disparity. It is turned into millimetres with this Kinect's own factory calibration, read from
the camera at start (zero-plane distance, pixel size, emitter baseline and shift); OpenKinect's fitted formula, good
to about 1 % from 0.5 to 4 m, stands in if the camera does not give it. The tilt motor is not driven: on the 1473
model it sits behind the audio chip, which needs Microsoft's firmware.

One Kinect object streams for the whole process; the dashboard and Hume share it.
"""
import atexit
import struct
import threading
import time

import cv2
import numpy as np
import usb1

VENDOR, CAMERA = 0x045E, 0x02AE
COLOR_EP, DEPTH_EP = 0x81, 0x82
W, H = 640, 480
TRANSFERS, PACKETS = 8, 32  # queued per stream, so a busy PC does not drop packets
# depth: projector always on, 11-bit, 640x480, 30 fps; colour: Bayer, 640x480, 30 fps. Both mirrored, the way the
# Kinect SDK gave them, so the calibrations in config/machine.json made with it stay valid.
START = ((0x105, 0x00), (0x06, 0x00), (0x12, 0x03), (0x13, 0x01), (0x14, 0x1E), (0x06, 0x02), (0x17, 0x01),
         (0x0C, 0x00), (0x0D, 0x01), (0x0E, 0x1E), (0x05, 0x01), (0x47, 0x01))
STOP = ((0x06, 0x00), (0x05, 0x00))
BAYER = cv2.COLOR_BayerGB2BGR  # the colour sensor's pattern starts G R / B G, which OpenCV calls GB
_raw = np.arange(2048, dtype=np.float64)
FITTED_MM = np.where(_raw < 1080, 123.6 * np.tan(_raw / 2842.5 + 1.1863), 0).round().astype(np.uint16)  # 2047: none


def factory_mm(shift, emitter_cm, reference_mm, pixel_mm):
    """Raw disparity (0..2047) to millimetres from the camera's own calibration; 0 where there is no depth."""
    metric = ((_raw - 4 * shift) / 4 - 0.375) * pixel_mm
    with np.errstate(divide="ignore", invalid="ignore"):
        mm = 10 * (metric * reference_mm / (emitter_cm - metric) + reference_mm)
    mm[~np.isfinite(mm) | (mm < 300) | (mm > 10000)] = 0
    mm[2047] = 0
    return mm.round().astype(np.uint16)


class _Stream:
    """Puts one stream's packets back together; keeps the latest whole frame."""

    def __init__(self, flag, payload, frame_bytes):
        self.flag, self.payload, self.frame_bytes = flag, payload, frame_bytes
        self.packets = -(-frame_bytes // payload)
        self.buf = bytearray(self.packets * payload)
        self.n = None  # packets of the frame being assembled; None until a frame's first packet arrives
        self.seq = 0
        self.frame, self.stamp, self.frames = None, 0.0, 0
        self.fresh = threading.Condition()

    def packet(self, data):
        if len(data) < 12 or data[0] != 0x52 or data[1] != 0x42:  # "RB"
            return
        flag, seq = data[3], data[5]
        if flag == self.flag | 1:
            self.n = 0
        elif self.n is None:
            return
        elif seq != (self.seq + 1) & 0xFF or self.n >= self.packets:  # a lost packet spoils the frame
            self.n = None
            return
        self.seq = seq
        body = data[12:]
        start = self.n * self.payload
        self.buf[start:start + len(body)] = body
        self.n += 1
        if flag == self.flag | 5:
            if self.n == self.packets:
                with self.fresh:
                    self.frame, self.stamp = bytes(self.buf[:self.frame_bytes]), time.time()
                    self.frames += 1
                    self.fresh.notify_all()
            self.n = None

    def latest(self, timeout):
        """The newest frame no older than timeout, waiting up to timeout for one; None if none comes."""
        with self.fresh:
            self.fresh.wait_for(lambda: self.frame is not None and time.time() - self.stamp < timeout, timeout)
            return self.frame if self.frame is not None and time.time() - self.stamp < timeout else None


class Kinect:
    def __init__(self):
        self._open()

    def _open(self):
        self.ctx = usb1.USBContext()
        self.ctx.open()
        device = next((d for d in self.ctx.getDeviceIterator(skip_on_error=True)
                       if (d.getVendorID(), d.getProductID()) == (VENDOR, CAMERA)), None)
        if device is None:
            self.ctx.close()
            raise RuntimeError("the Kinect's camera is not on the USB (check its 12 V adapter and the cable)")
        try:
            self.handle = device.open()
            self.handle.claimInterface(0)
        except (usb1.USBErrorNotSupported, usb1.USBErrorAccess, usb1.USBErrorNotFound) as e:
            self.ctx.close()
            raise RuntimeError("Windows has the Kinect's camera on another driver: give 'Kinect for Windows Camera' "
                               f"the WinUSB driver in Device Manager, as the README says ({e})")
        self.lock, self.tag = threading.Lock(), 0
        self.color_stream = _Stream(0x80, 1920 - 12, W * H)
        self.depth_stream = _Stream(0x70, 1760 - 12, W * H * 11 // 8)
        self.transfers, self.resubmit, self.pumping = [], True, True
        for stream, endpoint in ((self.depth_stream, DEPTH_EP), (self.color_stream, COLOR_EP)):
            size = device.getMaxISOPacketSize(endpoint)
            for _ in range(TRANSFERS):
                transfer = self.handle.getTransfer(iso_packets=PACKETS)
                transfer.setIsochronous(endpoint, PACKETS * size, callback=self._arrived, user_data=stream)
                transfer.submit()
                self.transfers.append(transfer)
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()
        # also when Lunes is stopped halfway (Ctrl-C while it starts): libusb's own clean-up at exit waits forever
        # for transfers nobody cancelled, and the process never ends
        atexit.register(self.close)
        for register, value in START:
            self._write(register, value)
        self.mm, self.calibration = self._factory_table()
        if self.depth_stream.latest(3.0) is None or self.color_stream.latest(3.0) is None:
            self.close()
            raise RuntimeError("the Kinect started but sends no pictures")

    def _pump(self):
        while self.pumping:
            try:
                self.ctx.handleEventsTimeout(0.2)
            except usb1.USBErrorInterrupted:
                pass

    def _arrived(self, transfer):
        if transfer.getStatus() != usb1.TRANSFER_COMPLETED:  # cancelled by close(), or the Kinect unplugged
            return
        stream = transfer.getUserData()
        for status, data in transfer.iterISO():
            if status == usb1.TRANSFER_COMPLETED and data:
                stream.packet(data)
        if self.resubmit:
            transfer.submit()

    def _command(self, cmd, data):
        with self.lock:
            self.tag = (self.tag + 1) & 0xFFFF
            self.handle.controlWrite(0x40, 0, 0, 0, struct.pack("<2sHHH", b"GM", len(data) // 2, cmd, self.tag) + data,
                                     timeout=1000)
            for _ in range(500):
                reply = self.handle.controlRead(0xC0, 0, 0, 0, 0x200, timeout=1000)
                if 8 <= len(reply) < 0x200:
                    break
                time.sleep(0.002)
            else:
                raise RuntimeError("the Kinect does not answer")
            magic, _, got_cmd, got_tag = struct.unpack_from("<2sHHH", reply)
            if (magic, got_cmd, got_tag) != (b"RB", cmd, self.tag):
                raise RuntimeError(f"the Kinect answered something else ({magic!r} {got_cmd:#x} {got_tag:#x})")
            return reply[8:]

    def _write(self, register, value):
        self._command(0x03, struct.pack("<HH", register, value))

    def _factory_table(self):
        """(raw-to-mm table, where it came from): the camera's factory depth calibration, or the fitted formula."""
        try:
            shift = struct.unpack_from("<H", self._command(0x16, struct.pack("<5H", 0x00, 0, 1, 30, 0)), 2)[0]
            emitter_cm, _, reference_mm, pixel_mm = struct.unpack_from("<4f", self._command(0x04, bytes(10)), 94)
        except (RuntimeError, struct.error, usb1.USBError):
            return FITTED_MM, "fitted formula (the camera gave no calibration)"
        if not (100 <= shift <= 400 and 5 <= emitter_cm <= 10 and 50 <= reference_mm <= 300 and 0.05 <= pixel_mm <= 0.2):
            return FITTED_MM, f"fitted formula (implausible calibration {shift} {emitter_cm} {reference_mm} {pixel_mm})"
        return factory_mm(shift, emitter_cm, reference_mm, pixel_mm), {
            "shift": shift, "emitter_cm": round(emitter_cm, 4), "reference_mm": round(reference_mm, 3),
            "pixel_mm": round(pixel_mm, 5)}

    def _frame(self, stream):
        frame = stream.latest(2.0)
        if frame is None:  # the cable was pulled or the Kinect hung: open it again once
            self.reconnect()
            frame = (self.color_stream if stream.flag == 0x80 else self.depth_stream).latest(2.0)
            if frame is None:
                raise RuntimeError("the Kinect sends no pictures")
        return frame

    def color(self):
        """BGR image, 640x480."""
        bayer = np.frombuffer(self._frame(self.color_stream), np.uint8).reshape(H, W)
        return cv2.cvtColor(bayer, BAYER)

    def depth(self):
        """Depth in millimetres, 640x480 uint16; 0 where the sensor could not measure."""
        b = np.frombuffer(self._frame(self.depth_stream), np.uint8).reshape(-1, 11).astype(np.uint16)
        # eight 11-bit values in every 11 bytes, most significant bit first
        raw = np.stack([(b[:, 0] << 3) | (b[:, 1] >> 5),
                        ((b[:, 1] & 0x1F) << 6) | (b[:, 2] >> 2),
                        ((b[:, 2] & 0x03) << 9) | (b[:, 3] << 1) | (b[:, 4] >> 7),
                        ((b[:, 4] & 0x7F) << 4) | (b[:, 5] >> 4),
                        ((b[:, 5] & 0x0F) << 7) | (b[:, 6] >> 1),
                        ((b[:, 6] & 0x01) << 10) | (b[:, 7] << 2) | (b[:, 8] >> 6),
                        ((b[:, 8] & 0x3F) << 5) | (b[:, 9] >> 3),
                        ((b[:, 9] & 0x07) << 8) | b[:, 10]], axis=1)
        return self.mm[raw.reshape(H, W)]

    def info(self):
        return {"color_frames": self.color_stream.frames, "depth_frames": self.depth_stream.frames,
                "depth_calibration": self.calibration}

    def reconnect(self):
        self.close()
        self._open()

    def close(self):
        if not self.pumping:
            return
        try:
            for register, value in STOP:
                self._write(register, value)
        except (usb1.USBError, RuntimeError):  # unplugged: nothing to stop
            pass
        self.resubmit = False
        for transfer in self.transfers:
            try:
                transfer.cancel()
            except usb1.USBErrorNotFound:  # already finished
                pass
        t0 = time.time()
        while any(t.isSubmitted() for t in self.transfers) and time.time() - t0 < 2:
            time.sleep(0.01)
        self.pumping = False
        self.pump.join(1)
        try:
            self.handle.releaseInterface(0)
        except usb1.USBError:
            pass
        self.handle.close()
        self.ctx.close()


def depth_colormap(depth_mm, near=800, far=2000):
    """False-colour view of a depth image; unknown pixels are black."""
    d = np.clip((depth_mm.astype(np.float32) - near) / (far - near), 0, 1)
    img = cv2.applyColorMap((255 - d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[depth_mm == 0] = 0
    return img
