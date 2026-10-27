"""Flashes an Intel HEX file to an ATmega2560 (Arduino Mega, MKS Gen) through its STK500v2 bootloader, then verifies it.

Opening the port resets the board into the bootloader, the same way the Arduino IDE does. Written because avrdude's
reset did not reach this board's bootloader through its FT232R, while a plain serial open does.

    python tools/flash_mks.py COM8 lunes_motion.ino.hex
"""
import sys
import time

import serial

PAGE = 256  # ATmega2560 flash page in bytes
CMD_SIGN_ON, CMD_LOAD_ADDRESS, CMD_ENTER_PROGMODE, CMD_LEAVE_PROGMODE = 0x01, 0x06, 0x10, 0x11
CMD_PROGRAM_FLASH, CMD_READ_FLASH = 0x13, 0x14
STATUS_OK = 0x00


def read_hex(path):
    """Returns the flash image from address 0, gaps filled with 0xFF."""
    data, base = {}, 0
    for line in open(path, encoding="ascii"):
        line = line.strip()
        if not line.startswith(":"):
            continue
        rec = bytes.fromhex(line[1:])
        if sum(rec) & 0xFF:
            raise ValueError(f"bad checksum in {line}")
        n, addr, kind, payload = rec[0], (rec[1] << 8) | rec[2], rec[3], rec[4:4 + rec[0]]
        if kind == 0x00:
            for i, b in enumerate(payload):
                data[base + addr + i] = b
        elif kind == 0x01:
            break
        elif kind == 0x02:
            base = ((payload[0] << 8) | payload[1]) << 4
        elif kind == 0x04:
            base = ((payload[0] << 8) | payload[1]) << 16
    image = bytearray(b"\xff" * (max(data) + 1))
    for a, b in data.items():
        image[a] = b
    return bytes(image)


class Bootloader:
    def __init__(self, port):
        self.s = serial.Serial(port, 115200, timeout=0.05)  # opening the port resets the board
        self.seq = 0

    def send(self, body):
        self.seq = (self.seq + 1) & 0xFF
        msg = bytes([0x1B, self.seq, len(body) >> 8, len(body) & 0xFF, 0x0E]) + bytes(body)
        chk = 0
        for b in msg:
            chk ^= b
        self.s.write(msg + bytes([chk]))

    def receive(self, timeout=1.0):
        deadline = time.time() + timeout
        buf = bytearray()
        while time.time() < deadline:
            buf += self.s.read(512)
            start = buf.find(b"\x1b")
            if start < 0 or len(buf) - start < 5:
                continue
            frame = buf[start:]
            size = (frame[2] << 8) | frame[3]
            if len(frame) < 6 + size:
                continue
            chk = 0
            for b in frame[:6 + size]:
                chk ^= b
            if frame[4] != 0x0E or chk != 0 or frame[1] != self.seq:
                raise IOError(f"bad frame from the bootloader: {bytes(frame[:16]).hex()}")
            return bytes(frame[5:5 + size])
        raise TimeoutError("the bootloader did not answer")

    def command(self, body, timeout=1.0):
        self.send(body)
        reply = self.receive(timeout)
        if reply[0] != body[0] or reply[1] != STATUS_OK:
            raise IOError(f"command 0x{body[0]:02x} failed: {reply[:4].hex()}")
        return reply

    def sign_on(self, window=2.0):
        t0 = time.time()
        while time.time() - t0 < window:
            self.send([CMD_SIGN_ON])
            try:
                reply = self.receive(0.1)
            except (TimeoutError, IOError):
                self.s.reset_input_buffer()
                continue
            if reply[0] == CMD_SIGN_ON and reply[1] == STATUS_OK:
                return reply[3:].decode("ascii", "replace")
        raise TimeoutError("no bootloader answered after the reset")

    def load_address(self, byte_addr):
        word = (byte_addr // 2) | 0x80000000  # bit 31: extended address for parts over 128 KB
        self.command([CMD_LOAD_ADDRESS, *word.to_bytes(4, "big")])


def main():
    port, path = sys.argv[1], sys.argv[2]
    image = read_hex(path)
    pages = (len(image) + PAGE - 1) // PAGE
    image = image.ljust(pages * PAGE, b"\xff")
    print(f"{path}: {len(image)} bytes, {pages} pages")

    bl = Bootloader(port)
    print("bootloader:", bl.sign_on())
    bl.command([CMD_ENTER_PROGMODE, 200, 100, 25, 32, 0, 0x53, 3, 0xAC, 0x53, 0, 0])
    try:
        # The bootloader erases pages in order as it programs them, so pages go in sequence from address 0.
        bl.load_address(0)
        for p in range(pages):
            chunk = image[p * PAGE:(p + 1) * PAGE]
            bl.command([CMD_PROGRAM_FLASH, PAGE >> 8, PAGE & 0xFF, 0xC1, 0x0A, 0x40, 0x4C, 0x20, 0, 0, *chunk])
            print(f"\r  writing {p + 1}/{pages}", end="", flush=True)
        print()
        bl.load_address(0)
        for p in range(pages):
            reply = bl.command([CMD_READ_FLASH, PAGE >> 8, PAGE & 0xFF, 0x20])
            if reply[2:2 + PAGE] != image[p * PAGE:(p + 1) * PAGE]:
                raise IOError(f"verify failed at page {p} (address 0x{p * PAGE:05x})")
            print(f"\r  verifying {p + 1}/{pages}", end="", flush=True)
        print("\nwritten and verified")
    finally:
        bl.command([CMD_LEAVE_PROGMODE, 1, 1])  # the bootloader then starts the new firmware
        bl.s.close()


if __name__ == "__main__":
    main()
