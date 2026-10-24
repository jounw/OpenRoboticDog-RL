"""BNO055 IMU driver (Raspberry Pi, standard library only).

Accesses /dev/i2c-1 directly via ioctl, so no smbus2/adafruit packages are needed.
Starts in NDOF mode (accel + gyro + magnetometer fusion) and fetches gyro, Euler,
quaternion, linear acceleration, gravity, temperature and calibration status in one burst read.
"""
import fcntl
import os
import struct
import time

I2C_SLAVE = 0x0703

CHIP_ID = 0xA0
ADDR_DEFAULT = 0x28

REG_CHIP_ID = 0x00
REG_PAGE_ID = 0x07
REG_GYR_DATA = 0x14     # burst start: GYR(6) EUL(6) QUA(8) LIA(6) GRV(6) TEMP(1) CALIB(1) = 34 bytes
REG_CALIB_STAT = 0x35
REG_UNIT_SEL = 0x3B
REG_OPR_MODE = 0x3D
REG_PWR_MODE = 0x3E
REG_SYS_TRIGGER = 0x3F

MODE_CONFIG = 0x00
MODE_NDOF = 0x0C
BURST_LEN = 34

# UNIT_SEL: bit0 accel 0=m/s², bit1 gyro 1=rad/s, bit2 Euler 0=deg, bit7 0=Windows orientation convention
UNIT_SEL_VALUE = 0x02

QUAT_SCALE = 1.0 / 16384.0   # 1 = 2^14 LSB
GYRO_SCALE = 1.0 / 900.0     # rad/s mode: 1 rad/s = 900 LSB
EULER_SCALE = 1.0 / 16.0     # 1° = 16 LSB
ACC_SCALE = 1.0 / 100.0      # 1 m/s² = 100 LSB


def decode_burst(buf):
    """Decode a 34-byte burst into a dict. Pure function, so it can be tested on a PC."""
    if len(buf) != BURST_LEN:
        raise ValueError(f"burst length {len(buf)} != {BURST_LEN}")
    vals = struct.unpack("<16h", buf[:32])
    gyro = [v * GYRO_SCALE for v in vals[0:3]]
    euler = [v * EULER_SCALE for v in vals[3:6]]          # heading, roll, pitch (deg)
    quat = [v * QUAT_SCALE for v in vals[6:10]]           # w, x, y, z
    lin_acc = [v * ACC_SCALE for v in vals[10:13]]
    gravity = [v * ACC_SCALE for v in vals[13:16]]
    temp = struct.unpack("<b", buf[32:33])[0]
    c = buf[33]
    calib = [(c >> 6) & 3, (c >> 4) & 3, (c >> 2) & 3, c & 3]   # sys, gyro, acc, mag (each 0-3)
    return {
        "q": quat,
        "gyro": gyro,
        "euler": euler,
        "lin_acc": lin_acc,
        "gravity": gravity,
        "temp": temp,
        "calib": calib,
    }


class BNO055:
    def __init__(self, bus="/dev/i2c-1", addr=ADDR_DEFAULT):
        self.fd = os.open(bus, os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, addr)

    def close(self):
        os.close(self.fd)

    def read(self, reg, n):
        os.write(self.fd, bytes([reg]))
        return os.read(self.fd, n)

    def write8(self, reg, val):
        os.write(self.fd, bytes([reg, val & 0xFF]))

    def begin(self, external_crystal=False):
        chip = self.read(REG_CHIP_ID, 1)[0]
        if chip != CHIP_ID:
            raise RuntimeError(f"chip ID 0x{chip:02x} != 0x{CHIP_ID:02x}: not a BNO055")
        self.write8(REG_OPR_MODE, MODE_CONFIG)
        time.sleep(0.03)
        self.write8(REG_PAGE_ID, 0)
        self.write8(REG_PWR_MODE, 0x00)          # normal
        self.write8(REG_UNIT_SEL, UNIT_SEL_VALUE)
        self.write8(REG_SYS_TRIGGER, 0x80 if external_crystal else 0x00)
        time.sleep(0.01)
        self.write8(REG_OPR_MODE, MODE_NDOF)
        time.sleep(0.03)
        # Fusion output (quaternion) appears a few hundred ms after the mode switch. Wait up to 2 s.
        for _ in range(100):
            if any(self.sample()["q"]):
                return
            time.sleep(0.02)
        raise RuntimeError("NDOF fusion output did not appear within 2 s")

    def sample(self):
        return decode_burst(self.read(REG_GYR_DATA, BURST_LEN))
