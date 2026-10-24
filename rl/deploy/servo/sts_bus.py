"""STS3215 bus (Pi): read 12 positions and voltage, SyncWrite goal angles, torque on/off. For the 50 Hz loop."""
import sys
from pathlib import Path

import scservo_sdk as scs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sts_reader import find_port, BAUDRATE, IDS  # noqa: E402

A_TORQUE, A_ACC, A_GOAL_POS, A_SPEED, A_POS, A_VOLT = 40, 41, 42, 46, 56, 62


class StsBus:
    def __init__(self, port="auto", ids=IDS):
        self.port_name = find_port(port)
        self.port = scs.PortHandler(self.port_name)
        if not (self.port.openPort() and self.port.setBaudRate(BAUDRATE)):
            raise IOError(f"Cannot open {self.port_name}")
        self.pk = scs.PacketHandler(0)
        self.ids = list(ids)
        self.rd = scs.GroupSyncRead(self.port, self.pk, A_POS, 2)
        for i in self.ids:
            self.rd.addParam(i)
        self.fail_count = 0

    def read_positions(self):
        """{id: position}. Non-responding servos keep the previous value (None if none)."""
        if self.rd.txRxPacket() != scs.COMM_SUCCESS:
            self.fail_count += 1
        out = {}
        for i in self.ids:
            out[i] = self.rd.getData(i, A_POS, 2) if self.rd.isAvailable(i, A_POS, 2) else None
        return out

    def read_voltage(self, servo_id=1):
        v, res, _ = self.pk.read1ByteTxRx(self.port, servo_id, A_VOLT)
        return v / 10.0 if res == scs.COMM_SUCCESS else None

    def write_goals(self, goals):
        """Write {id: servo value} in one packet."""
        g = scs.GroupSyncWrite(self.port, self.pk, A_GOAL_POS, 2)
        for i, v in goals.items():
            v = int(min(max(v, 0), 4095))
            g.addParam(i, [scs.SCS_LOBYTE(v), scs.SCS_HIBYTE(v)])
        return g.txPacket() == scs.COMM_SUCCESS

    def set_limits(self, acc=0, speed=0):
        """When running the policy, lift the servo's internal speed/acceleration limits (0 = max) to match the training model (firmware limits only)."""
        for i in self.ids:
            self.pk.write1ByteTxRx(self.port, i, A_ACC, acc)
            self.pk.write2ByteTxRx(self.port, i, A_SPEED, speed)

    def torque(self, on):
        for i in self.ids:
            self.pk.write1ByteTxRx(self.port, i, A_TORQUE, 1 if on else 0)

    def close(self):
        self.port.closePort()
