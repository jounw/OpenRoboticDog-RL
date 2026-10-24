"""Read the current positions of all 12 STS3215 servos at once (Pi, feetech-servo-sdk).

Torque is left untouched. Read-only, so the robot does not move.
"""
import glob

import scservo_sdk as scs

BAUDRATE = 1_000_000
PORT_CANDIDATES = ["/dev/ttyACM*", "/dev/ttyUSB*", "/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/ttyAMA0"]


def find_port(port="auto"):
    """If port is "auto", return the first available of USB serial (the HAT's CH343) → GPIO UART."""
    if port != "auto":
        return port
    for pat in PORT_CANDIDATES:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    raise IOError("Servo port not found: connect the HAT's servo USB-C port to the Pi with a cable and set the switch to USB")

ADDR_PRESENT_POSITION = 56
IDS = list(range(1, 13))


class StsReader:
    def __init__(self, port="auto", ids=IDS):
        port = find_port(port)
        self.port_name = port
        self.port = scs.PortHandler(port)
        if not (self.port.openPort() and self.port.setBaudRate(BAUDRATE)):
            raise IOError(f"Cannot open {port} (check HAT switch position, cable, and whether another program is using it)")
        self.packet = scs.PacketHandler(0)
        self.ids = list(ids)
        self.group = scs.GroupSyncRead(self.port, self.packet, ADDR_PRESENT_POSITION, 2)
        for i in self.ids:
            self.group.addParam(i)

    def read_positions(self):
        """{id: position or None}. None for non-responding servos."""
        out = {}
        if self.group.txRxPacket() != scs.COMM_SUCCESS:
            return {i: None for i in self.ids}
        for i in self.ids:
            if self.group.isAvailable(i, ADDR_PRESENT_POSITION, 2):
                out[i] = self.group.getData(i, ADDR_PRESENT_POSITION, 2)
            else:
                out[i] = None
        return out

    def ping(self):
        """List of responding IDs."""
        alive = []
        for i in self.ids:
            _, result, _ = self.packet.ping(self.port, i)
            if result == scs.COMM_SUCCESS:
                alive.append(i)
        return alive

    def close(self):
        self.port.closePort()
