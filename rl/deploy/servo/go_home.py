"""Slowly move all servos to the default standing pose (zero in servo_map.yaml) or a saved pose (poses.yaml). Torque stays on afterwards.

    ~/openroboticdog/.venv/bin/python go_home.py [--speed 300] [--acc 20] [--release]
    ~/openroboticdog/.venv/bin/python go_home.py --pose lie      # lie down
"""
import argparse
import sys
import time
from pathlib import Path

import scservo_sdk as scs
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sts_reader import find_port, BAUDRATE  # noqa: E402

SERVO_MAP = Path(__file__).resolve().parents[1] / "servo_map.yaml"
POSES = Path(__file__).resolve().parents[1] / "poses.yaml"
ADDR = {"Torque_Enable": (40, 1), "Acceleration": (41, 1), "Goal_Position": (42, 2), "Goal_Speed": (46, 2),
        "Present_Position": (56, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", default="auto")
    p.add_argument("--speed", type=int, default=300, help="steps/s (300 ≈ 26°/s)")
    p.add_argument("--acc", type=int, default=20)
    p.add_argument("--release", action="store_true", help="turn torque off after arrival")
    p.add_argument("--pose", default=None, help="pose name in poses.yaml (default: standing pose zero)")
    args = p.parse_args()
    servos = yaml.safe_load(SERVO_MAP.read_text(encoding="utf-8"))["servos"]
    if args.pose:
        pose = yaml.safe_load(POSES.read_text(encoding="utf-8"))[args.pose]
        for e in servos.values():
            e["zero"] = int(pose[e["id"]])
    port = scs.PortHandler(find_port(args.port))
    assert port.openPort() and port.setBaudRate(BAUDRATE)
    pk = scs.PacketHandler(0)

    def w(i, name, v):
        a, n = ADDR[name]
        r, _ = (pk.write1ByteTxRx if n == 1 else pk.write2ByteTxRx)(port, i, a, v)
        if r != scs.COMM_SUCCESS:
            raise IOError(f"ID {i} {name} write failed")

    def r(i, name):
        a, n = ADDR[name]
        v, res, _ = (pk.read1ByteTxRx if n == 1 else pk.read2ByteTxRx)(port, i, a)
        if res != scs.COMM_SUCCESS:
            raise IOError(f"ID {i} {name} read failed")
        return v

    try:
        for name, e in servos.items():
            i = e["id"]
            w(i, "Goal_Position", r(i, "Present_Position"))   # avoid a jump when torque turns on
            w(i, "Torque_Enable", 1)
            w(i, "Acceleration", args.acc)
            w(i, "Goal_Speed", args.speed)
        for name, e in servos.items():
            w(e["id"], "Goal_Position", e["zero"])
        t0 = time.time()
        while time.time() - t0 < 8.0:
            err = {e["id"]: abs(r(e["id"], "Present_Position") - e["zero"]) for e in servos.values()}
            if max(err.values()) <= 12:
                break
            time.sleep(0.2)
        print(f"Remaining error (steps) after {time.time() - t0:.1f}s: {err}")
        if args.release:
            for e in servos.values():
                w(e["id"], "Torque_Enable", 0)
            print("Torque off")
    finally:
        port.closePort()


if __name__ == "__main__":
    main()
