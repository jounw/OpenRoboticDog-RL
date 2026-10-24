"""Servo value ↔ joint angle (deploy version: uses only servo_map.yaml + robot_config.yaml)."""
import math
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
STEPS_PER_RAD = 4096 / (2 * math.pi)
JOINTS = [f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")]


class JointMap:
    def __init__(self, servo_map_path=HERE / "servo_map.yaml", config_path=HERE / "robot_config.yaml"):
        servos = yaml.safe_load(Path(servo_map_path).read_text(encoding="utf-8"))["servos"]
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        hp, lim = cfg["home_pose"], cfg["joint_limits"]
        self.names = JOINTS
        self.ids = [int(servos[n]["id"]) for n in JOINTS]
        self.direction = [int(servos[n]["direction"]) for n in JOINTS]
        self.zero = [int(servos[n]["zero"]) for n in JOINTS]
        self.home = [float(hp[n.split("_")[1]]) for n in JOINTS]
        self.lower, self.upper = [], []
        for n in JOINTS:
            part = n.split("_")[1]
            if part == "hip":
                lo, hi = lim["hip_right"] if n[1] == "R" else lim["hip_left"]
            else:
                lo, hi = lim[part]
            self.lower.append(float(lo)); self.upper.append(float(hi))

    def to_angles(self, positions):
        """{id: servo value} → [12] rad (JOINTS order)"""
        return [h + d * (positions[i] - z) / STEPS_PER_RAD
                for i, d, z, h in zip(self.ids, self.direction, self.zero, self.home)]

    def to_servo(self, angles):
        """[12] rad → {id: servo value} (clipped to joint limits)"""
        out = {}
        for k, (i, d, z, h) in enumerate(zip(self.ids, self.direction, self.zero, self.home)):
            a = min(max(angles[k], self.lower[k]), self.upper[k])
            out[i] = int(round(z + d * (a - h) * STEPS_PER_RAD))
        return out
