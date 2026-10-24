"""Hand-crafted trot gait (no learning). Joystick (vx, vy, yaw) → 12 joint targets at 50 Hz.

Diagonal pairs (FR+RL, FL+RR) alternate. During stance the foot pushes opposite to the body velocity;
during swing it returns forward along a half-circle arc. With a zero command the robot stands still.
Stride length = commanded speed × period, clipped to stride_max to stay within servo limits (slow acceleration).
"""
import math

from leg_constants import LEGS
from leg_ik import leg_fk, leg_ik

LEG_ORDER = ("FR", "FL", "RR", "RL")
PHASE = {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5}


class TrotGait:
    def __init__(self, home=(0.0, -0.5236, 0.2793), period=1.0, duty=0.6, step_height=0.03,
                 stride_max=0.10, body_drop=0.0, cmd_deadband=0.02, plane_yaw=None):
        """plane_yaw: {leg: rad} angle by which the real leg swing plane is rotated from the body x axis (CCW + seen from above).
        Compensates assembly twist: foot targets are rotated by -plane_yaw about the hip (2026-10-22 real robot: front legs bend right, rear legs bend left)."""
        self.plane_yaw = plane_yaw or {}
        self.period, self.duty, self.step_height = period, duty, step_height
        self.stride_max, self.deadband = stride_max, cmd_deadband
        self.phase = 0.0
        self.home_q = tuple(home)
        self.p0 = {}
        for leg in LEG_ORDER:
            x, y, z = leg_fk(leg, home)
            self.p0[leg] = (x, y, z + body_drop)      # raising the feet lowers the body
        self.stepping = False

    def foot_target(self, leg, cmd, phase):
        vx, vy, w = cmd
        px, py, pz = self.p0[leg]
        hx, hy, _ = LEGS[leg]["hip"]
        rx, ry = hx + px, hy + py                       # foot position relative to body center
        ux, uy = -(vx - w * ry), -(vy + w * rx)         # foot velocity relative to the body during stance
        ts = self.duty * self.period
        half = (ux * ts / 2, uy * ts / 2)
        n = math.hypot(*half)
        if n > self.stride_max / 2:
            half = (half[0] * self.stride_max / 2 / n, half[1] * self.stride_max / 2 / n)
        if phase < self.duty:                           # stance: -half → +half
            s = phase / self.duty
            return (px - half[0] + 2 * half[0] * s, py - half[1] + 2 * half[1] * s, pz)
        s = (phase - self.duty) / (1 - self.duty)       # swing: +half → -half, half-circle
        return (px + half[0] - 2 * half[0] * s, py + half[1] - 2 * half[1] * s,
                pz + self.step_height * math.sin(math.pi * s))

    def step(self, cmd, dt):
        """cmd=(vx, vy, yaw) m/s, rad/s → [12] joint angles (FR, FL, RR, RL × hip, thigh, calf)"""
        moving = max(abs(cmd[0]), abs(cmd[1]), abs(cmd[2]) * 0.1) > self.deadband
        if moving or self.stepping:
            self.phase = (self.phase + dt / self.period) % 1.0
            # when the command goes away, stop at the moment both pairs are in stance (phase≈0 or 0.5)
            if not moving and (self.phase < dt / self.period or abs(self.phase - 0.5) < dt / self.period):
                self.stepping = False
                self.phase = 0.0
            else:
                self.stepping = True
        q = []
        for leg in LEG_ORDER:
            if self.stepping:
                ph = (self.phase + PHASE[leg]) % 1.0
                tgt = self.foot_target(leg, cmd, ph)
            else:
                tgt = self.p0[leg]
            q.extend(leg_ik(leg, self._compensate(leg, tgt)))
        return q

    def _compensate(self, leg, tgt):
        """Leg-plane twist compensation: rotate the body-frame target (x, y) by -ψ about the hip origin → straight motion in the body frame on the real robot"""
        psi = self.plane_yaw.get(leg, 0.0)
        if not psi:
            return tgt
        c, s = math.cos(-psi), math.sin(-psi)
        x, y, z = tgt
        return (c * x - s * y, s * x + c * y, z)
