"""STS3215 servo model: mimics the servo's internal trajectory generator.

Measured (2026-10-12): when the goal position changes, the servo ramps its velocity up and down within an
acceleration limit (~7.7 rad/s²) to reach the goal (triangular/trapezoidal velocity). In the simulator this
trajectory is fed as the position actuator ctrl.
"""
import math


class ServoProfile:
    """Time-optimal tracking with acceleration and velocity limits. step(goal, dt) → (position, velocity)"""

    def __init__(self, a_max, v_max, x0=0.0, v0=0.0):
        self.a_max, self.v_max = a_max, v_max
        self.x, self.v = x0, v0

    def step(self, goal, dt):
        e = goal - self.x
        # speed at which braking must start now to stop at the goal (discrete-time correction: a·dt/2 margin)
        v_stop = math.sqrt(max(0.0, 2 * self.a_max * abs(e))) - self.a_max * dt / 2
        v_des = math.copysign(min(self.v_max, max(0.0, v_stop)), e)
        dv = max(-self.a_max * dt, min(self.a_max * dt, v_des - self.v))
        self.v += dv
        self.x += self.v * dt
        if abs(goal - self.x) < 1e-6 and abs(self.v) <= self.a_max * dt:
            self.x, self.v = goal, 0.0
        return self.x, self.v
