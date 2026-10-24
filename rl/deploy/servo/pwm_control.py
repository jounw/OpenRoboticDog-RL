"""Position-control all 12 servos in PWM open-loop mode (Mode 2) with PD running on the Pi. Bypasses the position-mode acceleration limit (≈8.5 rad/s²).

    duty = clip(kp·(q_target − q) − kd·q̇ , ±duty_max)     [duty units: ±1000]

- Loop target 250 Hz: SyncRead position (3 ms) + SyncWrite duty (0.5 ms). The 50 Hz policy only updates the target via set_target().
- Safety: zero outward duty beyond joint limits; full stop on temperature/current over-limit (duty 0 → back to position mode → torque off).
- Sign (measured 2026-10-22): +duty = decreasing servo value. Address 44 is sign-magnitude (bit10 = negative).
"""
import math
import sys
import threading
import time
from pathlib import Path

import scservo_sdk as scs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sts_reader import find_port, BAUDRATE, IDS  # noqa: E402

A_MODE, A_TORQUE, A_GOAL_TIME, A_POS, A_VOLT, A_TEMP, A_CURRENT = 33, 40, 44, 56, 62, 63, 69
STEPS_PER_RAD = 4096 / (2 * math.pi)
CURRENT_MA = 6.5


def duty_word(d):
    d = int(max(-1000, min(1000, round(d))))
    return abs(d) | (0x400 if d < 0 else 0)


class PwmServoLoop:
    def __init__(self, joint_map, kp=8000.0, kd=150.0, duty_max=800, rate_hz=250.0, port="auto",
                 temp_max=65, current_max_ma=2500, status_every=5, hold_hips=False):
        """kp: duty per rad, kd: duty per rad/s (PD on the Pi). Matched to the sim actuator kp/kv via torque conversion."""
        self.jm = joint_map
        self.kp, self.kd, self.duty_max = kp, kd, duty_max
        self.dt = 1.0 / rate_hz
        self.temp_max, self.current_max = temp_max, current_max_ma
        self.status_every = status_every
        self.port = scs.PortHandler(find_port(port))
        if not (self.port.openPort() and self.port.setBaudRate(BAUDRATE)):
            raise IOError("Cannot open servo port")
        self.pk = scs.PacketHandler(0)
        self.ids = list(joint_map.ids)
        # hold_hips: hip servos (name contains hip) stay in position mode holding the target instead of PWM (avoids jumps on mode switch)
        self.hip_ids = {i for n, i in zip(joint_map.names, joint_map.ids) if "hip" in n} if hold_hips else set()
        self.pwm_ids = [i for i in self.ids if i not in self.hip_ids]
        self.rd = scs.GroupSyncRead(self.port, self.pk, A_POS, 2)
        for i in self.ids:
            self.rd.addParam(i)
        self._status_idx = 0                      # status: read one servo individually per cycle (9-byte SyncRead returns shifted values)
        self._over = {i: 0 for i in self.ids}     # consecutive over-limit count
        self.temps = {i: None for i in self.ids}; self.currents = {i: None for i in self.ids}; self.volts = {i: None for i in self.ids}
        # Offset register (31, sign=bit11): position mode subtracts it from reads, PWM mode does not (measured 85 steps, 2026-10-23).
        # Zero calibration is in position-mode units, so subtract Offset from PWM-mode reads to match.
        self.offset = {}
        for i in self.ids:
            v = self.pk.read2ByteTxRx(self.port, i, 31)[0]
            self.offset[i] = -(v & 0x7FF) if v & 0x800 else (v & 0x7FF)
        self.in_pwm = False
        self.lock = threading.Lock()
        self.target = list(joint_map.home)
        self.q = list(joint_map.home); self.dq = [0.0] * 12
        self.pos_raw = dict(zip(self.ids, joint_map.zero))
        self.status = {"voltage": None, "temp_max": None, "current_max_ma": None, "loop_hz": 0.0, "fault": None,
                       "read_fail": 0, "stale_max": 0}
        self.stale = {i: 0 for i in self.ids}          # consecutive cycles with no position read
        self.stale_limit = 5
        self.running = False
        self.thread = None
        self.trace = []                           # (t, {id: pos_raw}, {id: duty}) for the first trace_s seconds
        self.trace_s = 2.0

    # ---- servo mode ----
    def _enter_pwm(self):
        """Switch only thigh/calf (pwm_ids) to PWM mode. Hips (hold_hips) stay in position mode with torque on.
        Send PD duty from the current error immediately to minimize the torque gap right after switching."""
        self._read()
        q = self.jm.to_angles(self.pos_raw)
        duties = {}
        for k, i in enumerate(self.ids):
            u = max(-self.duty_max, min(self.duty_max, self.kp * (self.target[k] - q[k])))
            duties[i] = -u * self.jm.direction[k]
        for k, i in enumerate(self.ids):
            self.pk.write1ByteTxRx(self.port, i, A_TORQUE, 1)
            if i in self.hip_ids:
                self.pk.write1ByteTxRx(self.port, i, A_MODE, 0)                 # hip: ensure position mode
                v = int(round(self.jm.zero[k] + self.jm.direction[k] * (self.target[k] - self.jm.home[k]) * STEPS_PER_RAD))
                self.pk.write2ByteTxRx(self.port, i, 42, v)
            else:
                self.pk.write2ByteTxRx(self.port, i, A_GOAL_TIME, 0)
                self.pk.write1ByteTxRx(self.port, i, A_MODE, 2)
                self.pk.write2ByteTxRx(self.port, i, A_GOAL_TIME, duty_word(duties[i]))
        self.in_pwm = True

    def _exit_pwm(self):
        self.in_pwm = False
        self._write_duty({i: 0 for i in self.pwm_ids})
        for i in self.ids:
            self.pk.write1ByteTxRx(self.port, i, A_TORQUE, 0)
            self.pk.write1ByteTxRx(self.port, i, A_MODE, 0)       # restore mode (EEPROM) after torque off

    def _write_duty(self, duties):
        g = scs.GroupSyncWrite(self.port, self.pk, A_GOAL_TIME, 2)
        for i, d in duties.items():
            w = duty_word(d)
            g.addParam(i, [scs.SCS_LOBYTE(w), scs.SCS_HIBYTE(w)])
        g.txPacket()

    def _read(self):
        ok = self.rd.txRxPacket() == scs.COMM_SUCCESS
        if not ok:
            self.status["read_fail"] += 1
        for i in self.ids:
            if ok and self.rd.isAvailable(i, A_POS, 2):
                v = self.rd.getData(i, A_POS, 2)
                if self.in_pwm and i in self.pwm_ids:
                    v -= self.offset[i]                       # PWM-mode read → position-mode units
                self.pos_raw[i] = v
                self.stale[i] = 0
            else:
                self.stale[i] += 1
        self.status["stale_max"] = max(self.status["stale_max"], max(self.stale.values()))

    def _read_status(self):
        """Read voltage/temperature/current of one servo (round-robin). Discard out-of-range values; fault only after 2 consecutive over-limits."""
        i = self.ids[self._status_idx % len(self.ids)]; self._status_idx += 1
        data, res, _ = self.pk.readTxRx(self.port, i, A_VOLT, 9)
        if res != scs.COMM_SUCCESS or len(data) < 9:
            return
        volt, temp = data[0] / 10.0, data[1]
        raw = data[7] | (data[8] << 8); cur = (raw & 0x7FFF) * CURRENT_MA
        if not (5.0 <= volt <= 16.0 and 0 <= temp <= 120 and cur <= 6000):
            return                                 # corrupted value
        prev = self.temps[i]
        if prev is not None and abs(temp - prev) > 15:
            return                                 # a jump of more than 15°C in one cycle is treated as a read error
        self.volts[i], self.temps[i], self.currents[i] = volt, temp, cur
        over = temp > self.temp_max or cur > self.current_max
        self._over[i] = self._over[i] + 1 if over else 0
        known_t = [v for v in self.temps.values() if v is not None]; known_c = [v for v in self.currents.values() if v is not None]
        known_v = [v for v in self.volts.values() if v is not None]
        self.status.update(voltage=min(known_v), temp_max=max(known_t), current_max_ma=max(known_c))
        if self._over[i] >= 2:
            self.status["fault"] = f"ID {i} temp {temp}°C current {cur:.0f} mA"

    # ---- public API ----
    def set_target(self, angles):
        with self.lock:
            self.target = [float(a) for a in angles]

    def state(self):
        with self.lock:
            return list(self.q), list(self.dq)

    def start(self):
        self._enter_pwm()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self._exit_pwm()
        self.port.closePort()

    def _loop(self):
        n, t_hz = 0, time.perf_counter()
        prev_q = None
        while self.running:
            t0 = time.perf_counter()
            self._read()
            q = self.jm.to_angles(self.pos_raw)
            dq = [0.0] * 12 if prev_q is None else [(a - b) / self.dt for a, b in zip(q, prev_q)]
            prev_q = q
            with self.lock:
                target = self.target
                self.q, self.dq = q, dq
            duties = {}
            for k, i in enumerate(self.ids):
                e = target[k] - q[k]
                u = self.kp * e - self.kd * dq[k]                  # joint angle units (rad) → duty
                if q[k] <= self.jm.lower[k] and u < 0:
                    u = 0.0                                        # never push further past the limit
                if q[k] >= self.jm.upper[k] and u > 0:
                    u = 0.0
                u = max(-self.duty_max, min(self.duty_max, u))
                if self.stale[i] >= self.stale_limit:
                    u = 0.0                                        # don't drive a servo whose position can't be read (no PD on stale values)
                duties[i] = -u * self.jm.direction[k]              # +duty = servo value decreases = joint angle × direction decreases
            if self.status["fault"]:
                duties = {i: 0 for i in self.ids}
                self.running = False
            if self.hip_ids:                                       # hips get a position-mode target
                g = scs.GroupSyncWrite(self.port, self.pk, 42, 2)
                for k, i in enumerate(self.ids):
                    if i in self.hip_ids:
                        v = int(round(self.jm.zero[k] + self.jm.direction[k] * (target[k] - self.jm.home[k]) * STEPS_PER_RAD))
                        g.addParam(i, [scs.SCS_LOBYTE(v), scs.SCS_HIBYTE(v)])
                g.txPacket()
            self._write_duty({i: duties[i] for i in self.pwm_ids})
            if n * self.dt < self.trace_s:
                self.trace.append((n * self.dt, dict(self.pos_raw), {i: int(d) for i, d in duties.items()}))
            n += 1
            if n % self.status_every == 0:
                self._read_status()
                now = time.perf_counter()
                self.status["loop_hz"] = self.status_every / (now - t_hz); t_hz = now
            time.sleep(max(0.0, self.dt - (time.perf_counter() - t0)))
        self._write_duty({i: 0 for i in self.pwm_ids})
