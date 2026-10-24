# Feetech STS3215 servo data sheet (OpenRoboticDog)

All measured and fitted servo data for this robot in one place.
Main subject: **PWM open-loop mode (Mode 2)**, which bypasses the firmware acceleration limit.
Position mode values are kept at the end for comparison.

Last updated: 2026-10-23.

## 1. Summary

| Item | PWM mode + host PD (current) | Position mode (firmware) |
|---|---|---|
| Who closes the loop | Host (Mac/Pi), 250 Hz target | Servo firmware |
| Acceleration limit | None (motor-limited) | 8.5 rad/s² trajectory generator |
| No-load speed | 5.06 rad/s @ duty 1000 | 4.7 rad/s (spec), never reached in joint range |
| Peak speed on a 0.8 rad step | ~4 rad/s (duty 800, open loop) | 2.5–2.6 rad/s |
| 0.3 rad step, 90 % time | 107–139 ms | ~400 ms trajectory |
| Stiffness (torque units) | 23.2 N·m/rad | 40 N·m/rad (fitted) |
| Max torque used | 2.32 N·m (duty 800) | 2.9 N·m (spec) |
| Sim model | `servo.mode = "pd"` / `"hybrid"` | `servo.mode = "profile"` |

Current robot configuration (`deploy/robot_config.yaml`, `pwm.hold_hips: true`):
thigh and calf servos run in PWM mode, hip servos stay in position mode. Sim equivalent: `hybrid`.

## 2. PWM motor model (open loop)

Model: `J·dω/dt = Kt·duty − b·ω`, duty in ±1000.

| Parameter | Value | How obtained |
|---|---|---|
| Speed per duty | 0.00506 rad/s per duty | Fit, open-loop duty steps |
| No-load speed | 5.06 rad/s @ 1000 | = speed per duty × 1000 (spec 4.7) |
| Speed time constant τ | 21 ms | Fit |
| Torque constant Kt | 0.0029 N·m per duty | **Spec** (2.9 N·m stall @ 1000), not measured |
| Damping b | 0.5735 N·m·s/rad | Kt / speed per duty |
| Reflected inertia J (incl. gearbox) | 0.0121 kg·m² | τ × b |
| Command delay | 8 ms | Fit to closed-loop data |

Source: `sim/sysid/pwm_params.yaml`.

### 2.1 Open-loop duty steps (raw data)

FR_thigh, leg in the air, 12.2 V battery, Mac ↔ HAT USB direct, sample period 0.8 ms.
Each step held 0.20 s. Steady speed = median of the second half of the step.

| Duty | Steady speed (rad/s) | Time to 63 % (ms) | Peak current (mA) | Travel in 0.2 s (rad) |
|---|---|---|---|---|
| +200 | −0.96 | 23 | 104 | −0.17 |
| −200 | +0.95 | 17 | 117 | +0.17 |
| +400 | −1.92 | 24 | 552 | −0.36 |
| −400 | +1.93 | 17 | 494 | +0.35 |
| +600 | −3.02 | 20 | 1112 | −0.53 |
| −600 | +2.95 | 20 | 1183 | +0.53 |
| +800 | −4.00 | 23 | 2080 | −0.71 |
| −800 | +3.91 | 20 | 2112 | +0.71 |

Speed is linear in duty (≈ 0.0049–0.0050 rad/s per duty) and symmetric in both directions.
Sign of speed is in servo raw units: +duty decreases the servo position value.

## 3. Host PD controller

`duty = clip(kp·(q_target − q) − kd·q̇, ±duty_max)`, computed on the host every loop.

| Parameter | Value | In torque units (× Kt) |
|---|---|---|
| kp | 8000 duty/rad | 23.2 N·m/rad (saturates at 0.1 rad error) |
| kd | 150 duty/(rad/s) | 0.435 N·m·s/rad |
| duty_max | 800 | 2.32 N·m |
| Loop rate | 250 Hz target | 221–227 Hz measured on Mac |
| Current limit | 2500 mA | Stop after 2 consecutive readings over limit |
| Temperature limit | 65 °C | Same |
| Stale position | 5 loops without a reading | Duty 0 for that servo |

Source: `deploy/robot_config.yaml` (`pwm:` block), `deploy/servo/pwm_control.py`.

### 3.1 Closed-loop step response (raw data)

FR_thigh, leg in the air, kp 8000, kd 150, Mac loop 221 Hz, sample period 5 ms.
Joint angles in rad (joint frame).

| From | To | Step (rad) | Time to 90 % (ms) | Overshoot (%) | Steady error (°) |
|---|---|---|---|---|---|
| −0.514 | −0.224 | +0.29 | 138 | 3.9 | 0.40 |
| −0.234 | −0.524 | −0.29 | 139 | 3.2 | 0.40 |
| −0.519 | −0.824 | −0.31 | 123 | 3.3 | 0.39 |
| −0.817 | −0.524 | +0.29 | 107 | 4.2 | 0.39 |
| −0.528 | +0.076 | +0.61 | 169 | 0.5 | 0.01 |
| +0.076 | −0.524 | −0.60 | 171 | 1.5 | 0.34 |
| −0.528 | −1.124 | −0.60 | 175 | 0.0 | 0.27 |
| −1.119 | −0.524 | +0.60 | 163 | 0.0 | 0.18 |

The sim PD model (Kt 0.0029, J 0.0121, b 0.5735, delay 8 ms) reproduces this log with 0.55° RMSE.

## 4. Sim actuator mapping

`sim/learn/servo_pd.py` converts the host PD into a MuJoCo position actuator:

| MuJoCo field | Value |
|---|---|
| actuator kp | kp × Kt = 23.2 |
| actuator kv | kd × Kt = 0.435 |
| forcerange | ±duty_max × Kt = ±2.32 N·m |
| joint damping | Kt × 1000 / no-load speed = 0.5735 |
| joint armature | 0.0121 |

`hybrid` mode puts the hips back to position-mode values: kp 40, kv 1.6, force ±2.9 N·m, damping 0.1, armature 0.01.

Domain randomization in training (`sim/learn/randomize.py`): PD kp and kv × U(0.8, 1.2) per servo.

## 5. Protocol details (must know before touching the code)

| Item | Value |
|---|---|
| Bus | 1 Mbps, STS protocol, IDs 1–12 |
| Mode register | addr 33: 0 = position, 2 = PWM open loop (EEPROM; switch with torque off) |
| Torque enable | addr 40 |
| PWM duty | addr 44 (same register as goal time), sign-magnitude, bit 10 = negative |
| Duty sign | +duty → servo position value decreases |
| Present position | addr 56, 2 bytes |
| Voltage / temperature / current | addr 62 / 63 / 69, current unit 6.5 mA |
| **Offset register** | addr 31, sign bit 11. **85 steps on every servo.** |

**Offset caveat.** Position mode subtracts the Offset from Present_Position. PWM mode does not.
All zero calibrations were done in position mode, so `pwm_control.py` subtracts the Offset from PWM-mode readings.
Without this, the robot read every PWM joint 85 steps (7.5°) off. That caused the asymmetric stance and hip jumps seen on 2026-10-22.
After the fix, readings in the two modes differ by 0–5 steps.

## 6. Position mode (for comparison)

Model: command → ServoProfile(a_max, v_max) → MuJoCo position actuator. Source: `sim/sysid/servo_params.yaml`.

| Parameter | Value |
|---|---|
| a_max (firmware trajectory) | 8.5 rad/s² (Acceleration register above 50 has no effect) |
| v_max | 4.7 rad/s (spec) |
| kp / kv | 40 / 1.6 (kv/kp = 0.04 s) |
| Force limit | 2.9 N·m (spec, not measured) |
| Delay | 0 (front hips fit 10 ms because of gravity load) |
| Fit RMSE, 116 air steps, all 12 servos | 0.33° overall, front hips 0.48° |
| Ground stand/crouch load | max 5.6 %, total current ≤ 0.58 A |
| Randomization range | kp 30–55, kv = 0.04 kp, a_max 7.5–9.5, delay 0–15 ms (hips +10 ms) |

Bus timing on the Pi (1 Mbps): 12-servo SyncRead of position 3.0 ms, 15-byte read 4.0 ms, SyncWrite 0.5 ms.

## 7. Raw data files

| File | Content |
|---|---|
| `sim/data/servo/20261022_215702_pwm_FR_thigh.csv` | PWM open-loop duty steps (t, duty, pos raw, speed, current) |
| `sim/data/servo/20261022_215805_pwm_FR_thigh_pd.csv` | PWM closed-loop PD steps (t, target, pos rad, speed, current) |
| `sim/data/servo/20261021_224030_air_hips.csv` | Position mode air steps, hips |
| `sim/data/servo/20261021_224107_air_legs.csv` | Position mode air steps, thigh and calf |

Scripts: `deploy/servo/pwm_sysid.py` (PWM tests), `deploy/servo/pwm_probe.py`, `sim/sysid/fit_servo.py` (position mode fit).

## 8. Not yet measured

- Stall torque (Kt is the 2.9 N·m spec value).
- PWM behavior under ground load. Only one servo (FR_thigh) was measured, in the air.
- Gear backlash. The sim has none. This is a candidate cause of the real-robot jitter that the sim does not show.
- Hips in PWM mode. They stay in position mode only because of the Offset bug, which is now fixed.
