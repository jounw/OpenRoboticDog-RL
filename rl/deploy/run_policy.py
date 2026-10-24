"""Policy run loop (stage 5 deployment). 50 Hz: sensors → 45-dim observation → policy → target angles → servos. Commands from a phone web joystick.

    # Pi (real robot)
    ~/openroboticdog/.venv/bin/python run_policy.py --policy policy.npz
    # Mac (same loop in MuJoCo sim, driven from the phone)
    sim/.venv/bin/mjpython deploy/run_policy.py --policy runs/g4_2/policy_latest.npz --sim

Flow: torque on and move slowly to the default pose → press START on the phone to run the policy → commands only while a stick is held (deadman);
E-STOP or a tilt fault turns torque off and waits for START; low voltage turns torque off and exits. Observation order matches sim/learn/env.py _get_obs.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "imu")); sys.path.insert(0, str(HERE / "servo"))
from gait import TrotGait  # noqa: E402
from joint_map import JointMap  # noqa: E402
from joystick_web import Command, start_server  # noqa: E402
from policy_numpy import NumpyPolicy  # noqa: E402


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- robot interfaces
class RealRobot:
    """Pi: BNO055 + STS3215 bus. With pwm=True the servos run in PWM mode and the Pi does 250 Hz PD control (bypasses the acceleration limit)."""

    def __init__(self, cfg, jm, pwm=False, use_imu=True):
        self.imu = None
        if use_imu:
            from bno055 import BNO055
            self.imu = BNO055(); self.imu.begin()
        self.jm = jm
        self.R_mount = quat_to_mat(cfg["imu_mount_quat"])
        self.last_pos = None
        self.pwm = None
        self.use_pwm, self.cfg = pwm, cfg
        self.halted = False
        self.q_last = np.array(jm.home, dtype=float)
        if pwm:
            self.pwm = self._make_pwm()
        else:
            from sts_bus import StsBus
            self.bus = StsBus()

    def _make_pwm(self):
        from pwm_control import PwmServoLoop
        pc = self.cfg.get("pwm", {})
        return PwmServoLoop(self.jm, kp=pc.get("kp", 8000.0), kd=pc.get("kd", 150.0), duty_max=pc.get("duty_max", 800),
                            rate_hz=pc.get("rate_hz", 250.0), temp_max=pc.get("temp_max", 65),
                            current_max_ma=pc.get("current_max_ma", 2500), hold_hips=pc.get("hold_hips", True))

    def halt(self):
        """Fall / E-STOP: turn torque off but keep running (IMU is still read). resume() stands the robot up again."""
        if self.halted:
            return
        if self.pwm is not None:
            self.pwm.stop()                       # duty 0 → back to position mode → torque off, port closed
            self.pwm = None
        else:
            self.bus.torque(False)
        self.halted = True

    def resume(self, home):
        """START: move slowly to the default pose in position mode (recreating the loop in PWM mode), then resume the policy."""
        if self.use_pwm:
            self.pwm = self._make_pwm()
        self.halted = False
        self.prepare(home)

    def read(self):
        """(gravity_body[3], gyro_body[3], q[12]) — gravity is a unit vector in the body frame (down is -z)"""
        if self.imu is None:                      # no IMU (direct Mac test): assume level and stationary
            gravity, gyro = np.array([0.0, 0.0, -1.0]), np.zeros(3)
        else:
            s = self.imu.sample()
            g = self.R_mount @ np.array(s["gravity"], dtype=float)
            gn = np.linalg.norm(g)
            gravity = -g / gn if gn > 1e-6 else np.array([0.0, 0.0, -1.0])   # BNO055 gravity points up (+z) → flip to point down
            gyro = self.R_mount @ np.array(s["gyro"], dtype=float)
        if self.halted:                           # torque off: use last joint values (do not touch the servo bus)
            return gravity, gyro, self.q_last.copy()
        if self.pwm is not None:
            q, _ = self.pwm.state()
            self.q_last = np.array(q)
            return gravity, gyro, np.array(q)
        pos = self.bus.read_positions()
        if self.last_pos is None:
            self.last_pos = {i: (v if v is not None else self.jm.zero[k]) for k, (i, v) in enumerate(pos.items())}
        for i, v in pos.items():
            if v is not None:
                self.last_pos[i] = v
        return gravity, gyro, np.array(self.jm.to_angles(self.last_pos))

    def write(self, angles):
        if self.pwm is not None:
            self.pwm.set_target(angles)
        else:
            self.bus.write_goals(self.jm.to_servo(angles))

    def voltage(self):
        if self.halted:
            return None
        if self.pwm is not None:
            if self.pwm.status.get("fault"):
                raise RuntimeError(self.pwm.status["fault"])
            return self.pwm.status.get("voltage")
        return self.bus.read_voltage()

    def prepare(self, home, seconds=2.0):
        if self.pwm is not None:
            # move slowly to the default pose in position mode, then start the PWM loop (so PD does not see a sudden large error)
            from sts_bus import StsBus
            bus = StsBus(); bus.torque(True); bus.set_limits(acc=20, speed=300)
            bus.write_goals(self.jm.to_servo(home)); time.sleep(seconds); bus.close()
            self.pwm.start(); time.sleep(0.05)
            return
        self.bus.torque(True); self.bus.set_limits(acc=20, speed=300)
        self.write(home); time.sleep(seconds)
        self.bus.set_limits(acc=0, speed=0)

    def stop(self, release=True, home=None):
        """release=False: move slowly to the default pose and keep torque on (normal exit). True: torque off (emergency)."""
        if self.halted and self.use_pwm:          # torque already off
            if not release and home is not None:
                from sts_bus import StsBus
                bus = StsBus(); bus.torque(True); bus.set_limits(acc=20, speed=300)
                bus.write_goals(self.jm.to_servo(home)); time.sleep(1.5); bus.close()
        elif self.pwm is not None:
            self.pwm.stop()                       # stop PWM loop (back to position mode, torque off)
            if not release and home is not None:
                from sts_bus import StsBus
                bus = StsBus(); bus.torque(True); bus.set_limits(acc=20, speed=300)
                bus.write_goals(self.jm.to_servo(home)); time.sleep(1.5); bus.close()
        else:
            if release:
                self.bus.torque(False)
            elif home is not None:
                self.bus.set_limits(acc=20, speed=300); self.write(home); time.sleep(1.5)
            self.bus.close()
        if self.imu is not None:
            self.imu.close()


class SimRobot:
    """Mac: CPU MuJoCo + ServoProfile (same servo model as the training env). viewer=True opens a window (mjpython)."""

    def __init__(self, cfg, jm, xml, viewer=True, delay=0, friction=None, servo_mode="profile"):
        import mujoco
        from servo_profile import ServoProfile
        self.mujoco = mujoco
        xml = Path(xml)
        assets = {p.name: p.read_bytes() for p in (xml.parent / "meshes").glob("*.stl")}
        self.m = mujoco.MjModel.from_xml_string(xml.read_text(encoding="utf-8"), assets)
        self.m.opt.timestep = 0.002
        if friction is not None:
            for f in ("FR_foot", "FL_foot", "RR_foot", "RL_foot"):
                self.m.geom_friction[self.m.geom(f).id, 0] = friction; self.m.geom_priority[self.m.geom(f).id] = 1
        self.servo_mode = servo_mode
        self.profile_mask = np.ones(12, dtype=bool)
        if servo_mode in ("pd", "hybrid"):          # mimic PWM mode + Pi PD: torque PD without the trajectory generator
            sys.path.insert(0, str(HERE.parent / "sim"))
            from learn.servo_pd import apply_hybrid_actuators, apply_pd_actuators
            pc = cfg.get("pwm", {})
            kw = dict(kp_duty=pc.get("kp", 8000.0), kd_duty=pc.get("kd", 150.0), duty_max=pc.get("duty_max", 800))
            if servo_mode == "pd":
                apply_pd_actuators(self.m, **kw); self.profile_mask[:] = False
            else:                                   # hip in position mode, thigh/calf PD (real-robot hold_hips setup)
                g = apply_hybrid_actuators(self.m, **kw); self.profile_mask[:] = False; self.profile_mask[g["hip_actuators"]] = True
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0); mujoco.mj_forward(self.m, self.d)
        self.imu = self.m.site("imu").id
        self.gyro_adr = self.m.sensor_adr[self.m.sensor("imu_gyro").id]
        self.profiles = [ServoProfile(8.5, 4.7, x0=h) for h in jm.home]
        self.n_sub = int(round(0.02 / 0.002))
        self.hist = [np.array(jm.home)] * 3; self.delay = delay
        self.v = None
        if viewer:
            from mujoco import viewer as mv
            self.v = mv.launch_passive(self.m, self.d); self.v.cam.distance, self.v.cam.elevation = 1.2, -20

    def read(self):
        R = self.d.site_xmat[self.imu].reshape(3, 3)
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        gyro = self.d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        return gravity, gyro, self.d.qpos[7:].copy()

    def write(self, angles):
        self.hist = [np.array(angles)] + self.hist[:-1]
        target = self.hist[self.delay]
        for _ in range(self.n_sub):
            for j, pf in enumerate(self.profiles):
                x, _ = pf.step(target[j], 0.002)
                self.d.ctrl[j] = x if self.profile_mask[j] else target[j]
            self.mujoco.mj_step(self.m, self.d)
        if self.v is not None:
            self.v.cam.lookat[:] = self.d.qpos[:3]; self.v.sync()

    def voltage(self):
        return 12.3

    def halt(self):
        self.halted = True

    def resume(self, home):
        self.mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.d.qpos[7:] = home; self.mujoco.mj_forward(self.m, self.d)
        self.halted = False
        self.prepare(home)

    def prepare(self, home, seconds=0.5):
        for _ in range(int(seconds / 0.02)):
            self.write(home)

    def stop(self, release=True, home=None):
        if self.v is not None:
            self.v.close()


# ---------------------------------------------------------------- control loop
class PolicyRunner:
    def __init__(self, cfg, jm, policy, stance_offset=None):
        self.cfg, self.jm, self.policy = cfg, jm, policy
        self.home = np.array(jm.home) + (np.zeros(12) if stance_offset is None else np.asarray(stance_offset, dtype=float))
        self.cmd_max = np.array(cfg["command_max"], dtype=float)
        self.last_act = np.zeros(12)               # for observation = previous filtered action
        self.alpha = float(getattr(policy, "action_filter_alpha", 1.0) or 1.0)
        self.prev_q = None
        self.dt = 1.0 / cfg["ctrl_hz"]
        self.idle_hold = float(cfg.get("idle_hold_s", 0.0))
        self.idle_blend = float(cfg.get("idle_blend_s", 0.5))
        self.idle_t = 0.0

    def reset(self):
        self.last_act = np.zeros(12); self.prev_q = None; self.idle_t = 0.0

    def idle_mix(self, target, cmd):
        """If the command stays zero longer than idle_hold, blend gradually into the default pose to stand fully still. Any command switches straight back to the policy."""
        if self.idle_hold <= 0:
            return target
        if np.any(np.abs(cmd) > 1e-6):
            self.idle_t = 0.0
            return target
        self.idle_t += self.dt
        if self.idle_t < self.idle_hold:
            return target
        a = min(1.0, (self.idle_t - self.idle_hold) / max(self.idle_blend, 1e-6))
        return (1 - a) * target + a * self.home

    def observation(self, gravity, gyro, q, cmd):
        dq = np.zeros(12) if self.prev_q is None else (q - self.prev_q) / self.dt
        self.prev_q = q.copy()
        return np.hstack([gravity, gyro, cmd, q - self.home, dq, self.last_act]).astype(np.float32)

    def step(self, gravity, gyro, q, cmd):
        obs = self.observation(gravity, gyro, q, cmd)
        raw = self.policy(obs)
        act = self.alpha * raw + (1.0 - self.alpha) * self.last_act      # same low-pass filter as the training env
        target = self.home + self.cfg["action_scale"] * act
        target = np.clip(target, np.array(self.jm.lower), np.array(self.jm.upper))
        step_max = self.cfg["safety"]["max_joint_step_rad"]
        target = np.clip(target, q - step_max, q + step_max)
        self.last_act = act
        return self.idle_mix(target, cmd)


def tilt_deg(gravity):
    return math.degrees(math.acos(max(-1.0, min(1.0, -gravity[2]))))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", default=None, help="policy.npz (omit with --gait for the hand-crafted gait)")
    p.add_argument("--gait", action="store_true", help="hand-crafted trot gait instead of the learned policy (deploy/gait.py)")
    p.add_argument("--gait-period", type=float, default=1.0)
    p.add_argument("--gait-stride", type=float, default=0.10)
    p.add_argument("--gait-height", type=float, default=0.03)
    p.add_argument("--gait-duty", type=float, default=0.6)
    p.add_argument("--gait-drop", type=float, default=0.0, help="lower the body height (m)")
    p.add_argument("--gait-yaw-gain", type=float, default=1.5, help="hand gait yaw feedback: gain that cancels unwanted rotation (gyro z)")
    p.add_argument("--gait-plane-yaw", default=None, help="leg-plane twist compensation (°), e.g. FR:7,FL:7,RR:-7,RL:-7 (default: robot_config leg_plane_yaw_deg)")
    p.add_argument("--cmd-const", nargs=3, type=float, default=None, help="fixed command instead of the joystick (for testing)")
    p.add_argument("--config", default=str(HERE / "robot_config.yaml"))
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--sim", action="store_true", help="run in MuJoCo sim (Mac, mjpython)")
    p.add_argument("--sim-xml", default=str(HERE.parent / "sim" / "assets" / "openroboticdog" / "openroboticdog.xml"))
    p.add_argument("--sim-delay", type=int, default=1)
    p.add_argument("--sim-friction", type=float, default=None)
    p.add_argument("--sim-servo", choices=["profile", "pd", "hybrid"], default="profile",
                   help="sim servo model: position mode (profile) / all PWM PD (pd) / hip position, rest PD (hybrid, real-robot --pwm setup)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--pwm", action="store_true", help="real robot: servo PWM mode + Pi PD (robot_config.yaml pwm section)")
    p.add_argument("--no-imu", action="store_true", help="real robot: no IMU (direct Mac test, on a stand only)")
    p.add_argument("--shadow-imu", action="store_true",
                   help="real robot (--no-imu): use the IMU (gravity, gyro) of a sim running alongside with the same joint targets")
    p.add_argument("--shadow-view", action="store_true", help="show the shadow sim in a MuJoCo window (run with mjpython)")
    p.add_argument("--cmd-from", default=None,
                   help="follow command and START state from another run_policy's /status (e.g. http://127.0.0.1:8080) — drive a sim window alongside the real robot")
    p.add_argument("--auto-enable", action="store_true", help="run the policy immediately without START (for sim testing)")
    p.add_argument("--seconds", type=float, default=0.0, help="if > 0, exit automatically after this many seconds")
    p.add_argument("--log", default=None, help="log joint targets and actual angles to CSV")
    p.add_argument("--no-idle-hold", action="store_true", help="disable default-pose hold at zero command (keep the policy pose)")
    p.add_argument("--action-scale", type=float, default=None, help="default: value stored in the policy file, else robot_config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.no_idle_hold:
        cfg["idle_hold_s"] = 0
    jm = JointMap(config_path=args.config)
    gait = None
    if args.gait:
        plane = {k: math.radians(float(v)) for k, v in cfg.get("leg_plane_yaw_deg", {}).items()}
        if args.gait_plane_yaw:
            plane = {kv.split(":")[0]: math.radians(float(kv.split(":")[1])) for kv in args.gait_plane_yaw.split(",")}
        gait = TrotGait(home=tuple(jm.home[:3]), period=args.gait_period, duty=args.gait_duty,
                        step_height=args.gait_height, stride_max=args.gait_stride, body_drop=args.gait_drop, plane_yaw=plane)
        if plane:
            print("leg-plane compensation (°):", {k: round(math.degrees(v), 1) for k, v in plane.items()}, flush=True)
        policy = lambda obs: np.zeros(12)
        print("hand-crafted gait (trot) mode", flush=True)
    else:
        if not args.policy:
            sys.exit("--policy or --gait is required")
        policy = NumpyPolicy(args.policy)
        assert policy.obs_size == 45 and policy.act_size == 12, (policy.obs_size, policy.act_size)
        scale = args.action_scale or policy.action_scale
        if scale is not None:
            cfg["action_scale"] = float(scale)
        print(f"action_scale = {cfg['action_scale']}", flush=True)
    stance = getattr(policy, "stance_offset", None)
    if stance is not None and np.any(stance):
        print("policy reference pose offset (°):", np.degrees(stance).round(1).tolist(), flush=True)
    cm = getattr(policy, "command_max", None)
    if cm is not None:
        cfg["command_max"] = [float(v) for v in cm]
        print("command max (from policy file):", cfg["command_max"], flush=True)
    runner = PolicyRunner(cfg, jm, policy, stance_offset=stance)
    if runner.alpha < 1.0:
        print(f"action low-pass filter α = {runner.alpha}", flush=True)
    cmd = Command(cfg["safety"]["command_timeout_s"])
    cmd.enable = args.auto_enable or args.cmd_const is not None
    start_server(cmd, args.port)
    follow = {"cmd": [0.0, 0.0, 0.0], "enable": False}
    if args.cmd_from:
        import json as _json
        import threading as _th
        import urllib.request as _ur

        def _poll():
            while True:
                try:
                    st = _json.loads(_ur.urlopen(args.cmd_from.rstrip("/") + "/status", timeout=0.5).read())
                    follow["cmd"] = st.get("cmd", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
                    follow["enable"] = bool(st.get("enable", False)) and not st.get("estop", False)
                except Exception:
                    follow["cmd"], follow["enable"] = [0.0, 0.0, 0.0], False
                time.sleep(0.05)
        _th.Thread(target=_poll, daemon=True).start()
        print(f"following commands from {args.cmd_from}", flush=True)
    print(f"web joystick: http://<this computer IP>:{args.port}", flush=True)

    robot = SimRobot(cfg, jm, args.sim_xml, viewer=not args.headless, delay=args.sim_delay,
                     friction=args.sim_friction, servo_mode=args.sim_servo) if args.sim else RealRobot(cfg, jm, pwm=args.pwm, use_imu=not args.no_imu)
    robot.prepare(runner.home)
    shadow = None
    if args.shadow_imu and not args.sim:
        shadow = SimRobot(cfg, jm, args.sim_xml, viewer=args.shadow_view, delay=args.sim_delay,
                          servo_mode="hybrid" if args.pwm else "profile")
        shadow.prepare(runner.home)
        print("using shadow sim IMU", flush=True)
    dt, t0, n, loop_t = runner.dt, time.monotonic(), 0, time.monotonic()
    next_t = time.monotonic()                     # absolute-time schedule (lateness is made up in the next cycle)
    work_ms = []
    reason = "stopped"
    halted, halt_reason = False, ""
    log_rows = []
    start_xy = np.array(robot.d.qpos[:2]) if args.sim else None
    yaw_of = lambda q: math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    start_yaw = yaw_of(robot.d.qpos[3:7]) if args.sim else 0.0
    try:
        while True:
            t_loop = time.monotonic()
            gravity, gyro, q = robot.read()
            if shadow is not None:
                gravity, gyro, _ = shadow.read()
                if tilt_deg(gravity) > 45:                 # if the shadow sim falls, reset only the sim instead of stopping the real robot
                    shadow.mujoco.mj_resetDataKeyframe(shadow.m, shadow.d, 0)
                    shadow.d.qpos[7:] = q; shadow.mujoco.mj_forward(shadow.m, shadow.d)
                    gravity, gyro, _ = shadow.read()
            tilt = tilt_deg(gravity)
            volt = robot.voltage() if n % 50 == 0 else volt if n else robot.voltage()
            if not halted and (cmd.estop or tilt > cfg["safety"]["max_tilt_deg"]):
                why = "E-STOP" if cmd.estop else f"tilt {tilt:.0f}°"
                print(f"{why}: torque off, press START on the web page to stand up again", flush=True)
                robot.halt(); runner.reset(); halted = True; halt_reason = why
                cmd.clear_enable()
            if halted:
                if cmd.enable and not cmd.estop:
                    if tilt > cfg["safety"]["max_tilt_deg"] * 0.5:
                        cmd.clear_enable()                     # still lying down: stand it up, then START again
                        cmd.set_status(halt=f"{halt_reason} — stand the robot up, then START (now {tilt:.0f}°)")
                    else:
                        print("START: standing up to the default pose, then resuming", flush=True)
                        robot.resume(runner.home); runner.reset(); halted = False
                        cmd.set_status(halt=None)                      # a single START stands up and is ready to drive
                        next_t = time.monotonic()
                        continue
                if halted:
                    cmd.set_status(halt=cmd.get_status().get("halt") or f"{halt_reason} — press START to stand up again",
                                   tilt_deg=round(tilt, 1), voltage=volt, loop_hz=round(1 / dt, 1))
                    next_t += dt
                    time.sleep(max(0.0, next_t - time.monotonic()))
                    continue
            if volt is not None and volt < cfg["safety"]["min_voltage"]:
                reason = f"voltage {volt:.1f} V"; break
            if getattr(robot, "pwm", None) is not None and robot.pwm.status.get("fault"):
                reason = robot.pwm.status["fault"]; break
            vx, vy, yaw = cmd.get()
            c = np.array([vx, vy, yaw]) * runner.cmd_max
            if args.cmd_const is not None:
                c = np.array(args.cmd_const, dtype=float)
            if args.cmd_from:
                c = np.array(follow["cmd"], dtype=float)          # already converted to m/s by the other side
                cmd.enable = follow["enable"]
            if cmd.enable and gait is not None:
                gc = c.copy()
                has_imu = args.sim or getattr(robot, "imu", None) is not None
                if has_imu:
                    gc[2] = c[2] - args.gait_yaw_gain * (gyro[2] - c[2])     # yaw feedback: cancel the difference between commanded and actual rotation
                target = np.array(gait.step(gc, dt))
                runner.observation(gravity, gyro, q, c)
            elif cmd.enable:
                target = runner.step(gravity, gyro, q, c)
            else:
                runner.observation(gravity, gyro, q, c)      # keep state (velocity estimate) up to date
                target = runner.home
            robot.write(target)
            if shadow is not None:
                shadow.write(target)
            if args.log:
                log_rows.append([round(time.monotonic() - t0, 4), *np.round(target, 4), *np.round(q, 4),
                                 *np.round(gravity, 4), *np.round(gyro, 4)])
            n += 1
            if n % 25 == 0:
                hz = 25 / max(1e-6, time.monotonic() - loop_t); loop_t = time.monotonic()
                work = float(np.mean(work_ms)) if work_ms else 0.0; work_ms.clear()
                cmd.set_status(q_deg={nm: round(math.degrees(v), 1) for nm, v in zip(jm.names, q)},
                               target_deg={nm: round(math.degrees(v), 1) for nm, v in zip(jm.names, target)},
                               loop_hz=round(hz, 1), work_ms=round(work, 2), voltage=volt, tilt_deg=round(tilt, 1),
                               cmd=[round(float(x), 2) for x in c])
            if args.seconds and time.monotonic() - t0 > args.seconds:
                reason = "timeout"; break
            work_ms.append((time.monotonic() - t_loop) * 1000)
            next_t += dt
            if next_t < time.monotonic() - dt:          # more than one cycle behind: reset the schedule
                next_t = time.monotonic()
            while True:                                  # coarse sleep, then busy-wait the last 1.5 ms (compensates macOS timer jitter)
                rem = next_t - time.monotonic()
                if rem <= 0:
                    break
                time.sleep(rem - 0.0015 if rem > 0.002 else 0)
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    finally:
        if args.log and log_rows:
            import csv
            with open(args.log, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["t"] + [f"tgt_{n}" for n in jm.names] + [f"q_{n}" for n in jm.names]
                                              + ["gx", "gy", "gz", "wx", "wy", "wz"]); w.writerows(log_rows)
            print(f"logged {len(log_rows)} rows → {args.log}", flush=True)
        if args.sim and args.seconds:
            dxy = np.array(robot.d.qpos[:2]) - start_xy
            dyaw = (yaw_of(robot.d.qpos[3:7]) - start_yaw + math.pi) % (2 * math.pi) - math.pi
            print(f"sim: moved {dxy.round(3)} m in {n * dt:.1f}s → mean {np.linalg.norm(dxy) / max(n * dt, 1e-6):.3f} m/s, "
                  f"turned {math.degrees(dyaw):.0f}° ({dyaw / max(n * dt, 1e-6):.2f} rad/s), height {robot.d.qpos[2]:.3f} m, tilt {tilt:.0f}°", flush=True)
        if getattr(robot, "pwm", None) is not None:
            print("pwm status:", robot.pwm.status, flush=True)
            tr = robot.pwm.trace
            if tr:
                for hid in (1, 4, 7, 10):
                    ps = [p[hid] for _, p, _ in tr]; ds = [d[hid] for _, _, d in tr]
                    print(f"  hip ID{hid}: servo value {min(ps)}~{max(ps)} (start {ps[0]}, end {ps[-1]}), duty {min(ds)}~{max(ds)}", flush=True)
        emergency = reason not in ("timeout", "Ctrl+C", "stopped")
        print(f"\n{reason}: " + ("torque off" if emergency else "returned to default pose, torque kept on"), flush=True)
        robot.stop(release=emergency, home=runner.home)


if __name__ == "__main__":
    main()
