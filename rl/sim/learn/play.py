"""Run a trained policy in CPU MuJoCo at 50 Hz and display it (stage 4 sim-to-sim). Includes servo model, delay and noise.

    .venv/bin/mjpython -m learn.play runs/xxx/policy.npz --cmd 0.2 0 0
    options: --delay 1 (control steps), --friction 0.3, --seconds 10, --headless (metrics only, no viewer)
"""
import argparse
import time

import mujoco
import numpy as np

from learn import scene
from learn.policy import NumpyPolicy
from sysid.servo_model import ServoProfile

CTRL_DT, SIM_DT = 0.02, 0.002


def run(policy, cmd, seconds=10.0, delay=0, friction=None, a_max=8.5, viewer=True, noise=True, seed=0, action_scale=None):
    model = scene.load_full_model()   # model with meshes for nicer visuals (dynamics same as the training model)
    model.opt.timestep = SIM_DT
    if friction is not None:
        for f in scene.FEET:
            model.geom_friction[model.geom(f).id, 0] = friction
            model.geom_priority[model.geom(f).id] = 1
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    home = data.qpos[7:].copy()
    so = getattr(policy, "stance_offset", None)
    if so is not None:
        home = home + so
        data.qpos[7:] = home; mujoco.mj_forward(model, data)
    imu = model.site("imu").id
    rng = np.random.default_rng(seed)
    profiles = [ServoProfile(a_max, 4.7, x0=h) for h in home]
    hist = [home.copy()] * 3
    last_act = np.zeros(12)
    n_sub = int(round(CTRL_DT / SIM_DT))
    lowers, uppers = model.jnt_range[1:].T
    scale = action_scale or getattr(policy, "action_scale", None) or 0.4
    log = []

    def obs():
        R = data.site_xmat[imu].reshape(3, 3)
        gravity = R.T @ np.array([0, 0, -1.0])
        gyro = data.sensordata[model.sensor_adr[model.sensor("imu_gyro").id]:][:3].copy()
        q, dq = data.qpos[7:], data.qvel[6:]
        if noise:
            gravity = gravity + rng.uniform(-0.03, 0.03, 3)
            gyro = gyro + rng.uniform(-0.1, 0.1, 3)
            q = q + rng.uniform(-0.01, 0.01, 12)
            dq = dq + rng.uniform(-1.0, 1.0, 12)
        return np.hstack([gravity, gyro, cmd, q - home, dq, last_act])

    def ctrl_step():
        nonlocal last_act, hist
        raw = policy(obs())
        a_ = getattr(policy, "action_filter_alpha", 1.0) or 1.0
        act = a_ * raw + (1 - a_) * last_act
        target = np.clip(home + scale * act, lowers, uppers)
        hist = [target] + hist[:-1]
        applied = hist[delay]
        for _ in range(n_sub):
            for j, pf in enumerate(profiles):
                data.ctrl[j], _ = pf.step(applied[j], SIM_DT)
            mujoco.mj_step(model, data)
        last_act = act
        R = data.site_xmat[imu].reshape(3, 3)
        vel_body = R.T @ data.qvel[:3]          # free-joint linear velocity is in world frame
        feet_z = [data.site_xpos[model.site(f).id][2] for f in scene.FEET]
        lifted = sum(z > 0.012 + 0.005 for z in feet_z)
        log.append((data.time, data.qpos[2], R[2, 2], vel_body[0], vel_body[1], data.qvel[5], lifted, max(feet_z)))
        return R[2, 2] < 0.0

    steps = int(seconds / CTRL_DT)
    if viewer:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as v:
            v.cam.distance, v.cam.elevation = 1.2, -20
            for _ in range(steps):
                if not v.is_running():
                    break
                t = time.monotonic()
                fell = ctrl_step()
                v.cam.lookat[:] = data.qpos[:3]
                v.sync()
                if fell:
                    print("fell", flush=True)
                    break
                time.sleep(max(0.0, CTRL_DT - (time.monotonic() - t)))
    else:
        for _ in range(steps):
            if ctrl_step():
                break
    return np.array(log)


def summarize(log, cmd):
    fell = bool(log[-1, 2] < 0.0)
    v = log[len(log) // 5:, 3:5].mean(axis=0) if len(log) > 5 else log[:, 3:5].mean(axis=0)
    return {"seconds": round(float(log[-1, 0]), 2), "fell": fell, "mean_vx": round(float(v[0]), 3),
            "mean_vy": round(float(v[1]), 3), "mean_yaw_rate": round(float(log[len(log) // 5:, 5].mean()), 3),
            "cmd": list(map(float, cmd)), "mean_height": round(float(log[:, 1].mean()), 3),
            "feet_lifted_avg": round(float(log[:, 6].mean()), 2), "max_foot_z": round(float(log[:, 7].max()), 3)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("policy")
    p.add_argument("--cmd", nargs=3, type=float, default=[0.2, 0.0, 0.0])
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--delay", type=int, default=0)
    p.add_argument("--friction", type=float, default=None)
    p.add_argument("--a-max", type=float, default=8.5)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--action-scale", type=float, default=None, help="default: value stored in the policy file, else 0.4")
    args = p.parse_args()
    log = run(NumpyPolicy(args.policy), np.array(args.cmd), args.seconds, args.delay, args.friction, args.a_max,
              viewer=not args.headless, noise=not args.no_noise, action_scale=args.action_scale)
    print(summarize(log, args.cmd))


if __name__ == "__main__":
    main()
