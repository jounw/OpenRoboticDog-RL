"""Render a 2x2 demo GIF of the policy in the sim: forward, backward, sideways, turning.

    python sim/learn/make_demo_gif.py --policy deploy/policies/g4s_6_hybrid_filter_282M.npz --out docs/sim_demo.gif

Uses the same sim robot and policy runner as deploy/run_policy.py (hybrid servo model, 1-step command delay).
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

RL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL / "deploy")); sys.path.insert(0, str(RL / "deploy" / "servo")); sys.path.insert(0, str(RL / "deploy" / "imu"))
from joint_map import JointMap  # noqa: E402
from policy_numpy import NumpyPolicy  # noqa: E402
from run_policy import PolicyRunner, SimRobot  # noqa: E402

CLIPS = [  # (label, command [vx m/s, vy m/s, yaw rad/s], camera azimuth: roughly side-on to the motion)
    ("Forward 0.3 m/s", (0.3, 0.0, 0.0), 100),
    ("Backward 0.2 m/s", (-0.2, 0.0, 0.0), 100),
    ("Sideways 0.2 m/s", (0.0, 0.2, 0.0), 190),
    ("Turn 1.0 rad/s", (0.0, 0.0, 1.0), 120),
]


def render_clip(cfg, jm, pol, xml, cmd, azimuth, seconds, w, h, every):
    robot = SimRobot(cfg, jm, xml, viewer=False, delay=1, servo_mode="hybrid")
    runner = PolicyRunner(cfg, jm, pol, stance_offset=pol.stance_offset)
    robot.prepare(runner.home)
    states = []
    for k in range(int(seconds / runner.dt)):
        g, wv, q = robot.read()
        robot.write(runner.step(g, wv, q, np.array(cmd, dtype=float)))
        if k % every == 0:
            states.append(robot.d.qpos.copy())
    # fixed camera framing the actual path, so the floor stays still and the motion is visible
    xy = np.array([st[:2] for st in states])
    lo, hi = xy.min(0), xy.max(0)
    renderer = mujoco.Renderer(robot.m, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.r_[(lo + hi) / 2, states[0][2]]
    cam.distance, cam.elevation, cam.azimuth = 0.8 + 0.9 * float(np.linalg.norm(hi - lo)), -18.0, azimuth
    frames = []
    for st in states:
        robot.d.qpos[:] = st
        mujoco.mj_forward(robot.m, robot.d)
        renderer.update_scene(robot.d, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=str(RL / "deploy" / "policies" / "g4s_6_hybrid_filter_282M.npz"))
    p.add_argument("--out", default=str(RL / "docs" / "sim_demo.gif"))
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--colors", type=int, default=96, help="GIF palette size (smaller file)")
    args = p.parse_args()

    cfg = yaml.safe_load((RL / "deploy" / "robot_config.yaml").read_text(encoding="utf-8"))
    cfg["idle_hold_s"] = 0
    pol = NumpyPolicy(args.policy)
    cfg["action_scale"] = pol.action_scale or cfg["action_scale"]
    jm = JointMap()
    xml = RL / "sim" / "assets" / "openroboticdog" / "openroboticdog.xml"
    every = max(1, round(cfg["ctrl_hz"] / args.fps))
    args.fps = cfg["ctrl_hz"] / every                # actual frame rate after skipping

    clips = [render_clip(cfg, jm, pol, xml, c, az, args.seconds, args.width, args.height, every) for _, c, az in CLIPS]
    try:
        font = ImageFont.truetype("Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default()
    out = []
    for i in range(min(len(c) for c in clips)):
        sheet = Image.new("RGB", (2 * args.width, 2 * args.height))
        for j, (label, _, _) in enumerate(CLIPS):
            tile = Image.fromarray(clips[j][i])
            d = ImageDraw.Draw(tile)
            d.rectangle([0, 0, args.width, 30], fill=(0, 0, 0))
            d.text((10, 5), label, fill=(255, 255, 255), font=font)
            sheet.paste(tile, ((j % 2) * args.width, (j // 2) * args.height))
        out.append(sheet.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out[0].save(args.out, save_all=True, append_images=out[1:], duration=round(1000 / args.fps), loop=0, optimize=True)
    print(f"{len(out)} frames, {Path(args.out).stat().st_size / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
