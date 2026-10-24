# OpenRoboticDog RL: usage

All commands run from `rl/`. Hardware and wiring: [`docs/HARDWARE.md`](docs/HARDWARE.md). Policy versions: [`deploy/policies/README.md`](deploy/policies/README.md).

```
rl/
├── sim/
│   ├── assets/openroboticdog/   MuJoCo model (MJCF + STL meshes)
│   ├── learn/                   training environment, PPO training, monitoring, sim playback, demo GIF
│   ├── sysid/                   measured servo parameters and data sheet
│   └── requirements.txt
├── docs/                        hardware doc, robot photo, demo GIFs
└── deploy/
    ├── run_policy.py            50 Hz policy runner (real robot or sim) with web joystick
    ├── policy_numpy.py          NumPy inference for exported policies
    ├── joystick_web.py, joystick.html   phone web joystick (port 8080)
    ├── joint_map.py, servo_map.yaml, robot_config.yaml   calibration and robot config
    ├── servo/                   Feetech STS3215 bus: position mode and PWM mode with host PD
    ├── imu/bno055.py            BNO055 IMU over I2C
    ├── gait.py, leg_ik.py       hand-coded trot gait (baseline, no learning)
    ├── robodog-joystick.service systemd unit to start the runner at boot on the Pi
    └── policies/                trained policies, training configs and progress plots
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r sim/requirements.txt        # add jax[cuda12] on a GPU machine
.venv/bin/pip install scservo-sdk pyserial           # real robot only
```

On macOS, use `mjpython` (installed with MuJoCo) instead of `python` whenever a viewer window opens.

## Run in the sim

```bash
# viewer + web joystick at http://127.0.0.1:8080 (press START, then hold and drag the pads)
.venv/bin/mjpython deploy/run_policy.py --policy deploy/policies/g4s_6_hybrid_filter_282M.npz --sim --sim-servo hybrid

# headless numeric check: walk forward at 0.3 m/s for 6 s
.venv/bin/python deploy/run_policy.py --policy deploy/policies/g4s_6_hybrid_filter_282M.npz \
    --sim --sim-servo hybrid --headless --seconds 6 --cmd-const 0.3 0 0

# regenerate the 2x2 demo GIF
.venv/bin/python sim/learn/make_demo_gif.py
```

## Run on the real robot

Always start with the robot on a stand and the legs in the air.

```bash
# on the Pi (servo USB + BNO055 IMU)
.venv/bin/python deploy/run_policy.py --policy deploy/policies/g4s_6_hybrid_filter_282M.npz --pwm

# on a laptop connected directly to the HAT USB, without IMU (stand testing only)
.venv/bin/python deploy/run_policy.py --policy deploy/policies/g4s_6_hybrid_filter_282M.npz --pwm --no-imu
```

Then open `http://<robot IP>:8080` on a phone on the same network, press START, and hold and drag the pads.

- `--pwm` runs thighs and calves in servo PWM mode with PD on the host (gains in `robot_config.yaml`), which bypasses the servo firmware's 8.5 rad/s² acceleration limit. Hips stay in position mode, matching the `hybrid` training model.
- Safety: tilt above 60° or E-STOP turns torque off and waits. Press START to stand up and resume. Low battery or servo over-temperature / over-current ends the program.
- `deploy/robodog-joystick.service` starts the runner automatically at boot on the Pi (`sudo cp` it to `/etc/systemd/system/` and `systemctl enable` it).

## Calibration

- `deploy/servo_map.yaml`: servo zero (default pose) and direction per joint. Put the robot in the default pose and run `deploy/servo/go_home.py` to check it.
- `deploy/robot_config.yaml` `imu_mount_quat`: IMU-to-body rotation. Measure gravity with the robot standing on a flat floor and rotate it onto the body −z axis.
- STS3215 Offset register (address 31): position mode subtracts it from the reported position, PWM mode does not. `servo/pwm_control.py` subtracts it so both modes use the same units.

## Train

```bash
cd sim
../.venv/bin/python -u -m learn.train --impl jax --out runs/my_run --num-envs 8192 --num-timesteps 300000000 --num-evals 60 \
  --set servo.mode='"hybrid"' --set action_scale=0.5 --set reward_config.max_foot_height=0.04 \
  --set 'command_config.a=[0.5,0.25,1.5]' --set reward_config.tracking_sigma_lin=0.03 \
  --set 'stance_offset=[0.0,-0.1499,0.0184,0.0,-0.1491,0.0176,0.0,-0.1497,0.0178,0.0,-0.1491,0.0176]' \
  --set action_filter_alpha=0.4
../.venv/bin/python -m learn.monitor --csv runs/my_run/progress.csv --out runs/my_run   # table + progress.png
```

This reproduces the current policy (trained on Google Colab with one G4 GPU, 281.8 M steps in 2 h 16 min).
Each evaluation writes a Brax checkpoint to `<out>/brax_ckpt/<step>` (resume with `--restore`) and exports `policy_latest.npz` for `deploy/run_policy.py`.
See [`sim/learn/README.md`](sim/learn/README.md) for details and [`sim/sysid/SERVO_DATASHEET.md`](sim/sysid/SERVO_DATASHEET.md) for the servo measurements behind the servo models.
