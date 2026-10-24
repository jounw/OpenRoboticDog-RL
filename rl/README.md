# OpenRoboticDog RL: joystick walking policy (sim-to-real)

This folder contains the code and the trained policy needed to train and run a joystick-controlled walking policy on OpenRoboticDog.
It is trained in MuJoCo Playground (MJX) with Brax PPO and runs on the real robot from a Raspberry Pi or a laptop.

| The robot | Walking on the real robot | The shared policy in the sim |
|---|---|---|
| <img src="docs/robot.jpg" alt="OpenRoboticDog standing in its default pose" width="280"> | <img src="docs/real_walk.gif" alt="OpenRoboticDog walking on the floor with the learned policy" width="280"> | <img src="docs/sim_demo.gif" alt="Policy in the sim: forward, backward, sideways, turning" width="280"> |
| 3D-printed body, 12 Feetech STS3215 servos, Raspberry Pi, BNO055 IMU, 1.5 kg | Learned policy on the floor, real time (1×), joystick control | Forward, backward, sideways and turning, 5 s each (hybrid servo model). Regenerate with `python sim/learn/make_demo_gif.py`. |

```
rl/
├── sim/
│   ├── assets/openroboticdog/   MuJoCo model (MJCF + STL meshes)
│   ├── learn/                   training environment, PPO training, monitoring, sim playback, demo GIF
│   ├── sysid/                   measured servo parameters and data sheet
│   └── requirements.txt
├── docs/                        robot photo, real walking GIF, sim demo GIF
└── deploy/
    ├── run_policy.py            50 Hz policy runner (real robot or sim) with web joystick
    ├── policy_numpy.py          NumPy inference for exported policies
    ├── joystick_web.py, joystick.html   phone web joystick (port 8080)
    ├── joint_map.py, servo_map.yaml, robot_config.yaml   calibration and robot config
    ├── servo/                   Feetech STS3215 bus: position mode and PWM mode with host PD
    ├── imu/bno055.py            BNO055 IMU over I2C
    ├── gait.py, leg_ik.py       hand-coded trot gait (baseline, no learning)
    ├── robodog-joystick.service systemd unit to start the runner at boot on the Pi
    └── policies/                trained policy, its training config and progress plot
```

## Hardware

| Part | Notes |
|---|---|
| Servos | 12 × Feetech STS3215, 1 Mbps bus, IDs 1–12 (FR, FL, RR, RL × hip, thigh, calf) |
| Servo driver | Bus servo driver HAT, connected over USB with the HAT switch on USB (`/dev/ttyACM0` on the Pi, `/dev/cu.usbmodem*` on macOS) |
| IMU | BNO055 on Pi I2C bus 1, address 0x28 (pins 1 = 3.3 V, 3 = SDA, 5 = SCL, 6 = GND) |
| Computer | Raspberry Pi (runs the policy), or a laptop connected directly to the HAT USB |
| Mass | 1.497 kg total |

## Policy

`deploy/policies/g4s_6_hybrid_filter_282M.npz`: 281.8 M PPO steps, MLP 512-256-128, 45 observations, 12 actions at 50 Hz.

| Setting | Value |
|---|---|
| Servo model | `hybrid`: hips in firmware position mode, thighs and calves in PWM mode with host PD |
| Command range | forward ±0.5 m/s, sideways ±0.25 m/s, yaw ±1.5 rad/s |
| Action scale | 0.5 rad |
| Action low-pass filter | α = 0.4 (same filter at training and deployment) |
| Stance offset | feet 2.6 cm back so front and rear legs carry equal load |
| Rewards | MuJoCo Playground Go1 joystick task, with tracking sigma scaled to the command range |
| Domain randomization | foot friction 0.05–1.5 (log-uniform), mass ±5 %, trunk payload, COM ±2 cm, servo gains ±20 %, command delay |

Command tracking in the sim (hybrid servo model, 1-step delay, 6 s per command):

| Command | Result |
|---|---|
| Forward 0.3 / 0.5 m/s | 0.31 / 0.49 m/s |
| Backward 0.2 m/s | 0.19 m/s |
| Sideways 0.2 m/s | 0.12 m/s |
| Yaw 0.5 / 1.0 / 1.5 rad/s | 0.45 / 0.84 / 1.06 rad/s |
| Stand still | 0 |

The full training configuration is in `deploy/policies/g4s_6_train_config.json` and the learning curves are in `deploy/policies/g4s_6_progress.png`.

## Setup

```bash
cd rl
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
```

## Run on the real robot

Always start with the robot on a stand and the legs in the air.

```bash
# on the Pi (servo USB + BNO055 IMU)
.venv/bin/python deploy/run_policy.py --policy deploy/policies/g4s_6_hybrid_filter_282M.npz --pwm

# on a laptop connected directly to the HAT USB, without IMU (stand testing only)
.venv/bin/python deploy/run_policy.py --policy deploy/policies/g4s_6_hybrid_filter_282M.npz --pwm --no-imu
```

Then open `http://<robot IP>:8080` on a phone on the same network.

- `--pwm` runs thighs and calves in servo PWM mode with PD on the host (gains in `robot_config.yaml`), which bypasses the servo firmware's 8.5 rad/s² acceleration limit. Hips stay in position mode, matching the `hybrid` training model.
- Safety: tilt above 60° or E-STOP turns torque off and waits; press START to stand up and resume. Low battery (< 10.5 V) or servo over-temperature / over-current ends the program.
- `deploy/robodog-joystick.service` starts the runner automatically at boot on the Pi (`sudo cp` it to `/etc/systemd/system/` and `systemctl enable` it).

### Calibration

- `deploy/servo_map.yaml`: servo zero (default pose) and direction per joint. Put the robot in the default pose and run `deploy/servo/go_home.py` to check it.
- `robot_config.yaml` `imu_mount_quat`: IMU-to-body rotation. Measure gravity with the robot standing on a flat floor and rotate it onto the body −z axis.
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

This reproduces the shared policy. 300 M steps took about 2.5 hours on a single Colab G4 GPU.
Each evaluation writes a Brax checkpoint to `<out>/brax_ckpt/<step>` (resume with `--restore`) and exports `policy_latest.npz` for `deploy/run_policy.py`.
See `sim/learn/README.md` for details and `sim/sysid/SERVO_DATASHEET.md` for the servo measurements behind the servo models.

## Known limitations

- Sideways walking reaches only about 60 % of the command.
- On the real robot the policy jitters more than in the sim while walking. Gear backlash and the IMU path are the main suspects and are not modeled yet.
- Hips are still in position mode on the real robot. Running them in PWM mode is untested.
