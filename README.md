# OpenRoboticDog RL

A reinforcement-learned, joystick-controlled walking policy for the OpenRoboticDog quadruped, built by HR Robotics, the robotics club at Haeryong High School (해룡고등학교).

The policy is trained in simulation (MuJoCo Playground + Brax PPO) and runs on the real robot from a Raspberry Pi, driven from a phone web joystick.
It is based on the robot design of [garciamathias/OpenRoboticDog](https://github.com/garciamathias/OpenRoboticDog). Everything this fork adds is in [`rl/`](rl/).

| The robot | Walking on the real robot | The policy in the sim |
|---|---|---|
| <img src="rl/docs/robot.jpg" alt="OpenRoboticDog standing in its default pose" width="280"> | <img src="rl/docs/real_walk.gif" alt="OpenRoboticDog walking on the floor with the learned policy" width="280"> | <img src="rl/docs/sim_demo.gif" alt="Policy in the sim: forward, backward, sideways, turning" width="280"> |
| 3D-printed body, 12 STS3215 servos, Raspberry Pi 5, BNO055 IMU, 1.5 kg | Learned policy on the floor, real time (1×), joystick control | Forward, backward, sideways and turning, 5 s each |

## Policy status

The policy is under active development and will keep being upgraded. New versions are added to `rl/deploy/policies/` with the date, where they were trained, and what changed.

| Version | Date | Training steps | Trained on | Notes |
|---|---|---|---|---|
| `g4s_6_hybrid_filter_282M` (current) | 2026-10-23 | 281.8 M | Google Colab, single G4 GPU runtime, 2 h 16 min | First release. Forward/backward and turning work well, sideways is weaker |

Training runs on Google Colab with JAX/MJX and 8192 parallel environments. Deployment and testing run on a Raspberry Pi 5 and a MacBook.

## What is in `rl/`

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
    └── policies/                trained policies, training configs and progress plots
```

## Hardware

| Part | Spec |
|---|---|
| Servos | 12 × Feetech STS3215, 12 V version, stall torque about 2.9 N·m, IDs 1–12 |
| Joint order | FR, FL, RR, RL legs × hip, thigh, calf (FR = IDs 1–3, FL = 4–6, RR = 7–9, RL = 10–12) |
| Servo driver | Waveshare Bus Servo Driver HAT (A), mounted on the Pi 40-pin header |
| Computer | Raspberry Pi 5 (runs the policy), or a laptop connected directly to the HAT USB |
| IMU | BNO055 |
| Body | 3D-printed (original STL files, PETG) |
| Mass | 1.497 kg total, battery included |

## Wiring and power

| Item | Spec |
|---|---|
| Battery | 3S LiPo 2200 mAh, 11.1 V nominal, 11.8–12.4 V measured |
| Bench supply | 12 V 30 A DC supply for development. Use the battery or the supply, never both at once |
| Power input | HAT green screw terminal (V = +, G = −), 16 AWG wire. The HAT DC jack is in parallel with it |
| Fuse | Inline 10 A (15 A for walking tests) |
| Servo buses | HAT TTL port A: front legs (IDs 1–6). Port B: rear legs (IDs 7–12). 12 V, 1 Mbps, daisy-chained |
| Servo link | HAT servo USB-C (CH343 USB-UART) to Pi USB-A, HAT switch set to USB. Appears as `/dev/ttyACM0` on the Pi and `/dev/cu.usbmodem*` on macOS. The ESP32 / GPIO UART path on the HAT is not used |
| Pi power | 5 V from the HAT through the 40-pin header. No undervoltage flags (`vcgencmd get_throttled` = 0x0) |
| IMU wiring | BNO055 VIN to Pi pin 1 (3.3 V), SDA to pin 3, SCL to pin 5, GND to pin 6. I2C bus 1, address 0x28 |
| Cooling | Pi fan forced to maximum speed |
| Measured current | Standing, all 12 servos: up to 0.58 A. One servo in PWM mode at duty 800, in the air: up to 2.1 A |
| Software cutoffs | Below 10.5 V, or above 2.5 A or 65 °C on any servo (2 consecutive readings) |
| Network | Phone web joystick over Wi-Fi on port 8080. The Pi can also run its own 5 GHz hotspot for low latency |
| Not measured yet | Total current while walking |

## Policy details

Current policy: `rl/deploy/policies/g4s_6_hybrid_filter_282M.npz`. MLP 512-256-128, 45 observations, 12 actions at 50 Hz.

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

The full training configuration is in `rl/deploy/policies/g4s_6_train_config.json` and the learning curves are in `rl/deploy/policies/g4s_6_progress.png`.

## Setup

```bash
cd rl
python3 -m venv .venv
.venv/bin/pip install -r sim/requirements.txt        # add jax[cuda12] on a GPU machine
.venv/bin/pip install scservo-sdk pyserial           # real robot only
```

On macOS, use `mjpython` (installed with MuJoCo) instead of `python` whenever a viewer window opens. All commands below run from `rl/`.

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

### Calibration

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

This reproduces the current policy. It was trained on Google Colab with one G4 GPU runtime (281.8 M steps in 2 h 16 min).
Each evaluation writes a Brax checkpoint to `<out>/brax_ckpt/<step>` (resume with `--restore`) and exports `policy_latest.npz` for `deploy/run_policy.py`.
See `rl/sim/learn/README.md` for details and `rl/sim/sysid/SERVO_DATASHEET.md` for the servo measurements behind the servo models.

## Known limitations

- Sideways walking reaches only about 60 % of the command.
- On the real robot the policy jitters more than in the sim while walking. Gear backlash and the IMU path are the main suspects and are not modeled yet.
- Hips are still in position mode on the real robot. Running them in PWM mode is untested.

## Credits

The robot design, 3D files, Fusion360 files and the original `max_v1` code are by [Mathias Garcia](https://github.com/garciamathias) in [garciamathias/OpenRoboticDog](https://github.com/garciamathias/OpenRoboticDog), released under the MIT License as stated in the original README.
This fork adds the `rl/` folder and this README. All other original files are unchanged, and the original README follows below.

---

# Original project

![OpenRoboticDog](illustration.png)

> **Note:** Currently, only the 3D files and Fusion360 files are available in this repository. The code implementation will be added soon.

## OpenRoboticDog

### Required Hardware
The following materials are needed to build the quadruped robot:

| Component | Quantity | Price per unit | Total |
|-----------|----------|----------------|--------|
| PLA Filament | ~500g | - | - |
| STS3215 Servo Motor | 12 | $15 | $180 |
| Motor Control Board | 2 | $11 | $22 |
| USB-C Cable | 2 | $7 | $14 |
| Power Supply | 2 | $10 | $20 |
| M3 Screws | 80 | - | $5 |

**Total Cost:** ~$241 (excluding PLA filament cost)

> **Note:** Most of these components are the same as those used in the [LeRobot project](https://github.com/TheRobotStudio/SO-ARM100?tab=readme-ov-file) by Hugging Face.

### Description
This project aims to create an affordable, autonomous quadruped robot that anyone can build at home. The goal is to develop a robot capable of performing basic autonomous movements like walking, turning, and basic navigation while keeping the total cost under $250 (excluding 3D printing material).

Key features of this project:
- **Affordable**: Total cost under $250 for all electronic components
- **Home-built**: All parts can be 3D printed at home or through a printing service
- **Autonomous**: Capable of performing basic movements without remote control
- **Open-source**: Complete documentation and code will be available for everyone
- **Educational**: Great project for learning robotics, programming, and 3D printing

The robot will be able to perform various autonomous movements such as:
- Walking forward and backward
- Turning left and right
- Basic obstacle avoidance
- Standing up and lying down
- Maintaining balance

This project is designed to be accessible to hobbyists, students, and robotics enthusiasts who want to learn about quadruped robotics without breaking the bank.

### Features
- Quadruped robot control
- Open-source hardware and software
- Modular design
- Easy to build and customize

### Getting Started
Instructions for setting up and using the project will be added soon.

### License
This project is open source and available under the MIT License.
