# OpenRoboticDog RL

A learned walking policy for [OpenRoboticDog](https://github.com/garciamathias/OpenRoboticDog), trained in simulation with reinforcement learning and run on the real robot with a phone joystick.

| The robot | Walking on the real robot | The policy in the sim |
|---|---|---|
| <img src="rl/docs/robot.jpg" alt="OpenRoboticDog standing in its default pose" width="280"> | <img src="rl/docs/real_walk.gif" alt="OpenRoboticDog walking on the floor with the learned policy" width="280"> | <img src="rl/docs/sim_demo.gif" alt="Policy in the sim: forward, backward, sideways, turning" width="280"> |

What this fork adds, all in [`rl/`](rl/):

- **Training**: MuJoCo Playground (MJX) + Brax PPO joystick task with servo models measured on the real STS3215 servos and wide domain randomization.
- **Trained policy**: forward/backward up to 0.5 m/s, turning up to about 1 rad/s, sideways walking (weaker), standing still.
- **Deployment**: a 50 Hz runner for the Raspberry Pi or a laptop, servo PWM mode with host-side PD (bypasses the servo's built-in acceleration limit), BNO055 IMU, and a phone web joystick.
- **Servo data sheet**: measured STS3215 parameters in position and PWM mode.

Start with [`rl/README.md`](rl/README.md) for setup, running in the sim, running on the robot, and training.

## Credits

The robot design, 3D files, Fusion360 files and the original `max_v1` code are by [Mathias Garcia](https://github.com/garciamathias) in [garciamathias/OpenRoboticDog](https://github.com/garciamathias/OpenRoboticDog), released under the MIT License as stated in the original README.
This fork adds the `rl/` folder and this introduction. All other original files are unchanged, and the original README follows below.

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
