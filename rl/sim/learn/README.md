# learn — locomotion reinforcement learning (stage 3)

| File | Role |
|---|---|
| `scene.py` | Robot MJCF -> training MJCF (meshes removed, explicit inertia, sensors added, foot friction priority) |
| `servo_jax.py` | JAX version of the servo trajectory generator (ServoProfile) |
| `servo_pd.py` | Simulated motor model for PWM drive mode (Pi-side duty PD converted to torque) |
| `env.py` | `RoboDogJoystick` env: 45 observations, 12 actions, rewards, termination, pushes, command sampling, servo model and delay |
| `randomize.py` | Domain randomization (foot friction 0.05~1.5 log-uniform, mass ±5 %, center of mass ±2 cm, kp/kv, joint friction) |
| `train.py` | Brax PPO training -> `params.pkl`, `policy.npz`, `progress.csv` |
| `policy.py` | Numpy inference for `policy.npz` (for Pi deployment) |
| `play.py` | 50 Hz sim-to-sim playback in CPU MuJoCo (delay, friction, noise options) |
| `monitor.py` | Show `progress.csv` as a table and plots |

## Mac (CPU, for validation)
```bash
cd rl/sim
.venv/bin/python -m learn.train --out runs/smoke --num-envs 8 --num-timesteps 4000 --num-evals 2 --batch-size 8 --num-minibatches 2
.venv/bin/mjpython -m learn.play runs/smoke/policy.npz --cmd 0.2 0 0
```

## H100 (Linux, CUDA 12)
```bash
git clone <origin> && cd <repo>/rl/sim
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "jax[cuda12]" playground brax warp-lang pyyaml pytest
python -c "import jax; print(jax.devices())"        # should list a CudaDevice
python -m learn.train --impl warp --out runs/h100_1 --num-envs 8192 --num-timesteps 200000000
```
- If `--impl warp` (mujoco_warp) does not work, use `--impl jax`. 200M steps with 8192 envs typically takes 30~60 min on an H100.
- Progress is in `runs/h100_1/progress.csv` (eval/episode_reward, eval/avg_episode_length). Episode length approaching 1000 means the robot no longer falls.
- When done, copy only `runs/h100_1/policy.npz` to the Mac, validate sim-to-sim with `learn.play`, then deploy to the Pi (stage 5, `rl/deploy`).

## Design summary
- Policy 50 Hz, physics 500 Hz. Action -> target angle = default pose + 0.4·action -> delay 0~2 steps (hips get an extra +1 with some probability) -> servo model -> actuator.
- Servo model (`servo.mode`): `profile` = position-mode servos, ServoProfile(a_max U(7.5, 9.5)) -> position actuator (kp U(30,55), kv=0.04kp); `pd` = all joints PWM + Pi PD; `hybrid` = hip in position mode, thigh/calf PWM PD (hardware setup).
- Observation (45): gravity 3, gyro 3, command 3, joint angle - default 12, joint velocity 12, previous action 12. No linear velocity (not measurable on hardware). The value network uses privileged_state (linear velocity, contacts, foot velocities, a_max, delay, etc.).
- Rewards: same terms as Go1 joystick, scaled to robot size: max foot height 0.03 m, command range vx ±0.3, vy ±0.15, yaw ±1.2.
- Termination: torso flipped (up_z<0) or IMU height below 6 cm.
- Contact detection assumes flat ground (foot height < radius + 5 mm). Switch to contact sensors for rough terrain.
