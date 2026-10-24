# Policies

The policy is under active development and will keep being upgraded. Each new version is added here with the date, where it was trained, and what changed.

| Version | Date | Training steps | Trained on | Notes |
|---|---|---|---|---|
| `g4s_6_hybrid_filter_282M` (current) | 2026-10-23 | 281.8 M | Google Colab, single G4 GPU runtime, 2 h 16 min | First release. Forward/backward and turning work well, sideways is weaker |

Training runs on Google Colab with JAX/MJX and 8192 parallel environments. Deployment and testing run on a Raspberry Pi 5 and a MacBook.

## g4s_6_hybrid_filter_282M

Files: `g4s_6_hybrid_filter_282M.npz` (policy), `g4s_6_train_config.json` (full training config), `g4s_6_progress.png` (learning curves).
MLP 512-256-128, 45 observations, 12 actions at 50 Hz.

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

## Known limitations

- Sideways walking reaches only about 60 % of the command.
- On the real robot the policy jitters more than in the sim while walking. Gear backlash and the IMU path are the main suspects and are not modeled yet.
- Hips are still in position mode on the real robot. Running them in PWM mode is untested.
