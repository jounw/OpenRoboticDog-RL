"""OpenRoboticDog joystick locomotion environment (MuJoCo Playground style, modeled on the Go1 joystick task).

Observation state (45): body-frame gravity 3, gyro 3, command 3, joint angle - default pose 12, joint velocity 12, previous action 12. No linear velocity.
Action (12): target angle = default pose + action x action_scale -> ServoProfile(a_max, delay) -> position actuator ctrl.
privileged_state additionally contains linear velocity, contacts, foot velocities, etc. (for the value network).
Contact detection assumes flat ground: foot center height < radius + 5 mm.
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco_playground._src import mjx_env

from learn import scene
from learn.servo_jax import servo_profile_step

FOOT_RADIUS = 0.012
CONTACT_EPS = 0.005


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.4,
        # Policy reference pose = model home (hardware zero-calibration pose) + stance_offset (rad, 12 joints). Used to balance weight front/back
        # (2026-10-22: home puts 60% of the load on the hind legs -> move all feet 2.6 cm back = thigh -8.6°, calf +1°)
        stance_offset=[0.0] * 12,
        # Action low-pass filter: a_f = α·a + (1−α)·a_f(prev). 1.0 disables it. The 'previous action' observation is the filtered value (keeps it Markov).
        # 2026-10-22: added because the policy oscillated at 10 Hz under PWM drive (could not turn). Hardware deployment (run_policy) uses the same filter.
        action_filter_alpha=1.0,
        soft_joint_pos_limit_factor=0.95,
        servo=config_dict.create(
            mode="profile",         # "profile": position-mode servos (ServoProfile) / "pd": all PWM + Pi PD / "hybrid": hip position mode, thigh/calf PWM PD (hardware setup)
            a_max=[7.5, 9.5],       # rad/s², U per episode (profile mode)
            v_max=4.7,
            delay_prob=0.5,         # probability of a 1-control-step (20 ms) delay
            hip_extra_delay_prob=0.5,
            pd=config_dict.create(kp_duty=8000.0, kd_duty=150.0, duty_max=800, kt=0.0029, omega_noload=5.06, armature=0.0121),  # sysid/pwm_params.yaml
        ),
        # The noise, reward, push and command settings below follow the MuJoCo Playground Go1 joystick (proven values) as-is.
        # Only two things changed for robot size: command velocity range (a) and max foot height (max_foot_height).
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(joint_pos=0.03, joint_vel=1.5, gyro=0.2, gravity=0.05, linvel=0.1),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Tracking.
                tracking_lin_vel=1.0, tracking_ang_vel=0.5,
                # Base reward.
                lin_vel_z=-0.5, ang_vel_xy=-0.05, orientation=-5.0,
                # Other.
                dof_pos_limits=-1.0, pose=0.5, termination=-1.0, stand_still=-1.0,
                # Regularization.
                torques=-0.0002, action_rate=-0.01, energy=-0.001,
                # Feet.
                # feet_slip: Go1 uses -0.1. Slow servos make foot-dragging gaits easy for this robot (g4_3: hind feet barely lifted), so it is stronger.
                feet_clearance=-2.0, feet_height=-0.2, feet_slip=-0.3, feet_air_time=0.1,
            ),
            # Go1 uses sigma 0.25 (command 1.5 m/s). sigma is in (m/s)², so it must scale with the square of the command range:
            # proportional scaling 0.25×(0.3/1.5)² ≈ 0.01; 0.02 (2x) is used for exploration signal.
            # With 0.25, standing still already earns a tracking reward of 0.87 averaged over the command distribution (0.70 even at the worst command 0.3), so perfect tracking gains
            # only 0.13 -> buried under foot/action_rate penalties and the policy did not move (2026-10-22 g4_2).
            # Next candidate: if tracking_lin_vel stays near 0.05 for long early on, raise to 0.03~0.05.
            tracking_sigma_lin=0.02,
            tracking_sigma_ang=0.25,   # yaw command range is the same as Go1
            max_foot_height=0.03,   # Go1 0.1 m (body height 0.28) scaled down to leg length
        ),
        pert_config=config_dict.create(
            enable=False, velocity_kick=[0.0, 3.0], kick_durations=[0.05, 0.2], kick_wait_times=[1.0, 3.0],
        ),
        # Commands: vx, vy, yaw rate sampled independently, each kept on with probability b (else 0) -> forward/backward, sidestep, turn in place, and stop all occur
        # Velocity range: reduced because at 0.5 the policy learned only to drag its feet without lifting (2026-10-22). The max walking speed achievable within servo limits is
        # unverified (rough estimate 0.1~0.2). If the actual speed after training falls short of 0.3, lower a[0] to 0.2.
        # zero_prob: fraction of episodes with the whole command set to 0. Go1-style sampling alone gives full stop only 3.75% of the time, so standing was not learned (2026-10-22 g4_3:
        # crept 6 cm in 6 s on a stop command). The stand_still penalty needs stop commands seen often enough to work.
        command_config=config_dict.create(a=[0.3, 0.15, 1.2], b=[0.9, 0.25, 0.5], zero_prob=0.25),
        impl="jax",
        naconmax=4 * 8192,
        njmax=40,
    )


class RoboDogJoystick(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict = default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
                 xml_path=scene.ROBOT_XML):
        super().__init__(config, config_overrides)
        self._xml_path = str(xml_path)
        self._mj_model = scene.load_training_model(xml_path)
        self._mj_model.opt.timestep = self._config.sim_dt
        mode = self._config.servo.mode
        self._pd_mode = mode in ("pd", "hybrid")
        nu_ = self._mj_model.nu
        if mode == "pd":
            from learn.servo_pd import apply_pd_actuators
            self._pd_gains = apply_pd_actuators(self._mj_model, **dict(self._config.servo.pd))
            self._profile_mask = np.zeros(nu_, dtype=bool)
        elif mode == "hybrid":
            from learn.servo_pd import apply_hybrid_actuators
            self._pd_gains = apply_hybrid_actuators(self._mj_model, **dict(self._config.servo.pd))
            self._profile_mask = np.zeros(nu_, dtype=bool); self._profile_mask[self._pd_gains["hip_actuators"]] = True
        else:
            self._profile_mask = np.ones(nu_, dtype=bool)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        m = self._mj_model
        self._imu_site_id = m.site("imu").id
        self._trunk_body_id = m.body("trunk").id
        self._torso_mass = float(m.body_subtreemass[self._trunk_body_id])
        self._feet_site_id = np.array([m.site(f).id for f in scene.FEET])
        self._feet_geom_id = np.array([m.geom(f).id for f in scene.FEET])
        off = np.array(self._config.stance_offset, dtype=float)
        init_q = np.array(m.keyframe("home").qpos); init_q[7:] += off
        self._init_q = jp.array(init_q)
        self._default_pose = jp.array(init_q[7:])
        self._lowers, self._uppers = m.jnt_range[1:].T
        self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor
        self._is_hip = jp.array([n.endswith("hip_joint") for n in scene.JOINTS])
        adr = []
        for f in scene.FEET:
            sid = m.sensor(f"{f}_global_linvel").id
            a = m.sensor_adr[sid]
            adr.append(list(range(a, a + 3)))
        self._foot_linvel_adr = jp.array(adr)
        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)
        self._cmd_zero_prob = float(self._config.command_config.get("zero_prob", 0.0))

    # ---- sensors ----
    def _sensor(self, data, name):
        return mjx_env.get_sensor_data(self._mj_model, data, name)

    def get_upvector(self, data):
        return self._sensor(data, "upvector")

    def get_gravity(self, data):
        return data.site_xmat[self._imu_site_id].T @ jp.array([0.0, 0.0, -1.0])

    def get_gyro(self, data):
        return self._sensor(data, "imu_gyro")

    def get_accelerometer(self, data):
        return self._sensor(data, "imu_acc")

    def get_local_linvel(self, data):
        return self._sensor(data, "local_linvel")

    def get_global_linvel(self, data):
        return self._sensor(data, "global_linvel")

    def get_global_angvel(self, data):
        return self._sensor(data, "global_angvel")

    def feet_contact(self, data):
        return data.site_xpos[self._feet_site_id][:, 2] < FOOT_RADIUS + CONTACT_EPS

    # ---- reset / step ----
    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        qpos = qpos.at[0:2].add(jax.random.uniform(k1, (2,), minval=-0.5, maxval=0.5))
        yaw = jax.random.uniform(k2, (1,), minval=-3.14, maxval=3.14)
        quat = mjx_math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
        qpos = qpos.at[3:7].set(mjx_math.quat_mul(qpos[3:7], quat))
        qvel = qvel.at[0:6].set(jax.random.uniform(k3, (6,), minval=-0.2, maxval=0.2))

        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
                                 impl=self.mjx_model.impl.value, naconmax=self._config.naconmax,
                                 njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)

        rng, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 8)
        pc = self._config.pert_config
        t_pert = jax.random.uniform(k1, minval=pc.kick_wait_times[0], maxval=pc.kick_wait_times[1])
        pert_dur_s = jax.random.uniform(k2, minval=pc.kick_durations[0], maxval=pc.kick_durations[1])
        pert_mag = jax.random.uniform(k3, minval=pc.velocity_kick[0], maxval=pc.velocity_kick[1])
        t_cmd = jax.random.exponential(k4) * 5.0
        cmd = self.sample_command(k5, jp.zeros(3))
        sc = self._config.servo
        a_max = jax.random.uniform(k6, minval=sc.a_max[0], maxval=sc.a_max[1])
        kd1, kd2 = jax.random.split(k7)
        delay = jax.random.bernoulli(kd1, sc.delay_prob).astype(jp.int32)
        delay = delay + (self._is_hip & jax.random.bernoulli(kd2, sc.hip_extra_delay_prob)).astype(jp.int32)

        info = {
            "rng": rng,
            "command": cmd,
            "steps_until_next_cmd": jp.round(t_cmd / self.dt).astype(jp.int32),
            "last_act": jp.zeros(self.mjx_model.nu),
            "filt_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "steps_until_next_pert": jp.round(t_pert / self.dt).astype(jp.int32),
            "pert_duration_seconds": pert_dur_s,
            "pert_duration": jp.round(pert_dur_s / self.dt).astype(jp.int32),
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": jp.zeros(3),
            "pert_mag": pert_mag,
            # servo model
            "a_max": a_max,
            "delay": delay,                      # (12,) 0~2 control steps
            "servo_x": self._default_pose,
            "servo_v": jp.zeros(self.mjx_model.nu),
            "target_hist": jp.tile(self._default_pose, (3, 1)),   # [0]=latest
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics["swing_peak"] = jp.zeros(())
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def _servo_substeps(self, data, info, target):
        """Run n_substeps physics steps, passing target angles through ServoProfile. Returns (data, x, v)"""
        dt = self._config.sim_dt
        a_max, v_max = info["a_max"], self._config.servo.v_max

        mask = jp.array(self._profile_mask)          # True: position mode (through trajectory generator), False: PWM PD (target as-is)

        def body(carry, _):
            data, x, v = carry
            xp, vp = servo_profile_step(x, v, target, dt, a_max, v_max)
            x = jp.where(mask, xp, target)
            v = jp.where(mask, vp, 0.0)
            data = data.replace(ctrl=x)
            data = mjx.step(self.mjx_model, data)
            return (data, x, v), None

        (data, x, v), _ = jax.lax.scan(body, (data, info["servo_x"], info["servo_v"]), (), self.n_substeps)
        return data, x, v

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state)
        info = state.info

        alpha = self._config.action_filter_alpha
        filt = alpha * action + (1.0 - alpha) * info["filt_act"]
        info["filt_act"] = filt
        target_now = jp.clip(self._default_pose + filt * self._config.action_scale, self._lowers, self._uppers)
        hist = jp.concatenate([target_now[None], info["target_hist"][:-1]], axis=0)   # [0]=now, [1]=1 step ago, ...
        applied = hist[info["delay"], jp.arange(self.mjx_model.nu)]
        data, x, v = self._servo_substeps(state.data, info, applied)
        info["target_hist"], info["servo_x"], info["servo_v"] = hist, x, v

        contact = self.feet_contact(data)
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0.0) * contact_filt
        info["feet_air_time"] += self.dt
        p_fz = data.site_xpos[self._feet_site_id][:, 2]
        info["swing_peak"] = jp.maximum(info["swing_peak"], p_fz)

        obs = self._get_obs(data, info)
        done = self._get_termination(data)
        rewards = self._get_reward(data, action, info, done, first_contact, contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        info["steps_until_next_cmd"] -= 1
        info["rng"], k1, k2 = jax.random.split(info["rng"], 3)
        info["command"] = jp.where(info["steps_until_next_cmd"] <= 0,
                                   self.sample_command(k1, info["command"]), info["command"])
        info["steps_until_next_cmd"] = jp.where(
            done | (info["steps_until_next_cmd"] <= 0),
            jp.round(jax.random.exponential(k2) * 5.0 / self.dt).astype(jp.int32),
            info["steps_until_next_cmd"])
        info["feet_air_time"] *= ~contact
        info["last_contact"] = contact
        info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(info["swing_peak"])
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))

    def _get_termination(self, data):
        fall = self.get_upvector(data)[-1] < 0.0
        too_low = data.site_xpos[self._imu_site_id][2] < 0.06
        return fall | too_low

    # ---- observations ----
    def _noisy(self, info, x, scale):
        info["rng"], k = jax.random.split(info["rng"])
        return x + (2 * jax.random.uniform(k, x.shape) - 1) * self._config.noise_config.level * scale

    def _get_obs(self, data, info):
        sc = self._config.noise_config.scales
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]
        state = jp.hstack([
            self._noisy(info, gravity, sc.gravity),                                 # 3
            self._noisy(info, gyro, sc.gyro),                                       # 3
            info["command"],                                                         # 3
            self._noisy(info, joint_angles, sc.joint_pos) - self._default_pose,      # 12
            self._noisy(info, joint_vel, sc.joint_vel),                              # 12
            info["filt_act"],                                                        # 12 (filter off = previous action)
        ])
        linvel = self.get_local_linvel(data)
        feet_vel = data.sensordata[self._foot_linvel_adr].ravel()
        privileged = jp.hstack([
            state, gyro, self.get_accelerometer(data), gravity, linvel, self.get_global_angvel(data),
            joint_angles - self._default_pose, joint_vel, data.actuator_force,
            info["last_contact"], feet_vel, info["feet_air_time"],
            data.xfrc_applied[self._trunk_body_id, :3],
            info["steps_since_last_pert"] >= info["steps_until_next_pert"],
            info["a_max"], info["delay"].astype(jp.float32),
            data.site_xpos[self._imu_site_id][2],
        ])
        return {"state": state, "privileged_state": privileged}

    # ---- rewards ----
    def _get_reward(self, data, action, info, done, first_contact, contact):
        cfg = self._config.reward_config
        cmd = info["command"]
        cmd_norm = jp.linalg.norm(cmd)
        local_v = self.get_local_linvel(data)
        gyro = self.get_gyro(data)
        gv, gw = self.get_global_linvel(data), self.get_global_angvel(data)
        up = self.get_upvector(data)
        q = data.qpos[7:]
        tau = data.actuator_force
        feet_vel = data.sensordata[self._foot_linvel_adr]
        vxy_sq = jp.sum(jp.square(feet_vel[..., :2]), axis=-1)
        foot_z = data.site_xpos[self._feet_site_id][:, 2]
        out = -jp.clip(q - self._soft_lowers, None, 0.0) + jp.clip(q - self._soft_uppers, 0.0, None)
        return {
            "tracking_lin_vel": jp.exp(-jp.sum(jp.square(cmd[:2] - local_v[:2])) / cfg.tracking_sigma_lin),
            "tracking_ang_vel": jp.exp(-jp.square(cmd[2] - gyro[2]) / cfg.tracking_sigma_ang),
            "lin_vel_z": jp.square(gv[2]),
            "ang_vel_xy": jp.sum(jp.square(gw[:2])),
            "orientation": jp.sum(jp.square(up[:2])),
            "dof_pos_limits": jp.sum(out),
            "pose": jp.exp(-jp.sum(jp.square(q - self._default_pose) * jp.array([1.0, 1.0, 0.1] * 4))),
            "termination": done,
            "stand_still": jp.sum(jp.abs(q - self._default_pose)) * (cmd_norm < 0.01),
            "torques": jp.sqrt(jp.sum(jp.square(tau))) + jp.sum(jp.abs(tau)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "energy": jp.sum(jp.abs(data.qvel[6:]) * jp.abs(tau)),
            "feet_slip": jp.sum(vxy_sq * contact) * (cmd_norm > 0.01),
            "feet_clearance": jp.sum(jp.abs(foot_z - cfg.max_foot_height) * jp.sqrt(jp.sqrt(vxy_sq))),
            "feet_height": jp.sum(jp.square(info["swing_peak"] / cfg.max_foot_height - 1.0) * first_contact)
                           * (cmd_norm > 0.01),
            "feet_air_time": jp.sum((info["feet_air_time"] - 0.1) * first_contact) * (cmd_norm > 0.01),
        }

    # ---- pushes and commands (same as Go1) ----
    def _maybe_apply_perturbation(self, state):
        def gen_dir(rng):
            a = jax.random.uniform(rng, minval=0.0, maxval=2 * jp.pi)
            return jp.array([jp.cos(a), jp.sin(a), 0.0])

        def apply_pert(state):
            t = state.info["pert_steps"] * self.dt
            u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
            force = u_t * self._torso_mass * state.info["pert_mag"] / state.info["pert_duration_seconds"]
            xfrc = jp.zeros((self.mjx_model.nbody, 6)).at[self._trunk_body_id, :3].set(force * state.info["pert_dir"])
            state = state.replace(data=state.data.replace(xfrc_applied=xfrc))
            state.info["steps_since_last_pert"] = jp.where(
                state.info["pert_steps"] >= state.info["pert_duration"], 0, state.info["steps_since_last_pert"])
            state.info["pert_steps"] += 1
            return state

        def wait(state):
            state.info["rng"], rng = jax.random.split(state.info["rng"])
            state.info["steps_since_last_pert"] += 1
            xfrc = jp.zeros((self.mjx_model.nbody, 6))
            due = state.info["steps_since_last_pert"] >= state.info["steps_until_next_pert"]
            state.info["pert_steps"] = jp.where(due, 0, state.info["pert_steps"])
            state.info["pert_dir"] = jp.where(due, gen_dir(rng), state.info["pert_dir"])
            return state.replace(data=state.data.replace(xfrc_applied=xfrc))

        return jax.lax.cond(state.info["steps_since_last_pert"] >= state.info["steps_until_next_pert"],
                            apply_pert, wait, state)

    def sample_command(self, rng, x_k):
        rng, y_rng, w_rng, z_rng, zero_rng = jax.random.split(rng, 5)
        y_k = jax.random.uniform(y_rng, (3,), minval=-self._cmd_a, maxval=self._cmd_a)
        z_k = jax.random.bernoulli(z_rng, self._cmd_b, (3,))
        w_k = jax.random.bernoulli(w_rng, 0.5, (3,))
        new = x_k - w_k * (x_k - y_k * z_k)
        return jp.where(jax.random.bernoulli(zero_rng, self._cmd_zero_prob), jp.zeros(3), new)

    # ---- properties ----
    @property
    def xml_path(self):
        return self._xml_path

    @property
    def action_size(self):
        return self._mjx_model.nu

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    def domain_randomize_fn(self):
        from learn.randomize import make_domain_randomize
        return make_domain_randomize(self._mj_model, self._feet_geom_id, self._trunk_body_id, pd_mode=self._pd_mode)
