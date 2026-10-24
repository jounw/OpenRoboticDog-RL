"""Simulated motor model for direct PWM drive mode: converts the Pi's duty PD to torque and feeds it to MuJoCo position actuators.

    torque = Kt·duty,  duty = kp_duty·(q_t − q) − kd_duty·q̇  (|duty| ≤ duty_max)
    -> actuator kp = kp_duty·Kt [N·m/rad], kv = kd_duty·Kt, forcerange = duty_max·Kt
    The motor torque-speed curve (zero torque at no-load speed) is modeled as joint damping b = Kt·1000 / ω_noload.

Defaults are from the spec (stall torque 2.9 N·m @ duty 1000, no-load 4.7 rad/s) and are updated from pwm_sysid.py results.
"""
import mujoco
import numpy as np

DEFAULT = dict(kp_duty=8000.0, kd_duty=150.0, duty_max=800, kt=2.9 / 1000, omega_noload=5.06, armature=0.0121)   # sysid/pwm_params.yaml


def apply_pd_actuators(mj_model, kp_duty=DEFAULT["kp_duty"], kd_duty=DEFAULT["kd_duty"], duty_max=DEFAULT["duty_max"],
                       kt=DEFAULT["kt"], omega_noload=DEFAULT["omega_noload"], armature=DEFAULT["armature"]):
    kp, kv, fmax = kp_duty * kt, kd_duty * kt, duty_max * kt
    mj_model.actuator_gainprm[:, 0] = kp
    mj_model.actuator_biasprm[:, 1] = -kp
    mj_model.actuator_biasprm[:, 2] = -kv
    mj_model.actuator_forcerange[:, 0] = -fmax
    mj_model.actuator_forcerange[:, 1] = fmax
    b = kt * 1000 / omega_noload
    hinge = np.array([mj_model.jnt_type[mj_model.dof_jntid[i]] == mujoco.mjtJoint.mjJNT_HINGE for i in range(mj_model.nv)])
    mj_model.dof_damping[hinge] = b            # only on hinge joint dofs, regardless of a free joint
    mj_model.dof_armature[hinge] = armature
    return dict(kp=kp, kv=kv, fmax=fmax, damping=b, armature=armature)


def apply_hybrid_actuators(mj_model, hip_kp=40.0, hip_kv=1.6, hip_force=2.9, hip_damping=0.1, hip_armature=0.01, **pd_kwargs):
    """Hardware setup (2026-10-22): hip in position mode (firmware trajectory generator + servo PD), thigh/calf in PWM mode + Pi PD.
    First switch everything to PD, then restore only the hip actuators/joints to position-mode values (servo_params.yaml)."""
    out = apply_pd_actuators(mj_model, **pd_kwargs)
    hips = []
    for a in range(mj_model.nu):
        j = mj_model.actuator_trnid[a, 0]
        if mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j).endswith("hip_joint"):
            hips.append(a)
            mj_model.actuator_gainprm[a, 0] = hip_kp
            mj_model.actuator_biasprm[a, 1] = -hip_kp
            mj_model.actuator_biasprm[a, 2] = -hip_kv
            mj_model.actuator_forcerange[a] = (-hip_force, hip_force)
            dof = mj_model.jnt_dofadr[j]
            mj_model.dof_damping[dof] = hip_damping
            mj_model.dof_armature[dof] = hip_armature
    out["hip_actuators"] = hips
    return out
