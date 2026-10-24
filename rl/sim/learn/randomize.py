"""Domain randomization (for Brax randomization_fn). Only changes model fields. a_max and delay are sampled per episode in env.reset.

- Foot friction: log-uniform U(log 0.05, log 1.5), independent per foot (down to near-zero friction by design)
- Link mass ×U(0.95, 1.05), torso +U(-0.05, 0.1) kg (range narrowed after measuring total mass 1.465 kg on 2026-10-22)
- Torso center of mass ±0.02 m (fixed 2026-10-22)
- Servo kp U(30, 55), kv = 0.04 kp (from sysid)
- Joint frictionloss ×U(0.8, 1.2), armature ×U(1.0, 1.1)
"""
import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

FRICTION_RANGE = (0.05, 1.5)
KP_RANGE = (30.0, 55.0)
KV_PER_KP = 0.04
COM_JITTER = 0.02    # m


def make_domain_randomize(mj_model, feet_geom_ids, trunk_body_id, pd_mode=False):
    """pd_mode: actuator kp/kv = model value ×U(0.8, 1.2) (PWM mode). Otherwise the position-mode sysid range (KP_RANGE)."""
    feet = jp.array(feet_geom_ids)
    nu = mj_model.nu
    kp0 = jp.array(mj_model.actuator_gainprm[:, 0]); kv0 = jp.array(-mj_model.actuator_biasprm[:, 2])

    def domain_randomize(model: mjx.Model, rng: jax.Array):
        @jax.vmap
        def rand(rng):
            rng, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 8)
            log_lo, log_hi = np.log(FRICTION_RANGE[0]), np.log(FRICTION_RANGE[1])
            mu = jp.exp(jax.random.uniform(k1, (4,), minval=log_lo, maxval=log_hi))
            geom_friction = model.geom_friction.at[feet, 0].set(mu)

            dmass = jax.random.uniform(k2, (model.nbody,), minval=0.95, maxval=1.05)
            body_mass = model.body_mass * dmass
            body_mass = body_mass.at[trunk_body_id].add(jax.random.uniform(k3, minval=-0.05, maxval=0.1))

            dpos = jax.random.uniform(k4, (3,), minval=-COM_JITTER, maxval=COM_JITTER)
            body_ipos = model.body_ipos.at[trunk_body_id].add(dpos)

            if pd_mode:
                f = jax.random.uniform(k5, (nu,), minval=0.8, maxval=1.2)
                kp, kv = kp0 * f, kv0 * f
            else:
                kp = jax.random.uniform(k5, (nu,), minval=KP_RANGE[0], maxval=KP_RANGE[1])
                kv = KV_PER_KP * kp
            gainprm = model.actuator_gainprm.at[:, 0].set(kp)
            biasprm = model.actuator_biasprm.at[:, 1].set(-kp)
            biasprm = biasprm.at[:, 2].set(-kv)

            fl = model.dof_frictionloss.at[6:].multiply(jax.random.uniform(k6, (nu,), minval=0.8, maxval=1.2))
            arm = model.dof_armature.at[6:].multiply(jax.random.uniform(k7, (nu,), minval=1.0, maxval=1.1))
            return geom_friction, body_mass, body_ipos, gainprm, biasprm, fl, arm

        geom_friction, body_mass, body_ipos, gainprm, biasprm, fl, arm = rand(rng)
        fields = {
            "geom_friction": geom_friction, "body_mass": body_mass, "body_ipos": body_ipos,
            "actuator_gainprm": gainprm, "actuator_biasprm": biasprm,
            "dof_frictionloss": fl, "dof_armature": arm,
        }
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({k: 0 for k in fields})
        model = model.tree_replace(fields)
        return model, in_axes

    return domain_randomize
