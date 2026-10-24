"""JAX version of sysid/servo_model.ServoProfile. All 12 joints at once, with a different a_max per episode.

State (x, v): current position/velocity of the servo's internal trajectory generator. Feed the policy target angle as goal and call every substep;
use the returned x as the MuJoCo position actuator ctrl.
"""
import jax
import jax.numpy as jp


def servo_profile_step(x, v, goal, dt, a_max, v_max):
    """Same computation as ServoProfile.step. x, v, goal: (n,); a_max, v_max: scalar or (n,)."""
    e = goal - x
    v_stop = jp.sqrt(jp.maximum(0.0, 2.0 * a_max * jp.abs(e))) - a_max * dt / 2.0
    v_des = jp.sign(e) * jp.minimum(v_max, jp.maximum(0.0, v_stop))
    dv = jp.clip(v_des - v, -a_max * dt, a_max * dt)
    v = v + dv
    x = x + v * dt
    settled = (jp.abs(goal - x) < 1e-6) & (jp.abs(v) <= a_max * dt)
    x = jp.where(settled, goal, x)
    v = jp.where(settled, 0.0, v)
    return x, v


servo_profile_step = jax.jit(servo_profile_step)
