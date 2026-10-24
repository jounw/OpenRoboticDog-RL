"""Brax PPO training (MuJoCo Playground style).

    # Mac CPU smoke test (a few minutes)
    .venv/bin/python -m learn.train --num-envs 16 --num-timesteps 20000 --out runs/smoke
    # H100
    .venv/bin/python -m learn.train --impl warp --num-envs 8192 --num-timesteps 200000000 --out runs/h100_1

Outputs: <out>/params.pkl (brax parameters), <out>/policy.npz (for numpy inference, learn/policy.py), <out>/progress.csv
"""
import argparse
import csv
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

# brax 0.14.2 uses jax.device_put_replicated, removed in jax 0.12 -> replace with equivalent behavior (stack along a leading device axis)
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices):
        return jax.tree_util.tree_map(lambda a: jp.stack([jp.asarray(a)] * len(devices)), x)
    jax.device_put_replicated = _device_put_replicated

from brax.io import model as brax_model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco_playground import wrapper

from learn.env import RoboDogJoystick, default_config


def ppo_config(num_envs, num_timesteps, episode_length=1000):
    """Start from the Playground Go1 joystick values."""
    return config_dict.create(
        num_timesteps=num_timesteps,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=num_envs,
        batch_size=256,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        num_resets_per_eval=10,
    )


def export_policy_npz(params, path, obs_size, act_size, hidden, action_scale=None, stance_offset=None, action_filter_alpha=None, command_max=None):
    """brax parameters -> npz for numpy inference (read by learn/policy.py). params = (normalizer, policy, value)"""
    norm, policy = params[0], params[1]
    mean, std = norm.mean, norm.std
    if isinstance(mean, dict):          # if observations are a dict, normalizer stats are a dict too
        mean, std = mean["state"], std["state"]
    layers = policy["params"]
    names = sorted(layers.keys(), key=lambda n: int(n.split("_")[1]))
    out = {"obs_mean": np.asarray(mean), "obs_std": np.asarray(std),
           "n_layers": np.array(len(names)), "obs_size": np.array(obs_size), "act_size": np.array(act_size)}
    if action_scale is not None:
        out["action_scale"] = np.array(float(action_scale))
    if stance_offset is not None:
        out["stance_offset"] = np.array(stance_offset, dtype=np.float32)
    if action_filter_alpha is not None:
        out["action_filter_alpha"] = np.array(float(action_filter_alpha))
    if command_max is not None:
        out["command_max"] = np.array(command_max, dtype=np.float32)
    for i, n in enumerate(names):
        out[f"w{i}"] = np.asarray(layers[n]["kernel"])
        out[f"b{i}"] = np.asarray(layers[n]["bias"])
    np.savez(path, **out)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--num-envs", type=int, default=8192)
    p.add_argument("--num-timesteps", type=int, default=200_000_000)
    p.add_argument("--impl", default="jax", help="jax (CPU/Mac) or warp (H100)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-randomize", action="store_true")
    p.add_argument("--num-evals", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-minibatches", type=int, default=None)
    p.add_argument("--restore", default=None, help="resume training: restore policy, value network and normalizer from a brax checkpoint folder (<out>/brax_ckpt/<step>)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override env config, e.g. --set reward_config.tracking_sigma_lin=0.04 --set command_config.a=[0.3,0.15,1.2]")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = default_config()
    env_cfg.impl = args.impl
    for item in args.set:
        key, _, val = item.partition("=")
        node = env_cfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node[k]
        node[parts[-1]] = json.loads(val)
        print(f"override: {key} = {node[parts[-1]]}")
    env = RoboDogJoystick(config=env_cfg)
    eval_env = RoboDogJoystick(config=env_cfg)

    cfg = ppo_config(args.num_envs, args.num_timesteps, env_cfg.episode_length)
    if args.num_evals is not None:
        cfg.num_evals = args.num_evals
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_minibatches is not None:
        cfg.num_minibatches = args.num_minibatches
    (out / "config.json").write_text(json.dumps({"env": env_cfg.to_dict(), "ppo": cfg.to_dict(), "args": vars(args)},
                                                indent=2, default=str), encoding="utf-8")

    net_cfg = dict(cfg.network_factory)
    ppo_kwargs = dict(cfg)
    del ppo_kwargs["network_factory"]
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_cfg)

    progress_path = out / "progress.csv"
    progress_rows = []
    t0 = time.time()

    def progress(num_steps, metrics):
        row = {"steps": num_steps, "elapsed_s": round(time.time() - t0, 1),
               **{k: float(v) for k, v in metrics.items()}}
        progress_rows.append(row)
        # Columns can grow between eval rows (training metrics start from the 1st eval), so rewrite every time with the union of all columns
        fields = list(dict.fromkeys(k for r in progress_rows for k in r))
        with open(progress_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(progress_rows)
        g = lambda k: float(metrics.get(k, float("nan")))
        print(f"[{row['elapsed_s']:7.0f}s] steps {num_steps:>12,}  reward {g('eval/episode_reward'):8.2f}  "
              f"len {g('eval/avg_episode_length'):6.1f}  "
              f"| lin {g('eval/episode_reward/tracking_lin_vel'):6.1f} ang {g('eval/episode_reward/tracking_ang_vel'):6.1f} "
              f"term {g('eval/episode_reward/termination'):5.2f} slip {g('eval/episode_reward/feet_slip'):6.1f} "
              f"air {g('eval/episode_reward/feet_air_time'):5.1f} | sps {g('eval/sps'):8.0f}", flush=True)

    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    def save_policy(step, make_policy, params):
        """Save the policy at every eval -> can check sim-to-sim mid-training"""
        del make_policy
        for path in (ckpt_dir / f"policy_{int(step):010d}.npz", out / "policy_latest.npz"):
            export_policy_npz(params, path, env.observation_size["state"][0], env.action_size,
                              net_cfg["policy_hidden_layer_sizes"], env_cfg.action_scale, list(env_cfg.stance_offset), env_cfg.action_filter_alpha, list(env_cfg.command_config.a))

    train_fn = functools.partial(
        ppo.train, **ppo_kwargs, network_factory=network_factory, seed=args.seed,
        randomization_fn=None if args.no_randomize else env.domain_randomize_fn(),
        progress_fn=progress, policy_params_fn=save_policy,
        save_checkpoint_path=str((out / "brax_ckpt").resolve()),          # save full state at every eval -> resume with --restore
        restore_checkpoint_path=(str(Path(args.restore).resolve()) if args.restore else None),
        restore_value_fn=True,
    )
    make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env,
                                            wrap_env_fn=wrapper.wrap_for_brax_training)
    brax_model.save_params(str(out / "params.pkl"), params)
    export_policy_npz(params, out / "policy.npz", env.observation_size["state"][0], env.action_size,
                      net_cfg["policy_hidden_layer_sizes"], env_cfg.action_scale, list(env_cfg.stance_offset), env_cfg.action_filter_alpha, list(env_cfg.command_config.a))
    print(f"Saved: {out / 'params.pkl'}, {out / 'policy.npz'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
