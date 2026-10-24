"""Run the trained policy (policy.npz) with numpy only. Same as sim/learn/policy.py (copied because the Pi has no sim package)."""
import numpy as np


class NumpyPolicy:
    def __init__(self, path):
        d = np.load(path)
        self.mean, self.std = d["obs_mean"], d["obs_std"]
        n = int(d["n_layers"])
        self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(n)]
        self.obs_size, self.act_size = int(d["obs_size"]), int(d["act_size"])
        self.action_scale = float(d["action_scale"]) if "action_scale" in d.files else None   # value used in training (falls back to config if absent)
        self.stance_offset = d["stance_offset"].astype(float) if "stance_offset" in d.files else None   # policy reference pose offset (rad)
        self.action_filter_alpha = float(d["action_filter_alpha"]) if "action_filter_alpha" in d.files else 1.0   # action low-pass filter
        self.command_max = d["command_max"].astype(float) if "command_max" in d.files else None   # training command range (full joystick deflection)

    def __call__(self, obs):
        x = (np.asarray(obs, dtype=np.float32) - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = x / (1.0 + np.exp(-x))   # swish
        return np.tanh(x[: self.act_size])
