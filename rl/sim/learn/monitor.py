"""Fetch progress.csv from Colab (or anywhere) and show it as a table and plots.

    .venv/bin/python -m learn.monitor --session robodog-g4 --remote /content/runs/g4_1 --out runs/g4_1
    .venv/bin/python -m learn.monitor --csv runs/g4_1/progress.csv            # only plot an already downloaded file

Outputs: <out>/progress.csv, <out>/train.log (tail), <out>/progress.png
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

# Outputs (tables, plots, logs) are in English
KEYS = [("eval/episode_reward", "Total reward"), ("eval/avg_episode_length", "Episode length (max 1000)"),
        ("eval/episode_reward/tracking_lin_vel", "Linear velocity tracking"),
        ("eval/episode_reward/tracking_ang_vel", "Angular velocity tracking"),
        ("eval/episode_reward/termination", "Termination (fall) penalty"),
        ("eval/episode_reward/feet_air_time", "Feet air time"),
        ("eval/episode_reward/feet_slip", "Feet slip penalty"), ("eval/episode_reward/orientation", "Orientation penalty")]


def fetch(session, remote, out):
    colab = str(Path(sys.executable).parent / "colab")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([colab, "download", "-s", session, f"{remote}/progress.csv", str(out / "progress.csv")],
                   capture_output=True, text=True)
    code = f"import subprocess;print(subprocess.run('tail -n 60 {remote}/train.log | grep -v -i warn', shell=True, capture_output=True, text=True).stdout)"
    r = subprocess.run([colab, "exec", "-s", session], input=code, capture_output=True, text=True)
    (out / "train.log").write_text(r.stdout, encoding="utf-8")


def load(csv_path):
    """For rows with more columns than the header (older train.py prepended training metrics), align only the trailing eval columns to the header."""
    rows = list(csv.reader(open(csv_path, encoding="utf-8")))
    hdr, out = rows[0], []
    for r in rows[1:]:
        if not r:
            continue
        if len(r) > len(hdr):
            r = r[:2] + r[len(r) - (len(hdr) - 2):]
        d = {}
        for k, v in zip(hdr, r):
            try:
                d[k] = float(v)
            except ValueError:
                pass
        out.append(d)
    return out


def table(rows):
    print(f"{'steps':>12s} {'min':>8s} {'reward':>8s} {'length':>7s} {'lin_vel':>7s} {'ang_vel':>7s} {'termin':>7s} {'air':>6s} {'sps':>8s}")
    for r in rows:
        print(f"{int(r['steps']):>12,d} {r['elapsed_s']/60:8.1f} {r['eval/episode_reward']:8.2f} "
              f"{r['eval/avg_episode_length']:7.1f} {r['eval/episode_reward/tracking_lin_vel']:7.1f} "
              f"{r['eval/episode_reward/tracking_ang_vel']:7.1f} {r['eval/episode_reward/termination']:7.2f} "
              f"{r['eval/episode_reward/feet_air_time']:6.1f} {r.get('eval/sps', float('nan')):8.0f}")


def plot(rows, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [r["steps"] / 1e6 for r in rows]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (k, title) in zip(axes.ravel(), KEYS):
        ax.plot(steps, [r.get(k, float("nan")) for r in rows], marker="o", ms=3)
        ax.set_title(title); ax.set_xlabel("Million steps"); ax.grid(alpha=0.3)
    fig.suptitle("OpenRoboticDog PPO training progress")
    fig.tight_layout(); fig.savefig(png, dpi=110)
    print("Plot saved:", png)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session", default=None)
    p.add_argument("--remote", default="/content/runs/g4_3")
    p.add_argument("--out", default="runs/g4_3")
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    out = Path(args.out)
    if args.session:
        fetch(args.session, args.remote, out)
        print((out / "train.log").read_text(encoding="utf-8")[-1500:])
    csv_path = Path(args.csv) if args.csv else out / "progress.csv"
    if not csv_path.exists():
        sys.exit(f"{csv_path} not found")
    rows = load(csv_path)
    table(rows)
    try:
        plot(rows, out / "progress.png")
    except ImportError:
        print("matplotlib missing: pip install matplotlib")


if __name__ == "__main__":
    main()
