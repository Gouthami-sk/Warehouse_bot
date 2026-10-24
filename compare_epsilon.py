"""Decay epsilon vs fixed epsilon: the numbers that show which one is better.

Run:  python compare_epsilon.py            5 seeds, default episodes
      python compare_epsilon.py 3          3 seeds (faster)
      python compare_epsilon.py 5 10000    5 seeds, 10000 episodes

Metrics (each averaged over the seeds):
  1. final training return  : average return of the last 10% of training episodes. What the bot
                              earns while it still has to act with its exploration setting.
  2. whole-run return       : average return over ALL training episodes. What the whole learning
                              period cost: a bot that learns fast and explores sensibly scores higher.
  3. episodes to profit     : first episode where the moving-average return stays above 0
                              (the bot delivers more than it wastes). Lower = learns faster.
  4. greedy from start      : after training, play with epsilon = 0 from the start cell:
                              delivered? how many steps? (compare with the shortest route)
"""
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import qlearning
from env import WarehouseEnv, bfs_dist, STEP_REWARD, PICKUP_REWARD, DELIVER_REWARD

E = WarehouseEnv
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else qlearning.N_EPISODES
DECAY = int(EPISODES * 0.75)                   # decay lasts 75% of the run, like the default
WINDOW = max(50, EPISODES // 100)              # moving-average window for "episodes to profit"
TAIL = max(100, EPISODES // 10)                # last 10% of the run

SETTINGS = [("decay 1.0 -> 0.01", dict(eps_start=1.0, eps_end=0.01, eps_decay_episodes=DECAY)),
            ("fixed 0.1", dict(eps_start=0.1, eps_end=0.1)),
            ("fixed 0.5", dict(eps_start=0.5, eps_end=0.5)),
            ("fixed 0.95", dict(eps_start=0.95, eps_end=0.95))]


def greedy(q):
    """One game with epsilon = 0 from the start cell. Returns (steps, delivered)."""
    env = E()
    state, done = env.reset(), False
    while not done:
        state, _, done = env.step(int(np.argmax(q[state])))
    return env.steps, env.delivered


def first_profit(returns):
    """First episode from which the moving average stays above 0 (None = never)."""
    ma = np.convolve(returns, np.ones(WINDOW) / WINDOW, mode="valid")
    above = ma > 0
    for i in range(len(above)):
        if above[i:].all():
            return i + WINDOW - 1
    return None


def job(args):
    idx, seed = args
    _, settings = SETTINGS[idx]
    q, returns = qlearning.train(seed=seed, n_episodes=EPISODES, **settings)
    steps, delivered = greedy(q)
    return idx, dict(tail=returns[-TAIL:].mean(), whole=returns.mean(),
                     profit=first_profit(returns), steps=steps, delivered=delivered)


def main():
    d1 = bfs_dist(E.SIZE, E.OBSTACLES, E.START, E.PACKAGE)
    d2 = bfs_dist(E.SIZE, E.OBSTACLES, E.PACKAGE, E.DELIVERY)
    best_ret = (d1 - 1) * STEP_REWARD + PICKUP_REWARD + (d2 - 1) * STEP_REWARD + DELIVER_REWARD
    print(f"Grid {E.SIZE}x{E.SIZE}, {EPISODES} episodes, {N_SEEDS} seeds. "
          f"Shortest route {d1 + d2} steps, best possible return {best_ret}\n")

    with Pool() as pool:
        results = pool.map(job, [(i, s) for i in range(len(SETTINGS)) for s in range(N_SEEDS)])

    names, tails, wholes = [], [], []
    header = f"{'setting':20s} {'final train return':>22s} {'whole-run return':>18s} {'episodes to profit':>20s} {'greedy from start':>22s}"
    print(header)
    print("-" * len(header))
    for i, (label, _) in enumerate(SETTINGS):
        runs = [r for idx, r in results if idx == i]
        tail = np.array([r["tail"] for r in runs])
        whole = np.array([r["whole"] for r in runs])
        profit = [r["profit"] for r in runs]
        ok = sum(r["delivered"] for r in runs)
        steps = [r["steps"] for r in runs if r["delivered"]]
        got = [p for p in profit if p is not None]
        profit_txt = (f"{int(np.mean(got))} ({len(got)}/{N_SEEDS} seeds)" if got else f"never (0/{N_SEEDS})")
        greedy_txt = f"{ok}/{N_SEEDS} ok" + (f", {int(np.mean(steps))} steps" if steps else "")
        print(f"{label:20s} {tail.mean():>14.1f} ± {tail.std():5.1f} {whole.mean():>18.1f} "
              f"{profit_txt:>20s} {greedy_txt:>22s}")
        names.append(label)
        tails.append((tail.mean(), tail.std()))
        wholes.append((whole.mean(), whole.std()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, title in ((axes[0], tails, f"final training return (last {TAIL} episodes)"),
                            (axes[1], wholes, "average return over the whole run")):
        means = [m for m, _ in data]
        stds = [s for _, s in data]
        bars = ax.bar(names, means, yerr=stds, capsize=4,
                      color=["tab:green", "tab:blue", "tab:orange", "tab:red"])
        ax.axhline(best_ret, color="green", linestyle="--", linewidth=1, label=f"best possible ({best_ret})")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=15)
        ax.legend(fontsize=8)
    fig.suptitle(f"{E.SIZE}x{E.SIZE} grid, mean of {N_SEEDS} seeds (error bar = std across seeds)")
    fig.tight_layout()
    out = f"compare_epsilon_{E.SIZE}.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
