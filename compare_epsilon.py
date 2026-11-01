"""Decay epsilon vs fixed epsilon: the numbers that show which one is better.

Run:  python compare_epsilon.py            5 seeds, default episodes
      python compare_epsilon.py 3          3 seeds (faster)
      python compare_epsilon.py 5 10000    5 seeds, 10000 episodes

Training uses the same start mix as qlearning.py (some episodes start on a random square).
Metrics (each averaged over the seeds):
  1. final training return  : average return of the last 10% of the episodes that began at the
                              start cell. What the bot earns while it still has to act with its
                              exploration setting. Comparable with the best possible return.
  2. whole-run return       : the same, averaged over ALL episodes that began at the start cell.
  3. episodes to profit     : first episode from which the moving-average return stays above 0.
                              Lower = learns faster.
  4. greedy from start      : after training, play with epsilon = 0 from the start cell:
                              delivered? how many steps? (compare with the shortest route)
  5. squares that deliver   : after training, play with epsilon = 0 from EVERY possible starting
                              square, without and with the package: the share that deliver.
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
WINDOW = max(50, EPISODES // 200)              # moving-average window for "episodes to profit"

SETTINGS = [("decay 1.0 -> 0.01", dict(eps_start=1.0, eps_end=0.01, eps_decay_episodes=DECAY)),
            ("fixed 0.1", dict(eps_start=0.1, eps_end=0.1)),
            ("fixed 0.5", dict(eps_start=0.5, eps_end=0.5)),
            ("fixed 0.95", dict(eps_start=0.95, eps_end=0.95))]


def greedy(q, cell=None, has=False):
    """One game with epsilon = 0. Returns (steps, delivered). Default: from the start cell."""
    env = E()
    state, done = env.reset(), False
    if cell is not None:
        env.pos, env.has_package = cell, has
        state = env.encode()
    while not done:
        state, _, done = env.step(int(np.argmax(q[state])))
    return env.steps, env.delivered


def coverage(q):
    """Share of possible starting squares from which the greedy bot delivers: (without, carrying)."""
    shares = []
    for has in (False, True):
        cells = [c for c in E.FREE_CELLS
                 if not (c == E.DELIVERY and has) and not (c == E.PACKAGE and not has)]
        shares.append(sum(greedy(q, c, has)[1] for c in cells) / len(cells))
    return shares


def first_profit(returns, idx):
    """Episode number from which the moving average stays above 0 (None = never).
    returns: returns of the start-cell episodes, idx: their episode numbers."""
    if len(returns) < WINDOW:
        return None
    ma = np.convolve(returns, np.ones(WINDOW) / WINDOW, mode="valid")
    below = np.flatnonzero(ma <= 0)
    first = below[-1] + 1 if len(below) else 0     # first point after the last time it was not above 0
    return int(idx[first + WINDOW - 1]) if first < len(ma) else None


def job(args):
    idx, seed = args
    _, settings = SETTINGS[idx]
    mask = []
    q, returns = qlearning.train(seed=seed, n_episodes=EPISODES, fixed_start_mask=mask, **settings)
    mask = np.array(mask)
    episode_numbers = np.flatnonzero(mask)
    from_start = returns[mask]                     # only the episodes that began at the start cell
    tail = max(100, len(from_start) // 10)
    steps, delivered = greedy(q)
    cov_without, cov_carrying = coverage(q)
    return idx, dict(tail=from_start[-tail:].mean(), whole=from_start.mean(),
                     profit=first_profit(from_start, episode_numbers), steps=steps, delivered=delivered,
                     cov_without=cov_without, cov_carrying=cov_carrying)


def main():
    d1 = bfs_dist(E.SIZE, E.OBSTACLES, E.START, E.PACKAGE)
    d2 = bfs_dist(E.SIZE, E.OBSTACLES, E.PACKAGE, E.DELIVERY)
    best_ret = (d1 - 1) * STEP_REWARD + PICKUP_REWARD + (d2 - 1) * STEP_REWARD + DELIVER_REWARD
    print(f"Grid {E.SIZE}x{E.SIZE}, {EPISODES} episodes, {N_SEEDS} seeds, "
          f"{qlearning.RANDOM_START_FRACTION:.0%} of episodes start on a random square. "
          f"Shortest route {d1 + d2} steps, best possible return {best_ret}\n")

    with Pool() as pool:
        results = pool.map(job, [(i, s) for i in range(len(SETTINGS)) for s in range(N_SEEDS)])

    names, tails, wholes, cov_no, cov_has = [], [], [], [], []
    header = (f"{'setting':20s} {'final train return':>22s} {'whole-run return':>18s} "
              f"{'episodes to profit':>20s} {'greedy from start':>22s} {'squares that deliver':>24s}")
    print(header)
    print(" " * 113 + "(no package / carrying)")
    print("-" * len(header))
    for i, (label, _) in enumerate(SETTINGS):
        runs = [r for idx, r in results if idx == i]
        tail = np.array([r["tail"] for r in runs])
        whole = np.array([r["whole"] for r in runs])
        profit = [r["profit"] for r in runs]
        ok = sum(r["delivered"] for r in runs)
        steps = [r["steps"] for r in runs if r["delivered"]]
        got = [p for p in profit if p is not None]
        cn = np.array([r["cov_without"] for r in runs]) * 100
        ch = np.array([r["cov_carrying"] for r in runs]) * 100
        profit_txt = (f"{int(np.mean(got))} ({len(got)}/{N_SEEDS} seeds)" if got else f"never (0/{N_SEEDS})")
        greedy_txt = f"{ok}/{N_SEEDS} ok" + (f", {np.mean(steps):.1f} steps" if steps else "")
        print(f"{label:20s} {tail.mean():>14.1f} ± {tail.std():5.1f} {whole.mean():>18.1f} "
              f"{profit_txt:>20s} {greedy_txt:>22s} {cn.mean():>11.1f}% / {ch.mean():.1f}%")
        names.append(label)
        tails.append((tail.mean(), tail.std()))
        wholes.append((whole.mean(), whole.std()))
        cov_no.append((cn.mean(), cn.std()))
        cov_has.append((ch.mean(), ch.std()))

    colours = ["tab:green", "tab:blue", "tab:orange", "tab:red"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, data, title in ((axes[0], tails, "final training return\n(last 10% of start-cell episodes)"),
                            (axes[1], wholes, "average return over the whole run\n(start-cell episodes)")):
        ax.bar(names, [m for m, _ in data], yerr=[s for _, s in data], capsize=4, color=colours)
        ax.axhline(best_ret, color="green", linestyle="--", linewidth=1, label=f"best possible ({best_ret})")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=15)
        ax.legend(fontsize=8)
    x = np.arange(len(names))
    axes[2].bar(x - 0.2, [m for m, _ in cov_no], 0.4, yerr=[s for _, s in cov_no], capsize=3,
                label="start without package", color="tab:purple")
    axes[2].bar(x + 0.2, [m for m, _ in cov_has], 0.4, yerr=[s for _, s in cov_has], capsize=3,
                label="start carrying package", color="tab:cyan")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=15)
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("% of squares")
    axes[2].set_title("squares from which the greedy bot delivers", fontsize=10)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{E.SIZE}x{E.SIZE} grid, mean of {N_SEEDS} seeds (error bar = std across seeds)")
    fig.tight_layout()
    out = f"compare_epsilon_{E.SIZE}.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
