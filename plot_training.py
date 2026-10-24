"""Draw what the bot learned: the reward curve, epsilon, and the best move in every cell.

It retrains using qlearning.py (same settings, same seed), then plots.

Run:  python plot_training.py                  open a window
      python plot_training.py --save out.png   save a picture instead
"""
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import qlearning                               # our training code (train, epsilon_for, settings)
from env import (WarehouseEnv, MOVES, bfs_dist, STEP_REWARD, PICKUP_REWARD, DELIVER_REWARD, UP, DOWN, LEFT, RIGHT)

SIZE = WarehouseEnv.SIZE
WINDOW = 50                                    # moving-average window


def optimal_return():
    """Return of the shortest route, worked out from the layout (24 on this grid)."""
    d1 = bfs_dist(SIZE, WarehouseEnv.OBSTACLES, WarehouseEnv.START, WarehouseEnv.PACKAGE)
    d2 = bfs_dist(SIZE, WarehouseEnv.OBSTACLES, WarehouseEnv.PACKAGE, WarehouseEnv.DELIVERY)
    return (d1 - 1) * STEP_REWARD + PICKUP_REWARD + (d2 - 1) * STEP_REWARD + DELIVER_REWARD


def measure_random_baseline(n_episodes=100):
    """Average return of a bot that moves at random (own random generator: training is unaffected)."""
    rng = random.Random(0)
    env = WarehouseEnv()
    totals = []
    for _ in range(n_episodes):
        env.reset()
        total, done = 0, False
        while not done:
            _, reward, done = env.step(rng.choice([UP, DOWN, LEFT, RIGHT]))
            total += reward
        totals.append(total)
    return float(np.mean(totals))


OPTIMAL_RETURN = optimal_return()              # the shortest route's return
RANDOM_BASELINE = measure_random_baseline()    # measured on THIS grid


def moving_average(values, window):
    """Average of each run of `window` neighbouring values: smooths out the noise."""
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def draw_curve(ax, returns):
    """Reward per episode (faint), its moving average (bold) and epsilon (second axis)."""
    episodes = np.arange(len(returns))
    ax.plot(episodes, returns, color="lightsteelblue", linewidth=0.5, label="return per episode")
    window = WINDOW if len(returns) < 5000 else 200
    smooth = moving_average(returns, window)
    ax.plot(np.arange(window - 1, len(returns)), smooth, color="tab:blue", linewidth=2,
            label=f"moving average ({window} episodes)")
    ax.axhline(OPTIMAL_RETURN, color="tab:green", linestyle="--", label=f"best possible ({OPTIMAL_RETURN})")
    ax.axhline(RANDOM_BASELINE, color="tab:red", linestyle="--", label=f"random agent ({RANDOM_BASELINE:.0f})")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("Training progress")

    eps_ax = ax.twinx()
    eps_ax.plot(episodes, [qlearning.epsilon_for(e) for e in episodes],
                color="tab:orange", linewidth=1.5, label="epsilon")
    eps_ax.set_ylabel("epsilon")
    eps_ax.set_ylim(0, 1.05)

    lines = ax.get_legend_handles_labels()
    eps_lines = eps_ax.get_legend_handles_labels()
    ax.legend(lines[0] + eps_lines[0], lines[1] + eps_lines[1], loc="center right", fontsize=8)


def draw_policy(ax, q_table, has_package, title):
    """Draw the grid and an arrow in each cell for the action with the highest Q-value."""
    big = SIZE > 20
    step = 5 if big else 1                      # on a big grid label every 5th row and column
    ticks = list(range(0, SIZE, step))
    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)                       # flipped so row 0 is at the top
    ax.set_aspect("equal")
    ax.set_xticks([c + 0.5 for c in ticks])
    ax.set_xticklabels(ticks, fontsize=7 if big else 10)
    ax.set_yticks([r + 0.5 for r in ticks])
    ax.set_yticklabels(ticks, fontsize=7 if big else 10)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_title(title, pad=22)

    edge_width = 0.3 if big else 1.0
    head = 0.35 if big else 0.18                # arrowhead size, bigger on a big grid so it stays visible
    for r in range(SIZE):
        for c in range(SIZE):
            cell = (r, c)
            if cell in WarehouseEnv.OBSTACLES:
                colour = "#444444"
            elif cell == WarehouseEnv.DELIVERY:
                colour = "gold"
            elif cell == WarehouseEnv.PACKAGE:
                colour = "lightgreen"
            else:
                colour = "white"
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=colour, edgecolor="gray", linewidth=edge_width))

            if cell in WarehouseEnv.OBSTACLES or cell == WarehouseEnv.DELIVERY:
                continue
            state = (r * SIZE + c) * 2 + int(has_package)    # same formula as env.encode()
            q_values = q_table[state]
            if q_values.max() == q_values.min():
                continue                       # all four values equal: the bot never learned anything here
            d_row, d_col = MOVES[int(np.argmax(q_values))]
            ax.arrow(c + 0.5 - d_col * 0.2, r + 0.5 - d_row * 0.2,
                     d_col * 0.4, d_row * 0.4,
                     head_width=head, color="tab:blue", length_includes_head=True)


def main():
    if "--save" in sys.argv:
        plt.switch_backend("Agg")
    random.seed(qlearning.SEED)
    q_table, returns = qlearning.train()

    side = 10 if SIZE <= 20 else 18
    fig = plt.figure(figsize=(side, side * 0.9))
    grid = fig.add_gridspec(2, 2)
    draw_curve(fig.add_subplot(grid[0, :]), returns)
    draw_policy(fig.add_subplot(grid[1, 0]), q_table, False, "Best move: bot WITHOUT the package")
    draw_policy(fig.add_subplot(grid[1, 1]), q_table, True, "Best move: bot WITH the package")
    fig.tight_layout()

    if "--save" in sys.argv:
        path = sys.argv[sys.argv.index("--save") + 1]
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
