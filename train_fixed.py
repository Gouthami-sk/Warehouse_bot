"""Train a bot with a FIXED epsilon (no decay) and save its Q-table to q_fixed.npy.

Run:  python train_fixed.py
"""
import numpy as np

import qlearning
from env import WarehouseEnv
from test_bot import run                        # plays one greedy game (no exploring) from any square

EPSILON = 0.95                                  # explore 95% of the time, for the whole training
SAVE_FILE = "q_fixed.npy"                       # separate file: q_table.npy stays untouched


def main():
    # same number of episodes as the decay run, so the comparison is fair
    q_table, returns = qlearning.train(eps_start=EPSILON, eps_end=EPSILON,
                                       n_episodes=qlearning.N_EPISODES)
    np.save(SAVE_FILE, q_table)
    tail = max(200, qlearning.N_EPISODES // 10)
    print(f"Grid {WarehouseEnv.SIZE}x{WarehouseEnv.SIZE}: trained {len(returns)} episodes "
          f"with fixed epsilon {EPSILON}")
    print(f"Training average of the last {tail} episodes: {returns[-tail:].mean():.1f}")
    last = run(q_table, WarehouseEnv.START, False)[-1]    # one game from the start cell, no exploring
    print(f"Greedy test from the start: {last['step']} steps, return {last['total']}, delivered {last['delivered']}")
    print(f"Saved {SAVE_FILE}")


if __name__ == "__main__":
    main()
