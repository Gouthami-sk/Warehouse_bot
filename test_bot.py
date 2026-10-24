"""Test bot: load the saved Q-table and watch the bot work using only what it learned
(epsilon = 0: no exploring, always the highest Q-value).

Run:  python test_bot.py                              start cell, no package
      python test_bot.py --from 0 0 --carrying        another cell (row col), already holding the package
      python test_bot.py --save test.gif              save a GIF instead of opening a window
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from env import WarehouseEnv
from visual import draw                        # draws one frame: grid, shelves, P, D, bot

Q_FILE = "q_table.npy"                         # the file qlearning.py saved
MAX_FRAMES = WarehouseEnv.MAX_STEPS            # show the whole game (a stuck bot loops to the step limit)
INTERVAL = 100                                 # milliseconds per move


def run(q_table, start_pos, has_package):
    """Play one game greedily from the given square. Returns one dict per step, like visual.record()."""
    env = WarehouseEnv()
    env.reset()
    env.pos = start_pos                        # put the bot where we want it
    env.has_package = has_package
    state = env.encode()
    total, done = 0, False
    frames = [{"pos": env.pos, "has": env.has_package, "step": 0, "action": None,
               "reward": None, "total": 0, "delivered": False}]
    while not done:                            # the step limit also ends the loop
        action = int(np.argmax(q_table[state]))        # the best-scored move on this page
        state, reward, done = env.step(action)
        total += reward
        frames.append({"pos": env.pos, "has": env.has_package, "step": env.steps,
                       "action": action, "reward": reward, "total": total,
                       "delivered": env.delivered})
    return frames


def main():
    args = sys.argv[1:]
    start = WarehouseEnv.START
    if "--from" in args:
        i = args.index("--from")
        start = (int(args[i + 1]), int(args[i + 2]))
    carrying = "--carrying" in args
    save = args[args.index("--save") + 1] if "--save" in args else None

    row, col = start                           # reject squares the bot can never stand on
    if not (0 <= row < WarehouseEnv.SIZE and 0 <= col < WarehouseEnv.SIZE) or start in WarehouseEnv.OBSTACLES:
        sys.exit(f"{start} is off the grid or a shelf. Pick another square.")
    if start == WarehouseEnv.DELIVERY or (start == WarehouseEnv.PACKAGE and not carrying):
        sys.exit(f"{start} is not a valid starting state. Pick another square.")

    try:
        q_table = np.load(Q_FILE)
    except FileNotFoundError:
        sys.exit(f"{Q_FILE} not found. Run python qlearning.py first.")

    frames = run(q_table, start, carrying)
    last = frames[-1]
    verdict = "delivered" if last["delivered"] else "NOT delivered (stuck until the step limit)"
    print(f"From {start}, carrying={carrying}: {last['step']} steps, return {last['total']}, {verdict}")
    shown = frames[:MAX_FRAMES + 1]

    if save:
        plt.switch_backend("Agg")              # draw off-screen: no window needed when saving
    fig, ax = plt.subplots(figsize=(9, 9.8))
    anim = FuncAnimation(fig, lambda i: draw(ax, shown[i], [f["pos"] for f in shown[:i + 1]]),
                         frames=len(shown), interval=INTERVAL, repeat=False)
    if save:
        anim.save(save, writer=PillowWriter(fps=1000 // INTERVAL))
        print(f"Saved {save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
