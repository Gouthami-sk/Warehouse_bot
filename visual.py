"""Draw the warehouse with Matplotlib and animate the bot.

Run:  python visual.py                  watch a random agent
      python visual.py --save run.gif   save a GIF instead of opening a window
"""
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

from env import WarehouseEnv, ACTION_NAMES, COLLISION_REWARD, UP, DOWN, LEFT, RIGHT

SIZE = WarehouseEnv.SIZE
BIG = SIZE > 20                                # big grids get thinner lines, smaller text, a bigger bot


def record(actions):
    """Run the actions in the environment and remember what happened at each step."""
    env = WarehouseEnv()
    env.reset()
    total = 0
    frames = [{"pos": env.pos, "has": env.has_package, "step": 0, "action": None,
               "reward": None, "total": 0, "delivered": False}]
    for action in actions:
        _, reward, done = env.step(action)
        total += reward
        frames.append({"pos": env.pos, "has": env.has_package, "step": env.steps,
                       "action": action, "reward": reward, "total": total,
                       "delivered": env.delivered})
        if done:
            break
    return frames


def label(ax, row, col, text, color="black", size=None):
    """Write a letter in the middle of a cell."""
    if size is None:
        size = 9 if BIG else 16
    ax.text(col + 0.5, row + 0.5, text, ha="center", va="center",
            color=color, fontsize=size, fontweight="bold", zorder=4)


def draw(ax, f, trail=None):
    """Draw one frame: the grid, the shelves, P, D and the bot.

    trail: optional list of (row, col) cells the bot has already visited, drawn as a faint line.
    """
    ax.clear()

    # the floor is one image (fast, even for 50x50): white floor, dark shelves, gold bay, green package
    floor = np.ones((SIZE, SIZE, 3))
    for r, c in WarehouseEnv.OBSTACLES:
        floor[r, c] = (0.27, 0.27, 0.27)
    dr, dc = WarehouseEnv.DELIVERY
    floor[dr, dc] = (1.0, 0.84, 0.0)
    if not f["has"]:                           # the package disappears once picked up
        pr, pc = WarehouseEnv.PACKAGE
        floor[pr, pc] = (0.56, 0.93, 0.56)
    ax.imshow(floor, extent=(0, SIZE, SIZE, 0), interpolation="nearest", zorder=0)

    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)                       # flipped so row 0 is at the top
    ax.set_aspect("equal")

    step = 5 if BIG else 1                     # label every 5th row and column on a big grid
    ticks = list(range(0, SIZE, step))
    ax.set_xticks([c + 0.5 for c in ticks])
    ax.set_xticklabels(ticks, fontsize=8 if BIG else 10)
    ax.set_yticks([r + 0.5 for r in ticks])
    ax.set_yticklabels(ticks, fontsize=8 if BIG else 10)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(0, SIZE + 1), minor=True)       # thin grid lines between cells
    ax.set_yticks(np.arange(0, SIZE + 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linewidth=0.25 if BIG else 0.8, zorder=1)
    ax.tick_params(which="minor", length=0)

    if not BIG:                                # on a big grid the dark squares already say "shelf"
        for r, c in WarehouseEnv.OBSTACLES:
            label(ax, r, c, "X", color="white")

    label(ax, dr, dc, "D")
    sr, sc = WarehouseEnv.START
    label(ax, sr, sc, "S", color="gray")
    if not f["has"]:
        pr, pc = WarehouseEnv.PACKAGE
        label(ax, pr, pc, "P")

    if trail and len(trail) > 1:               # where the bot has been so far
        ax.plot([c + 0.5 for _, c in trail], [r + 0.5 for r, _ in trail],
                color="tab:blue", alpha=0.45, linewidth=1.2 if BIG else 2, zorder=2)

    r, c = f["pos"]
    colour = "tab:red" if f["reward"] == COLLISION_REWARD else "tab:blue"
    radius = 0.8 if BIG else 0.3
    ax.add_patch(Circle((c + 0.5, r + 0.5), radius, color=colour, zorder=5))
    if f["has"]:                               # small green box on the bot = carrying
        half = 0.4 if BIG else 0.12
        ax.add_patch(Rectangle((c + 0.5 - half, r + 0.5 - half), 2 * half, 2 * half,
                               facecolor="tab:green", zorder=6))

    if f["action"] is None:
        title = "Start"
    else:
        title = (f"step {f['step']}: {ACTION_NAMES[f['action']]}   "
                 f"reward {f['reward']}   total {f['total']}")
    if f["delivered"]:
        title += "   DELIVERED!"
    ax.set_title(title, pad=22)


def replay(actions, interval=500, save=None):
    """Animate a list of actions. Opens a window, or saves a GIF if save is a filename."""
    frames = record(actions)
    if save:
        plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(9, 9.8) if BIG else (5, 5.5))
    anim = FuncAnimation(fig, lambda i: draw(ax, frames[i], [fr["pos"] for fr in frames[:i + 1]]),
                         frames=len(frames), interval=interval, repeat=False)
    if save:
        anim.save(save, writer=PillowWriter(fps=1000 // interval))
        print(f"Saved {save}")
    else:
        plt.show()
    return anim


if __name__ == "__main__":
    save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    actions = [random.choice([UP, DOWN, LEFT, RIGHT]) for _ in range(WarehouseEnv.MAX_STEPS)]
    replay(actions, interval=150, save=save)
