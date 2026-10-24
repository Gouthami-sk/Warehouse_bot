"""Click-to-start test bot: click any square, and the bot starts there and plays using only
the saved Q-table (epsilon = 0: no exploring, always the highest Q-value).

Run:  python test_bot_click.py                  uses q_table.npy
      python test_bot_click.py q_fixed.npy      uses another saved table

In the window:  click a square = start the bot there
                c             = toggle "carrying the package" (then click a square)
                q             = quit
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from env import WarehouseEnv
from test_bot import run, Q_FILE, MAX_FRAMES, INTERVAL   # run() plays one greedy game from any square
from visual import draw                                  # draws one frame: grid, shelves, P, D, bot


def square_problem(cell, carrying):
    """Why the bot cannot start on this square (None if it can)."""
    if cell in WarehouseEnv.OBSTACLES:
        return "that square is a shelf"
    if cell == WarehouseEnv.DELIVERY and carrying:
        return "the bay with the package is already a delivery: pick another square"
    if cell == WarehouseEnv.PACKAGE and not carrying:
        return "the package square only works while carrying (press c, then click again)"
    return None


def main():
    q_file = sys.argv[1] if len(sys.argv) > 1 else Q_FILE
    try:
        q_table = np.load(q_file)
    except FileNotFoundError:
        sys.exit(f"{q_file} not found. Check the name with: ls *.npy")

    me = {"carrying": False, "timer": None, "i": 0, "shown": [], "verdict": ""}
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.subplots_adjust(top=0.80)

    def show(frame, note, trail=None):
        draw(ax, frame, trail)                   # grid, shelves, P, D, bot, trail (sets its own title)
        mode = "YES" if me["carrying"] else "no"
        ax.set_title(ax.get_title() + f"\n{note}\ntable: {q_file}   |   carrying: {mode}\n"
                     "click a square to start   |   c = toggle carrying   |   q = quit",
                     pad=22, fontsize=9)
        fig.canvas.draw_idle()

    def preview(note):
        frame = {"pos": WarehouseEnv.START, "has": me["carrying"], "step": 0, "action": None,
                 "reward": None, "total": 0, "delivered": False}
        show(frame, note)

    def stop():
        if me["timer"] is not None:
            me["timer"].stop()
            me["timer"] = None

    def tick():
        i = me["i"]
        last = i >= len(me["shown"]) - 1
        trail = [f["pos"] for f in me["shown"][:i + 1]]   # the path walked so far
        show(me["shown"][i], me["verdict"] if last else "running...  (click another square to restart)", trail)
        if last:
            stop()
        else:
            me["i"] = i + 1

    def on_click(event):
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and str(getattr(toolbar, "mode", "")) != "":
            return                                        # zoom or pan tool is active: ignore the click
        if event.inaxes is not ax or event.xdata is None:
            return
        cell = (int(event.ydata), int(event.xdata))       # (row, col); the y-axis points down
        stop()
        problem = square_problem(cell, me["carrying"])
        if problem:
            preview(f"{cell}: {problem}")
            return
        frames = run(q_table, cell, me["carrying"])       # the whole greedy game from that square
        final = frames[-1]
        if final["delivered"]:
            me["verdict"] = f"from {cell}: delivered in {final['step']} steps, return {final['total']}"
        else:
            me["verdict"] = (f"from {cell}: NOT delivered, stuck until the {final['step']}-step limit "
                             f"(return {final['total']})")
        me["shown"] = frames[:MAX_FRAMES + 1]
        me["i"] = 0
        tick()                                            # show the first frame right away
        me["timer"] = fig.canvas.new_timer(interval=INTERVAL)
        me["timer"].add_callback(tick)
        me["timer"].start()

    def on_key(event):
        if event.key == "c":
            stop()
            me["carrying"] = not me["carrying"]
            preview("now click a square to start from")
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    preview("click a square to start the bot there")
    plt.show()


if __name__ == "__main__":
    main()
