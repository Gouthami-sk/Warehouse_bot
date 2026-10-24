"""Warehouse environment: the world and its rules. No learning happens here.

The warehouse is a 50 x 50 grid. The bot starts in the top-left corner, the package sits at
(12, 37) and the delivery bay is the bottom-right corner. Shelves are generated with a fixed
layout seed, so every run sees the same warehouse, and a route to the package and on to the
bay is always kept open.
"""
import random
from collections import deque

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}
# (row change, column change) for each action
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Rewards
STEP_REWARD = -1         # normal move
COLLISION_REWARD = -100  # wall or shelf
PICKUP_REWARD = 20       # replaces the -1 on the pickup move
DELIVER_REWARD = 100     # replaces the -1 on the delivery move

GRID_SIZE = 50


def bfs_dist(size, obstacles, src, dst):
    """Length of the shortest path src -> dst avoiding shelves (None if unreachable)."""
    seen, queue = {src: 0}, deque([src])
    while queue:
        cell = queue.popleft()
        if cell == dst:
            return seen[cell]
        for d_row, d_col in MOVES.values():
            nxt = (cell[0] + d_row, cell[1] + d_col)
            if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in obstacles and nxt not in seen:
                seen[nxt] = seen[cell] + 1
                queue.append(nxt)
    return None


def make_layout(size):
    """Returns (start, package, delivery, obstacles, max_steps) for a grid of this size."""
    start, delivery = (0, 0), (size - 1, size - 1)
    scale = (size - 1) / 8
    package = (round(2 * scale), round(6 * scale))
    protected = {start, package, delivery}
    rng = random.Random(1234)
    obstacles = set()
    target = int(0.10 * size * size)                       # about 10% of the cells are shelf
    longest = max(3, size // 8)
    tries = 0
    while len(obstacles) < target and tries < 100000:
        tries += 1
        row, col = rng.randrange(size), rng.randrange(size)
        length = rng.randint(2, longest)
        d_row, d_col = (0, 1) if rng.random() < 0.5 else (1, 0)
        seg = {(row + i * d_row, col + i * d_col) for i in range(length)}
        if any(not (0 <= r < size and 0 <= c < size) for r, c in seg):
            continue
        if seg & protected or seg & obstacles:
            continue
        trial = obstacles | seg
        if bfs_dist(size, trial, start, package) is None or bfs_dist(size, trial, package, delivery) is None:
            continue                                        # never wall the bot in
        obstacles = trial
    # enough steps for a bumbling bot to still finish: a few times the shortest route
    return start, package, delivery, obstacles, max(200, 6 * size)


_START, _PACKAGE, _DELIVERY, _OBSTACLES, _MAX_STEPS = make_layout(GRID_SIZE)


class WarehouseEnv:
    """Square warehouse. Coordinates are (row, col), origin top-left.

    State = (row, col, has_package) -> SIZE * SIZE * 2 states.
    """

    SIZE = GRID_SIZE
    START = _START
    PACKAGE = _PACKAGE
    DELIVERY = _DELIVERY
    OBSTACLES = _OBSTACLES
    MAX_STEPS = _MAX_STEPS

    n_actions = 4
    n_states = SIZE * SIZE * 2

    def __init__(self, step_reward=STEP_REWARD):
        self.step_reward = step_reward
        self.reset()

    # ---- state encoding -------------------------------------------------
    def encode(self):
        row, col = self.pos
        return (row * self.SIZE + col) * 2 + int(self.has_package)

    def decode(self, state):
        cell, has = divmod(state, 2)
        row, col = divmod(cell, self.SIZE)
        return row, col, bool(has)

    # ---- core API -------------------------------------------------------
    def reset(self):
        self.pos = self.START
        self.has_package = False
        self.steps = 0
        self.delivered = False
        return self.encode()

    def step(self, action):
        """Apply one action. Returns (state, reward, done)."""
        row, col = self.pos
        d_row, d_col = MOVES[action]
        new_pos = (row + d_row, col + d_col)

        out_of_bounds = not (0 <= new_pos[0] < self.SIZE and 0 <= new_pos[1] < self.SIZE)
        if out_of_bounds or new_pos in self.OBSTACLES:
            reward = COLLISION_REWARD
        else:
            self.pos = new_pos
            reward = self.step_reward
            if self.pos == self.PACKAGE and not self.has_package:
                self.has_package = True
                reward = PICKUP_REWARD
            elif self.pos == self.DELIVERY and self.has_package:
                self.delivered = True
                reward = DELIVER_REWARD

        self.steps += 1
        done = self.delivered or self.steps >= self.MAX_STEPS
        return self.encode(), reward, done

    # ---- display --------------------------------------------------------
    def render(self):
        """Print the grid. B = bot, b = bot carrying package."""
        lines = ["    " + " ".join(str(c % 10) for c in range(self.SIZE))]
        for r in range(self.SIZE):
            cells = []
            for c in range(self.SIZE):
                cell = (r, c)
                if cell == self.pos:
                    ch = "b" if self.has_package else "B"
                elif cell in self.OBSTACLES:
                    ch = "X"
                elif cell == self.PACKAGE and not self.has_package:
                    ch = "P"
                elif cell == self.DELIVERY:
                    ch = "D"
                else:
                    ch = "."
                cells.append(ch)
            lines.append(f"{r:2d} | " + " ".join(cells))
        print("\n".join(lines))
