"""Warehouse environment: the world and its rules. No learning happens here."""
 
# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}
# (row change, column change) for each action
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
 
# Rewards
STEP_REWARD = -1       # normal move
COLLISION_REWARD = -100  # wall or shelf
PICKUP_REWARD = 20     # replaces the -1 on the pickup move
DELIVER_REWARD = 100   # replaces the -1 on the delivery move

class WarehouseEnv:
    """9x9 warehouse. Coordinates are (row, col), origin top-left.

    State = (row, col, has_package) -> 81 * 2 = 162 states (134 of them can really happen).
    """

    SIZE = 9
    START = (0, 0)
    PACKAGE = (2, 6)
    DELIVERY = (8, 8)
    OBSTACLES = {(0, 3), (1, 3), (2, 3),          # shelf A: top, vertical
                 (4, 1), (4, 2), (4, 3),          # shelf B: horizontal
                 (3, 6), (4, 6), (5, 6),          # shelf C: under the package
                 (6, 3), (6, 4),                  # shelf D
                 (7, 5), (7, 6)}                  # shelf E: near the bay
    MAX_STEPS = 200

    n_actions = 4
    n_states = SIZE * SIZE * 2
    
    def __init__(self, step_reward=STEP_REWARD):
        self.step_reward = step_reward     # cost of a normal move; Day 3 also tests 0
        self.reset()
    # ---- state encoding -------------------------------------------------
    def encode(self):
        """Turn (row, col, has_package) into one integer 0..49."""
        row, col = self.pos
        return (row * self.SIZE + col) * 2 + int(self.has_package)
 
    def decode(self, state):
        """Inverse of encode(): integer -> (row, col, has_package)."""
        cell, has = divmod(state, 2)
        row, col = divmod(cell, self.SIZE)
        return row, col, bool(has)
 
    # ---- core API -------------------------------------------------------
    def reset(self):
        """Start a new episode. Returns the starting state."""
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
            # Collision: bot stays where it is
            reward = COLLISION_REWARD
        else:
            self.pos = new_pos
            reward = self.step_reward #was : reward = STEP_REWARD
            if self.pos == self.PACKAGE and not self.has_package:
                self.has_package = True      # pickup happens only once
                reward = PICKUP_REWARD
            elif self.pos == self.DELIVERY and self.has_package:
                self.delivered = True
                reward = DELIVER_REWARD
 
        self.steps += 1
        # done = delivered (true terminal) or step limit reached (timeout)
        done = self.delivered or self.steps >= self.MAX_STEPS
        return self.encode(), reward, done
 
    # ---- display --------------------------------------------------------
    def render(self):
        """Print the grid. B = bot, b = bot carrying package."""
        lines = ["    " + " ".join(str(c) for c in range(self.SIZE))]
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
            lines.append(f"{r} | " + " ".join(cells))
        print("\n".join(lines))
 



