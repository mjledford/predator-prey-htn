import os
# Action codes
DO_NOTHING, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4

# Local observation cell codes
EMPTY, WALL, PRED, PREY = 0, 1, 2, 3

# Directional offsets (dx, dy)
DIRS = {
    UP:    (0, -1),
    DOWN:  (0, 1),
    LEFT:  (-1, 0),
    RIGHT: (1, 0),
}

# Useful iteration order
ORDERED_DIRS = [UP, DOWN, LEFT, RIGHT]

ACTION_NAMES = {
    0: "STAY",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
}

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(ROOT, "figs")
os.makedirs(FIG_DIR, exist_ok=True)