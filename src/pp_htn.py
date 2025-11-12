import gtpyhop
import random

random.seed(42)  # for reproducibility
# env action ids
DO_NOTHING, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4

# local observation cell codes
EMPTY, WALL, PRED, PREY = 0, 1, 2, 3

# offsets for neighbors in the obs window (dx, dy)
# if agent is at (X,Y), then neighbor at (X+dx, Y+dy)
# Note: +x is right, +y is down
# For example (5x5 observation window), if agent is at (2,2) then UP target cell/neighbor is at (2,1), DOWN (2,3), LEFT (1,2), RIGHT (3,2)
DIRS = {
    UP:    (0, -1),
    DOWN:  (0, 1),
    LEFT:  (-1, 0),
    RIGHT: (1, 0),
}

ORDERED_DIRS = [UP, DOWN, LEFT, RIGHT]

DEBUG = False

# primitive action
def do(state, agent_id, action_id):
    """
    Record the chosen discrete action for agent_id
    The simiulator (gym env) will eventually execute it outsideGTPyhop
    """
    if not hasattr(state, "last_action"):
        state.last_action = {}
    state.last_action[agent_id] = int(action_id)
    return state # returning the (modified) state signals success

# helper function to pick action from local obs
def action_from_obs(obs, obs_dim):
    """
    Greedy chase if any PREY cells visible in local obs window;
    """
    size = 2 * obs_dim + 1
    center = obs_dim

    # find all visible prey cells in the (2*obs_dim+1)^2 egocentric window
    prey_idxs = [k for k, v in enumerate(obs) if v == PREY]
    # if none visible, do nothing
    if not prey_idxs:
        return DO_NOTHING  

    # pick the nearest prey by Manhattan distance
    best_md = 10**9
    best_dx = 0
    best_dy = 0
    for k in prey_idxs:
        r, c = divmod(k, size)
        dx = c - center         # +x = right
        dy = r - center         # +y = down
        md = abs(dx) + abs(dy)  
        if md < best_md:
            best_md = md
            best_dx, best_dy = dx, dy

    dx, dy = best_dx, best_dy   

    # move greedily to reduce distance; break ties horizontally
    if abs(dx) >= abs(dy):
        if dx > 0:  return RIGHT
        if dx < 0:  return LEFT
        if dy > 0:  return DOWN
        if dy < 0:  return UP
        return DO_NOTHING
    else:
        if dy > 0:  return DOWN
        if dy < 0:  return UP
        if dx > 0:  return RIGHT
        if dx < 0:  return LEFT
        return DO_NOTHING

def legal_moves_from_obs(obs, obs_dim):
    """
    Given local obs, return list of legal move action_ids (no WALL)
    """
    size = 2 * obs_dim + 1
    center = obs_dim
    legal_moves = []
    for a in ORDERED_DIRS:
        dx, dy = DIRS[a]
        nx, ny = center + dx, center + dy
        # bounds check (should always be in-bounds for local obs window)
        if not (0 <=nx < size and 0 <= ny < size):
            continue
        idx = ny * size + nx
        if obs[idx] != WALL:
            legal_moves.append(a)
    return legal_moves

def _default_fallback_action():
    return DO_NOTHING

def _default_fallback_plan(agent_id):
    return [("do", agent_id, DO_NOTHING)]     
# METHODS for 'choose_action'

def m_chase_if_visible(state, agent_id):
    """If prey visible in local obs, chase it; else fail so next method can try"""
    
    if DEBUG:
        print("[DEBUG] Chase (prey visible)")
        
    obs = state.obs[agent_id]
    action = action_from_obs(obs, state.obs_dim)
    if action == DO_NOTHING:
        return False
    print(f"[INFO] Agent {agent_id} attempting to chase prey with action {action}")
    return [("do", agent_id, action)]

def m_patrol_if_not_visible(state, agent_id):
    """
    Fallback patrol:
    - compute legal directions (not into WALL)
    - optionally include or exclude the last executed action (prev_action)
      depending on the boolean flag state.keep_prev_action
    - pick uniformly at random among remaining legal directions
    """
    if DEBUG:
        print(f"[DEBUG] Patrol (prey not visible)")
    
    obs = state.obs[agent_id]
    # smart planner determines legal moves from local obs (don't go left if the wall is left, therefore choose up, down, right aka legal moves)
    legal_moves = legal_moves_from_obs(obs, state.obs_dim)
    
    if not legal_moves:
        return _default_fallback_plan(agent_id) # default = [("do", agent_id, DO_NOTHING)]
    
    prev = getattr(state, "prev_action", DO_NOTHING)
    keep_prev = bool(getattr(state, "keep_prev_action", True))
    
    if not keep_prev and prev in legal_moves:
        # user requested to to filter out the previous direction if possible
        filtered = [a for a in legal_moves if a != prev]
        if filtered:
            legal_moves = filtered
    
    
    rng = getattr(state, "rng", random)
    
    # Choose default RIGHT (original progress report 1 basic action to move right)
    #action = RIGHT
    # Choose randomly among legal moves
    action = rng.choice(legal_moves)
    print(f"[INFO] Agent {agent_id} attempting to patrol with action {action}")
    
    return [("do", agent_id, action)]


# REGISTER domain
domain_name = "pp_htn"
gtpyhop.Domain(domain_name)
gtpyhop.declare_actions(do)
gtpyhop.declare_task_methods("choose_action", m_chase_if_visible, m_patrol_if_not_visible)