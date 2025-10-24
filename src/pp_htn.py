import gtpyhop

# env action ids
DO_NOTHING, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4

# local observation cell codes
EMPTY, WALL, PRED, PREY = 0, 1, 2, 3

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
    size = 2 * obs_dim + 1
    center = obs_dim

    # find all visible prey cells in the (2*obs_dim+1)^2 egocentric window
    prey_idxs = [k for k, v in enumerate(obs) if v == PREY]
    if not prey_idxs:
        return DO_NOTHING  

    # pick the nearest prey by Manhattan distance
    best_md = 10**9
    best_dx = best_dy = 0
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


# METHODS for 'choose_action'

def m_chase_if_visible(state, agent_id):
    """If prey visible in local obs, chase it; else fail so next method can try"""
    print("[DEBUG] Chase if visible")
    obs = state.obs[agent_id]
    action = action_from_obs(obs, state.obs_dim)
    if action == DO_NOTHING:
        return False
    return [("do", agent_id, action)]

def m_patrol_if_not_visible(state, agent_id):
    """Fallback: simple patrol/right-bias when nothing visible."""
    # Example: drift right, otherwise down (modify to taste)
    # You could also keep stateful patrol direction per agent in state.patrol_dir
    print(f"[DEBUG] Patrol if not visible")
    return [("do", agent_id, RIGHT)]


# REGISTER domain
domain_name = "pp_htn"
gtpyhop.Domain(domain_name)
gtpyhop.declare_actions(do)
gtpyhop.declare_task_methods("choose_action", m_chase_if_visible, m_patrol_if_not_visible)